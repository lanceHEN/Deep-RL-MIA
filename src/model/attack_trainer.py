"""
This module contains implementations for attack classifiers trainers in both individual
and collective modes.
"""

from abc import ABC, abstractmethod
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pytorch_tcn import TCN
from torchvision.models import resnet18

from config import IndividualAttackTrainerConfig, CollectiveAttackTrainerConfig


class AttackTrainer(ABC):
    """
    An abstracted version of attack trainers for individual or collective modes
    to de-duplicate common code between them. The individual and collective
    modes should each extend this.

    Attributes:
        action_dim (int): Dimension of action space.
        T_max (int): Max trajectory length.
        classification_threshold (float): Minimum probability for a
            positive classification.
    """

    def __init__(
        self,
        action_dim: int,
        T_max: int,
        classification_threshold: float = 0.5
    ):
        """
        Initializes an AttackTrainer with the given action dimension, max
        trajectory length, and classification probability threshold.

        Args:
            action_dim (int): Dimension of action space.
            T_max (int): Max trajectory length.
            classification_threshold (float): Minimum probability for a
                positive classification.
        """
        self.action_dim = action_dim
        self.T_max = T_max
        self.classification_threshold = classification_threshold
        
    @property
    @abstractmethod
    def device(self) -> str:
        pass
    
    @property
    @abstractmethod
    def verbose(self) -> int:
        pass

    @property
    @abstractmethod
    def classifier(self) -> nn.Module:
        pass

    @property
    @abstractmethod
    def criterion(self) -> nn.Module:
        pass

    @property
    @abstractmethod
    def optimizer(self) -> torch.optim.Optimizer:
        pass

    @property
    @abstractmethod
    def scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        pass

    def train(
        self,
        stacked_trajectories: np.ndarray,
        train_labels: np.ndarray,
        epochs: int,
        batch_size: int,
    ) -> None:
        """
        Trains the classification model for the given number of epochs,
        on the given stacked trajectory pairs and labels.

        Args:
            stacked_trajectories (np.ndarray): Vertically stacked query and
                policy output trajectories. Either [n_samples, (2 * action_dim), T_max]
                or [n_samples, 2 * action_dim, T_max, minibatch_size], depending on
                whether in individual or collective mode. Note only actions are stored,
                and they don't have to be shuffled.
            train_labels (np.ndarray): [n_samples,] array of labels for the trajectory
                stacks.
            epochs (int): Number of epochs to train the classifier.
            batch_size (int): Batch size for stochastic gradient descent.
        """
        if self.verbose > 0:
            print("Training attack classifier")
        train_loader = self._make_dataloader(
            stacked_trajectories, train_labels, batch_size
        )

        self.classifier.train()

        for _ in range(epochs):
            for inputs, labels in train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                self.optimizer.zero_grad()

                out = self.classifier(inputs)
                #print(out.shape, labels.shape)
                loss = self.criterion(out, labels)

                loss.backward()
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.classifier.parameters(), max_norm=self.grad_clip
                )

                self.optimizer.step()
                
            self.scheduler.step()

        if self.verbose > 0:
            print("Finished training attack classifier")
    
    @torch.no_grad()
    def predict(self, stacked_trajectories: np.ndarray) -> np.ndarray:
        """
        Given the stacked trajectories, produces classification predictions.

        Args:
            stacked_trajectories (np.ndarray): Vertically stacked query and
                policy output trajectories. Either [n_samples, (2 * action_dim), T_max]
                or [n_samples, 2 * action_dim, T_max, minibatch_size, depending on
                whether in individual or collective mode.  Note only actions are stored.
        Returns:
            np.ndarray: [n_samples,] array of predictions for the trajectory
                stacks, using classification_threshold to round up or down
                (if prob < classification_threshold round down, else round up).
        """
        inputs = torch.from_numpy(stacked_trajectories).to(self.device)
        raw = self.classifier(inputs).cpu().numpy()
        return np.where(raw < self.classification_threshold, 0.0, 1.0)

    @staticmethod
    def _make_dataloader(
        stacked_trajectories: np.ndarray, train_labels: np.ndarray, batch_size: int
    ) -> DataLoader:
        """
        Produces a torch DataLoader shuffled over the given trajectories and labels.
        """
        dataset = TensorDataset(
            torch.from_numpy(stacked_trajectories), torch.from_numpy(train_labels).float()
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)


class IndividualAttackTrainer(AttackTrainer):
    """
    Used to train the individual attack classifier, i.e. IndividualAttackClassifier.

    Attributes:
        config (IndividualAttackTrainerConfig): Stores config info.
        action_dim (int): Dimension of action space.
        T_max (int): Max trajectory length.
        classification_threshold (float): Minimum probability for a
            positive classification.
        classifier (nn.Module): The attack classifier to train.
        criterion (nn.Module): BCE Loss.
        optimizer (torch.optim.Optimizer): Adam optimizer.
        scheduler (torch.optim.lr_scheduler.LRScheduler): LR scheduler to decay LR
            every scheduler_steps.
        grad_clip (float): Threshold value for gradient clipping.
    """

    def __init__(self, config: IndividualAttackTrainerConfig):
        """
        Initializes an IndividualAttackTrainer with the given config.

        Args:
            config (IndividualAttackTrainerConfig): Stores config info.
        """
        super().__init__(config.action_dim, config.T_max, config.classification_threshold)

        self.config = config
        
        self._device = self.config.device
        
        self._verbose = self.config.verbose

        self._classifier = IndividualAttackClassifier(
            self.config.action_dim,
            self.config.num_channels,
            self.config.kernel_size,
            self.config.dropout,
        ).to(self.device)
        self._criterion = nn.BCELoss()
        self._optimizer = torch.optim.Adam(self._classifier.parameters(), lr=self.config.lr)
        self._scheduler = torch.optim.lr_scheduler.StepLR(
            self._optimizer,
            self.config.scheduler_step,
            gamma=self.config.scheduler_decay,
        )
        self._grad_clip = self.config.grad_clip
        
    @property
    def device(self):
        return self._device
    
    @property
    def verbose(self):
        return self._verbose

    @property
    def classifier(self) -> nn.Module:
        return self._classifier

    @property
    def criterion(self) -> nn.Module:
        return self._criterion

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self._optimizer

    @property
    def scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        return self._scheduler

    @property
    def grad_clip(self) -> float:
        return self._grad_clip
    
    @device.setter
    def device(self, device):
        self._device = device
        
    @verbose.setter
    def verbose(self, verbose):
        self._verbose = verbose

    @classifier.setter
    def classifier(self, clf):
        self._classifier = clf

    @criterion.setter
    def criterion(self, crit):
        self._criterion = crit

    @optimizer.setter
    def optimizer(self, opt):
        self._optimizer = opt

    @scheduler.setter
    def scheduler(self, sched):
        self._scheduler = sched

    @grad_clip.setter
    def grad_clip(self, clip):
        self._grad_clip = clip


class CollectiveAttackTrainer(AttackTrainer):
    """
    Used to train the collective attack classifier, i.e. CollectiveAttackClassifier.

    Attributes:
        action_dim (int): Dimension of action space.
        T_max (int): Max trajectory length.
        classification_threshold (float): Minimum probability for a
            positive classification.
        classifier (nn.Module): The attack classifier to train.
        criterion (nn.Module): BCE Loss.
        optimizer (torch.optim.Optimizer): Adam optimizer.
        scheduler (torch.optim.lr_scheduler.LRScheduler): LR scheduler to decay LR
            every scheduler_steps.
        grad_clip (float): Threshold value for gradient clipping.
    """

    def __init__(self, config: CollectiveAttackTrainerConfig):
        """
        Initializes an IndividualAttackTrainer with the given config.

        Args:
            config (IndividualAttackTrainer): Stores config info.
        """
        super().__init__(config.action_dim, config.T_max, config.classification_threshold)

        self.config = config
        
        self._device = self.config.device
        
        self._verbose = self.config.verbose

        self._classifier = CollectiveAttackClassifier(self.config.action_dim).to(self.device)
        self._criterion = nn.BCELoss()
        self._optimizer = torch.optim.Adam(self._classifier.parameters(), lr=self.config.lr)
        self._scheduler = torch.optim.lr_scheduler.StepLR(
            self._optimizer,
            self.config.scheduler_step,
            gamma=self.config.scheduler_decay,
        )
        self._grad_clip = self.config.grad_clip
        
    @property
    def device(self):
        return self._device
    
    @property
    def verbose(self):
        return self._verbose

    @property
    def classifier(self) -> nn.Module:
        return self._classifier

    @property
    def criterion(self) -> nn.Module:
        return self._criterion

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self._optimizer

    @property
    def scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        return self._scheduler

    @property
    def grad_clip(self) -> float:
        return self._grad_clip
    
    @device.setter
    def device(self, device):
        self._device = device
        
    @verbose.setter
    def verbose(self, verbose):
        self._verbose = verbose

    @classifier.setter
    def classifier(self, clf):
        self._classifier = clf

    @criterion.setter
    def criterion(self, crit):
        self._criterion = crit

    @optimizer.setter
    def optimizer(self, opt):
        self._optimizer = opt

    @scheduler.setter
    def scheduler(self, sched):
        self._scheduler = sched

    @grad_clip.setter
    def grad_clip(self, clip):
        self._grad_clip = clip


class IndividualAttackClassifier(nn.Module):
    """
    A TCN-based classifier used for individual trajectories.
    """

    def __init__(
        self,
        action_dim: int,
        num_channels: List = [600, 600],
        kernel_size: int = 3,
        dropout: float = 0.45,
    ):
        """
        Initializes an IndividualAttackClassifier with the given parameters.

        Args:
            action_dim (int): Dimension of action space.
            num_channels (List): number of feature channels in each residual block of the network.
            kernel_size (int): Kernel size for convolutions.
            dropout (float): Dropout rate.
        """
        super().__init__()

        self.tcn = TCN(
            num_inputs=2 * action_dim,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        self.fc = nn.Linear(num_channels[-1], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, 2d_A, T]
        x = self.tcn(x)  # [B, num_channels[-1], T]
        x = torch.mean(x, dim=2)  # Mean over time - [B, num_channels[-1]]
        x = self.fc(x).squeeze(1)  # [B,]
        return self.sigmoid(x)


class CollectiveAttackClassifier(nn.Module):
    """
    A Resnet-18-based classifier that, given pairs of batched candidate and
    output trajectories, determines if the candidate trajectories were used
    in training.
    """

    def __init__(self, action_dim: int):
        """
        Initializes a CollectiveAttackClassifier with the given parameters.

        Args:
            action_dim (int): Dimension of action space.
        """
        super().__init__()

        input_channels = 2 * action_dim
        self.resnet = resnet18()
        # Modify conv layer to work with actions
        self.resnet.conv1 = nn.Conv2d(
            input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        self.resnet.fc = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #print(x.shape)
        # x: [B, 2d_A, T, m]
        x = self.resnet(x)
        x = x.squeeze(1)  # [B,]
        return self.sigmoid(x)
