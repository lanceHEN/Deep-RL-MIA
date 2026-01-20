"""
This module contains implementations for attack classifiers trainers in both individual
and collective modes.
"""

from abc import ABC, abstractmethod
from typing import ArrayLike

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pytorch_tcn import TCN
from torchvision.models import resnet18


class AttackTrainer(ABC):
    """
    An abstracted version of attack trainers for individual or collective modes
    to de-duplicate common code between them. The individual and collective
    modes should each extend this.

    Attributes:
        action_dim (int): Dimension of action space.
        T_max (int): Max trajectory length.
    """

    def __init__(
        self,
        action_dim: int,
        T_max: int,
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
        
    @property
    @abstractmethod
    def classifier(self) -> nn.Module:
        pass
        
    @property
    @abstractmethod
    def criterion(self)-> nn.Module:
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
        stacked_trajectories: torch.Tensor,
        train_labels: torch.Tensor,
        epochs: int,
        batch_size: int,
    ) -> None:
        """
        Trains the classification model for the given number of epochs,
        on the given stacked trajectory pairs and labels.

        Args:
            stacked_trajectories (torch.Tensor): Vertically stacked query and
                policy output trajectories. Either [n_samples, (2 * action_dim), T_max]
                or [n_samples, 2 * action_dim, T_max, minibatch_size, depending on
                whether in individual or collective mode.
            train_labels (torch.Tensor): [B,] tensor of labels for the trajectory
                stacks.
            epochs (int): Number of epochs to train the classifier.
            batch_size (int): Batch size for stochastic gradient descent.
        """
        
        train_loader = self._make_dataloader(
            stacked_trajectories, train_labels, batch_size
        )

        self.classifier.train()

        for _ in range(epochs):
            for inputs, labels in train_loader:
                self.optimizer.zero_grad()

                out = self.classifier(inputs)

                loss = self.criterion(out, labels)

                loss.backward()
                # Gradient clipping
                torch.nn.utils.clip_grad_value_(
                    self.classifier.parameters(), clip_value=self.grad_clip
                )

                self.optimizer.step()

    @abstractmethod
    def forward(self, x):
        pass

    def _make_dataloader(
        stacked_trajectories: torch.Tensor, train_labels: torch.Tensor, batch_size: int
    ) -> DataLoader:
        """
        Produces a torch DataLoader over the given trajectories and labels.
        """
        dataset = TensorDataset(stacked_trajectories, train_labels)
        return DataLoader(dataset, batch_size=batch_size)


class IndividualAttackTrainer(AttackTrainer):
    """
    Used to train the individual attack classifier, i.e. IndividualAttackClassifier.

    Attributes:
        action_dim (int): Dimension of action space.
        T_max (int): Max trajectory length.
        classifier (nn.Module): The attack classifier to train.
        criterion (nn.Module): BCE Loss.
        optimizer (torch.optim.Optimizer): Adam optimizer.
        scheduler (torch.optim.lr_scheduler.LRScheduler): LR scheduler to decay LR
            every scheduler_steps.
        grad_clip (float): Threshold value for gradient clipping.
    """

    def __init__(
        self,
        action_dim: int,
        T_max: int,
        num_channels: ArrayLike = [600, 600],
        kernel_size: int = 3,
        dropout: float = 0.45,
        lr: float = 0.0003,
        grad_clip: float = 0.35,
        scheduler_step=100,
        scheduler_decay=0.1,
    ):
        """
        Initializes an IndividualAttackTrainer with the given parameters.

        Args:
            action_dim (int): Dimension of action space.
            T_max (int): Max trajectory length.
            num_channels (ArrayLike): number of feature channels in each residual block of the network.
            kernel_size (int): Kernel size for convolutions.
            dropout (float): Dropout rate.
            lr (float): Initial learning rate.
            grad_clip (float): Threshold value for gradient clipping.
            scheduler_step (int): The learning rate will update every scheduler_step
                epochs.
            scheduler_decay (float): Learning rate decay rate.
        """
        super().__init__(
            action_dim, T_max
        )
        
        self.classifier = IndividualAttackClassifier(
            action_dim, num_channels, kernel_size, dropout
        )
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.classifier.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, scheduler_step, gamma=scheduler_decay
        )
        self.grad_clip = grad_clip
        
    @property
    def classifier(self) -> nn.Module:
        return self.classifier
        
    @property
    def criterion(self) -> nn.Module:
        return self.criterion
    
    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self.optimizer
    
    @property
    def scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        return self.scheduler

class CollectiveAttackTrainer(AttackTrainer):
    """
    Used to train the collective attack classifier, i.e. CollectiveAttackClassifier.

    Attributes:
        action_dim (int): Dimension of action space.
        T_max (int): Max trajectory length.
        classifier (nn.Module): The attack classifier to train.
        criterion (nn.Module): BCE Loss.
        optimizer (torch.optim.Optimizer): Adam optimizer.
        scheduler (torch.optim.lr_scheduler.LRScheduler): LR scheduler to decay LR
            every scheduler_steps.
        grad_clip (float): Threshold value for gradient clipping.
    """

    def __init__(
        self,
        action_dim: int,
        T_max: int,
        dropout: float = 0.0,
        lr: float = 0.0008,
        grad_clip: float = 0.35,
        scheduler_step=100,
        scheduler_decay=1.0,
    ):
        """
        Initializes an IndividualAttackTrainer with the given parameters.

        Args:
            action_dim (int): Dimension of action space.
            T_max (int): Max trajectory length.
            kernel_size (int): Kernel size for convolutions.
            dropout (float): Dropout rate.
            lr (float): Initial learning rate.
            grad_clip (float): Threshold value for gradient clipping.
            scheduler_step (int): The learning rate will update every scheduler_step
                epochs.
            scheduler_decay (float): Learning rate decay rate.
        """
        super().__init__(
            action_dim, T_max
        )
        
        self.classifier = CollectiveAttackClassifier(action_dim)
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.classifier.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, scheduler_step, gamma=scheduler_decay
        )
        self.grad_clip = grad_clip

    @property
    def classifier(self) -> nn.Module:
        return self.classifier
        
    @property
    def criterion(self) -> nn.Module:
        return self.criterion
    
    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self.optimizer
    
    @property
    def scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        return self.scheduler
    
class IndividualAttackClassifier(nn.Module):
    """
    A TCN-based classifier used for individual trajectories.
    """

    def __init__(
        self,
        action_dim: int,
        num_channels: ArrayLike = [600, 600],
        kernel_size: int = 3,
        dropout: float = 0.45,
    ):
        """
        Initializes an IndividualAttackClassifier with the given parameters.

        Args:
            action_dim (int): Dimension of action space.
            num_channels (ArrayLike): number of feature channels in each residual block of the network.
            kernel_size (int): Kernel size for convolutions.
            dropout (float): Dropout rate.
        """

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
        x = self.fc(x)  # [B,]
        return self.sigmoid(x)


class CollectiveAttackClassifier(nn.Module):
    """
    A Resnet-18-based classifier that, given pairs of batched candidate and
    output trajectories, determines if the candidate trajectories were used
    in training.
    """

    def __init__(self, action_dim: int):
        """
        Initializes a CollectiveAttackClassi for __init__
        
        :param self: Description
        :param action_dim: Description
        :type action_dim: int
        """
        input_channels = 2 * action_dim
        self.resnet = resnet18()
        # Modify conv layer to work with actions
        self.resnet.conv1 = nn.Conv2d(
            input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        self.resnet.fc = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, 2d_A, T, m]
        x = self.resnet(x)
        x = self.fc(x)
        return self.sigmoid(x)
