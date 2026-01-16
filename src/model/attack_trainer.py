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


class AttackTrainer(ABC):
    """
    An abstracted version of attack trainers for individual or collective modes
    to de-duplicate common code between them. The individual and collective
    modes should each extend this.
    """

    def __init__(
        self, action_dim: int, T_max: int, classification_threshold: float = 0.5
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
        self.classification_threshold = classification_threshold
        self.classifier = None

    @abstractmethod
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
        pass

    @property
    def classifier(self):
        return self.classifier

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
        classifier (nn.Module): The attack classifier to train.
        bce (nn.BCELoss): BCE Loss.
        optimizer (torch.optim.Adam): Adam optimizer.
        scheduler (torch.optim.lr_scheduler.StepLR): LR scheduler to decay LR
            every scheduler_steps.
        grad_clip (float): Threshold value for gradient clipping.
    """

    def __init__(
        self,
        action_dim: int,
        T_max: int,
        classification_threshold: float = 0.5,
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
            classification_threshold (float): Minimum probability for a
                positive classification.
            num_channels (ArrayLike): number of feature channels in each residual block of the network.
            kernel_size (int): Kernel size for convolutions.
            dropout (float): Dropout rate.
            lr (float): Initial learning rate.
            grad_clip (float): Threshold value for gradient clipping.
            scheduler_step (int): The learning rate will update every scheduler_step
                epochs.
            scheduler_decay (float): Learning rate decay rate.
        """
        super().__init__(action_dim, T_max, classification_threshold)
        self.classifier = IndividualAttackClassifier(
            action_dim, num_channels, kernel_size, dropout
        )
        self.bce = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.classifier.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, scheduler_step, gamma=scheduler_decay
        )
        self.grad_clip = grad_clip

    def train(
        self,
        stacked_trajectories: torch.Tensor,
        train_labels: torch.Tensor,
        epochs: int = 300,
        batch_size: int = 16,
    ) -> None:
        train_loader = self._make_dataloader(
            stacked_trajectories, train_labels, batch_size
        )

        for _ in range(epochs):
            for inputs, labels in train_loader:
                self.optimizer.zero_grad()

                out = self.classifier(inputs)

                loss = self.bce(out, labels)

                loss.backward()
                # Gradient clipping
                torch.nn.utils.clip_grad_value_(
                    self.classifier.parameters(), clip_value=self.grad_clip
                )

                self.optimizer.step()


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

        self.fc = nn.Linear(num_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.tcn(x)
        x = self.fc(x)
        return self.sigmoid(x)
