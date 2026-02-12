"""
This module provides several dataclasses for modular configuration of different
components in the Deep RL MIA pipeline.
"""

from dataclasses import dataclass, field
from typing import List

import gymnasium as gym


@dataclass
class DataOracleConfig:
    """
    Stores config info for DataOracle.
    """

    env: gym.Env
    verbose: int = 0
    ddpg_buffer_size: int = 10000
    ddpg_learning_starts: int = 1000
    ddpg_learn_timesteps: int = 200000


@dataclass
class TrainerOracleConfig:
    """
    Stores config info for TrainerOracle.
    """

    env: gym.Env
    verbose: int = 0
    bcq_device: str = "cpu:0"
    bcq_batch_size: int = 100
    discount_factor: float = 0.99


@dataclass
class IndividualAttackTrainerConfig:
    """
    Stores config info for IndividualAttackTrainer.
    """

    action_dim: int
    T_max: int
    verbose: int = 0
    device: str = "mps"
    classification_threshold: float = 0.5
    num_channels: List = field(default_factory=lambda: [600, 600])
    kernel_size: int = 3
    dropout: float = 0.45
    lr: float = 0.0003
    grad_clip: float = 0.35
    scheduler_step = 100
    scheduler_decay = 0.1


@dataclass
class CollectiveAttackTrainerConfig:
    """
    Stores config info for CollectiveAttackTrainer.
    """

    action_dim: int
    T_max: int
    verbose: int = 0
    device: str = "mps"
    classification_threshold: float = 0.5
    dropout: float = 0.0
    lr: float = 0.0008
    grad_clip: float = 0.35
    scheduler_step = 100
    scheduler_decay = 1.0
