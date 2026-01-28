"""
This module provides a dataclass to configure an entire experiment.
"""

from dataclasses import dataclass
from typing import Union

import gymnasium as gym

from .component_config import (
    IndividualAttackTrainerConfig,
    CollectiveAttackTrainerConfig,
)


@dataclass
class ExperimentConfig:
    """
    Stores all config info necessary for running an MIA experiment.
    """
    
    # Individual / collective attack trainer info
    individual_attack: bool
    attack_trainer_config: Union[
        IndividualAttackTrainerConfig, CollectiveAttackTrainerConfig
    ]
    collective_batch_size: int

    env: gym.Env
    T_max: int
    train_trajs: int = 1000
    train_seed: int = None
    attack_trainer_train_test_split_seed: int = None
    attack_trainer_train_test_split_test_size: float = 0.2
    external_trajs: int = 1000
    external_train_tolerance: float = 1e-6
    external_seed: int = None
    train_output_seed: int = None
    external_output_seed: int = None
    attack_trainer_epochs: int = 100
    attack_trainer_batch_size: int = 16

    # Data Oracle info
    data_oracle_ddpg_verbose: int = 0
    data_oracle_ddpg_buffer_size: int = 10000
    data_oracle_ddpg_learning_starts: int = 1000
    data_oracle_ddpg_learn_timesteps: int = 200000

    # Trainer Oracle info
    trainer_oracle_bcq_epochs: int = 100000
    trainer_oracle_bcq_device: str = "cpu:0"
    trainer_oracle_bcq_batch_size: int = 100
    trainer_oracle_discount_factor: float = 0.99