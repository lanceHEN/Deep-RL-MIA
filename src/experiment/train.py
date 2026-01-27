import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

from .experiment_runner import ExperimentRunner

root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))
from config import (
    ExperimentConfig,
    DataOracleConfig,
    TrainerOracleConfig,
    IndividualAttackTrainerConfig,
    CollectiveAttackTrainerConfig,
)

src_dir = Path(__file__).parent.parent
sys.path.insert(0, str(src_dir))
from model import (
    IndividualAttackTrainer,
    CollectiveAttackTrainer,
    DataFormatter,
    DataOracle,
    TrainerOracle,
)

attack_trainer_config = IndividualAttackTrainerConfig(
    action_dim=3,
    T_max=100
)

config = ExperimentConfig(
    individual_attack=True,
    attack_trainer_config=attack_trainer_config,
    env_name="Hopper-v5",
    T_max=100,
    train_trajs=1000,
    train_seed=1,
    external_trajs=1000,
    external_seed=2,
    output_seed=3,
    attack_trainer_epochs=100,
    attack_trainer_batch_size=16,
    data_oracle_ddpg_verbose=0,
    data_oracle_ddpg_buffer_size=10000,
    data_oracle_ddpg_learning_starts=1000,
    data_oracle_ddpg_learn_timesteps=200000,
    trainer_oracle_bcq_device="cpu:0",
    trainer_oracle_bcq_batch_size=100,
    trainer_oracle_discount_factor=0.99,
)

experiment_runner = ExperimentRunner(config)

experiment_runner.run_experiment()