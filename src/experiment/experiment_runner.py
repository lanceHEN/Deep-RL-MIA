"""
A single flexible class to run all experiments from.
"""

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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


class ExperimentRunner:
    """
    A single flexible class to run all experiments from via the run_experiment
    method.

    Attributes:
        config (ExperimentConfig): Config info.
    """

    def __init__(self, config: ExperimentConfig):
        """
        Initializes an ExperimentRunner with the given config.

        Args:
            config (ExperimentConfig): Config info.
        """
        self.config = config

    def run_experiment(self):

        # Initialize Data Oracle
        data_oracle = DataOracle(
            DataOracleConfig(
                self.config.env,
                self.config.data_oracle_ddpg_verbose,
                self.config.data_oracle_ddpg_buffer_size,
                self.config.data_oracle_ddpg_learning_starts,
                self.config.data_oracle_ddpg_learn_timesteps,
            )
        )

        # Collect training trajectories
        train_trajectories = data_oracle.collect_trajectories(
            self.config.train_trajs, self.config.T_max, self.config.train_seed
        )

        # Initialize Trainer Oracle
        trainer_oracle = TrainerOracle(
            TrainerOracleConfig(
                self.config.env,
                self.config.trainer_oracle_bcq_device,
                self.config.trainer_oracle_bcq_batch_size,
                self.config.trainer_oracle_discount_factor,
            )
        )

        # Train for some epochs
        trainer_oracle.train(train_trajectories, epochs=2)

        # Get some external trajectories - use train_trajectories to ensure different.
        external_trajectories = data_oracle.collect_trajectories(
            self.config.external_trajs, self.config.T_max, self.config.external_seed
        )

        # Get output trajectories from trainer oracle.

        # First make initial states to generate outputs from.
        # We condition on initial state because we'd waste time making trajectories
        # with other initial states that would never be matched by the data formatteer.
        train_initial_states = np.array(
            [traj["states"][0] for traj in train_trajectories]
        )  # [train_trajs, state_dim]
        external_initial_states = np.array(
            [traj["states"][0] for traj in external_trajectories]
        )  # [external_trajs, state_dim]

        # Then get each output trajectory type
        train_output_trajectories = trainer_oracle.get_output_trajectories(
            train_initial_states, T_max=self.config.T_max, seed=self.config.output_seed
        )  # [train_trajs,]

        external_output_trajectories = trainer_oracle.get_output_trajectories(
            external_initial_states,
            T_max=self.config.T_max,
            seed=self.config.output_seed,
        )  # [external_trajs,]

        # Initialize data formatter
        data_formatter = DataFormatter

        # Get positive and negative pairs
        # positive_pairs: [train_trajs, 2*action_dim, T_max]
        # positive_labels: [train_trajs,]
        # negative_pairs: [external_trajs, 2*action_dim, T_max]
        # negative_labels: [train_trajs,]
        positive_pairs, positive_labels = data_formatter.pair_train_output_trajectories(
            train_trajectories, train_output_trajectories, T_max=self.config.T_max
        )
        negative_pairs, negative_labels = (
            data_formatter.pair_external_output_trajectories(
                external_trajectories,
                external_output_trajectories,
                T_max=self.config.T_max,
            )
        )
        all_pairs = np.vstack((positive_pairs, negative_pairs))
        all_labels = np.concatenate((positive_labels, negative_labels))
        
        train_pairs, test_pairs, train_labels, test_labels = train_test_split(all_pairs, all_labels, test_size=self.config.attack_trainer_train_test_split_test_size,
                                                                              random_state=self.config.attack_trainer_train_test_split_seed)

        # Initialize attack trainer
        if self.config.individual_attack:
            attack_trainer = IndividualAttackTrainer(self.config.attack_trainer_config)
        else:
            attack_trainer = CollectiveAttackTrainer(self.config.attack_trainer_config)

        # Train on positive and negative pairs. NOTE that they do not need to be
        # shuffled.
        attack_trainer.train(
            train_pairs,
            train_labels,
            epochs=self.config.attack_trainer_epochs,
            batch_size=self.config.attack_trainer_batch_size,
        )

        test_preds = attack_trainer.predict(test_pairs)
        
        print(test_labels.shape)
        print(test_preds.shape)
        
        print(f"Accuracy: {100*accuracy_score(test_labels, test_preds)}%")


attack_trainer_config = IndividualAttackTrainerConfig(
    action_dim=3,
    T_max=100,
    classification_threshold=0.5
)

config = ExperimentConfig(
    individual_attack=True,
    attack_trainer_config=attack_trainer_config,
    env=gym.make("Hopper-v5"),
    T_max=100,
    train_trajs=1000,
    train_seed=1,
    external_trajs=1000,
    external_seed=2,
    output_seed=3,
    attack_trainer_epochs=100,
    attack_trainer_train_test_split_seed=4,
    attack_trainer_train_test_split_test_size=0.2,
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