import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

src_dir = Path(__file__).parent.parent
sys.path.insert(0, str(src_dir))
from model import (
    IndividualAttackTrainer,
    CollectiveAttackTrainer,
    DataFormatter,
    DataOracle,
    TrainerOracle,
)

# envs = ['Hopper-v4', 'HalfCheetah-v4', 'Ant-v4']


def main(
    env_name: str,
    T_max: int,
    train_trajs: int = 1000,
    train_seed: int = None,
    external_seed: int = None,
    output_seed: int = None,
    attack_trainer_epochs: int = 100,
    attack_trainer_batch_size: int = 16
):
    # Initialize env
    env = gym.make(env_name)

    # Initialize Data Oracle
    data_oracle = DataOracle(env)

    # Collect training trajectories
    train_trajectories = data_oracle.collect_trajectories(
        train_trajs, T_max, train_seed
    )

    # Initialize Trainer Oracle
    trainer_oracle = TrainerOracle(env)

    # Train for some epochs
    trainer_oracle.train(train_trajectories, epochs=2)

    # Get some external trajectories - use train_trajectories to ensure different.
    external_trajectories = data_oracle.collect_trajectories(
        train_trajs, T_max, external_seed
    )

    # Get output trajectories from trainer oracle.

    # First make initial states to generate outputs from.
    # We condition on initial state because we'd waste time making trajectories
    # with other initial states that would never be matched by the data formatteer.
    train_initial_states = np.array([traj["states"][0] for traj in train_trajectories])
    external_initial_states = np.array([traj["states"][0] for traj in external_trajectories])

    # Then get each output trajectory type
    train_output_trajectories = trainer_oracle.get_output_trajectories(
        train_initial_states, T_max=T_max, seed=output_seed
    )
    
    external_output_trajectories = trainer_oracle.get_output_trajectories(
        external_initial_states, T_max=T_max, seed=output_seed
    )

    # Initialize data formatter
    data_formatter = DataFormatter
    
    # Get positive and negative pairs
    positive_pairs, positive_labels = data_formatter.pair_train_output_trajectories(train_trajectories, train_output_trajectories)
    negative_pairs, negative_labels = data_formatter.pair_external_output_trajectories(external_trajectories, external_output_trajectories)
    all_pairs = np.vstack((positive_pairs, negative_pairs))
    all_labels = np.vstack((positive_pairs, negative_pairs))

    # Initialize attack trainer
    attack_trainer = IndividualAttackTrainer(
        action_dim=env.action_space.shape[0], T_max=T_max
    )
    
    # Train on positive and negative pairs. NOTE that they do not need to be
    # shuffled.
    attack_trainer.train(all_pairs, all_labels, epochs=attack_trainer_epochs, batch_size=attack_trainer_batch_size)
    
    predictions = attack_trainer.predict(train_output_trajectories)
    print(predictions)


env_name = "Hopper-v5"

if __name__ == "__main__":
    main(env_name=env_name, T_max=3, train_trajs=2, train_seed=1, external_seed=2, output_seed=3, attack_trainer_epochs=2, attack_trainer_batch_size=16)
