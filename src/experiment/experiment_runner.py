"""
A single flexible class to run all experiments from.
"""

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch

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
        train_trajectories = data_oracle.collect_training_trajectories(
            self.config.train_trajs, self.config.T_max, self.config.train_seed
        )
        
        print("\n" + "="*60)
        print("DIAGNOSTIC 1: TRAINING DATA QUALITY")
        print("="*60)

        # Check episode rewards
        episode_rewards = [traj["rewards"].sum() for traj in train_trajectories[:100]]
        print(f"Sample episode rewards: {episode_rewards[:10]}")
        print(f"Average episode reward: {np.mean(episode_rewards):.2f}")
        print(f"Std episode reward: {np.std(episode_rewards):.2f}")

        # Check episode lengths
        episode_lengths = [len(traj["actions"]) for traj in train_trajectories[:100]]
        print(f"Average episode length: {np.mean(episode_lengths):.1f}")
    
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
        trainer_oracle.train(train_trajectories, epochs=self.config.trainer_oracle_bcq_epochs)
        
        print("\n" + "="*60)
        print("DIAGNOSTIC 2: BCQ POLICY QUALITY")
        print("="*60)
    
        total_reward = 0
        for ep in range(10):
            state = self.config.env.reset()[0]
            ep_reward = 0
        
            for t in range(self.config.T_max):
                action = trainer_oracle.bcq.predict(np.expand_dims(state, axis=0))
                action = action.reshape(-1)
                state, reward, done, _, _ = self.config.env.step(action)
                ep_reward += reward
            
                if done:
                    break
        
            total_reward += ep_reward
    
        bcq_avg_reward = total_reward / 10
        print(f"BCQ Average Reward: {bcq_avg_reward:.2f}")

        # Get some external trajectories - use train_trajectories to ensure different.
        external_trajectories = data_oracle.collect_external_trajectories(
            train_trajectories, self.config.external_trajs, self.config.T_max, self.config.external_train_tolerance, self.config.external_seed,
            random_policy=self.config.external_random_policy
        )

        # Get output trajectories from trainer oracle.

        # First make initial states to generate outputs from.
        # We condition on initial state because we'd waste time making trajectories
        # with other initial states that would never be matched by the data formatteer.
        
        train_qpos = np.array(
            [traj["qpos"] for traj in train_trajectories]
        )  # [train_trajs, pos_dim]
        train_qvel = np.array(
            [traj["qvel"] for traj in train_trajectories]
        )  # [train_trajs, vel_dim]
        external_qpos = np.array(
            [traj["qpos"] for traj in external_trajectories]
        )  # [external_trajs, pos_dim]
        external_qvel = np.array(
            [traj["qvel"] for traj in external_trajectories]
        )  # [external_trajs, vel_dim]
        
        # Then get each output trajectory
        train_output_trajectories = trainer_oracle.get_output_trajectories(
            train_qpos, train_qvel, T_max=self.config.T_max, seed=self.config.train_output_seed
        )  # [train_trajs,]
        
        external_output_trajectories = trainer_oracle.get_output_trajectories(
            external_qpos,
            external_qvel,
            T_max=self.config.T_max,
            seed=self.config.external_output_seed,
        )  # [external_trajs,]
        
        print("\n" + "="*60)
        print("DIAGNOSTIC 3: OVERFITTING SIGNAL")
        print("="*60)
    
        # Check action similarity
        member_diffs = []
        for i in range(min(50, len(train_trajectories))):
            train_act = train_trajectories[i]["actions"]
            output_act = train_output_trajectories[i]["actions"]
        
            min_len = min(len(train_act), len(output_act))
            diff = np.mean(np.abs(train_act[:min_len] - output_act[:min_len]))
            member_diffs.append(diff)
    
        nonmember_diffs = []
        for i in range(min(50, len(external_trajectories))):
            ext_act = external_trajectories[i]["actions"]
            ext_output_act = external_output_trajectories[i]["actions"]
        
            min_len = min(len(ext_act), len(ext_output_act))
            diff = np.mean(np.abs(ext_act[:min_len] - ext_output_act[:min_len]))
            nonmember_diffs.append(diff)
    
        member_avg = np.mean(member_diffs)
        nonmember_avg = np.mean(nonmember_diffs)
    
        print(f"Member action diff:     {member_avg:.4f} ± {np.std(member_diffs):.4f}")
        print(f"Non-member action diff: {nonmember_avg:.4f} ± {np.std(nonmember_diffs):.4f}")
        print(f"Ratio (member/non-member): {member_avg/nonmember_avg:.3f}")

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
        
        print("\n" + "="*60)
        print("DIAGNOSTIC 4: DATA FORMAT")
        print("="*60)
    
        print(f"Pairs shape: {all_pairs.shape}")
        print(f"Labels shape: {all_labels.shape}")
        print(f"Positive samples: {(all_labels == 1).sum()}")
        print(f"Negative samples: {(all_labels == 0).sum()}")
        print(f"Label balance: {(all_labels == 1).sum() / len(all_labels):.2%}")
    
        # Sample a few pairs
        print(f"\nSample pair 0:")
        print(f"  Shape: {all_pairs[0].shape}")
        print(f"  Label: {all_labels[0]}")
        print(f"  Data range: [{all_pairs[0].min():.3f}, {all_pairs[0].max():.3f}]")
        
        # Batch if necessary if in collective mode
        if not self.config.individual_attack:
            # [train_trajs + external_trajs, 2*action_dim, T_max] -> 
            # [(train_trajs + external_trajs) // collective_batch_size, collective_batch_size, 2*action_dim, T_max]
            m = self.config.collective_batch_size
            all_pairs = np.array([all_pairs[i:i+ m] for i in range(0, (len(all_pairs) // m) * m, m)])
            
            # [train_trajs + external_trajs,] -> [(train_trajs + external_trajs) // collective_batch_size, collective_batch_size]
            all_labels = np.array([all_labels[i:i+ m] for i in range(0, (len(all_labels) // m) * m, m)])
            
            # Tranpose so minibatch size is final dim, as done in the paper
            all_pairs = np.transpose(all_pairs, (0, 2, 3, 1))
        
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
        
        # Check training set accuracy
        train_acc = accuracy_score(train_labels, attack_trainer.predict(train_pairs))
        print(f"Training set accuracy: {train_acc:.2%}")

        test_acc = accuracy_score(test_labels, attack_trainer.predict(test_pairs))
        print(f"Test set accuracy: {test_acc:.2%}")

def main():
    env = gym.make("Hopper-v5")

    attack_trainer_config = IndividualAttackTrainerConfig(
        action_dim=env.action_space.shape[0],
        T_max=100,
        device="mps",
        classification_threshold=0.5
    )

    config = ExperimentConfig(
       individual_attack=True,
       attack_trainer_config=attack_trainer_config,
       collective_batch_size=50,
       env=env,
       T_max=100,
       train_trajs=500,
       train_seed=1,
       external_trajs=500,
       external_train_tolerance=1e-6,
       external_seed=2,
       train_output_seed=3,
       external_output_seed=4,
       external_random_policy=False,
       attack_trainer_epochs=300,
       attack_trainer_train_test_split_seed=4,
       attack_trainer_train_test_split_test_size=0.2,
       attack_trainer_batch_size=256,
       data_oracle_ddpg_verbose=0,
       data_oracle_ddpg_buffer_size=100000,
       data_oracle_ddpg_learning_starts=1000,
       data_oracle_ddpg_learn_timesteps=1000000,
       trainer_oracle_bcq_epochs=150000,
       trainer_oracle_bcq_device="cpu:0",
       trainer_oracle_bcq_batch_size=256,
       trainer_oracle_discount_factor=0.99,
   )


    experiment_runner = ExperimentRunner(config)

    experiment_runner.run_experiment()
    
if __name__ == "__main__":
    main()