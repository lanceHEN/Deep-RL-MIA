"""
A single flexible class to run all experiments from.
"""

import sys
from pathlib import Path
from typing import List

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
        
    def test_external_random_ratios(self, external_random_ratios: List[float] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]):
        """
        Tests different ratios of random trajectories in the external set,
        and plots test accuracy over them.
        """
        for ratio in external_random_ratios:
            print(f"\nTesting external random ratio: {ratio:.2f}")
            self.run_experiment(ratio)

    def run_experiment(self, external_random_ratio: float = 0.0) -> float:
        """
        Runs a particular experiment with the given config, external random ratio, and verbose setting. Returns test accuracy.
        """

        # Initialize Data Oracle
        data_oracle = DataOracle(
            DataOracleConfig(
                self.config.env,
                self.config.verbose,
                self.config.data_oracle_ddpg_buffer_size,
                self.config.data_oracle_ddpg_learning_starts,
                self.config.data_oracle_ddpg_learn_timesteps,
            )
        )

        # Collect training trajectories
        train_trajectories = data_oracle.collect_training_trajectories(
            self.config.train_trajs, self.config.T_max, self.config.train_seed
        )
        
        if self.config.verbose > 0:
        
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
                self.config.verbose,
                self.config.trainer_oracle_bcq_device,
                self.config.trainer_oracle_bcq_batch_size,
                self.config.trainer_oracle_discount_factor,
            )
        )

        # Train for some epochs
        trainer_oracle.train(train_trajectories, epochs=self.config.trainer_oracle_bcq_epochs)
        
        if self.config.verbose > 0:
        
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
            random_traj_ratio=external_random_ratio
        )
        
        if self.config.verbose > 0:
        
            # Similarity between training and extenral trajectories
            # ========================================
            # 2. FULL STATE TRAJECTORY DISTRIBUTION  
            # ========================================
            print("\n2. FULL STATE TRAJECTORY DISTRIBUTION")
    
            # Flatten all states from all trajectories
            train_all_states = np.concatenate([traj["states"] for traj in train_trajectories], axis=0)
            ext_all_states = np.concatenate([traj["states"] for traj in external_trajectories], axis=0)
    
            print(f"   Train states: {train_all_states.shape}")
            print(f"   External states: {ext_all_states.shape}")
    
            # Statistical comparison across all states
            train_state_mean = train_all_states.mean(axis=0)
            ext_state_mean = ext_all_states.mean(axis=0)
            train_state_std = train_all_states.std(axis=0)
            ext_state_std = ext_all_states.std(axis=0)
    
            state_mean_diff = np.linalg.norm(train_state_mean - ext_state_mean)
            state_std_diff = np.linalg.norm(train_state_std - ext_state_std)
    
            print(f"   All states mean L2 difference: {state_mean_diff:.6f}")
            print(f"   All states std L2 difference:  {state_std_diff:.6f}")
        
            # ========================================
            # 3. STATE TRAJECTORY SIMILARITY
            # ========================================
            print("\n3. STATE TRAJECTORY SIMILARITY")
    
            # Compare state sequences (not just initial states)
            def state_sequence_distance(traj1, traj2):
                """Compute distance between two state trajectories"""
                states1 = traj1["states"]
                states2 = traj2["states"]
        
                min_len = min(len(states1), len(states2))
        
                # Mean absolute difference across all states and dimensions
                return np.mean(np.abs(states1[:min_len] - states2[:min_len]))
    
            # Intra-train distances
            n_sample = min(30, len(train_trajectories))
            intra_train_state_dists = [] # All pairs
            for i in range(n_sample):
                for j in range(i+1, n_sample):
                    dist = state_sequence_distance(train_trajectories[i], train_trajectories[j])
                    intra_train_state_dists.append(dist)
    
            # Intra-external distances
            intra_ext_state_dists = [] # All pairs
            for i in range(n_sample):
                for j in range(i+1, n_sample):
                    dist = state_sequence_distance(external_trajectories[i], external_trajectories[j])
                    intra_ext_state_dists.append(dist)
    
            # Inter-set distances (train vs external)
            inter_state_dists = [] # All pairs
            for i in range(n_sample):
                for j in range(n_sample):
                    dist = state_sequence_distance(train_trajectories[i], external_trajectories[j])
                    inter_state_dists.append(dist)
    
            print(f"   Intra-train state sequence distance:  {np.mean(intra_train_state_dists):.4f}")
            print(f"   Intra-external state sequence distance: {np.mean(intra_ext_state_dists):.4f}")
            print(f"   Inter-set state sequence distance:    {np.mean(inter_state_dists):.4f}")
    
            intra_avg = (np.mean(intra_train_state_dists) + np.mean(intra_ext_state_dists)) / 2
    
            if intra_avg > 0:
                ratio = np.mean(inter_state_dists) / intra_avg
                print(f"   Ratio (inter/intra): {ratio:.3f}")
        
                if ratio < 1.1:
                    print(f"   → State trajectories from SAME DISTRIBUTION")
                else:
                    print(f"   → State trajectories from DIFFERENT DISTRIBUTIONS")
        
            print("\n2. ACTION DISTRIBUTION")
    
            train_actions = np.concatenate([traj["actions"] for traj in train_trajectories])
            ext_actions = np.concatenate([traj["actions"] for traj in external_trajectories])
    
            print(f"   Train actions: {train_actions.shape}")
            print(f"   External actions: {ext_actions.shape}")
    
            # Overall action statistics
            train_action_mean = train_actions.mean(axis=0)
            ext_action_mean = ext_actions.mean(axis=0)
            train_action_std = train_actions.std(axis=0)
            ext_action_std = ext_actions.std(axis=0)
    
            print(f"\n   Train action mean: {train_action_mean}")
            print(f"   Ext action mean:   {ext_action_mean}")
            print(f"   Mean difference:   {np.abs(train_action_mean - ext_action_mean)}")
            
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
        
        if self.config.verbose > 0:
        
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
        
        if self.config.verbose > 0:
        
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
            
            # [train_trajs + external_trajs,] -> [(train_trajs + external_trajs) // collective_batch_size,] #
            # CRITICAL: ALL LABELS IN MIINBATCH MUST BE THE SAME.
            all_labels = np.array([all_labels[i] for i in range(0, (len(all_labels) // m) * m, m)])
            
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
        
        

        test_acc = accuracy_score(test_labels, attack_trainer.predict(test_pairs))
        
        if self.config.verbose > 0:
            # Check training set accuracy
            train_acc = accuracy_score(train_labels, attack_trainer.predict(train_pairs))
            print(f"Training set accuracy: {train_acc:.2%}")
            print(f"Test set accuracy: {test_acc:.2%}")
        
        return test_acc

def main():
    env = gym.make("Hopper-v5")
    T_max = 100
    verbose = 1

    attack_trainer_config = IndividualAttackTrainerConfig(
        action_dim=env.action_space.shape[0],
        T_max=T_max,
        verbose=verbose,
        device="mps",
        classification_threshold=0.5
    )


    config = ExperimentConfig(
       individual_attack=False,
       attack_trainer_config=attack_trainer_config,
       collective_batch_size=50,
       env=env,
       T_max=T_max,
       verbose=verbose,
       train_trajs=100,
       train_seed=1,
       external_trajs=100,
       external_train_tolerance=1e-6,
       external_seed=2,
       train_output_seed=3,
       external_output_seed=4,
       attack_trainer_epochs=1,
       attack_trainer_train_test_split_seed=4,
       attack_trainer_train_test_split_test_size=0.2,
       attack_trainer_batch_size=256,
       data_oracle_ddpg_buffer_size=100000,
       data_oracle_ddpg_learning_starts=1,
       data_oracle_ddpg_learn_timesteps=100,
       trainer_oracle_bcq_epochs=100,
       trainer_oracle_bcq_device="cpu:0",
       trainer_oracle_bcq_batch_size=256,
       trainer_oracle_discount_factor=0.99,
   )

    
    experiment_runner = ExperimentRunner(config)
    
    experiment_runner.run_experiment(0.5)
    
if __name__ == "__main__":
    main()