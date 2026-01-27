from typing import List, Dict, Union, Tuple

import gymnasium as gym
from stable_baselines3 import DDPG
import numpy as np

from config import DataOracleConfig


class DataOracle:
    """
    An implementation of the Data Oracle. This will interact with a
    specified environment and return i.i.d. trajectories for use by the
    Model Trainer Oracle, via the collect_training_trajectories method. It
    can also return external trajectories that are explicitly different
    from the training trajectories via the collect_external_trajectories method.

    Here, we use DDPG as the exploration policy.

    Attributes:
        config (DataOracleConfig): Stores config information.
        env (gym.Env): Gym environment to interact with.
        policy (DDPG): DDPG-based policy for exploration.
    """

    def __init__(self, config: DataOracleConfig):
        """
        Initializes a DataOracle object over the given config values.
        Also initializes and trains the DDPG exploration policy.

        Args:
            config (DataOracleConfig): Stores config information.
        """
        self.config = config
        self.env = config.env

        # Initialize DDPG with the given env
        self.policy = DDPG(
            "MlpPolicy",
            self.env,
            verbose=self.config.ddpg_verbose,
            buffer_size=self.config.ddpg_buffer_size,
            learning_starts=self.config.ddpg_learning_starts,
        )

        # Learn for some steps.
        print("Learning DDPG policy for data oracle.")
        self.policy.learn(
            total_timesteps=self.config.ddpg_learn_timesteps, progress_bar=True
        )
        print("Finished learning DDPG policy")
        
    def _gen_trajectory(self, T_max: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates a single trajectory, returning numpy arrays for states, actions,
        and rewards.
        """
        states = []
        actions = []
        rewards = []

        current_state = self.env.reset()[0]  # Always start a new episode.

        for _ in range(T_max):
            # print(current_state)
            # Get the action
            action, _ = self.policy.predict(current_state)

            # Step through environment
            next_state, reward, done, _, _ = self.env.step(action)

            states.append(current_state)
            actions.append(action)
            rewards.append(reward)

            if done:
                states.append(next_state)
                break

            current_state = next_state
                
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
            
        return states, actions, rewards

    def collect_training_trajectories(
        self, n_trajectories: int, T_max: int, seed: int = None
    ) -> List[Dict[str, np.ndarray]]:
        """
        Returns the requested number of i.i.d. trajectories from the supplied
        environment, stopping early within each trajectory if T_max steps are taken.

        Args:
            n_trajectories (int): Number of trajectories to generate.
            T_max (int): Maximal number of steps in a trajectory.
            seed (int): Optional random seed for reproducibility.

        Returns:
            List[Dict[str, np.ndarray]]: List of trajectories, where each trajectory
                contains an a 'states' key mapping to a [T, state_dim] array of states,
                a 'actions' key mapping to a [T, action_dim] array of actions,
                a 'rewards' key mapping to a [T,] array of rewards, and a 'terminals'
                key mapping to whether each transition was the last one or not.
        """
        print(f"Generating {n_trajectories} training trajectories.")
        if seed is not None:  # For reuse
            self.env.reset(seed=seed)
            np.random.seed(seed)

        trajectories = []
        for _ in range(n_trajectories):
            states, actions, rewards = self._gen_trajectory(T_max)

            # Create terminal flags array
            T = len(actions)
            terminals = np.zeros(T, dtype=np.float32)
            terminals[-1] = 1.0
            

            trajectories.append(
                {
                    "states": states,  # [T, state_dim]
                    "actions": actions,  # [T, action_dim]
                    "rewards": rewards,  # [T,]
                    "terminals": terminals,  # [T,]
                }
            )

        print("Finished generating training trajectories.")
        return trajectories

    def collect_external_trajectories(
        self,
        train_trajectories: List[Dict[str, np.ndarray]],
        n_trajectories: int,
        T_max: int,
        seed: int = None
    ) -> List[Dict[str, np.ndarray]]:
        """
        Performs the same as collect_training_trajectories, except it also
        checks, for each outputted trajectory, that it is not a duplicate of
        a trajectory in train_trajectories (according to states and actions).

        Args:
            train_trajectories (List[Dict[str, np.ndarray]]): List of training
                trajectories to ensure no duplicates are made from them.
            n_trajectories (int): Number of trajectories to generate.
            T_max (int): Maximal number of steps in a trajectory.
            seed (int): Optional random seed for reproducibility.

        Returns:
            List[Dict[str, np.ndarray]]: List of trajectories different from
                anything in train_trajectories by at least one state or action, where each trajectory
                contains an a 'states' key mapping to a [T, state_dim] array of states,
                a 'actions' key mapping to a [T, action_dim] array of actions,
                a 'rewards' key mapping to a [T,] array of rewards, and a 'terminals'
                key mapping to whether each transition was the last one or not.
        """
        print(f"Generating {n_trajectories} external trajectories.")
        if seed is not None:  # For reuse
            self.env.reset(seed=seed)
            np.random.seed(seed)

        trajectories = []
        
        while len(trajectories) < n_trajectories:
            states, actions, rewards = self._gen_trajectory(T_max)
                
            skip = False
            # Check not in train_trajectories
            for train_traj in train_trajectories:
                if len(train_traj["actions"]) == len(actions):
                    if np.allclose(actions, train_traj["actions"], atol=1e-6) and np.allclose(states, train_traj["states"], atol=1e-6):
                        skip = True
                        print("Skipping generated trajectory because it is identical to a training one")
                        break
                        
            if skip:
                continue

            # Create terminal flags array
            T = len(actions)
            terminals = np.zeros(T, dtype=np.float32)
            terminals[-1] = 1.0  

            trajectories.append(
                {
                    "states": states,  # [T, state_dim]
                    "actions": actions,  # [T, action_dim]
                    "rewards": rewards,  # [T,]
                    "terminals": terminals,  # [T,]
                }
            )

        print("Finished generating external trajectories.")
        return trajectories