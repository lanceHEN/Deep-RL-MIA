from typing import List, Dict, Union, Tuple

import gymnasium as gym
from stable_baselines3 import DDPG
import numpy as np


class DataOracle:
    """
    An implementation of the Data Oracle. This will interact with a
    specified environment and return i.i.d. trajectories for use by the
    Model Trainer Oracle, via the collect_trajectories method.

    Here, we use DDPG as the exploration policy.

    Attributes:
        env (gym.Env): Gym environment to interact with.
        policy (DDPG): DDPG-based policy for exploration.
    """

    def __init__(self, env: gym.Env):
        """
        Initializes a DataOracle object over the given environment.
        Also initializes and trains the DDPG exploration policy.

        Args:
            env (gym.Env): Gym environment to interact with.
        """
        self.env = env

        # Initialize DDPG with the given env
        self.policy = DDPG(
            "MlpPolicy", self.env, verbose=0, buffer_size=10000, learning_starts=1000
        )

        # Learn for some steps.
        print("Learning DDPG policy for data oracle.")
        self.policy.learn(total_timesteps=100, progress_bar=True)
        print("Finished learning DDPG policy")

    def collect_trajectories(
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
        print(f"Generating {n_trajectories} trajectories.")
        if seed is not None:  # For reuse
            self.env.reset(seed=seed)
            np.random.seed(seed)

        trajectories = []
        for _ in range(n_trajectories):
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

            # Create terminal flags array
            T = len(actions)
            terminals = np.zeros(T, dtype=np.float32)
            terminals[-1] = 1.0

            trajectories.append(
                {
                    "states": np.array(states),
                    "actions": np.array(actions),
                    "rewards": np.array(rewards),
                    "terminals": terminals,
                }
            )

        print("Finished generating trajectories.")
        return trajectories
