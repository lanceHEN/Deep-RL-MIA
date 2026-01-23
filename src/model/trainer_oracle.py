from typing import List, Dict, Union, Tuple

import gymnasium as gym
from d3rlpy.algos import BCQConfig
from d3rlpy.dataset import MDPDataset
import numpy as np


class TrainerOracle:
    """
    The TrainerOracle class is an implementation of the Training Oracle.
    Given trajectories for a given env, it will train a BCQ algorithm for some
    number of epochs. After training is complete, one can get output
    trajectories starting from given initial states.

    Attributes:
        env (gym.Env): Gym environment to interact with.
        bce (BCQ): d3rlpy BCQ implementation.
    """

    def __init__(self, env: gym.Env):
        """
        Initializes a TrainerOracle over the given environment. Note
        bcq will be None until train is called.

        Args:
             env (gym.Env): Gym environment to interact with.
        """
        self.env = env
        self.bcq = None

    def train(
        self,
        trajectories: List[Dict[str, np.ndarray]],
        epochs: int,
    ) -> None:
        """
        Initializes the BCQ model and fits it on the given trajectories, for
        the given number of epochs.

        This must be called before get_output_trajectories, otherwise there
        will be no underlying BCQ.

        Args:
            trajectories (List[Dict[str, np.ndarray]]): List of trajectories,
                where each trajectory contains an a 'states' key mapping to a
                [T, state_dim] array of states, a 'actions' key mapping to a
                [T, action_dim] array of actions, a 'rewards' key mapping to
                a 'rewards' key mapping to a [T,] array of rewards, and a
                'terminals' key mapping to whether each transition was the
                last one or not via binary flags.
            epochs (int): Number of epochs to train BCQ for.
        """
        # We already have the trajectories, so we just need to prepare them
        # for self.bcq.

        # To do so, convert each into an Episode object
        observations = []
        actions = []
        rewards = []
        terminals = []
        for traj in trajectories:
            observations.append(traj["states"])
            actions.append(traj["actions"])
            rewards.append(traj["rewards"])
            terminals.append(traj["terminals"])

        observations = np.concatenate(observations, axis=0)
        actions = np.concatenate(actions, axis=0)
        rewards = np.concatenate(rewards, axis=0)
        terminals = np.concatenate(terminals, axis=0)

        # Create dataset
        dataset = MDPDataset(
            observations=observations,
            actions=actions,
            rewards=rewards,
            terminals=terminals,
        )

        # Create BCQ config
        config = BCQConfig()

        # Build BCQ from config
        self.bcq = config.create(device="cpu:0")

        print(f"Training BCQ for {epochs} epochs...")
        self.bcq.fit(dataset, n_steps=epochs, show_progress=True)

        print("BCQ training complete!")

    def get_output_trajectories(
        self, initial_states: np.ndarray, T_max: int, seed: int = None
    ) -> List[Dict[str, np.ndarray]]:
        """
        Produces trajectories formed from applying the learned BCQ policy on
        the list of initial states.

        This must be called after train, otherwise there
        will be no underlying BCQ.

        Args:
            initial_states (np.ndarray): Array of initial states to start the
                produced trajectories from.
            T_max (int): Max number of steps in a trajectory.
            seed (int): Optional random seed for reproducibility.

        Returns:
            List[Dict[str, np.ndarray]]: List of outputted trajectories,
                where each trajectory contains an a 'states' key mapping to a
                [T, state_dim] array of states, a 'rewards' key mapping to a [T,]
                array of rewards, and a 'terminals' key mapping to whether each
                transition was the last one or not via binary flags.
            epochs (int): Number of epochs to train BCQ for.
        Raises:
            RuntimeError: if train is not called before
        """
        if self.bcq is None:
            raise RuntimeError("train method must be called first")

        if seed is not None:
            self.env.reset(seed=seed)

        trajectories = []

        for initial_state in initial_states:
            states = []
            actions = []
            rewards = []

            current_state = initial_state

            for _ in range(T_max):
                # Get the action
                action = self.bcq.predict(current_state)

                # Step through environment
                next_state, reward, done, _, _ = self.env.step(action)

                states.append(current_state)
                actions.append(action)
                rewards.append(reward)

                if done:
                    states.append(next_state)
                    break

                current_state = next_state

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

        return trajectories
