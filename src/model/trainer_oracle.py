from typing import List, Dict, Union, Tuple

import gym
from d3rlpy.algos import BCQ
from d3rlpy.dataset import Episode
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
        trajectories: List[Dict[str, List[object]]],
        epochs: int,
    ) -> None:
        """
        Initializes the BCQ model and fits it on the given trajectories, for
        the given number of epochs.

        This must be called before get_output_trajectories, otherwise there
        will be no underlying BCQ.

        Args:
            trajectories (List[Dict[str, List[object]]]): List of trajectories,
                where each trajectory contains an a 'states' key mapping to a
                list of states, a 'actions' key mapping to a list of actions,
                and a 'rewards' key mapping to a list of rewards.
            epochs (int): Number of epochs to train BCQ for.
        """
        # We already have the trajectories, so we just need to prepare them
        # for self.bcq.

        # To do so, convert each into an Episode object
        episodes = []
        for traj in trajectories:

            episode = Episode(
                observations=np.array(traj['states']),
                actions=np.array(traj['actions']),
                rewards=np.array(traj['rewards']),
            )

            episodes.append(episode)

        # Construct the BCQ
        self.bcq = BCQ(
            actor_learning_rate=1e-3,
            critic_learning_rate=1e-3,
            imitator_learning_rate=1e-3,
            batch_size=100,
            gamma=0.99,
            tau=0.005,
            n_critics=2,
            lam=0.75,
            n_action_samples=100,
            action_flexibility=0.05,
            use_gpu=False,
            n_epochs=epochs,
        )

        self.bcq.fit(episodes)

    def get_output_trajectories(
        self, initial_states: List[object], T_max: int, seed: int = None
    ) -> List[Dict[str, List[object]]]:
        """
        Produces trajectories formed from applying the learned BCQ policy on
        the list of initial states.

        This must be called after train, otherwise there
        will be no underlying BCQ.

        Args:
            initial_states (List[object]): List of initial states to start the
                produced trajectories from.
            T_max (int): Max number of steps in a trajectory.
            seed (int): Optional random seed for reproducibility.
            
        Returns:
            List[Dict[str, List[object]]]: List of outputted trajectories, where
                each trajectory contains an a 'states' key mapping to a list
                of states, a 'actions' key mapping to a list of actions, and
                a 'rewards' key mapping to a list of rewards.

        Raises:
            RuntimeError: if train is not called before
        """
        if self.bcq is None:
            raise RuntimeError("train method must be called first")

        if seed is not None:
            self.env.seed(seed)

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

            trajectories.append(
                {"states": states, "actions": actions, "rewards": rewards}
            )

        return trajectories