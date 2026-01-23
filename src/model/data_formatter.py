from typing import List, Dict, Union, Tuple
from collections import defaultdict

import torch
import numpy as np


class DataFormatter:

    @staticmethod
    def _set_action_length(actions: np.ndarray, T_max: int) -> np.ndarray:
        """
        Sets actions to have length T_max, trimming if it's too long or repeating
        the last action if too short.
        """
        n = len(actions)
        if n == T_max:
            return actions
        elif n > T_max:
            return actions[:T_max]
        else:
            last_action = actions[-1]
            actions = np.concatenate(
                actions, np.full((T_max - n, len(last_action)), last_action)
            )

            return actions

    @staticmethod
    def pair_train_output_trajectories(
        train_trajectories: List[List[Dict[str, np.ndarray]]],
        output_trajectories: List[List[Dict[str, np.ndarray]]],
        T_max: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pairs output with training trajectories, returning arrays
        of the stacked trajectories (actions only) and positive (1) labels,
        respectively.

        CRITICAL: Assumes train_trajectories and output_trajectories are given in
        the same ordered such that the initial state of train_trajectories[i]
        is the initial state of output_trajectories[i]. This speeds up computation.

        Args:
            train_trajectories (List[Dict[str, np.ndarray]]): List of training trajectories,
                where each trajectory contains an a 'states' key mapping to a
                [T, state_dim] array of states, a 'actions' key mapping to a
                [T, action_dim] array of actions, a 'rewards' key mapping to a
                [T,] array of rewards, and a 'terminals' key mapping to whether each
                transition was the last one or not via binary flags.
            output_trajectories (List[Dict[str, np.ndarray]]): List of output
                trajectories of the same format as train_trajectories. Should
                be given in the same order as train_trajectories such that the
                initial state of train_trajectories[i] is the initial state of
                output_trajectories[i]. This speeds up computation.
            T_max (int): Maximum trajectory length--smaller trajectories have
                their last action repeated to get to length T_max while larger
                ones are trimmed.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The stacked train and
                output trajectories (actions only), along with their labels.
        """
        return DataFormatter._pair_trajectories_with_label(
            train_trajectories, output_trajectories, T_max, 1.0
        )

    @staticmethod
    def pair_external_output_trajectories(
        external_trajectories: List[List[Dict[str, np.ndarray]]],
        output_trajectories: List[List[Dict[str, np.ndarray]]],
        T_max: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pairs output with external trajectories, returning arrays
        of the stacked trajectories (actions only) and negative (0) labels,
        respectively.

        CRITICAL: Assumes external_trajectories and output_trajectories are given in
        the same ordered such that the initial state of external_trajectories[i]
        is the initial state of output_trajectories[i]. This speeds up computation.

        Args:
            external_trajectories (List[Dict[str, np.ndarray]]): List of external trajectories,
                where each trajectory contains an a 'states' key mapping to a
                [T, state_dim] array of states, a 'actions' key mapping to a
                [T, action_dim] array of actions, a 'rewards' key mapping to a
                [T,] array of rewards, and a 'terminals' key mapping to whether each
                transition was the last one or not via binary flags.
            output_trajectories (List[Dict[str, np.ndarray]]): List of output
                trajectories of the same format as external_trajectories. Should
                be given in the same order as external_trajectories such that the
                initial state of external_trajectories[i] is the initial state of
                output_trajectories[i]. This speeds up computation.
            T_max (int): Maximum trajectory length--smaller trajectories have
                their last action repeated to get to length T_max while larger
                ones are trimmed.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The stacked external and
                output trajectories (actions only), along with their labels.
        """
        return DataFormatter._pair_trajectories_with_label(
            external_trajectories, output_trajectories, T_max, 0.0
        )

    @staticmethod
    def _pair_trajectories_with_label(
        trajectories: List[List[Dict[str, np.ndarray]]],
        output_trajectories: List[List[Dict[str, np.ndarray]]],
        T_max: int,
        label: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pairs trajectories with ouptut trajectories, returning arrays
        of the stacked trajectories (actions only) and given labels,
        respectively.

        CRITICAL: Assumes trajectories and output_trajectories are given in
        the same ordered such that the initial state of trajectories[i]
        is the initial state of output_trajectories[i]. This speeds up computation.

        Args:
            trajectories (List[Dict[str, np.ndarray]]): List of non-output trajectories,
                where each trajectory contains an a 'states' key mapping to a
                [T, state_dim] array of states, a 'actions' key mapping to a
                [T, action_dim] array of actions, a 'rewards' key mapping to a
                [T,] array of rewards, and a 'terminals' key mapping to whether each
                transition was the last one or not via binary flags.
            output_trajectories (List[Dict[str, np.ndarray]]): List of output
                trajectories of the same format as trajectories. Should
                be given in the same order as trajectories such that the
                initial state of trajectories[i] is the initial state of
                output_trajectories[i]. This speeds up computation.
            T_max (int): Maximum trajectory length--smaller trajectories have
                their last action repeated to get to length T_max while larger
                ones are trimmed.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The stacked non-output and
                output trajectories (actions only), along with their labels.
        """
        stacked_trajs = []
        for traj, output_traj in zip(trajectories, output_trajectories):
            # Note here we transpose to get shape [action_dim, T_max]. That way,
            # by vertically stacking we get the [2*action_dim, T_max] shape used
            # in the paper
            traj_actions = DataFormatter._set_action_length(
                traj["actions"], T_max
            )  # [T_max, action_dim]

            output_traj_actions = DataFormatter._set_action_length(
                output_traj["actions"], T_max
            )  # [T_max, action_dim]

            traj_actions = np.swapaxes(
                traj_actions, 0, 1
            )  # Swapaxes allows use with 2-d and 3-d
            output_traj_actions = np.swapaxes(output_traj_actions, 0, 1)

            # Vertically stack traj and output traj to get the [2*action_dim, T_max,...] shape
            stacked_trajs.append(np.vstack((traj_actions, output_traj_actions)))

        return np.array(stacked_trajs), np.full((len(stacked_trajs)), label)
