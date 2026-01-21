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
            for _ in range(T_max - n):
                actions = np.append(actions, last_action, axis=0)
            
            return actions
    
    def pair_trajectories(self, train_trajectories: List[
            List[Dict[str, np.ndarray]]
        ], output_trajectories: List[
            List[Dict[str, np.ndarray]]
        ], external_trajectories: List[
            List[Dict[str, np.ndarray]]
        ], T_max: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pairs output with training and/or external trajectories, returning arrays
        of the stacked trajectories (actions only) and labels, respectively.
        
        Args:
            train_trajectories (List[Dict[str, np.ndarray]]): List of training trajectories,
                where each trajectory contains an a 'states' key mapping to a
                [T, state_dim] array of states, a 'actions' key mapping to a
                [T, action_dim] array of actions, and a 'rewards' key mapping
                to a [T,] array of rewards.
            output_trajectories (List[Dict[str, np.ndarray]]): List of output
                trajectories of the same format as train_trajectories.
            external_trajectories (List[Dict[str, np.ndarray]]): List of external
                trajectories of the same format as train_trajectories.
            T_max (int): Maximum trajectory length--smaller trajectories have
                their last action repeated to get to length T_max while larger
                ones are trimmed.
                
        Returns:
            Tuple[np.ndarray, np.ndarray]: The stacked train/external and
                output trajectories (actions only), along with their labels (1 for train, 0
                for external).
        """
        # First get trajectories (actions only) organized by first state
        train_trajs_by_initial_state = defaultdict(list)
        output_trajs_by_initial_state = defaultdict(list)
        external_trajs_by_initial_state = defaultdict(list)
        
        for trajs, mapping in zip([train_trajectories, output_trajectories, external_trajectories], [train_trajs_by_initial_state, output_trajs_by_initial_state, external_trajs_by_initial_state]):
            for traj in trajs:
                initial_state = traj["states"][0]
                actions = DataFormatter._set_action_length(traj["actions"], T_max)
                mapping[initial_state].append(actions)
                
        final_trajs = []
        labels = []
                
        # Now match up output trajs with either
        for initial_state, output_trajs in output_trajs_by_initial_state.items():
            train_trajs = train_trajs_by_initial_state[initial_state]
            external_trajs = external_trajs_by_initial_state[initial_state]
            
            for output_traj in output_trajs:         
                for label, other_trajs in zip([1, 0], [train_trajs, external_trajs]):
                    for other_traj in other_trajs:
                        final_trajs.append(np.vstack((other_traj, output_traj)))
                        labels.append(label)
                        
        return np.arrayk(final_trajs), np.array(labels)
                
                
        