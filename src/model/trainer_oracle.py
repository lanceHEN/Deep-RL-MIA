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
        self.bcq =  None
        
    def train(self, trajectories: List[Dict[str, Union[object, List[Tuple[object, object, float, object]]]]], epochs: int) -> None:
        """
        Initializes the BCQ model and fits it on the given trajectories, for
        the given number of epochs.
        
        This must be called before get_output_trajectories, otherwise there
        will be no underlying BCQ.
        
        Args:
            trajectories (List[Dict[str, Union[object, List[Tuple[object, object,
                float, object]]]]]): List of trajectories, where each trajectory
                contains an 'initial_state' key mapping to the initial state,
                and a 'transitions' key mapping to a list of (s, a, r, s')
                tuples.
            epochs (int): Number of epochs to train BCQ for.
        """
        # We already have the trajectories, so we just need to prepare them
        # for self.bcq.
        
        # To do so, convert each into an Episode object
        episodes = []
        for traj in trajectories:
            transitions = traj['transitions']
            
            observations = []
            actions = []
            rewards = []
            
            for (s, a, r, s_next) in transitions:
                observations.append(s)
                actions.append(a)
                rewards.append(r)
            
            # Also include the final state (doesn't have its own tuple)
            if len(transitions) > 0:
                final_s_next = transitions[-1][3]
                observations.append(final_s_next)
                
            episode = Episode(
                observations = np.array(observations),
                actions = np.array(actions),
                rewards = np.array(rewards)
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
            n_epochs=epochs
        )

        self.bcq.fit(episodes)
    
    def get_output_trajectories(self, initial_states: List[object], T_max: int, seed: int=None):
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
            
        Raises:
            RuntimeError: if train is not called before
        """
        if self.bcq is None:
            raise RuntimeError("train method must be called first")
        
        if seed is not None:
            self.env.seed(seed)
        
        trajectories = []
        
        for initial_state in initial_states:
            transitions = []
            
            current_state = initial_state

            for _ in range(T_max):
                # Get the action
                action = self.bcq.predict(current_state)

                # Step through environment
                next_state, reward, done, _, _ = self.env.step(action)

                transition = (current_state, action, reward, next_state)
                transitions.append(transition)

                if done:
                    break

                current_state = next_state

            trajectories.append(
                {"initial_state": initial_state, "transitions": transitions}
            )

        return trajectories