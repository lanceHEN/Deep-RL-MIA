import gym
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
        self.policy = DDPG("MlpPolicy", self.env, verbose=0)
        # Learn for some steps.
        self.policy.learn(total_timesteps=100000)
        
    def collect_trajectories(self, n_trajectories: int, T_max: int, seed: int=None):
        if seed is not None: # For reuse
            self.env.seed(seed)
            np.random.seed(seed)
            
        trajectories = []
        for _ in range(n_trajectories):
            initial_state = self.env.reset() # Always start a new episode.
            
            transitions = []
            
            current_state = initial_state
            
            for _ in range(T_max):
                # Get the action
                action, _ = self.policy.predict(current_state)
                
                # Step through environment
                next_state, reward, done, _, _ = self.env.step(action)
                
                transition = (current_state, action, reward, next_state)
                transitions.append(transition)
                
                current_state = next_state
                
            trajectories.append({'initial_state': initial_state,
                                 'transitions': transitions})
            
        return trajectories