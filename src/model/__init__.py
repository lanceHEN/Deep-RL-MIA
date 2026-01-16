import gym
from stable_baselines3 import DDPG

class DataOracle:
    
    
    def __init__(self, env: gym.Env):
        self.env = env
        self.policy = self.