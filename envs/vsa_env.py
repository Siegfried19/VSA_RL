import numpy as np
import gym
from gym import spaces
from scipy.linalg import expm
import torch

class VSAEnv(gym.Env):
    """Custom Environment that follows SafetyGym interface"""
    
    metadata = {'render.modes':['human']}
    
    def __init__(self):
        super(VSAEnv, self).__init__()
        
        self.dynamics_mode = 'VSA'
        
        # VSA parameters
        self.M   = 0.072
        self.Jp  = 0.1055
        self.Js  = 0.000795
        
        self.Dq  = 0.2
        self.Dp  = 10.2763
        self.Ds  = 0.1316
        
        self.n   = 0.006
        self.ks  = 10000
        self.la  = 0.015
        
        # Disturbance parameters
        
        # Simulator parameters
        self.dt = 0.01
        self.max_episode_steps = 5000
        
        # Disturbance estimator parameters
        
        # State and action spaces
        action_space = np.array([50, 10])
        safe_action_space = np.array([40, 8])
        self.action_space = spaces.Box(low=-action_space, high=action_space, shape=(2,))
        self.safe_action_space = spaces.Box(low=-safe_action_space, high=safe_action_space, shape=(2,))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,)) # [q, θp, θs, dq, dθp, dθs, tau_e, k]
        
        # Initialize Env
        self.state = np.zeros((6,))
        self.obs = np.zeros((8,))
        self.uncertainty = np.zeros((3,))
        self.episode_step = 0
        
        self.reset()
        
        # Generate desired trajectory
    def generate_trajectory(self):
        pass
        
    def step(self, action):
        pass
        
    def get_reward(self):
        pass
    
    def get_done(self):
        pass
    
    def reset(self):
        
        self.episode_step = 0
        self.state = np.zeros((6,))
        self.uncertainty = np.zeros((3,))
        self.obs = np.zeros((8,))
        
        return self.state, self.obs
    
    def seed(self, s):
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(s)

    def get_dynamics(self):
        pass
    
    def get_cbf(self):
        pass
    
    def get_obs(self, states, step):
        pass