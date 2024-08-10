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
        m = 2
        g = 9.81
        d = 0.3
        
        M   = 0.072
        Jp  = 0.1055
        Js  = 0.000795
        
        Dq  = 0.2
        Dp  = 10.2763
        Ds  = 0.1316
        
        n   = 0.006
        ks  = 10000
        la  = 0.015
        
        # Disturbance parameters
        delta_Js    = 0.01 * Js
        delta_Dq    = 0.02 * Dq
        delta_Dp    = 0.02 * Dp
        delta_Ds    = 0.02 * Ds
        delta_G     = 0.01 * m * g
        tau_fq  = 0.05
        tau_fp  = 0.05
        tau_fs  = 0.05
        
        # Simulator parameters
        self.dt = 0.01
        self.max_episode_steps = 5000
        
        # Disturbance estimator parameters
        self.L1_theta_q = 0.05
        self.L1_theta_p = 0.09
        self.L1_theta_s = 0.12
        
        # variables for plotting
        self.overall_disturbance = []
        self.wind_disturbance = []
        self.friction_disturbance = []
        
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
        self.q_ref_traj, self.k_ref_traj = self.generate_trajectory()
        
        # Generate desired trajectory
    def generate_trajectory(self, q_range = 5, k_range = [30,50]):
        rate = 0.5*np.pi
        q_ref = []
        k_ref = []
        for t in np.arange(0, self.max_episode_steps*self.dt + self.dt, self.dt):
            q_ref.append(q_range*np.cos(rate * t)/2)
            k_ref.append((k_range[1]-k_range[0])/2 + (k_range[1]-k_range[0])*np.cos(rate * t)/2)
        return q_ref, k_ref
        
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
    
if __name__ == "__main__":

    import matplotlib.pyplot as plt
    env = VSAEnv()
    q_ref_traj, k_ref_traj = env.generate_trajectory()
    
    time = np.arange(0, env.max_episode_steps*env.dt + env.dt, env.dt)
    plt.figure(figsize = (12, 6))
    plt.subplot(2,1,1)
    plt.plot(time,q_ref_traj,label='q_ref')
    plt.title('k_ref vs Time')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(time, k_ref_traj, label='k_ref', color='orange')
    plt.title('k_ref vs Time')
    plt.legend()
    
    plt.tight_layout()
    plt.show()