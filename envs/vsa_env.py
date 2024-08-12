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
        self.get_f, self.get_g = self.get_dynamics()
        # VSA parameters
        self.m = 2
        self.g = 9.81
        self.d = 0.3
        
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
        self.delta_Js    = 0.01 * self.Js
        self.delta_Dq    = 0.02 * self.Dq
        self.delta_Dp    = 0.02 * self.Dp
        self.delta_Ds    = 0.02 * self.Ds
        self.delta_G     = 0.01 * self.m * self.g
        self.tau_fq  = 0.05
        self.tau_fp  = 0.05
        self.tau_fs  = 0.05
        
        # environment parameters
        self.dt = 0.01
        self.max_episode_steps = 5000
        
        # Disturbance estimator parameters
        self.Ae = 1.0
        self.L1_theta_q = 0.09
        self.L1_theta_p = 0.09
        self.L1_theta_s = 0.12
        
        # variables for plotting
        self.dwq_record = []
        self.dwp_record = []
        self.dws_record = []
        
        # State and action spaces
        action_space = np.array([50, 10])
        safe_action_space = np.array([40, 8])
        
        self.action_space = spaces.Box(low=-action_space, high=action_space, shape=(2,))
        self.safe_action_space = spaces.Box(low=-safe_action_space, high=safe_action_space, shape=(2,))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,)) # [q, θp, θs, dq, dθp, dθs, q_ref, k_ref]
        
        # Initialize Env
        self.state = np.zeros((6,))
        self.obs = np.zeros((8,))
        self.uncertainty = np.zeros((3,))
        self.episode_step = 0
        
        # Generate desired trajectory
        self.q_ref_traj, self.k_ref_traj = self.generate_trajectory()
        
        self.reset() 
        
    # Generate desired trajectory
    def generate_trajectory(self, q_range = 5, k_range = [30,50], rate = 0.5*np.pi):
        q_ref = []
        k_ref = []
        for t in np.arange(0, self.max_episode_steps*self.dt + self.dt, self.dt):
            q_ref.append(q_range*np.cos(rate * t)/2)
            k_ref.append((k_range[1]-k_range[0])/2 + (k_range[1]-k_range[0])*np.cos(rate * t)/2)
        return q_ref, k_ref
    
    def generate_tau_ext(self):
        tau_e = []
        tau_e.append(np.random.uniform(1, 5))
        for t in np.arange(0, self.max_episode_steps*self.dt + self.dt, self.dt):
            delta_tau_e = np.random.uniform(-2*self.dt, 2*self.dt)
            tau_e.append(tau_e[-1] + delta_tau_e)
            tau_e[-1] = np.clip(tau_e[-1], 1, 5)
        return tau_e
        
    def step(self, action, use_reward = True):
        # Calculate the disturbance
        dwq = 0.0
        dwp = 0.0
        dws = 0.0
        
        x1 = self.state[0]
        x2 = self.state[1]
        x3 = self.state[2]
        x4 = self.state[3]
        x5 = self.state[4]
        x6 = self.state[5]    
        
        phi = x1 - x2
        tau_e , tau_r, k = self.get_intermediate()
        
        self.delta_M = 0.01*self.M + 0.01*self.M*abs(phi) + 0.05*self.M*x3
        self.delta_Jp = 0.1*self.Jp + 0.03*self.Jp*x3
        delta_tau_e = 0.05 * 0.1 * phi + 0.04*x3
        delta_tau_r = 0.015 + 0.04*phi + 0.02*x3
        tau_gp = 0.1 * self.g * np.sin(x2)
        
        dwq_pos = 1/self.M * (self.Dq*x4 + self.m*self.g*self.d*np.sin(x1) + tau_e) + self.tau_ext[self.episode_step]/(self.M + self.delta_M)
        dwq_neg = ((self.Dq + self.delta_Dq)*x4 + (tau_e + delta_tau_e) + (self.m*self.g + self.delta_G)*self.d*np.sin(x1) + self.tau_fq) / (self.M + self.delta_M)
        dwq = dwq_pos - dwq_neg
        dwp = 1/self.Jp * (self.Dp*x5 - tau_e - action[0] - tau_gp) - ((self.Dp + self.delta_Dp)*x5 + tau_e + delta_tau_e + tau_gp - self.tau_fp + action[0])/(self.Jp + self.delta_Jp)
        dws = 1/self.Js * (self.Ds*x6 + tau_r - action[1]) - ((self.Ds + self.delta_Ds)*x6 - tau_r - delta_tau_r - self.tau_fs + action[1])/(self.Js + self.delta_Js)
        
        # Update the state
        self.state = (self.get_f(self.state) + self.get_g(self.state) @ action + np.array([0, 0, 0, dwq, dwp, dws])) * self.dt + self.state
        self.episode_step += 1
        reward = 0
        
        info = dict()
        
        # Get the reward
        if use_reward:
            reward = self.get_reward()
        
        # Check if the episode is done
        if (self.state[0] - self.state[1]) > np.pi/3 or (self.state[0] - self.state[1]) < -np.pi/3:
            info['max_deflection_hitted'] = True
            print('Max deflection hitted')
            penalty = -100 * (self.state[0] - self.state[1])^2
            reward += penalty
            done = True
        else:
            done = self.episode_step >= self.max_episode_steps
        if done:
            info['reach_max_steps'] = True
        
        return self.state, reward, done, info
        
    def get_reward(self):
        index = min(self.episode_step, self.max_episode_steps-1)
        q_error = 10 * np.abs(self.q_ref_traj[index] - self.state[0])
        k_error = 0.5 * np.abs(self.k_ref_traj[index] - self.state[2])
        dist = q_error + k_error
        reward = -dist
        return reward
    
    def get_done(self):
        if (self.state[0] - self.state[1]) > np.pi/3 or (self.state[0] - self.state[1]) < -np.pi/3:
            return True
        else:
            return False
    
    def reset(self):
            
        self.episode_step = 0
        self.state = np.zeros((6,))
        self.uncertainty = np.zeros((3,))
        self.obs = np.zeros((8,))
        self.tau_ext = self.generate_tau_ext()
        
        return self.state, self.obs
    
    def seed(self, s):
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(s)

    def get_dynamics(self):
        tau_e, tau_r, k = self.get_intermediate()
        def get_f(state):
            f_x = np.zeros(state.shape)
            f_x[0] = state[3]
            f_x[1] = state[4]
            f_x[2] = state[5]
            f_x[3] = 1/self.M * (-self.Dq*state[3] - self.m*self.g*self.d*np.sin(state[0]) - tau_e)
            f_x[4] = 1/self.Jp * (-self.Dp*state[4] + tau_e)
            f_x[5] = 1/self.Js * (-self.Ds*state[5] - tau_r)
            return f_x
        def get_g(state):
            g_x = np.zeros((state.shape[0], 2))
            g_x[4,0] = 1/self.Jp
            g_x[5,1] = 1/self.Js
            return g_x
    
    def get_intermediate(self):
        phi = self.state[0] - self.state[1]
        tau_e = (2*self.ks * (self.n*self.state[2])^2 * self.la^2 * phi) / (self.la - self.n * self.state[2])^2
        tau_r = 2*self.ks * self.n^2 * self.state[2] * self.la^3 * phi^2 / (self.la - self.n * self.state[2])^3
        k = 2*self.ks * (self.n * self.state[2])^2 * self.la^2 / (self.la - self.n * self.state[2])^2
        return tau_e, tau_r, k
    
    def get_state(obs):
        return obs
    
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