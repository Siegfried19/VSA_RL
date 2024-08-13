import numpy as np
import gym
from gym import spaces
from scipy.linalg import expm
import torch
# from utils import *
import matplotlib.pyplot as plt
from datetime import datetime
import os

def expand_dim(state):
    expand_dim = len(state.shape) == 1
    if expand_dim:
        if isinstance(state, np.ndarray):
            state = np.expand_dims(state, axis=0)
        else:
            state = state.unsqueeze(0)
    return state,expand_dim

def narrow_dim(state,expand_dim):
    if expand_dim:
        if isinstance(state, np.ndarray):
            state = np.squeeze(state, axis=0)
        else:
            state = state.squeeze(0)
    return state

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
        
        self.def_max = np.pi/3
        
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
        
        # Disturbance estimator parameters, fro dynamics of q, theta_p, theta_s
        self.Ae = np.array([0,0,0,1,1,1])
        self.L1_delta = np.array([0,0,0,0.09, 0.09, 0.12])
        
        # variables for plotting
        self.dwq_record = []
        self.dwp_record = []
        self.dws_record = []
        
        # State and action spaces
        action_space = np.array([15, 6])
        safe_action_space = np.array([12, 4])
        
        self.action_space = spaces.Box(low=-action_space, high=action_space, shape=(2,))
        self.safe_action_space = spaces.Box(low=-safe_action_space, high=safe_action_space, shape=(2,))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,)) # [q, θp, θs, dq, dθp, dθs, q_ref, k_ref]
        
        # Initialize Env
        self.state = np.zeros((6,))
        self.obs = np.zeros((8,))
        self.uncertainty = np.zeros((3,))
        self.episode_step = 0
        
        # # Generate desired trajectory
        # self.q_ref_traj, self.k_ref_traj = self.generate_trajectory()
        
        # Render trjectory
        self.render_flag = False
        # self.reset()
            
    # Generate desired trajectory
    def generate_trajectory(self, q_start, k_start, rate_q, rate_k, q_range = 3*np.pi, k_range = [5,55]):
        q_ref = []
        k_ref = []
        q_phase_shift = np.arccos(2*q_start/q_range)
        k_phase_shift = np.arcsin(k_start/(k_range[1]-k_range[0]))
        
        for t in np.arange(0, self.max_episode_steps*self.dt + self.dt, self.dt):
            q_ref.append(q_range*np.cos(rate_q * t + q_phase_shift)/2)
            k_ref.append((k_range[1]-k_range[0])/2 + (k_range[1]-k_range[0])*np.cos(rate_k * t + k_phase_shift)/2)
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
        tau_e , tau_r, k = self.get_intermediate(self.state)
        
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
        
        self.uncertainty = np.array([dwq, dwp, dws])
        # Update the state
        self.state = (self.get_f(self.state) + self.get_g(self.state) @ action + np.array([0, 0, 0, dwq, dwp, dws])) * self.dt + self.state
        self.episode_step += 1
        reward = 0
        
        info = dict()
        
        # Get the reward
        if use_reward:
            reward = self.get_reward()
        
        # Check if the episode is done
        if (self.state[0] - self.state[1]) > self.def_max or (self.state[0] - self.state[1]) < -self.def_max:
            info['max_deflection_hitted'] = True
            penalty = -20 * (self.state[0] - self.state[1])**2
            reward += penalty
            done = True
        else:
            done = self.episode_step >= self.max_episode_steps
        if done:
            info['reach_max_steps'] = True
        
        return self.state, reward, done, info
    
    def get_reward(self):
        index = min(self.episode_step, self.max_episode_steps)
        q_error = 0.3 * np.abs(self.q_ref_traj[index] - self.state[0])
        k_error = 0.1 * np.abs(self.k_ref_traj[index] - self.state[2])
        dist = q_error + k_error
        reward = -dist
        return reward
    
    def get_done(self):
        if (self.state[0] - self.state[1]) > np.pi/3 or (self.state[0] - self.state[1]) < -np.pi/3:
            return True
        else:
            return False
    
    def reset(self, random_init = False):
            
        self.episode_step = 0
        self.state = np.zeros((6,))
        self.uncertainty = np.zeros((3,))
        self.obs = np.zeros((8,))
        self.tau_ext = self.generate_tau_ext()  
        
        if random_init:
            self.state[0] = np.random.uniform(-np.pi/2, np.pi/2)
            self.state[1] = self.state[0] + np.random.uniform(-np.pi/8, np.pi/8)
            self.state[2] = np.random.uniform(np.pi*0.45, np.pi*0.6)
            rate_q = np.random.uniform(0.1*np.pi, 0.5*np.pi)
            rate_k = rate_q*np.random.uniform(0.3, 1)
        else:
            self.state[0] = np.pi/4
            self.state[1] = np.pi/8
            self.state[2] = np.pi*0.6
            rate_q = 0.3*np.pi
            rate_k = rate_q
        _, _, k = self.get_intermediate(self.state)
        self.q_ref_traj, self.k_ref_traj = self.generate_trajectory(self.state[0], k, rate_q, rate_k)
        
        if self.render_flag:
            self.render_start()
        
        return self.state, self.obs
    
    def seed(self, s):
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(s)

    def get_dynamics(self):
        def get_f(state):
            state, expand = expand_dim(state)
            tau_e, tau_r, k = self.get_intermediate(state)
            if isinstance(state, torch.Tensor):
                f_x = torch.zeros(state.shape)
            else:
                f_x = np.zeros(state.shape)
            f_x[:,0] = state[:,3]
            f_x[:,1] = state[:,4]
            f_x[:,2] = state[:,5]
            f_x[:,3] = 1/self.M * (-self.Dq*state[:,3] - self.m*self.g*self.d*np.sin(state[:,0])) - 1/self.M * tau_e
            f_x[:,4] = 1/self.Jp * (-self.Dp*state[:,4] + tau_e)
            f_x[:,5] = 1/self.Js * (-self.Ds*state[:,5] - tau_r)
            f_x = narrow_dim(f_x, expand)
            return f_x
        
        def get_g(state):
            state, expand = expand_dim(state)
            if isinstance(state, torch.Tensor):
                g_x = torch.zeros((state.shape[0], state.shape[1], 2))
            else:
                g_x = np.zeros((state.shape[0], state.shape[1], 2))
            g_x[:,4,0] = 1/self.Jp
            g_x[:,5,1] = 1/self.Js
            g_x = narrow_dim(g_x, expand)
            return g_x
        return get_f, get_g
    
    def get_intermediate(self, state):
        state, expand = expand_dim(state)
        phi = state[:,0] - state[:,1]
        tau_e = (2*self.ks * (self.n * state[:,2])**2 * self.la**2 * phi) / (self.la - self.n * state[:,2])**2
        tau_r = 2*self.ks * self.n**2 * state[:,2] * self.la**3 * phi**2 / (self.la - self.n * state[:,2])**3
        k = 2*self.ks * (self.n * state[:,2])**2 * self.la**2 / (self.la - self.n * state[:,2])**2
        
        tau_e = narrow_dim(tau_e, expand)
        tau_r = narrow_dim(tau_r, expand)
        k = narrow_dim(k, expand)
        return tau_e, tau_r, k
    
    def get_obs(self, state, episode_step):
        obs = np.zeros((8,))
        obs[:6] = state
        obs[6] = self.q_ref_traj[episode_step]
        obs[7] = self.k_ref_traj[episode_step]
        return obs
    
    def get_state(obs):
        return np.copy(obs[:6])
    
    def render_start(self):
        self.q_real_traj_plot = []
        self.k_real_traj_plot = []
        self.q_ref_traj_plot = []
        self.k_ref_traj_plot = []
        self.save_folder = "plots"
        os.makedirs(self.save_folder, exist_ok=True)
    
    def render_save(self):
        _, _, k = self.get_intermediate(self.state)
        self.q_real_traj_plot.append(self.state[0])
        self.k_real_traj_plot.append(k)
        self.q_ref_traj_plot.append(self.q_ref_traj[self.episode_step])
        self.k_ref_traj_plot.append(self.k_ref_traj[self.episode_step])
        
    def render_activate(self):
        time = np.linspace(0, len(self.q_real_traj_plot)*self.dt, len(self.q_real_traj_plot))
        plt.figure(figsize = (12, 6))
        
        plt.subplot(2,1,1)
        plt.plot(time, self.q_real_traj_plot, label='q_real', color='orange')
        plt.plot(time, self.q_ref_traj_plot, label='q_ref', color='green')
        plt.title('q trajectory')
        plt.legend()
        
        plt.subplot(2, 1, 2)
        plt.plot(time, self.k_real_traj_plot, label='k_real', color='orange')
        plt.plot(time, self.k_ref_traj_plot, label='k_ref', color='green')
        plt.title('k trajectory')
        plt.legend()
        
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d%H%M%S")
        filename = f"{self.save_folder}/VSA_{timestamp}.png"
        plt.savefig(filename)
        plt.close()
        

if __name__ == "__main__":

    env = VSAEnv()   
    env.reset(random_init = True)
    q_ref_traj, k_ref_traj = env.q_ref_traj, env.k_ref_traj
    print(env.state[0])
    _,_,k = env.get_intermediate(env.state)
    print(k)
    
    time = np.arange(0, env.max_episode_steps*env.dt + env.dt, env.dt)
    plt.figure(figsize = (12, 6))
    plt.subplot(2,1,1)
    plt.plot(time, q_ref_traj,label='q_ref')
    plt.title('q_ref vs Time')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(time, k_ref_traj, label='k_ref', color='orange')
    plt.title('k_ref vs Time')
    plt.legend()
    
    plt.tight_layout()
    plt.show()