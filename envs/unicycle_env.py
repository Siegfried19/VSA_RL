import numpy as np
import gym
from gym import spaces
from scipy.linalg import expm
# For plotting
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from datetime import datetime
import os


class UnicycleEnv(gym.Env):
    """Custom Environment that follows SafetyGym interface"""

    metadata = {'render.modes': ['human']}

    def __init__(self):

        super(UnicycleEnv, self).__init__()

        self.dynamics_mode = 'Unicycle'
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))
        self.safe_action_space = spaces.Box(low=-1.5, high=1.5, shape=(2,))
        self.observation_space = spaces.Box(low=-1e10, high=1e10, shape=(7,))
        
        # Environment parameters
        self.boundary = np.array([[-3., -3.], [3., 3.]])    #TODO: Add boundary cost
        self.dt = 0.2
        self.max_episode_steps = 400
        self.reward_goal = 1.0
        self.goal_size = 0.3
        
        # Build Hazerds
        self.hazards = []
        self.hazards.append({'type': 'circle', 'radius': 0.6, 'location': 1.5*np.array([0., 0.])})
        self.hazards.append({'type': 'circle', 'radius': 0.6, 'location': 1.5*np.array([-1., 1.])})
        self.hazards.append({'type': 'circle', 'radius': 0.6, 'location': 1.5*np.array([-1., -1.])})
        self.hazards.append({'type': 'circle', 'radius': 0.6, 'location': 1.5*np.array([1., -1.])})
        self.hazards.append({'type': 'circle', 'radius': 0.6, 'location': 1.5*np.array([1., 1.])})
        self.hazards_locations = np.array([hazard['location'] for hazard in self.hazards])
        self.hazards_radius = np.array([hazard['radius'] for hazard in self.hazards])
        
        # Disturbance estimator parameters L1 bound
        self.safe_bound = 1
        self.a = 1
        self.lt = 0.0 * self.safe_bound
        self.ld = 0.1 * np.sqrt(2) * self.safe_bound
        self.bd = 0.1 * self.safe_bound
        self.L1_theta = self.ld * 2.6 + self.bd
        self.L1_phi = self.L1_theta + 2.2
        self.L1_eta = self.lt + self.ld + self.L1_phi
        self.L1_gamma = 0.05 * 2.0 * np.sqrt(3) * self.L1_eta * self.dt + np.sqrt(3) * (1.0 - np.exp(-self.a * self.dt)) * self.L1_theta
        
        # Initialize Env
        self.state = None
        self.uncertainty = None
        self.episode_step = 0
        self.goal_pos = np.array([2.5, 2.5])
        
        self.render_flag = False
        self.reset()
        
        # Get Dynamics
        self.get_f, self.get_g = self._get_dynamics()

        # Viewer
        self.viewer = None

    def step(self, action):
        """Organize the observation to understand what's going on

        Parameters
        ----------
        action : ndarray
                Action that the agent takes in the environment

        Returns
        -------
        new_obs : ndarray
          The new observation with the following structure:
          [pos_x, pos_y, cos(theta), sin(theta), xdir2goal, ydir2goal, dist2goal]

        """

        # action = np.clip(action, -1.0, 1.0)
        state, reward, done, info = self._step(action)
        return self.get_obs(), reward, done, info

    def _step(self, action):
        """

        Parameters
        ----------
        action

        Returns
        -------
        state : ndarray
            New internal state of the agent.
        reward : float
            Reward collected during this transition.
        done : bool
            Whether the episode terminated.
        info : dict
            Additional info relevant to the environment.
        """

        # Start with our prior for continuous time system x' = f(x) + g(x)u
        self.uncertainty = -0.1 * 1 * self.get_g(self.state) @ np.array([np.cos(self.state[2]),  0])
        self.state = self.dt * (self.get_f(self.state) + self.get_g(self.state) @ action) + self.state
        self.state = self.dt * self.uncertainty + self.state

        self.episode_step += 1

        info = dict()

        dist_goal = self._goal_dist()
        reward = (self.last_goal_dist - dist_goal)
        self.last_goal_dist = dist_goal
        # Check if goal is met
        if self.goal_met():
            info['goal_met'] = True
            reward += self.reward_goal
            done = True
        else:
            done = self.episode_step >= self.max_episode_steps

        # Include constraint cost in reward
        info['cost'] = 0
        for hazard in self.hazards:
            if hazard['type'] == 'circle': # They should all be circles if 'default'
                penalty = 0.1 * (np.sum((self.state[:2] - hazard['location']) ** 2) < hazard['radius'] ** 2)
                info['cost'] += penalty
                reward -= penalty * 10
        if info['cost'] != 0:
            print("Warning, collision happened")
            done = True
        
        return self.state, reward, done, info

    def goal_met(self):
        """Return true if the current goal is met this step

        Returns
        -------
        goal_met : bool
            True if the goal condition is met.

        """

        return np.linalg.norm(self.state[:2] - self.goal_pos) <= self.goal_size

    def reset(self):
        """ Reset the state of the environment to an initial state.

        Returns
        -------
        observation : ndarray
            Next observation.
        """

        self.episode_step = 0

        # Re-initialize state
        self.state = np.array([-2.5, -2.5, 0.])
        self.uncertainty = np.array([0, 0, 0.])
        
        # Re-initialize last goal dist
        self.last_goal_dist = self._goal_dist()
        
        if self.render_flag:
            self.render_start()

        # TODO: Randomize this
        return self.get_obs(), self.get_obs()

    def get_obs(self):
        """Given the state, this function returns it to an observation akin to the one obtained by calling env.step

        Parameters
        ----------

        Returns
        -------
        observation : ndarray
          Observation: [pos_x, pos_y, cos(theta), sin(theta), xdir2goal, ydir2goal, exp(-dist2goal)]
        """

        rel_loc = self.goal_pos - self.state[:2]
        goal_dist = np.linalg.norm(rel_loc)
        goal_compass = self.obs_compass()  # compass to the goal

        return np.array([self.state[0], self.state[1], np.cos(self.state[2]), np.sin(self.state[2]), goal_compass[0], goal_compass[1], np.exp(-goal_dist)])

    def _get_dynamics(self):
        """Get affine CBFs for a given environment.

        Parameters
        ----------

        Returns
        -------
        get_f : callable
                Drift dynamics of the continuous system x' = f(x) + g(x)u
        get_g : callable
                Control dynamics of the continuous system x' = f(x) + g(x)u
        """

        def get_f(state):
            f_x = np.zeros(state.shape)
            return f_x

        def get_g(state):
            theta = state[2]
            g_x = np.array([[np.cos(theta), 0],
                            [np.sin(theta), 0],
                            [            0, 1.0]])
            return g_x

        return get_f, get_g

    # Transfer the degree to the robot frame
    def obs_compass(self):
        """
        Return a robot-centric compass observation of a list of positions.
        Compass is a normalized (unit-lenght) egocentric XY vector,
        from the agent to the object.
        This is equivalent to observing the egocentric XY angle to the target,
        projected into the sin/cos space we use for joints.
        (See comment on joint observation for why we do this.)
        """

        # Get ego vector in world frame
        vec = self.goal_pos - self.state[:2]
        # Rotate into frame
        R = np.array([[np.cos(self.state[2]), -np.sin(self.state[2])], [np.sin(self.state[2]), np.cos(self.state[2])]])
        vec = np.matmul(vec, R)
        # Normalize
        vec /= np.sqrt(np.sum(np.square(vec))) + 0.001
        return vec

    def _goal_dist(self):
        return np.linalg.norm(self.goal_pos - self.state[:2])
    
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
    
    # All the rendering functions
    def render_start(self):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(self.boundary[0][0], self.boundary[1][0])
        self.ax.set_ylim(self.boundary[0][1], self.boundary[1][1])
        self.ax.set_aspect('equal')
        
        for hazard in self.hazards:
            hazards = plt.Circle(hazard['location'], hazard['radius'], color = 'red', fill = True)
            self.ax.add_patch(hazards)
            
        target = plt.Circle(self.goal_pos, 0.3, color = 'green', fill = True)
        self.ax.add_patch(target)
        
        self.path_data, = self.ax.plot([], [], 'g--', linewidth=1)
        self.robot, = self.ax.plot([], [], 'ro', markersize=5)
        self.target, = self.ax.plot([], [], 'bo', markersize=5)
        self.x_robot = []
        self.y_robot = []
        
        self.save_folder = "animations"
        os.makedirs(self.save_folder, exist_ok=True)
        
    def render_save(self):
        self.x_robot.append(self.state[0])
        self.y_robot.append(self.state[1])
        
    def init(self):
        self.path_data.set_data([], [])
        self.robot.set_data([], [])
        return self.path_data, self.robot
    
    def update(self, frame):
        i = int(frame)
        self.robot.set_data(self.x_robot[i], self.y_robot[i])
        old_x, old_y = self.path_data.get_data()
        new_x = np.append(old_x, self.x_robot[i])
        new_y = np.append(old_y, self.y_robot[i])
        self.path_data.set_data(new_x, new_y)
        
        return self.path_data, self.robot
    
    def render_activate(self):  
        print("Rendering")     
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d%H%M%S")
        filename = f"{self.save_folder}/Unicycle_{timestamp}.mp4"
        
        ani = FuncAnimation(self.fig, self.update, frames=np.linspace(0, len(self.x_robot)-1, len(self.x_robot)),init_func=self.init, blit=True, interval=(len(self.x_robot)-1)/24)
        ani.save(filename, fps=24, extra_args=['-vcodec', 'libx264'])
        plt.close(self.fig)  
        plt.close('all') 
    
    def get_random_hazard_locations(n_hazards, hazard_radius, bds=None):
        """

        Parameters
        ----------
        n_hazards : int
            Number of hazards to create
        hazard_radius : float
            Radius of hazards
        bds : list, optional
            List of the form [[x_lb, x_ub], [y_lb, y_ub] denoting the bounds of the 2D arena

        Returns
        -------
        hazards_locs : ndarray
            Numpy array of shape (n_hazards, 2) containing xy locations of hazards.
        """

        if bds is None:
            bds = np.array([[-3., -3.], [3., 3.]])

        # Create buffer with boundaries
        buffered_bds = bds
        buffered_bds[0] += hazard_radius
        buffered_bds[1] -= hazard_radius

        hazards_locs = np.zeros((n_hazards, 2))

        for i in range(n_hazards):
            successfully_placed = False
            iter = 0
            while not successfully_placed and iter < 500:
                hazards_locs[i] = (bds[1] - bds[0]) * np.random.random(2) + bds[0]
                successfully_placed = np.all(np.linalg.norm(hazards_locs[:i] - hazards_locs[i], axis=1) > 3*hazard_radius)
                iter += 1

            if iter >= 500:
                raise Exception('Could not place hazards in arena.')
        return hazards_locs