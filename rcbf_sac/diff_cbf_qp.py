import argparse
import numpy as np
import torch
from rcbf_sac.dynamics import DYNAMICS_MODE
from rcbf_sac.utils import to_tensor, prRed
from time import time
from qpth.qp import QPFunction
from quadprog import solve_qp

class CBFQPLayer:

    def __init__(self, env, args, gamma_b=10, k_d=1.5, l_p=0.03):
        """Constructor of CBFLayer.

        Parameters
        ----------
        env : gym.env
            Gym environment.
        gamma_b : float, optional
            gamma of control barrier certificate.
        k_d : float, optional
            confidence parameter desired (2.0 corresponds to ~95% for example).
        """

        self.device = torch.device("cuda" if args.cuda else "cpu")

        self.env = env
        self.u_min, self.u_max = self.get_control_bounds()
        self.gamma_b = gamma_b

        if self.env.dynamics_mode not in DYNAMICS_MODE:
            raise Exception('Dynamics mode not supported.')

        if self.env.dynamics_mode == 'Unicycle':
            self.num_cbfs = len(env.hazards_locations)
            self.k_d = k_d  # Useless in regular use
            self.l_p = l_p
        elif self.env.dynamics_mode == 'SimulatedCars':
            self.num_cbfs = 2
        elif self.env.dynamics_mode == 'Quadrotor':
            self.num_cbfs = 1
        elif self.env.dynamics_mode == 'VSA': # TODO: Check this later
            self.num_cbfs = 1

        self.action_dim = env.action_space.shape[0]
        self.num_ineq_constraints = self.num_cbfs + 2 * self.action_dim

    def get_safe_action(self, state_batch, action_batch, mean_pred_batch, sigma_batch, use_L1=False):
        """

        Parameters
        ----------
        state_batch : torch.tensor or ndarray
        action_batch : torch.tensor or ndarray
            State batch
        mean_pred_batch : torch.tensor or ndarray
            Mean of disturbance
        sigma_batch : torch.tensor or ndarray
            Standard deviation of disturbance
        use_L1 : bool, optional for turning on L1 estimation

        Returns
        -------
        final_action_batch : torch.tensor
            Safe actions to take in the environment.
        """

        # batch form if only a single data point is passed
        expand_dims = len(state_batch.shape) == 1
        if expand_dims:
            action_batch = action_batch.unsqueeze(0)
            state_batch = state_batch.unsqueeze(0)
            mean_pred_batch = mean_pred_batch.unsqueeze(0)
            sigma_batch = sigma_batch.unsqueeze(0)

        start_time = time()
        Ps, qs, Gs, hs, h_value = self.get_cbf_qp_constraints(state_batch, action_batch, mean_pred_batch, sigma_batch, use_L1)
        build_qp_time = time()
        safe_action_batch = self.solve_qp(Ps, qs, Gs, hs)
        # prCyan('Time to get constraints = {} - Time to solve QP = {} - time per qp = {} - batch_size = {} - device = {}'.format(build_qp_time - start_time, time() - build_qp_time, (time() - build_qp_time) / safe_action_batch.shape[0], Ps.shape[0], Ps.device))
        # The actual safe action is the cbf action + the nominal action
        # if self.env.dynamics_mode == 'VSA':
        #     final_action = torch.clamp(action_batch, self.u_min.repeat(action_batch.shape[0], 1), self.u_max.repeat(action_batch.shape[0], 1))
        #     final_action[:,0] = torch.clamp(action_batch[:,0] + safe_action_batch[:,0], self.u_min.repeat(action_batch.shape[0], 1)[:,0], self.u_max.repeat(action_batch.shape[0], 1)[:,0])
        # else:    
        #     final_action = torch.clamp(action_batch + safe_action_batch, self.u_min.repeat(action_batch.shape[0], 1), self.u_max.repeat(action_batch.shape[0], 1))
        final_action = torch.clamp(action_batch + safe_action_batch, self.u_min.repeat(action_batch.shape[0], 1), self.u_max.repeat(action_batch.shape[0], 1))
        return final_action if not expand_dims else final_action.squeeze(0), h_value

    def solve_qp(self, Ps: torch.Tensor, Qs: torch.Tensor, Gs: torch.Tensor, Hs: torch.Tensor):
        """Solves:
            minimize_{u,eps} 0.5 * u^T P u + q^T u
                subject to G[u,eps]^T <= h

        Parameters
        ----------
        Ps : torch.Tensor
            (batch_size, n_u+1, n_u+1)
        Qs : torch.Tensor
            (batch_size, n_u+1)
        Gs : torch.Tensor
            (batch_size, num_ineq_constraints, n_u+1)
        Hs : torch.Tensor
            (batch_size, num_ineq_constraints)
        Returns
        -------
        safe_action_batch : torch.tensor
            The solution of the qp without the last dimension (the slack).
        """
        Ghs = torch.cat((Gs, Hs.unsqueeze(2)), -1)
        Ghs_norm = torch.max(torch.abs(Ghs), dim=2, keepdim=True)[0]
        Gs /= Ghs_norm
        hs = Hs / Ghs_norm.squeeze(-1)
        # sol = self.cbf_layer(Ps, qs, Gs, hs, solver_args={"check_Q_spd": False, "maxIter": 100000, "notImprovedLim": 10, "eps": 1e-4})
        sol = to_tensor(self.cbf_layer_quadprog(Ps, Qs, Gs, hs), torch.FloatTensor, self.device)
        sol = sol.unsqueeze(0) if len(sol.shape) == 1 else sol
        safe_action_batch = sol[:, :-1]
        return safe_action_batch

    def cbf_layer(self, Ps, Qs, Gs, hs, As=None, bs=None, solver_args=None):
        """

        Parameters
        ----------
        Ps : torch.Tensor
        Qs : torch.Tensor
        Gs : torch.Tensor
            shape (batch_size, num_ineq_constraints, num_vars)
        hs : torch.Tensor
            shape (batch_size, num_ineq_constraints)
        As : torch.Tensor, optional
        bs : torch.Tensor, optional
        solver_args : dict, optional

        Returns
        -------
        result : torch.Tensor
            Result of QP
        """

        if solver_args is None:
            solver_args = {}

        if As is None or bs is None:
            As = torch.Tensor().to(self.device).double()
            bs = torch.Tensor().to(self.device).double()

        result = QPFunction(verbose=0, **solver_args)(Ps.double(), Qs.double(), Gs.double(), hs.double(), As, bs).float()
        if torch.any(torch.isnan(result)):
            prRed('QP Failed to solve - result is nan == {}!'.format(torch.any(torch.isnan(result))))
            raise Exception('QP Failed to solve')
        return result
    
    def cbf_layer_quadprog(self, Ps, Qs, Gs, hs, As=None, bs=None):
        P_np = Ps.cpu().detach().numpy() if isinstance(Ps, torch.Tensor) else Ps
        q_np = Qs.cpu().detach().numpy() if isinstance(Qs, torch.Tensor) else Qs
        G_np = Gs.cpu().detach().numpy() if isinstance(Gs, torch.Tensor) else Gs
        h_np = hs.cpu().detach().numpy() if isinstance(hs, torch.Tensor) else hs
        
        P_np = np.squeeze(P_np).astype(np.double)
        q_np = np.squeeze(q_np).astype(np.double)
        G_np = np.squeeze(G_np).astype(np.double)
        h_np = np.squeeze(h_np).astype(np.double)
        
        if len(G_np.shape) == 1:
            h_np = np.expand_dims(h_np, axis=0)
            G_np = np.expand_dims(G_np, axis=0)
        G_qp = -G_np
        h_qp = -h_np
        try:
            solution = solve_qp(P_np, q_np, G_qp.T, h_qp)[0]   
        except ValueError:
            print("Infeasible QP, unsaft action execute")
            solution = torch.zeros((Gs.shape[0], Gs.shape[2]))
        return solution

    def get_cbf_qp_constraints(self, state_batch, action_batch, mean_pred_batch, sigma_pred_batch, use_L1=False):
        """Build up matrices required to solve qp
        Program specifically solves:
            minimize_{u,eps} 0.5 * u^T P u + q^T u
                subject to G[u,eps]^T <= h

        Each control barrier certificate is of the form:
            dh/dx^T (f_out + g_out u) >= -gamma^b h_out^3 where out here is an output of the state.

        In the case of SafetyGym_point dynamics:
        state = [x y θ v ω]
        state_d = [v*cos(θ) v*sin(θ) omega ω u^v u^ω]

        Quick Note on batch matrix multiplication for matrices A and B:
            - Batch size should be first dim
            - Everything needs to be 3-dimensional
            - E.g. if B is a vec, i.e. shape (batch_size, vec_length) --> .view(batch_size, vec_length, 1)

        Parameters
        ----------
        state_batch : torch.tensor
            current state (check dynamics.py for details on each dynamics' specifics)
        action_batch : torch.tensor
            Nominal control input.
        mean_pred_batch : torch.tensor
            mean disturbance prediction state, dimensions (n_s, n_u)
        sigma_pred_batch : torch.tensor
            standard deviation in additive disturbance after undergoing the output dynamics.
        gamma_b : float, optional
            CBF parameter for the class-Kappa function

        Returns
        -------
        P : torch.tensor
            Quadratic cost matrix in qp (minimize_{u,eps} 0.5 * u^T P u + q^T u)
        q : torch.tensor
            Linear cost vector in qp (minimize_{u,eps} 0.5 * u^T P u + q^T u)
        G : torch.tensor
            Inequality constraint matrix (G[u,eps] <= h) of size (num_constraints, n_u + 1)
        h : torch.tensor
            Inequality constraint vector (G[u,eps] <= h) of size (num_constraints,)
        """

        assert len(state_batch.shape) == 2 and len(action_batch.shape) == 2 and len(mean_pred_batch.shape) == 2 and len(sigma_pred_batch.shape) == 2, print(state_batch.shape, action_batch.shape, mean_pred_batch.shape,sigma_pred_batch.shape)

        batch_size = state_batch.shape[0]
        gamma_b = self.gamma_b
        # Expand dims
        state_batch = torch.unsqueeze(state_batch, -1)
        action_batch = torch.unsqueeze(action_batch, -1)
        mean_pred_batch = torch.unsqueeze(mean_pred_batch, -1)
        sigma_pred_batch = torch.unsqueeze(sigma_pred_batch, -1)
        
        h_value = 0
        self.use_L1 = use_L1

        if self.env.dynamics_mode == 'Unicycle':
            
            # Constrain setup
            num_cbfs = self.num_cbfs
            n_u = action_batch.shape[1]
            num_constraints = self.num_cbfs + 2 * n_u
            # Inequality constraints (G[u, eps] <= h)
            G = torch.zeros((batch_size, num_constraints, n_u + 1)).to(self.device)  # the extra variable is for epsilon (to make sure qp is always feasible)
            h = torch.zeros((batch_size, num_constraints)).to(self.device)
            ineq_constraint_counter = 0
            
            # Set up the CBFs
            hazards_radius = self.env.hazards_radius
            hazards_locations = to_tensor(self.env.hazards_locations, torch.FloatTensor, self.device)
            collision_radius = 1.2 * hazards_radius  # add a little buffer
            l_p = self.l_p
            
            # Build the dynamics that we used to analyze the CBF
            thetas = state_batch[:, 2, :].squeeze(-1)
            c_thetas = torch.cos(thetas)
            s_thetas = torch.sin(thetas)

            # p(x): lookahead output (batch_size, 2), we use this as current state of x,y
            ps = torch.zeros((batch_size, 2)).to(self.device)
            ps[:, 0] = state_batch[:, 0, :].squeeze(-1) + l_p * c_thetas
            ps[:, 1] = state_batch[:, 1, :].squeeze(-1) + l_p * s_thetas

            # The real dynamics we consider p_dot(x) = f_p(x) + g_p(x)u + D_p where f_p(x) = 0,  g_p(x) = RL and D_p is the disturbance that contains all
            # Ideal f_p(x) = [0,...,0]^T
            f_ps = torch.zeros((batch_size, 2, 1)).to(self.device)

            # g_p(x) = RL where L = diag([1, l_p])
            Rs = torch.zeros((batch_size, 2, 2)).to(self.device)
            Rs[:, 0, 0] = c_thetas
            Rs[:, 0, 1] = -s_thetas
            Rs[:, 1, 0] = s_thetas
            Rs[:, 1, 1] = c_thetas
            Ls = torch.zeros((batch_size, 2, 2)).to(self.device)
            Ls[:, 0, 0] = 1
            Ls[:, 1, 1] = l_p
            g_ps = torch.bmm(Rs, Ls)  # (batch_size, 2, 2)
            
            # The disturbance esitmation for CBF terms
            # D_p(x) = g_p [0 D_θ]^T + [D_x1 D_x2]^T
            mu_theta_aug = torch.zeros([batch_size, 2, 1]).to(self.device)
            mu_theta_aug[:, 1, :] = mean_pred_batch[:, 2, :]
            mu_ps = torch.bmm(g_ps, mu_theta_aug) + mean_pred_batch[:, :2, :]
            
            # Extra one for GP
            sigma_theta_aug = torch.zeros([batch_size, 2, 1]).to(self.device)
            sigma_theta_aug[:, 1, :] = sigma_pred_batch[:, 2, :]
            sigma_ps = torch.bmm(torch.abs(g_ps), sigma_theta_aug) + sigma_pred_batch[:, :2, :]
            
            # Disturbance For l1
            fD = torch.add(f_ps, mu_ps)

            # h_cbf (batch_size, hazards_locations)
            ps_hzds = ps.repeat((1, num_cbfs)).reshape((batch_size, num_cbfs, 2))
            h_cbf = torch.zeros((batch_size, num_cbfs))
            h_cbf = 0.5 * ((ps_hzds[:, :, 0] - hazards_locations[:, 0]) ** 2 + (ps_hzds[:, :, 1] - hazards_locations[:, 1]) ** 2 - collision_radius ** 2)
            if  torch.any(h_cbf < 0):
                    h_value = 1

            # Partial derivative of h_cbf w.r.t to x for cbf with relative degree 1 (batch, num_cbfs, 2)
            dh_cbf = ps_hzds - hazards_locations
            
            if self.use_L1:
                G[:, :num_cbfs, :n_u] = -torch.bmm(dh_cbf, g_ps)
                G[:, :num_cbfs, n_u] = -1
                h[:, :num_cbfs] = gamma_b * (h_cbf ** 3) + (torch.bmm(dh_cbf, fD) + torch.bmm(torch.bmm(dh_cbf, g_ps), action_batch)).squeeze(-1) - torch.mul(torch.norm(dh_cbf, dim = 2), torch.full((batch_size,), self.env.L1_gamma))
                # print(h,torch.mul(torch.norm(dh_cbf.view(batch_size, -1), dim = 1), torch.full((batch_size,), self.env.L1_gamma)))
                ineq_constraint_counter += num_cbfs
            else:
                G[:, :num_cbfs, :n_u] = -torch.bmm(dh_cbf, g_ps)
                G[:, :num_cbfs, n_u] = -1
                h[:, :num_cbfs] = gamma_b * (h_cbf ** 3) + (torch.bmm(dh_cbf, fD) - torch.bmm(torch.abs(dh_cbf), sigma_ps) + torch.bmm(torch.bmm(dh_cbf, g_ps), action_batch)).squeeze(-1)
                ineq_constraint_counter += num_cbfs
                
            P = torch.diag(torch.tensor([1.e0, 1.e-2, 1e5])).repeat(batch_size, 1, 1).to(self.device)
            q = torch.zeros((batch_size, n_u + 1)).to(self.device)

        elif self.env.dynamics_mode == 'SimulatedCars':

            n_u = action_batch.shape[1]  # dimension of control inputs
            num_constraints = self.num_cbfs + 2 * n_u  # each cbf is a constraint, and we need to add actuator constraints (n_u of them)
            collision_radius = 3.5

            # Inequality constraints (G[u, eps] <= h)
            G = torch.zeros((batch_size, num_constraints, n_u + 1)).to(self.device)  # the extra variable is for epsilon (to make sure qp is always feasible)
            h = torch.zeros((batch_size, num_constraints)).to(self.device)
            ineq_constraint_counter = 0

            # Current State
            pos = state_batch[:, ::2, 0]
            vels = state_batch[:, 1::2, 0]

            # Action (acceleration)
            vels_des = 30.0 * torch.ones((state_batch.shape[0], 5)).to(self.device)  # Desired velocities
            # vels_des[:, 0] -= 10 * torch.sin(0.2 * t_batch)
            accels = self.env.kp * (vels_des - vels)
            accels[:, 1] -= self.env.k_brake * (pos[:, 0] - pos[:, 1]) * ((pos[:, 0] - pos[:, 1]) < 6.0)
            accels[:, 2] -= self.env.k_brake * (pos[:, 1] - pos[:, 2]) * ((pos[:, 1] - pos[:, 2]) < 6.0)
            accels[:, 3] = 0.0  # Car 4's acceleration is controlled directly
            accels[:, 4] -= self.env.k_brake * (pos[:, 2] - pos[:, 4]) * ((pos[:, 2] - pos[:, 4]) < 13.0)

            # f(x)
            f_x = torch.zeros((state_batch.shape[0], state_batch.shape[1])).to(self.device)
            f_x[:, ::2] = vels
            f_x[:, 1::2] = accels

            # f_D(x) - disturbance in the drift dynamics
            fD_x = torch.zeros((state_batch.shape[0], state_batch.shape[1])).to(self.device)
            fD_x[:, 1::2] = sigma_pred_batch[:, 1::2, 0].squeeze(-1)

            # g(x)
            g_x = torch.zeros((state_batch.shape[0], state_batch.shape[1], 1)).to(self.device)
            g_x[:, 7, 0] = 50.0  # Car 4's acceleration

            # h1
            h13 = 0.5 * (((pos[:, 2] - pos[:, 3]) ** 2) - collision_radius ** 2)
            h15 = 0.5 * (((pos[:, 4] - pos[:, 3]) ** 2) - collision_radius ** 2)

            # dh1/dt = Lfh1
            h13_dot = (pos[:, 3] - pos[:, 2]) * (vels[:, 3] - vels[:, 2])
            h15_dot = (pos[:, 3] - pos[:, 4]) * (vels[:, 3] - vels[:, 4])

            # Lffh1
            dLfh13dx = torch.zeros((batch_size, 10)).to(self.device)
            dLfh13dx[:, 4] = (vels[:, 2] - vels[:, 3])  # Car 3 pos
            dLfh13dx[:, 5] = (pos[:, 2] - pos[:, 3])  # Car 3 vel
            dLfh13dx[:, 6] = (vels[:, 3] - vels[:, 2])
            dLfh13dx[:, 7] = (pos[:, 3] - pos[:, 2])
            Lffh13 = torch.bmm(dLfh13dx.view(batch_size, 1, -1), f_x.view(batch_size, -1, 1)).squeeze()
            LfDfh13 = torch.bmm(torch.abs(dLfh13dx.view(batch_size, 1, -1)), fD_x.view(batch_size, -1, 1)).squeeze()

            dLfh15dx = torch.zeros((batch_size, 10)).to(self.device)
            dLfh15dx[:, 8] = (vels[:, 4] - vels[:, 3])  # Car 5 pos
            dLfh15dx[:, 9] = (pos[:, 4] - pos[:, 3])  # Car 5 vels
            dLfh15dx[:, 6] = (vels[:, 3] - vels[:, 4])
            dLfh15dx[:, 7] = (pos[:, 3] - pos[:, 4])
            Lffh15 = torch.bmm(dLfh15dx.view(batch_size, 1, -1), f_x.view(batch_size, -1, 1)).squeeze()
            LfDfh15 = torch.bmm(torch.abs(dLfh15dx.view(batch_size, 1, -1)), fD_x.view(batch_size, -1, 1)).squeeze()

            # Lfgh1
            Lgfh13 = torch.bmm(dLfh13dx.view(batch_size, 1, -1), g_x)
            Lgfh15 = torch.bmm(dLfh15dx.view(batch_size, 1, -1), g_x)
            # Inequality constraints (G[u, eps] <= h)
            h[:, 0] = Lffh13 - LfDfh13 + (gamma_b + gamma_b) * h13_dot + gamma_b * gamma_b * h13 + torch.bmm(Lgfh13, action_batch).squeeze()
            h[:, 1] = Lffh15 - LfDfh15 + (gamma_b + gamma_b) * h15_dot + gamma_b * gamma_b * h15 + torch.bmm(Lgfh15, action_batch).squeeze()
            G[:, 0, 0] = -Lgfh13.squeeze()
            G[:, 1, 0] = -Lgfh15.squeeze()
            G[:, :self.num_cbfs, n_u] = -2e2  # for slack
            ineq_constraint_counter += self.num_cbfs

            # Let's also build the cost matrices, vectors to minimize control effort and penalize slack
            P = torch.diag(torch.tensor([0.1, 1e1])).repeat(batch_size, 1, 1).to(self.device)
            q = torch.zeros((batch_size, n_u + 1)).to(self.device)
        
        elif self.env.dynamics_mode == 'Quadrotor':
            # The actual state, not observe
            state = state_batch[:, :, 0]
            theta = state[:, 2]
            n_u = action_batch.shape[1]
            num_constraints = self.num_cbfs + 2 * n_u

            G = torch.zeros((batch_size, num_constraints, n_u + 1)).to(self.device)  # the extra variable is for epsilon (to make sure qp is always feasible)
            h = torch.zeros((batch_size, num_constraints)).to(self.device)
            ineq_constraint_counter = 0

            # h = 0.5 * (1.7^2 - (x^2 + z^2))
            h_cbf = 0.5 * (self.env.circle_bound_radius**2 - (state[:, 0]**2 + state[:, 1]**2))
            # If hit happened
            if h_cbf < 0:
                h_value = 1
            # x z theta dx dz dtheta
            # Derivative w.r.t to time
            dh_cbf = -state[:, 0] * state[:, 3] - state[:, 1] * state[:, 4]
            
            # f(x)
            f_x = torch.zeros((state_batch.shape[0], state_batch.shape[1])).to(self.device)
            f_x[:, 0] = state[:, 3]
            f_x[:, 1] = state[:, 4]
            f_x[:, 2] = state[:, 5]
            f_x[:, 4] = -self.env.g
            
            # f(x) with disturbance, notice that this disturbance contains all the disturbance in the dynamics
            f = torch.add(f_x, mean_pred_batch.squeeze(-1))
            
            # This is for GP model
            fD_x = torch.zeros((state_batch.shape[0], state_batch.shape[1])).to(self.device)
            fD_x[:, 3:5] = sigma_pred_batch[:, 3:5, 0].squeeze(-1)

            # g(x)
            g_x = torch.zeros((state_batch.shape[0], state_batch.shape[1], n_u)).to(self.device)
            g_x[:, 3, 0] = -torch.sin(theta)/self.env.mass
            g_x[:, 3, 1] = -torch.sin(theta)/self.env.mass
            g_x[:, 4, 0] = torch.cos(theta)/self.env.mass
            g_x[:, 4, 1] = torch.cos(theta)/self.env.mass
            g_x[:, 5, 0] = -self.env.d/self.env.Iyy
            g_x[:, 5, 1] = self.env.d/self.env.Iyy

            # First take derivaitve of t, then take partial dirivative of x
            dLfhdx = torch.zeros((batch_size, 6)).to(self.device)
            dLfhdx[:, 0] = -state[:, 3]
            dLfhdx[:, 1] = -state[:, 4]
            dLfhdx[:, 3] = -state[:, 0]
            dLfhdx[:, 4] = -state[:, 1]

            # LDffh --- Why abs?
            LDffh = torch.bmm(torch.abs(dLfhdx.view(batch_size, 1, -1)), fD_x.view(batch_size, -1, 1)).squeeze()

            # Lffh
            Lffh = torch.bmm(dLfhdx.view(batch_size, 1, -1), f.view(batch_size, -1, 1)).squeeze()

            # Lgfh
            Lgfh = torch.bmm(dLfhdx.view(batch_size, 1, -1), g_x)
            
            if self.use_L1:
                # L1_gamma, L1_theta = self.env.update_param()  # get new L1 parameters
                # L1_penalty = self.env.L1_gamma# self.env.L1_theta - 2*self.env.L1_gamma 
                # L1_penalty = torch.bmm(torch.norm(dLfhdx.view(batch_size, 1, -1)), L1_penalty).squeeze()
                
                # LDffh = torch.mul(torch.norm(dLfhdx.view(batch_size, -1), dim = 1), torch.full((batch_size,), self.env.L1_gamma)).squeeze()
                LDffh = torch.bmm(dLfhdx.view(batch_size, 1, -1), mean_pred_batch.view(batch_size, -1, 1)).squeeze()
                h[:, 0] = Lffh + (gamma_b + gamma_b) * dh_cbf + gamma_b * gamma_b * h_cbf + torch.bmm(Lgfh, action_batch).squeeze() + LDffh #- Hight order CBF under L1
                # Notice, this is not the same with the paper, but it works better, might just becaue that extra bound is not needed
            else:
                h[:, 0] = Lffh + (gamma_b + gamma_b) * dh_cbf + gamma_b * gamma_b * h_cbf + torch.bmm(Lgfh, action_batch).squeeze() - LDffh
                
            
            G[:, :self.num_cbfs, 0] = -Lgfh.squeeze()[0]
            G[:, :self.num_cbfs, 1] = -Lgfh.squeeze()[1]
            G[:, :self.num_cbfs, n_u] = -2e2  # for slack
            ineq_constraint_counter += self.num_cbfs
            P = torch.diag(torch.tensor([0.001, 0.001, 1e1])).repeat(batch_size, 1, 1).to(self.device)
            q = torch.zeros((batch_size, n_u + 1)).to(self.device)

        elif self.env.dynamics_mode == 'VSA':
            
            # In cbf functions, we assume alpha1(x) = x^2, alpha2(x) = x^2
            L1_delta = torch.unsqueeze(to_tensor(np.stack(self.env.L1_delta, axis = 0), torch.FloatTensor, self.device), -1)
            num_cbfs = self.num_cbfs
            n_u = action_batch.shape[1]
            state = state_batch[:, :, 0]
            num_constraints = self.num_cbfs + 2 * n_u
            # num_constraints = self.num_cbfs
            
            # Constraints init
            G = torch.zeros((batch_size, num_constraints, n_u + 1)).to(self.device)  # the extra variable is for epsilon (to make sure qp is always feasible)
            h = torch.zeros((batch_size, num_constraints)).to(self.device)
            ineq_constraint_counter = 0
            
            # Ser up the CBFs
            h_cbf = 0.5 * (state[:,0] - state[:,1])**2 - self.env.def_max**2
            if h_cbf < 0:
                h_value = 1
                
            f_x = self.env.get_f(state)
            g_x = self.env.get_g(state)
            d_x = mean_pred_batch
            
            dhdx = torch.zeros((batch_size, state.shape[-1]))
            dhdx[:, 0] = state[:, 0] - state[:, 1]
            dhdx[:, 1] = state[:, 1] - state[:, 0]
            
            Lfh = torch.bmm(dhdx.view(batch_size, 1, -1), f_x.view(batch_size, -1, 1)).squeeze()
              
            dLfhdx = torch.zeros((batch_size, state.shape[-1]))
            dLfhdx[:, 0] = state[:, 3] - state[:, 4]
            dLfhdx[:, 1] = state[:, 4] - state[:, 3]
            dLfhdx[:, 3] = state[:, 0] - state[:, 1]
            dLfhdx[:, 4] = state[:, 1] - state[:, 0]
            
            L2fh = torch.bmm(dLfhdx.view(batch_size, 1, -1), f_x.view(batch_size, -1, 1)).squeeze()
            LfLph = torch.bmm(dLfhdx.view(batch_size, 1, -1), d_x.view(batch_size, -1, 1)).squeeze()
            LfLgh = torch.bmm(dLfhdx.view(batch_size, 1, -1), g_x.view(batch_size, -1, n_u)) 
            
            alpha2_phi1 = self.gamma_b * (Lfh**2 + self.gamma_b * 2*Lfh*h_cbf**2 + self.gamma_b * h_cbf**4)
            Lfalpha_1_h = self.gamma_b * 2*h_cbf*Lfh
            LfLgh_u = torch.bmm(LfLgh.view(batch_size, 1, -1), action_batch.view(batch_size, -1, 1)).squeeze()

            if self.use_L1:
                Lfbd_bound = torch.bmm(torch.abs(dLfhdx.view(batch_size, 1, -1)), L1_delta.view(batch_size, -1, 1)).squeeze()
            
            h[:, :self.num_cbfs] = alpha2_phi1 + Lfalpha_1_h  + L2fh + LfLph + LfLgh_u
            G[:, :self.num_cbfs, :n_u] = -LfLgh.squeeze()
            G[:, :self.num_cbfs, n_u] = -20  # for slack
            
            ineq_constraint_counter += self.num_cbfs
            P = torch.diag(torch.tensor([0.05, 0.01, 1e1])).repeat(batch_size, 1, 1).to(self.device)
            q = torch.zeros((batch_size, n_u + 1)).to(self.device)
    
        else:
            raise Exception('Dynamics mode unknown!')

        # Second let's add actuator constraints
        n_u = action_batch.shape[1]  # dimension of control inputs

        for c in range(n_u):

            # u_max >= u_nom + u ---> u <= u_max - u_nom
            if self.u_max is not None:
                G[:, ineq_constraint_counter, c] = 1
                h[:, ineq_constraint_counter] = self.u_max[c] - action_batch[:, c].squeeze(-1)
                ineq_constraint_counter += 1

            # u_min <= u_nom + u ---> -u <= u_min - u_nom
            if self.u_min is not None:
                G[:, ineq_constraint_counter, c] = -1
                h[:, ineq_constraint_counter] = -self.u_min[c] + action_batch[:, c].squeeze(-1)
                ineq_constraint_counter += 1

        return P, q, G, h, h_value

    def get_control_bounds(self):
        """

        Returns
        -------
        u_min : torch.tensor
            min control input.
        u_max : torch.tensor
            max control input.
        """

        u_min = torch.tensor(self.env.safe_action_space.low).to(self.device)
        u_max = torch.tensor(self.env.safe_action_space.high).to(self.device)

        return u_min, u_max


if __name__ == "__main__":

    from build_env import build_env
    from rcbf_sac.dynamics import DynamicsModel
    from copy import deepcopy
    from rcbf_sac.utils import to_numpy, prGreen


    def simple_controller(env, state, goal):
        goal_xy = goal[:2]
        goal_dist = -np.log(goal[2])  # the observation is np.exp(-goal_dist)
        v = 0.02 * goal_dist
        relative_theta = 1.0 * np.arctan2(goal_xy[1], goal_xy[0])
        omega = 1.0 * relative_theta

        return np.clip(np.array([v, omega]), env.action_space.low, env.action_space.high)


    parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
    # Environment Args
    parser.add_argument('--env-name', default="SafetyGym", help='Options are Unicycle or SafetyGym')
    parser.add_argument('--robot_xml', default='xmls/point.xml',
                        help="SafetyGym Currently only supporting xmls/point.xml")
    parser.add_argument('--k_d', default=3.0, type=float)
    parser.add_argument('--gamma_b', default=100, type=float)
    parser.add_argument('--l_p', default=0.03, type=float)
    parser.add_argument('--gp_model_size', default=2000, type=int, help='gp')
    parser.add_argument('--cuda', action='store_true', help='run on CUDA (default: False)')
    args = parser.parse_args()
    # Environment
    env = build_env(args)

    device = torch.device('cuda' if args.cuda else 'cpu')


    def to_def_tensor(ndarray):

        return to_tensor(ndarray, torch.FloatTensor, device)


    diff_cbf_layer = CBFQPLayer(env, args, args.gamma_b, args.k_d, args.l_p)
    dynamics_model = DynamicsModel(env, args)

    obs = env.reset()
    done = False

    ep_ret = 0
    ep_cost = 0
    ep_step = 0

    for i_step in range(3000):

        if done:
            prGreen('Episode Return: %.3f \t Episode Cost: %.3f' % (ep_ret, ep_cost))
            ep_ret, ep_cost, ep_step = 0, 0, 0
            obs = env.reset()

        state = dynamics_model.get_state(obs)

        print('state = {}, dist2hazards = {}'.format(state[:2],
                                                     np.sqrt(np.sum((env.hazards_locations - state[:2]) ** 2, 1))))

        disturb_mean, disturb_std = dynamics_model.predict_disturbance(state)

        action = simple_controller(env, state, obs[-3:])  # TODO: observations last 3 indicated
        # action = 2*np.random.rand(2) - 1.0
        assert env.action_space.contains(action)
        final_action = diff_cbf_layer.get_safe_action(to_def_tensor(state), to_def_tensor(action),
                                                      to_def_tensor(disturb_mean), to_def_tensor(disturb_std))
        final_action = to_numpy(final_action)

        # Env Step
        observation2, reward, done, info = env.step(final_action)
        observation2 = deepcopy(observation2)

        # Update state and store transition for GP model learning
        next_state = dynamics_model.get_state(observation2)
        if ep_step % 2 == 0:
            dynamics_model.append_transition(state, final_action, next_state)

        # print('reward', reward)
        ep_ret += reward
        ep_cost += info.get('cost', 0)
        ep_step += 1
        # env.render()

        obs = observation2
        state = next_state