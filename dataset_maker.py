
import os
import numpy as np
import glob
from math import sqrt, sin, cos
from policy.high_policy import GaussianPolicy
import torch.autograd
from mpc_codes import mpc

class Dataset(object):
    
    def __init__(self, obs_dim, act_dim, max_size=int(1e6)):
        #
        self._obs_buf = np.zeros(shape=(max_size, obs_dim), dtype=np.float32)
        self._act_buf = np.zeros(shape=(max_size, act_dim), dtype=np.float32)
        #
        self._max_size = max_size
        self._ptr = 0
        self._size = 0

    def add(self, obs, act):
        self._obs_buf[self._ptr] = obs
        self._act_buf[self._ptr] = act
        #
        self._ptr = (self._ptr + 1) % self._max_size
        self._size = min(self._size + 1, self._max_size)
    
    def sample(self, batch_size):
        index = np.random.randint(0, self._size, size=batch_size)
        #
        return [ self._obs_buf[index], self._act_buf[index] ]
    
    def save(self, save_dir, n_iter):
        save_dir = save_dir + "/dataset"
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        data_path = save_dir + "/data_{0}".format(n_iter)
        np.savez(data_path, obs=self._obs_buf, act=self._act_buf)

    def load(self, data_dir):
        self._obs_buf, self._act_buf = None, None
        for i, data_file in enumerate(glob.glob(os.path.join(data_dir, "*.npz"))):
            size = int(data_file.split("/")[-1].split("_")[-1].split(".")[0])
            np_file = np.load(data_file)
            #
            obs_array = np_file['obs'][:size, :]
            act_array = np_file['act'][:size, :]
            if i == 0:
                self._obs_buf = obs_array
                self._act_buf = act_array    
            else:
                self._obs_buf = np.append(self._obs_buf, obs_array, axis=0)
                self._act_buf = np.append(self._act_buf, act_array, axis=0)
        #
        self._size  = self._obs_buf.shape[0]
        self._ptr = self._size - 1

SIM_TIME = 1/48
rpy_rates = np.zeros((1, 3))

class DroneDyanamics(torch.nn.Module):
    def forward(self, state, action):

        coordinate = state[:,:3].view(-1,1)
       
        velocity = state[:,10:13].view(-1,1)
        rpy = state[:,7:10].view(-1,1)
        quaternion = state[:,3:7].view(-1,1)
        rpy_rates = state[:,13:16].view(-1,1)
        u = action
        u = torch.clamp(u, 0, 21702.645)
        self.KF = 0.0
        self.KM = 0.0
        self.GRAVITY = 0.2646
        
        ixx= 0.000014 
        iyy=0.000014
        izz= 0.000022
       
        self.J = torch.diag(torch.tensor([ixx, iyy, izz]))
        self.J_INV = torch.linalg.inv(self.J) 
        self.M = 0.027
        self.L = 0.0397
        rotation = quaternion_rotation_matrix(quaternion).reshape(3,3)
        
        forces = (u**2).mul_(self.KF)
        thrust = torch.reshape(torch.tensor([0, 0, torch.sum(forces)]), (3,1))
        thrust_world_frame = torch.matmul(rotation, thrust) 
        force_world_frame = thrust_world_frame - torch.reshape(torch.tensor([0, 0, self.GRAVITY]), (3,1))
        z_torques = ((u**2)*self.KM)
        z_torque = (-z_torques[0][0] + z_torques[0][1] - z_torques[0][2] + z_torques[0][3])
        x_torque = (forces[0][0] + forces[0][1] - forces[0][2] - forces[0][3]) * (self.L/sqrt(2))
        y_torque = (- forces[0][0] + forces[0][1] + forces[0][2] - forces[0][3]) * (self.L/sqrt(2))
        torques = torch.reshape(torch.tensor([x_torque, y_torque, z_torque]), (3,1))
        torques = torques - torch.cross(rpy_rates.float(), torch.matmul(self.J, rpy_rates.float()))
        rpy_rates_deriv = torch.matmul(self.J_INV, torques.float())
        no_pybullet_dyn_accs = (force_world_frame / self.M)
       
        velocity.data+=(no_pybullet_dyn_accs.data.mul_(SIM_TIME))
        '''This gave me the angular acceleration'''
        rpy_rates.data+=(rpy_rates_deriv.data.mul_(SIM_TIME))
        coordinate.data+=(velocity.data.mul_(SIM_TIME))
        rpy.data+=(rpy_rates.data.mul_(SIM_TIME))
        quaternion = eular_to_quat(rpy[0],rpy[1], rpy[2])+(u*np.finfo(np.float64).eps)
        quaternion = torch.reshape(quaternion, (1,4))
        velocity = torch.clamp(velocity, -0.5, 0.2)
        rpy_rates = torch.clamp(rpy_rates, -1.0, 1.0)
        rpy = torch.clamp(rpy, -0.053, 0.053)
        quaternion = torch.clamp(quaternion, -1,1)
        state = torch.cat((torch.t(coordinate),quaternion,torch.t(rpy),torch.t(velocity),torch.t(rpy_rates)), dim=1)
        return state

def eular_to_quat(roll, pitch, yaw):
    qx = torch.sin(roll/2)*cos(pitch/2)*cos(yaw/2) - cos(roll/2)*torch.sin(pitch/2)*torch.sin(yaw/2)
    qy = torch.cos(roll/2)*torch.sin(pitch/2)*torch.cos(yaw/2) + torch.sin(roll/2)*torch.cos(pitch/2)*torch.sin(yaw/2)
    qz = torch.cos(roll/2)*torch.cos(pitch/2)*torch.sin(yaw/2) - torch.sin(roll/2)*torch.sin(pitch/2)*torch.cos(yaw/2)
    qw = torch.cos(roll/2)*torch.cos(pitch/2)*torch.cos(yaw/2) + torch.sin(roll/2)*torch.sin(pitch/2)*torch.sin(yaw/2)

    return torch.tensor([qw, qx, qy, qz])

def quaternion_rotation_matrix(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.
 
    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 
 
    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix. 
             This rotation matrix converts a point in the local reference 
             frame to a point in the global reference frame.
    """
    # Extract the values from Q
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]

    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
     
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
     
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
     
    # 3x3 rotation matrix
    rot_matrix = torch.tensor([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
                            
    return rot_matrix

nx = 16
nu = 4
TIMESTEPS = 10  # T
N_BATCH = 1
LQR_ITER = 5
ACTION_LOW = -2.0
ACTION_HIGH = 2.0
u_init = None
action = None
render = True
retrain_after_iter = 50
run_iter = 500

class Collector(object):

    def __init__(self, env, logger, save_dir, max_samples, 
        max_wml_iter, beta0, n_samples):

        self.env = env
        self.logger = logger
        self.save_dir = save_dir
        self.max_samples = 5000 #max_samples
        self.max_wml_iter = 15 #max_wml_iter
        self.beta0 = beta0
        self.n_samples = 15 #n_samples

        self.mpc = mpc.MPC
        self.plan_dt = 0.0208
        # self.plan_T = 2  # horizon time
        self.curr_time = 0

    def data_collection(self):

        # swingup goal (observe theta and theta_dt)
        goal_weights = torch.tensor((1.,1.,1.,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.5,0.5,0.5,0.1,0.1,0.1))#,0.1,0.1,0.1,0.1))  # nx
        goal_state = torch.tensor((0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.))#,0.,0.,0.,0.))  # nx
        ctrl_penalty = 0.001
        q = torch.cat((
            goal_weights,
            ctrl_penalty * torch.ones(nu)
        ))  # nx + nu
        px = -torch.sqrt(goal_weights) * goal_state
        p = torch.cat((px, torch.zeros(nu)))
        Q = torch.diag(q).repeat(TIMESTEPS, N_BATCH, 1, 1)  # T x B x nx+nu x nx+nu
        p = p.repeat(TIMESTEPS, N_BATCH, 1)
        costs = mpc.QuadCost(Q, p)  # T x B x nx+nu (linear component of cost)
        obs_dim = 16
        act_dim = 4
        act_low = 0 
        act_high = 21702.645 
        clip_value = [act_low, act_high]
        data_set = Dataset(obs_dim, act_dim)

        # Data collection
        
        obs, done = self.env.reset(), True
        save_iter = 100
        obs = obs['0']['state'][:16]
        obs = torch.tensor(obs).view(1,-1)
        for n in range(self.max_samples):
            if done:
                obs = self.env.reset()
                obs = obs['0']['state'][:16]
                obs = torch.tensor(obs).view(1,-1)
            # # # # # # # # # # # # # # # # # # 
            # ---- Weighted Maximum Likelihood 
            # # # # # # # # # # # # # # # # # # 
            pi = GaussianPolicy(act_dim, clip_value=clip_value)
            # online optimization
            opt_success, pass_index = True, 0

            # if obs[0] >= 0.2: # 
            #     act = np.array( [4.0] )
            # # elif obs[0] >= -0.4 and obs[0] <= 0.1:
            # #     act = np.array( [0.0] )
            # else: # weighted maximum likelihood optimization....

            for i in range(self.max_wml_iter):  # max_wml_iter = 15
                rewards = np.zeros(self.n_samples)
                Actions = np.zeros(shape=(self.n_samples, pi.act_dim))
                for j in range(self.n_samples):  # n_samples = 15
                    act = pi.sample()
                    act = torch.tensor(act).view(1,-1)
                    reward, pass_index = self.episode(act, obs,costs, self.curr_time)
                    rewards[j] = np.mean(reward)
                    #
                    Actions[j, :] = act
                    #
                if not (np.max(rewards) - np.min(rewards)) <= 1e-10:
                    # compute weights 
                    beta = self.beta0 / (np.max(rewards) - np.min(rewards))
                    weights = np.exp(beta * (rewards - np.max(rewards)))
                    Weights = weights / np.mean(weights)
                    # 
                    pi.fit(Weights, Actions)
                    opt_success = True   
                else:
                    opt_success = False     
                    #
                self.logger.log("********** Sample %i ************"%n)
                self.logger.log("---------- Wml_iter %i ----------"%i)
                self.logger.log("---------- Reward %f ----------"%np.mean(rewards))
                for i in range(4):
                    self.logger.log("---------- Pass index %i----------"%pass_index[0][i])
                if abs(np.mean(rewards)) <= 0.001:
                        self.logger.log("---------- Converged %f ----------"%np.mean(rewards))
                        break
            
            if opt_success:
                act = pi()
                self.logger.log("---------- Success {0} ----------".format(act))
            else:
                act = np.array( [(pass_index+1) * self.plan_dt] )
                self.logger.log("---------- Faild {0} ----------".format(act))
            # collect optimal action
            data_set.add(obs, act)

            # # # # # # # # # # # # # # # # # # 
            # ---- Execute the action --------
            # # # # # # # # # # # # # # # # # # 
            #
            action = {}
            action['0']=act
            obs, rewarde, done, _ = self.env.step(action)
            obs = obs['0']['state'][:16]
            obs = torch.tensor(obs).view(1,-1)
            self.curr_time += 1
            self.logger.log("---------- Obs{0}  ----------".format(obs))
            self.logger.log("---------- Reward from env{0}  ----------".format(rewarde))
            self.logger.log("                          ")

            #
            if (n+1) % save_iter == 0:
                data_set.save(self.save_dir, n)

    def planner(self, t):

        #### Initialize the simulation #############################
        H = .1
        H_STEP = .05
        R = .3
        INIT_XYZS = np.array([[R*np.cos((i/6)*2*np.pi+np.pi/2), R*np.sin((i/6)*2*np.pi+np.pi/2)-R, H+i*H_STEP] for i in range(1)])
        INIT_RPYS = np.array([[0, 0,  i * (np.pi/2)/1] for i in range(1)], dtype=np.float32)
        AGGR_PHY_STEPS = int(240/48) if True else 1

        PERIOD = 10
        NUM_WP = 48*PERIOD
        TARGET_POS = np.zeros((NUM_WP,3)).astype(np.float32)

        for i in range(t, NUM_WP):
            TARGET_POS[i, :] = R*np.sin((i/NUM_WP)*(2*np.pi)+np.pi, dtype=np.float32)+INIT_XYZS[0, 0], R*np.sin((i/NUM_WP)*(2*np.pi)+np.pi/2, dtype=np.float32)*np.cos((i/NUM_WP)*(2*np.pi)+np.pi/2, dtype=np.float32)+INIT_XYZS[0, 1], 0
            #TARGET_POS[i, :] = R*np.sin((i/NUM_WP)*(2*np.pi)+np.pi/2)*np.cos((i/NUM_WP)*(2*np.pi)+np.pi/2)+INIT_XYZS[0, 1],R*np.sin((i/NUM_WP)*(2*np.pi)+np.pi)+INIT_XYZS[0, 0],  0
        '''Uncomment the above if you want to rotate the 8 figure by 90 degress clockwise'''
        '''This will give poistion at each wp_counters marks.'''
        
        return torch.tensor(TARGET_POS)

    def episode(self, act, obs,costs, t, u_init=None): # where t is current_time or index in states

            opt_t = act
   
            obs = obs.float()
            ref_traj = self.planner(t) 
            ref_traj = ref_traj.float()
            pred_traj = []
            for i in range(t, 480): # where T = 480
                ctrl = mpc.MPC(nx, nu, TIMESTEPS, u_lower=ACTION_LOW, u_upper=ACTION_HIGH, lqr_iter=LQR_ITER,
                       exit_unconverged=False, eps=1e-2,
                       n_batch=N_BATCH, backprop=False, verbose=0, u_init=u_init,
                       grad_method=mpc.GradMethods.AUTO_DIFF)


                nom_state, nom_action, nom_cost = ctrl(obs, costs, DroneDyanamics())
                action = nom_action[0]
                u_init = torch.cat((nom_action[1:], torch.zeros(1, N_BATCH, nu)), dim=0)
                nom_s = nom_state.detach().numpy()
                app_state = [i[0][:3] for i in nom_s]
                pred_traj.append(app_state)

            tens = (opt_t/self.plan_dt).detach().numpy()
            opt_node = np.clip(np.int32(tens), 0, len(pred_traj)-1)# pred_traj.shape[0]-1)
           
            pred_traj=np.array(pred_traj)
            ref_traj=ref_traj.detach().numpy()
            ref_traj = np.reshape(ref_traj, (-1,10,3))
            loss = np.linalg.norm(pred_traj[t:480]  - ref_traj[t:480])
            print("loss  :  ",loss)
            rew = - loss
            #    
            return rew, opt_node


