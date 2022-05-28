
import os
import datetime
import argparse
import numpy as np
import matplotlib.pyplot as plt


from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary

from gym_pybullet_drones.utils.utils import str2bool
import torch
from common import logger
from common import util as U
from deep_model import train, Actor
from dataset_maker import Collector

#
def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--option', type=int, default=2,
        help="0 - Data collection; 1 - train the deep high-level policy; 2 - test the trained policy.")
    parser.add_argument('--save_dir', type=str, default=os.path.dirname(os.path.realpath(__file__)),
        help="Directory where to save the checkpoints and training metrics")
    parser.add_argument('--save_video', type=bool, default=False,
        help="Save the animation as a video file")
    parser.add_argument('--load_dir', type=str, help="Directory where to load weights")
    parser.add_argument('--drone',              default="cf2x",     type=DroneModel,    help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--num_drones',         default=1,          type=int,           help='Number of drones (default: 3)', metavar='')
    parser.add_argument('--physics',            default="pyb",      type=Physics,       help='Physics updates (default: PYB)', metavar='', choices=Physics)
    parser.add_argument('--vision',             default=False,      type=str2bool,      help='Whether to use VisionAviary (default: False)', metavar='')
    parser.add_argument('--gui',                default=True,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=False,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot',               default=True,       type=str2bool,      help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui',     default=False,      type=str2bool,      help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--aggregate',          default=True,       type=str2bool,      help='Whether to aggregate physics steps (default: True)', metavar='')
    parser.add_argument('--obstacles',          default=True,       type=str2bool,      help='Whether to add obstacles to the environment (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=240,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=48,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec',       default=120,         type=int,           help='Duration of the simulation in seconds (default: 5)', metavar='')
    return parser
    
def run_deep_high_mpc(env, actor_params, load_dir):
    obs_dim = 16 #env.observation_space  # check whether dim needed or not
    act_dim = 4 #env.action_space
    logs = []
    #
    model = Actor(obs_dim, act_dim)
    # actor = PolicyNet(*args, **kwargs)
    model.load_weights(load_dir)
    print(model)
    #
    ep_len, ep_reward =  0, 0
    sim_T=4#480#30.
    sim_dt=1#1/48.#0.02
    for i in range(10):
        obs = env.reset()
        obs = obs['0']['state'][:16]
        obs = torch.tensor(obs).view(1,-1)
        obs = obs.float()
        t = 0
        while t < sim_T:
            t += sim_dt
            act = model(obs)
            # execute action
            act = act.detach().numpy()
            action = {}
            action['0']=act
            next_obs, reward, _, info = env.step(action)
            next_obs = next_obs['0']['state'][:16]
            next_obs = torch.tensor(next_obs).view(1,-1)
            next_obs = next_obs.float()
            #
            obs = next_obs
            ep_reward += reward
            #
            env.render()
            ep_len += 1
            #
            update = False
            if t >= sim_T:
                update = True
            logs.append([reward,info, t, update])
    return logs

#### Define and parse (optional) arguments for the script ##

args = arg_parser().parse_args()
H = .1
H_STEP = .05
R = .3
INIT_XYZS = np.array([[R*np.cos((i/6)*2*np.pi+np.pi/2), R*np.sin((i/6)*2*np.pi+np.pi/2)-R, H+i*H_STEP] for i in range(1)])
INIT_RPYS = np.array([[0, 0,  i * (np.pi/2)/1] for i in range(1)])
AGGR_PHY_STEPS = int(240/48) if True else 1

env = CtrlAviary(drone_model=args.drone,
                           num_drones=args.num_drones,
                           initial_xyzs=INIT_XYZS,
                           initial_rpys=INIT_RPYS,
                           physics=args.physics,
                           neighbourhood_radius=10,
                           freq=args.simulation_freq_hz,
                           aggregate_phy_steps=AGGR_PHY_STEPS,
                           gui=args.gui,
                           record=args.record_video,
                           obstacles=args.obstacles   # if want can put this false
                           )
def main():
    #
    args = arg_parser().parse_args()
    #
    plan_T = 2.0   # Prediction horizon for MPC and local planner
    plan_dt = 0.04 # Sampling time step for MPC and local planner

    actor_params = dict(
        hidden_units=[32, 32],
        learning_rate=1e-4,
        activation='relu',
        train_epoch=1000,
        batch_size=128
    )

    # training
    training_params = dict(
        max_samples =5000,
        max_wml_iter=15,
        beta0=3.0,
        n_samples=15,
    )
    
    # if in training mode, create new dir to save the model and checkpoints
    if args.option == 0: # collect data
        save_dir = U.get_dir(args.save_dir + "/Dataset")
        save_dir = os.path.join(save_dir, datetime.datetime.now().strftime("deep_highmpc-%m-%d-%H-%M-%S"))

        #
        logger.configure(dir=save_dir)
        logger.log("***********************Log & Store Hyper-parameters***********************")
        logger.log("actor params")
        logger.log(actor_params)
        logger.log("training params")
        logger.log(training_params)
        logger.log("***************************************************************************")

        # 
        collector = Collector(env=env, logger=logger, \
            save_dir=save_dir, **training_params)
        collector.data_collection()
    elif args.option == 1: # train the policy
        data_dir = args.save_dir + "/Dataset"
        train(env, logger, data_dir, **actor_params)
    elif args.option == 2: # evaluate the policy
        load_dir = args.save_dir + "/Dataset/deep_highmpc-05-26-15-47-14/dataset/act_net/weights_900.h5"
        print(load_dir)
        a = run_deep_high_mpc(env, actor_params, load_dir)
        print("Logssssssssssssssssssssssssssssssssssssssssssssssssss  : ",a)
        
        
if __name__ == "__main__":
    main()