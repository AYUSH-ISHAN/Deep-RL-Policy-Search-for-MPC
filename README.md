# Abstract : 
The combination of policy search and deep neural networks holds the promise of automating a variety of decision- making tasks. Model Predictive Control (MPC) provides robust solutions to robot control tasks by making use of a dynamical model of the system and solving an optimization problem online over a short planning horizon. Policy Search and Model Predictive Control (MPC) are two different paradigms for robot control: policy search has the strength of automatically learning complex policies using experienced data, while MPC can offer optimal control performance using models and trajectory optimization. An open research question is how to leverage and combine the advantages of both approaches.

# Introduction :
In practice, the closed-loop performance of MPC for a specific task is sensitive to several design choices, including cost function formulation,2 hyperparameters, and the prediction horizon. As a result, a series of approximations, heuristics, and parameter tuning is
employed, producing sub-optimal solutions. On the other hand, reinforcement learning (RL) [19] methods, like policy search, allow solving continuous control problems with minimum prior knowledge about the task. The key idea of RL is to automatically train the policy via trial and error and maximize the task performance measured by the given reward function. While RL has achieved impressive results in solving a wide range of robot control tasks, the lack of interpretability of an end-to-end controller trained using RL is of significant concern by the control community.
Ideally, the control framework should be able to combine the advantages of both methods—the ability of model-based controllers, like MPC, to safely control a physical robot using the well-established knowledge in dynamic modeling and optimization and the power of RL to learn complex policies using experienced data automatically. Therefore, the resulting control framework can handle large-scale inputs, reduce human-in-the-loop design and tuning, and eventually achieve adaptive and optimal control performance. Despite these valuable features, designing such a system remains a
significant challenge.

# The Algorithm :
<p align="center"><img src = "./media/1.png"/></p>



































# Try with ans without Gaussian. 
Without gaussiam means.. Vanilla MPC dataset.. Trained on RL.. With directly mean squared error as reward aur custom reward

# Different Swarm based algorithm like Dragon fly algo, Grasshopper algo
# UAV-Confrontation-using-H-MARL
This repo is related to UAV Confrontation using Heirarchial MultiAgent Reinforcement Learning

# Make it a bidriectional project..
<ol>
  <li>Cooperative Env -->  Form clustering, try out A2C type architecture</li>
  <li>Confronatation env -->  Use competitive RL, H-MADDPG as found in paper</li>
</ol>

# Use MPC.Pytorch for MPC and for RL algos use the baselines and Ray type thing. 
https://github.com/locuslab/mpc.pytorch
# Resources :

1. RL Part :
- https://github.com/utiasDSL/gym-pybullet-drones  [Contains 2 papers -- really important papers]
- https://github.com/AboudyKreidieh/h-baselines
- https://github.com/Jackiemesser/Autonomous-UAVs  [Can see this ]
- See the papers listed in the paper uploaded in the github

2. RL and MPC Part :
- https://github.com/uzh-rpg/high_mpc     https://www.youtube.com/watch?v=Qei7oGiEIxY  Video to understand the papers in this link
- https://github.com/utiasDSL/safe-control-gym [**Important Repo**]  <br> Here, ion this put the files from Pybullet drone gym official ..
https://github.com/utiasDSL/safe-control-gym/tree/main/safe_control_gym/envs/gym_pybullet_drones    So, that we can connect MARL with the MPC

3. Mutliagent approach: 
- https://github.com/CORE-Robotics-Lab/MAGIC

5. ROS part :
- DRDO Inter IIT Files
- https://github.com/ethz-asl/mav_control_rw  [Ros + Controls]

# In two part ..

First the H_MARL approach and then release the ROS extensions.
