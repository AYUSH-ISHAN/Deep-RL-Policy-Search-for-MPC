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
