# UAV-Confrontation-using-H-MARL
This repo is related to UAV Confrontation using Heirarchial MultiAgent Reinforcement Learning
# Look at A* and other algorithms

# Make it a bidriectional project..
<ol>
  <li>Cooperative Env -->  Form clustering, try out A2C type architecture</li>
  <li>Confronatation env -->  Use competitive RL, H-MADDPG as found in paper</li>
</ol>


# Resources :

1. RL Part :
- https://github.com/utiasDSL/gym-pybullet-drones  [Contains 2 papers -- really important papers]
- https://github.com/AboudyKreidieh/h-baselines
- https://github.com/Jackiemesser/Autonomous-UAVs  [Can see this ]
- [17] P. Dayan and G. Hinton, “Feudal reinforcement learning,”
Advances in Neural Information Processing Systems, vol. 5,
p. 9, 2000.<br>
[18] S. Ahilan and P. Dayan, “Feudal multi-agent hierarchies for
cooperative reinforcement learning,” in Annual Conference
of the American Library Association, 2019, http://arxiv.org/
abs/1901.08492.<br>
[19] Q. Yang, Y. Zhu, J. Zhang, S. Qiao, and J. Liu, Eds., “UAV air
combat autonomous maneuver decision based on ddpg algo-
rithm,” in 2019 IEEE 15th International Conference on Control
and Automation (ICCA), pp. 37–42, Edinburgh, UK, July 2019.<br>
[20] S. Xuan and L. Ke, “Study on attack-defense countermeasure
of UAV swarms based on multi-agent reinforcement learn-
ing,” Radio Engineering, vol. 51, pp. 360–366, 2021.<br>
[21] S. Xuan and L. Ke, “UAV swarm attack-defense confrontation
based on multi-agent reinforcement learning,” in Advances in
Guidance, Navigation and Control. Lecture Notes in Electrical
Engineering, vol 644, L. Yan, H. Duan, and X. Yu, Eds.,
Springer, Singapore, 2020.<br>
[22] I. Mordatch and P. Abbeel, “Emergence of grounded composi-
tional language in multi-agent populations,” 2017, http://arxiv
.org/abs/1703.04908.<br>
[23] M. L. Littman, “Markov games as a framework for multi-agent
reinforcement learning,” in Proceedings of the Eleventh Inter-
national Conference, Rutgers University, pp. 157–163, New
Brunswick, NJ, USA, July 1994.<br>
[24] W. Kim, M. Cho, and Y. Sung, “Message-dropout: an efficient
training method for multi-agent deep reinforcement learning,”
Proceedings of the AAAI Conference on Artificial Intelligence,
vol. 33, pp. 6079–6086, 2019.<br>
[25] C. Yu, A. Velu, E. Vinitsky, Y. Wang, A. Bayen, and Y. Wu,
“The surprising effectiveness of mappo in cooperative, multi-
agent games,” 2021, http://arxiv.org/abs/2103.01955.<br>

2. RL and MPC Part :
- https://github.com/uzh-rpg/high_mpc     https://www.youtube.com/watch?v=Qei7oGiEIxY  Video to understand the papers in this link
- https://github.com/utiasDSL/safe-control-gym [**Important Repo**]  <br> Here, ion this put the files from Pybullet drone gym official ..
https://github.com/utiasDSL/safe-control-gym/tree/main/safe_control_gym/envs/gym_pybullet_drones    So, that we can connect MARL with the MPC


3. ROS part :
- DRDO Inter IIT Files
- https://github.com/ethz-asl/mav_control_rw  [Ros + Controls]

# In two part ..

First the H_MARL approach and then release the ROS extensions.
