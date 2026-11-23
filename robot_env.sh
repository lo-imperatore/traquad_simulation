#!/bin/bash
# Environment to talk to the real omniquad from the laptop (source it, do not run it):
#   docker compose exec ros2_humble_sim bash
#   source robot_env.sh
# The container must use network_mode: host (compose.yaml). Do NOT run Gazebo in a shell with this environment:
# simulated topics (/cmd_vel, ...) would reach the robot network.
source /opt/ros/humble/setup.bash
[ -f install/setup.bash ] && source install/setup.bash
export ROS_DOMAIN_ID=10
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
unset ROS_LOCALHOST_ONLY
ros2 daemon stop >/dev/null 2>&1   # restart the daemon with the new domain / RMW
echo "robot env: ROS_DOMAIN_ID=$ROS_DOMAIN_ID RMW_IMPLEMENTATION=$RMW_IMPLEMENTATION"
echo "check: ros2 topic list | grep -E 'state_broadcaster|omni_controller|imu_sensor'"
