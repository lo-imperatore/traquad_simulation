"""Rest / stand sequence in Gazebo (see getup_omniquad_sim.py). model: folder in models/ with params/env.yaml (default = stand pose)."""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context):
    lc = lambda n: LaunchConfiguration(n).perform(context)
    model_dir = os.path.join(get_package_share_directory('rlg_quad_controller'), 'models', lc('model'))
    env = os.path.join(model_dir, 'params', 'env.yaml')
    if not os.path.isfile(env):
        env = os.path.join(model_dir, 'env.yaml')
    return [Node(
        package='rlg_quad_controller', executable='getup_omniquad_sim', name='getup_omniquad_sim', output='screen',
        parameters=[{'env_cfg_path': env, 'use_sim_time': True, 'auto_stand': lc('auto_stand').lower() == 'true',
                     'standup_time': float(lc('standup_time')),
                     'activation_hip_offset_deg': float(lc('activation_hip_offset_deg')),
                     **({'rest_pose': [float(x) for x in lc('rest_pose').split(',')]} if lc('rest_pose') else {})}])]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('model', default_value='lstm_climb', description='Folder in models/ (env.yaml: default pose)'),
        DeclareLaunchArgument('auto_stand', default_value='false'),
        DeclareLaunchArgument('standup_time', default_value='5.0'),
        DeclareLaunchArgument('activation_hip_offset_deg', default_value='7.5'),
        DeclareLaunchArgument('rest_pose', default_value='', description='8 comma separated values, POLICY_LEGS order; empty = node default'),
        OpaqueFunction(function=launch_setup),
    ])
