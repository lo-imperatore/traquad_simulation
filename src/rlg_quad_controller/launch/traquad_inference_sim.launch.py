"""Traquad RSL-RL policy (IsaacLab) in Gazebo.

    ros2 launch rlg_quad_controller traquad_inference_sim.launch.py [model:=traquad_omnistyle] [checkpoint:=model_1400.pt]

The model folder in models/ is an IsaacLab RSL-RL run: params/{env,agent}.yaml + model_<iter>.pt
(the highest iteration is used when checkpoint is empty).
"""
import os
import re
from glob import glob

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context):
    pkg_share = get_package_share_directory('rlg_quad_controller')
    model_dir = os.path.join(pkg_share, 'models', LaunchConfiguration('model').perform(context))
    checkpoint = LaunchConfiguration('checkpoint').perform(context)
    if checkpoint:
        model_path = os.path.join(model_dir, checkpoint)
    else:
        pts = glob(os.path.join(model_dir, 'model_*.pt'))
        if not pts:
            raise RuntimeError(f'No model_<iter>.pt in {model_dir}')
        model_path = max(pts, key=lambda p: int(re.search(r'model_(\d+)\.pt$', p).group(1)))

    return [Node(
        package='rlg_quad_controller',
        executable='inference_traquad_sim',
        name='inference_traquad_sim',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'model_path': model_path,
            'env_cfg_path': os.path.join(model_dir, 'params', 'env.yaml'),
            'agent_cfg_path': os.path.join(model_dir, 'params', 'agent.yaml'),
            'acc_filter_alpha': float(LaunchConfiguration('acc_filter_alpha').perform(context)),
            'log_obs': LaunchConfiguration('log_obs').perform(context) == 'true',
        }],
    )]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('model', default_value='traquad_omnistyle'),
        DeclareLaunchArgument('checkpoint', default_value=''),
        DeclareLaunchArgument('acc_filter_alpha', default_value='1.0',
                              description='low-pass on the accelerometer, 1.0 = none (as in training)'),
        DeclareLaunchArgument('log_obs', default_value='false'),
        OpaqueFunction(function=launch_setup),
    ])
