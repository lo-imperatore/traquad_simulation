"""
Policy on the laptop -> real omniquad (see inference_omniquad_robot.py).
Example:
  ros2 launch rlg_quad_controller omniquad_robot_bridge.launch.py                    # dry run
  ros2 launch rlg_quad_controller omniquad_robot_bridge.launch.py dry_run:=false wheel_action_gain:=0.0
"""
import os
import re
from glob import glob

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def resolve_model(model_dir, checkpoint):
    """params/{env,agent}.yaml and model_<iter>.pt (highest iteration if checkpoint is empty)."""
    cfg_dir = os.path.join(model_dir, 'params')
    if not os.path.isfile(os.path.join(cfg_dir, 'env.yaml')):
        cfg_dir = model_dir
    if checkpoint:
        model_path = os.path.join(model_dir, checkpoint)
    else:
        pts = glob(os.path.join(model_dir, 'model_*.pt'))
        if not pts:
            raise RuntimeError(f'No model_<iter>.pt in {model_dir}')
        model_path = max(pts, key=lambda p: int(re.search(r'model_(\d+)\.pt$', p).group(1)))
    return os.path.join(cfg_dir, 'env.yaml'), os.path.join(cfg_dir, 'agent.yaml'), model_path


def launch_setup(context):
    lc = lambda n: LaunchConfiguration(n).perform(context)
    pkg_share = get_package_share_directory('rlg_quad_controller')
    env_path, agent_path, model_path = resolve_model(
        os.path.join(pkg_share, 'models', lc('model')), lc('checkpoint'))

    return [Node(
        package='rlg_quad_controller',
        executable='inference_omniquad_robot',
        name='inference_omniquad_robot',
        output='screen',
        parameters=[
            os.path.join(pkg_share, 'config', 'omniquad_robot_config.yaml'),
            {'env_cfg_path': env_path, 'agent_cfg_path': agent_path, 'model_path': model_path,
             'dry_run': lc('dry_run').lower() == 'true',
             'auto_start': lc('auto_start').lower() == 'true',
             'leg_action_gain': float(lc('leg_action_gain')),
             'wheel_action_gain': float(lc('wheel_action_gain')),
             'log_path': lc('log_path')},
        ],
    )]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('model', default_value='2026-11-09_21-26-58_finetune_curbs',
                              description='Folder in models/ (IsaacLab RSL-RL run)'),
        DeclareLaunchArgument('checkpoint', default_value='',
                              description='model_<iter>.pt; empty = highest iteration'),
        DeclareLaunchArgument('dry_run', default_value='true',
                              description='true: compute and print, publish nothing'),
        DeclareLaunchArgument('auto_start', default_value='false',
                              description='true: policy active right after the standup (else ~/start)'),
        DeclareLaunchArgument('leg_action_gain', default_value='1.0'),
        DeclareLaunchArgument('wheel_action_gain', default_value='1.0'),
        DeclareLaunchArgument('log_path', default_value='',
                              description='.npz with obs/actions/targets written at Ctrl-C'),
        OpaqueFunction(function=launch_setup),
    ])
