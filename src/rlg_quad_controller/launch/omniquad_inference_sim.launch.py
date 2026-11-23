import os
import re
from glob import glob
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def resolve_model(model_dir, checkpoint):
    """
    Trova env.yaml, agent.yaml e i pesi in una cartella modello:
    - rl-games: <dir>/env.yaml, <dir>/agent.yaml, <dir>/*.pth
    - RSL-RL (run IsaacLab): <dir>/params/{env,agent}.yaml, <dir>/model_<iter>.pt
    Se checkpoint è vuoto si usa il .pth oppure il model_<iter>.pt con iterazione più alta.
    """
    cfg_dir = os.path.join(model_dir, 'params')
    if not os.path.isfile(os.path.join(cfg_dir, 'env.yaml')):
        cfg_dir = model_dir

    if checkpoint:
        model_path = os.path.join(model_dir, checkpoint)
    else:
        pth = sorted(glob(os.path.join(model_dir, '*.pth')))
        pts = glob(os.path.join(model_dir, 'model_*.pt'))
        if pth:
            model_path = pth[0]
        elif pts:
            model_path = max(pts, key=lambda p: int(re.search(r'model_(\d+)\.pt$', p).group(1)))
        else:
            raise RuntimeError(f'Nessun checkpoint trovato in {model_dir}')

    return os.path.join(cfg_dir, 'env.yaml'), os.path.join(cfg_dir, 'agent.yaml'), model_path


def launch_setup(context):
    pkg_share = get_package_share_directory('rlg_quad_controller')
    params    = os.path.join(pkg_share, 'config', 'mulinex_simulation_config.yaml')
    model_dir = os.path.join(pkg_share, 'models', LaunchConfiguration('model').perform(context))
    env_path, agent_path, model_path = resolve_model(
        model_dir, LaunchConfiguration('checkpoint').perform(context))

    node = Node(
        package    = 'rlg_quad_controller',
        executable = 'inference_controller_sim',
        name       = 'inference_controller_sim',
        parameters = [
            params,
            {'env_cfg_path':    env_path},
            {'agent_cfg_path':  agent_path},
            {'model_path':      model_path},
            # se vuoi, puoi aggiungere override per gli scale:
            # {'angular_velocity_scale': 1.0},
            # {'cmd_vel_scale':         1.0},
        ]
    )
    return [node]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('model', default_value='omni_flat_v1',
                              description='Cartella in models/ (rl-games o run RSL-RL)'),
        DeclareLaunchArgument('checkpoint', default_value='',
                              description='File dei pesi nella cartella (es. model_3450.pt); vuoto = automatico'),
        OpaqueFunction(function=launch_setup),
    ])
