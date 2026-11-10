from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

# default pose of the IsaacLab policies (env.yaml scene.robot.init_state.joint_pos), same as init_pos in omniquad_mf.yaml
DEFAULT_POSE = {'LF_HFE': 2.0, 'LH_HFE': -2.0, 'RF_HFE': -2.0, 'RH_HFE': 2.0,
                'LF_KFE': -1.2, 'LH_KFE': 1.2, 'RF_KFE': 1.2, 'RH_KFE': -1.2}


def generate_launch_description():
    # Omniquad (legs + omni wheels) in Gazebo with PD leg controller and wheel
    # velocity controller, ready for rlg_quad_controller omniquad_inference_sim.launch.py.
    # Extra arguments (world_name, use_lidar_3D, headless, spawn_x, ...) are forwarded to the included launch.
    # The legs are spawned in the default pose (as after a reset in IsaacLab): spawned at 0 the robot can land
    # with the hind legs folded before the PD controller starts, and 5 N m cannot unfold them.
    pd_wheels_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('mulinex_ignition'),
                'launch',
                'gz_harmonic_sim_w_rbt_PD_wheels.launch.py'
            ])
        ),
        launch_arguments={
            'robot_xacro': 'mul_ignition.xacro',
            'control_cfg_file_name': LaunchConfiguration('control_cfg_file_name'),
            'spawn_z': LaunchConfiguration('spawn_z'),
            'xacro_args': [' '.join(f'{k}:={v}' for k, v in DEFAULT_POSE.items()),
                           ' leg_velocity_limit:=', LaunchConfiguration('leg_velocity_limit')],
        }.items(),
    )

    return LaunchDescription([
        DeclareLaunchArgument('control_cfg_file_name', default_value='omniquad_mf.yaml'),
        DeclareLaunchArgument('leg_velocity_limit', default_value='10',
                              description='leg joint velocity limit [rad/s] (IsaacLab velocity_limit_sim = 10)'),
        DeclareLaunchArgument('spawn_z', default_value='0.45', description='IsaacLab init_state.pos z = 0.4'),
        pd_wheels_launch,
    ])
