"""Traquad in Gazebo set up like the IsaacLab training of the RSL-RL policies (traquad_omnistyle):
3 rollers of 14 mm per ground wheel (dry friction 0.175 N m), wheel drives capped at 0.694 N m, leg PD 60 / 0.6 and default pose
HFE +-1.47 (config/traquad_policy.yaml). Then run the policy with

    ros2 launch rlg_quad_controller traquad_inference_sim.launch.py

Every argument of gz_harmonic_sim_w_rbt_PD_wheels.launch.py can be passed (world_name, spawn_x, ...).
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    args = {
        'roller_n': '3',
        'roller_radius': '0.014',
        'wheel_effort': '0.694',   # IsaacLab wheels actuator effort_limit_sim
        'roller_friction': '0.175',  # IsaacLab roller joint dry friction (0.1 x 14/8)
        'ankle_center': '0.1009',  # IsaacLab ankle limits: front -24.2 .. +35.8 deg, hind mirrored
        'ankle_range': '0.5236',
        'ankle_damping': '0.01',   # IsaacLab ankles actuator damping
        'idler_sphere': 'false',     # sphere collider for the raised idler (cylinder-edge contacts stall in DART)
        'track_nose': 'false',       # true for the Legs3+ policies (track box cut along the belt line, frictionless)
        'wheel_mu': '0.75',          # IsaacLab contact friction: robot mu 0.4-1.1 x terrain 1.0 (multiply)
        'spawn_z': '0.35',          # IsaacLab init_state.pos z = 0.29
        'headless': 'false',
    }
    decls = [DeclareLaunchArgument(k, default_value=v) for k, v in args.items()]
    xacro_args = ['roller_n:=', LaunchConfiguration('roller_n'),
                  ' roller_radius:=', LaunchConfiguration('roller_radius'),
                  ' wheel_effort:=', LaunchConfiguration('wheel_effort'),
                  ' roller_friction:=', LaunchConfiguration('roller_friction'),
                  ' ankle_center:=', LaunchConfiguration('ankle_center'),
                  ' ankle_range:=', LaunchConfiguration('ankle_range'),
                  ' ankle_damping:=', LaunchConfiguration('ankle_damping'),
                  ' wheel_mu:=', LaunchConfiguration('wheel_mu'),
                  ' track_nose:=', LaunchConfiguration('track_nose'),
                  ' idler_sphere:=', LaunchConfiguration('idler_sphere'),
                  # spawn the legs in the default pose of the policy (as after a reset in IsaacLab): spawned at 0,
                  # the robot can land on its front legs before the PD controller starts and stay stuck there
                  ' LF_HFE:=1.47 LH_HFE:=-1.47 RF_HFE:=-1.47 RH_HFE:=1.47']
    base = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution(
            [FindPackageShare('mulinex_ignition'), 'launch', 'gz_harmonic_sim_w_rbt_PD_wheels.launch.py'])),
        launch_arguments={
            'robot_xacro': 'traquad.xacro',
            'control_cfg_file_name': 'traquad_policy.yaml',
            'spawn_z': LaunchConfiguration('spawn_z'),
            'headless': LaunchConfiguration('headless'),
            'xacro_args': xacro_args,
        }.items(),
    )
    return LaunchDescription(decls + [base])
