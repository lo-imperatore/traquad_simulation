import rclpy
from rclpy.node import Node

import yaml
import torch
import numpy as np
import transforms3d as t3d

from sensor_msgs.msg import Imu, JointState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from pi3hat_moteus_int_msgs.msg import JointsCommand, JointsStates

from .utils.torch_utils import quat_rotate_inverse_numpy
from .utils.rlg_utils import build_player, run_inference
from custom_interfaces.msg import WheelVelocityCommand 

import time

class InferenceController(Node):
    """
    ROS2 node per eseguire inference con una policy rl‑games o RSL-RL
    e pubblicare comandi articolari a rate fissa.
    """

    # dimensione di ogni termine di osservazione supportato
    OBS_TERM_DIMS = {
        'base_lin_vel': 3, 'base_ang_vel': 3, 'projected_gravity': 3, 'velocity_commands': 3,
        'joint_pos': 8, 'joint_vel': 12, 'actions': 12,
    }

    def __init__(self):
        super().__init__('inference_controller')

        # --- Dichiarazione parametri ROS ---
        self.declare_parameter('model_path', '')
        self.declare_parameter('env_cfg_path', '')
        self.declare_parameter('agent_cfg_path', '')
        self.declare_parameter('simulation', False)
        self.declare_parameter('joint_state_topic', '/Joint_Feedback')
        self.declare_parameter('wheels_state_topic', '/Wheel_Feedback')
        self.declare_parameter('joint_target_topic', '/pd_controller/command')                      # topic for joint commands (PD controller)
        self.declare_parameter('wheels_target_topic', '/wheels_vel_controller/wheels_velocity_cmd') # topic for wheels commands (wheels velocity controller)
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('imu_topic', '/mulinex/imu')
        # velocità lineare della base (nel frame base), usata solo se la policy osserva base_lin_vel
        self.declare_parameter('base_lin_vel_topic', '/mulinex/gt_odom')
        # nuovi parametri per scaling
        self.declare_parameter('angular_velocity_scale', 1.0)
        self.declare_parameter('cmd_vel_scale', 1.0)

        # --- Lettura parametri ---
        self.model_path       = self.get_parameter('model_path').value
        self.env_cfg_path     = self.get_parameter('env_cfg_path').value
        self.agent_cfg_path   = self.get_parameter('agent_cfg_path').value
        self.simulation       = self.get_parameter('simulation').value
        self.joint_state_topic= self.get_parameter('joint_state_topic').value
        self.wheels_state_topic= self.get_parameter('wheels_state_topic').value
        self.joint_target_topic = self.get_parameter('joint_target_topic').value
        self.wheels_target_topic = self.get_parameter('wheels_target_topic').value
        self.cmd_vel_topic    = self.get_parameter('cmd_vel_topic').value
        self.angular_vel_scale= self.get_parameter('angular_velocity_scale').value
        self.cmd_vel_scale    = self.get_parameter('cmd_vel_scale').value
        imu_topic             = self.get_parameter('imu_topic').value


        # init vars
        self.base_lin_vel = np.zeros((3,1))
        self.base_ang_vel = np.zeros((3,1))
        self.projected_gravity = np.zeros((3,1))
        self.cmd_vel = np.zeros((3,1))

        # --- Caricamento YAML di configurazione ---
        with open(self.env_cfg_path, 'r') as f:
            self.env_cfg = yaml.load(f, Loader=yaml.Loader)
        with open(self.agent_cfg_path, 'r') as f:
            self.agent_cfg = yaml.load(f, Loader=yaml.Loader)

        # --- Termini di osservazione della policy (ordine e history presi da env.yaml) ---
        # base_lin_vel è opzionale: si attiva solo se presente nelle osservazioni di training.
        # history_length > 0: il termine contiene gli ultimi N valori, dal più vecchio al più nuovo
        # (come il CircularBuffer di IsaacLab); la history di gruppo, se c'è, vale per tutti i termini.
        policy_cfg = self.env_cfg['observations']['policy']
        self.obs_terms = [k for k, v in policy_cfg.items() if isinstance(v, dict) and 'func' in v]
        unknown = [t for t in self.obs_terms if t not in self.OBS_TERM_DIMS]
        if unknown:
            raise RuntimeError(f'Termini di osservazione non supportati: {unknown}')
        group_hist = policy_cfg.get('history_length')
        self.obs_hist = {t: max(1, (group_hist if group_hist is not None else policy_cfg[t].get('history_length')) or 1)
                         for t in self.obs_terms}
        self.obs_buffers = None   # riempiti alla prima inference, come al reset in IsaacLab
        self.use_base_lin_vel = 'base_lin_vel' in self.obs_terms
        self.base_lin_vel_received = False

        # --- Calcolo rate di inference ---
        dt = self.env_cfg['sim']['dt']
        decimation = self.env_cfg['decimation']
        self.rate_hz = 1.0 / (dt * decimation)
        self.get_logger().info(f'Inference rate: {self.rate_hz:.2f} Hz')

        # --- Scaling azioni ---
        leg_scale   = self.env_cfg['actions']['joint_pos']['scale']
        wheel_scale = self.env_cfg['actions']['joint_vel']['scale']
        self.action_scale = np.array([leg_scale]*8 + [wheel_scale]*4).reshape((12,1))

        # --- Caricamento modello (rl-games o RSL-RL) ---
        self.get_logger().info(f"Loading checkpoint: {self.model_path}")
        self.player = build_player(
            weights_path   = self.model_path,
            env_cfg_path   = self.env_cfg_path,
            agent_cfg_path = self.agent_cfg_path,
        )
        self.get_logger().info(f'Model successfully loaded ({type(self.player).__name__}).')
        obs_dim = sum(self.OBS_TERM_DIMS[t] * self.obs_hist[t] for t in self.obs_terms)
        self.get_logger().info(f'Observation: {obs_dim} values, history {self.obs_hist}')
        if getattr(self.player, 'obs_dim', obs_dim) != obs_dim:
            raise RuntimeError(f'La policy si aspetta {self.player.obs_dim} osservazioni, il nodo ne costruisce {obs_dim}')

        # --- Pubblicatori/Sottoscrittori ---
        if self.simulation:
            # il PD controller di Gazebo (rbt_pd_cnt) si sottoscrive a JointsCommand
            self.joint_pub = self.create_publisher(JointsCommand, self.joint_target_topic, 10)
            self.wheels_pub = self.create_publisher(WheelVelocityCommand, self.wheels_target_topic, 10)

            self.joint_sub = self.create_subscription(
                JointState, self.joint_state_topic, self.joint_state_callback, 10
            )
            self.wheels_sub = self.create_subscription(
                JointState, self.wheels_state_topic, self.wheels_state_callback, 10
            )
        else:
            self.joint_pub = self.create_publisher(JointsCommand, self.joint_target_topic, 10)
            self.wheels_pub = self.create_publisher(WheelVelocityCommand, self.wheels_target_topic, 10)

            self.joint_sub = self.create_subscription(
                JointState, self.joint_state_topic, self.joint_state_callback, 10
            )
            self.wheels_sub = self.create_subscription(
                JointState, self.wheels_state_topic, self.wheels_state_callback, 10
            )
        self.cmd_sub = self.create_subscription(
            Twist, self.cmd_vel_topic, self.cmd_vel_callback, 10
        )

        self.declare_parameter('use_imu', False)
        self.use_imu = self.get_parameter('use_imu').value


        if self.use_imu:
            self.imu_sub = self.create_subscription(Imu, imu_topic, self.imu_callback, 10) 

        if self.use_base_lin_vel:
            base_lin_vel_topic = self.get_parameter('base_lin_vel_topic').value
            self.get_logger().info(f'Policy uses base_lin_vel, subscribing to {base_lin_vel_topic}')
            self.odom_sub = self.create_subscription(
                Odometry, base_lin_vel_topic, self.odom_callback, 10
            )

        # --- Buffer di stato ---
        self.n_joints_tot = 12
        self.n_joints_pos = 8
        self.joint_names_pos = [
            'LF_HFE','LH_HFE','RF_HFE','RH_HFE',
            'LF_KFE','LH_KFE','RF_KFE','RH_KFE',
        ]
        self.joint_names_vel = [
            'LF_HFE','LH_HFE','RF_HFE','RH_HFE',
            'LF_KFE','LH_KFE','RF_KFE','RH_KFE',
            'LF_WHEEL_JNT','LH_WHEEL_JNT','RF_WHEEL_JNT','RH_WHEEL_JNT'
        ]
        self.wheels_names = [
            'RF_WHEEL_JNT', 'LF_WHEEL_JNT', 'RH_WHEEL_JNT', 'LH_WHEEL_JNT'
        ]
        self.joint_pos = {n:0.0 for n in self.joint_names_pos}  # 8 joints
        self.joint_vel = {n:0.0 for n in self.joint_names_vel}  # 12 joints
        self.prev_action = np.zeros((self.n_joints_tot,1))

        # --- Posa di default (dal config di training IsaacLab) e warmup ---
        # In IsaacLab le ruote si chiamano *_ANKLE, in Gazebo *_WHEEL_JNT
        init_joint_pos = self.env_cfg['scene']['robot']['init_state']['joint_pos']
        self.default_pose = np.array(
            [init_joint_pos[n] for n in self.joint_names_pos] +
            [init_joint_pos.get(n.replace('_WHEEL_JNT', '_ANKLE'), 0.0) for n in self.joint_names_vel[8:]]
        )
        self.get_logger().info(f'Default pose: {self.default_pose}')
        self._warmup_duration = 3.0
        self.start_time = self.get_clock().now()

        # --- Timer di inference ---
        self.timer = self.create_timer(1.0/self.rate_hz, self.inference_callback)
        self.get_logger().info('Node initialized and ready.')

    def cmd_vel_callback(self, msg: Twist):
        self.cmd_vel = np.array([
            msg.linear.x, msg.linear.y, msg.angular.z
        ]).reshape((3,1))
        # self.cmd_vel = np.zeros((3,1))
        # self.cmd_vel = np.array([1, 0, 0]).reshape((3,1))
        self.get_logger().info(f'\nCommand velocity received: linear-x: {msg.linear.x}, linear-y: {msg.linear.y}, angular-z: {msg.angular.z}\n')


    def imu_callback(self, msg: Imu):
        """
        Update base angular velocity and orientation from IMU.
        Project gravity vector into robot frame.
        """
        self.base_ang_vel = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ]).reshape((3, 1))

        quat = [
            msg.orientation.w,
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z
        ]
        yaw, pitch, roll = t3d.euler.quat2euler(quat, axes='szyx')
        self.yaw_bias = 0.0
        corrected = t3d.euler.euler2quat(
            yaw, pitch, roll, axes='szyx'
        )
        self.base_quat = np.array(corrected).reshape((4, 1))
        self.projected_gravity = quat_rotate_inverse_numpy(
            self.base_quat, np.array([0, 0, -1]).reshape((3, 1))
        )

    def odom_callback(self, msg: Odometry):
        """
        Update base linear velocity. Il twist di nav_msgs/Odometry è espresso nel
        child_frame (base_link), come base_lin_vel in IsaacLab.
        """
        self.base_lin_vel = np.array([
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z
        ]).reshape((3, 1))
        self.base_lin_vel_received = True

    def joint_state_callback(self, msg: JointState):
        """
        Update current joint positions and velocities.
        """
        for name, pos, vel in zip(msg.name, msg.position, msg.velocity):
            if name in self.joint_pos:
                self.joint_pos[name] = pos
                self.joint_vel[name] = vel
        # self.get_logger().info(f"Joint state (callback):  pos: {self.joint_pos}\nvel: {self.joint_vel}\n")


    def wheels_state_callback(self, msg: JointState):
        """
        Update current wheels positions and velocities.
        """
        for name, pos, vel in zip(msg.name, msg.position, msg.velocity):
            if name in self.wheels_names:
                self.joint_vel[name] = vel
        # self.get_logger().info(f"Wheels vel (callback): {self.joint_vel}\n\n")


    def inference_callback(self):
        now = self.get_clock().now()
        # costruisco l’osservazione

        # +---------------------------------------------------------+
        # | Active Observation Terms in Group: 'policy' (shape: (41,)) |
        # +-----------+---------------------------------+-----------+
        # |   Index   | Name                            |   Shape   |
        # +-----------+---------------------------------+-----------+
        # |     0     | base_ang_vel                    |    (3,)   |
        # |     1     | projected_gravity               |    (3,)   |
        # |     2     | velocity_commands               |    (3,)   |
        # |     3     | joint_pos                       |    (8,)   |
        # |     4     | joint_vel                       |   (12,)   |
        # |     5     | actions                         |   (12,)   |
        # +-----------+---------------------------------+-----------+

        # for name in self.wheels_names:
        #     self.joint_vel[name] = 0.0

        # i modelli RSL-RL rough hanno anche base_lin_vel (in testa) e alcuni la history dei termini
        if self.use_base_lin_vel and not self.base_lin_vel_received:
            self.get_logger().warn('No base_lin_vel received yet, using zeros', throttle_duration_sec=2.0)
        current = {
            'base_lin_vel': self.base_lin_vel,
            'base_ang_vel': self.base_ang_vel * self.angular_vel_scale,
            'projected_gravity': self.projected_gravity,
            'velocity_commands': self.cmd_vel * self.cmd_vel_scale,
            'joint_pos': (np.array(list(self.joint_pos.values()) - self.default_pose[0:8])).reshape((self.n_joints_pos,1)), # 8
            'joint_vel': np.fromiter(self.joint_vel.values(), dtype=float).reshape((self.n_joints_tot,1)), # 12
            'actions': self.prev_action, # 12
        }
        if self.obs_buffers is None:
            self.obs_buffers = {t: [current[t].flatten()] * self.obs_hist[t] for t in self.obs_terms}
        else:
            for t in self.obs_terms:
                self.obs_buffers[t] = self.obs_buffers[t][1:] + [current[t].flatten()]
        obs = np.concatenate([np.concatenate(self.obs_buffers[t]) for t in self.obs_terms]).reshape((1,-1))
        # self.get_logger().info(f"OSSERVAZIONE RL: {obs}")


        # obs = np.array([[
        #     -7.3413e-01, -4.9844e-01, -1.8524e-01,  1.0361e-02,  3.6034e-02,
        #     -1.0303e+00,  1.0000e+00,  0.0000e+00, -0.0000e+00, -1.0728e-01,
        #     -4.0791e-01,  3.2821e-01, -1.8986e-01,  8.9756e-03, -6.2530e-02,
        #     -1.1988e-02, -5.1432e-02,  2.5427e+00, -2.2847e-01, -1.6544e+00,
        #     -1.8129e+00,  7.4929e-01,  7.1045e+00, -2.7388e+00,  8.2183e+00,
        #     9.6702e+00,  9.6817e+00,  5.3614e+00,  1.0157e+01,  1.7997e-01,
        #     3.2596e-01, -8.4879e-01, -2.2894e-01,  2.0105e-01,  9.0680e-02,
        #     -3.5459e-01,  3.5399e-01,  1.0000e+00,  1.0000e+00,  5.0096e-01,
        #     1.0000e+00
        # ]])

        action = run_inference(self.player, obs, det=True).flatten()

        # action = np.array([ 0.5663, -0.1318, -0.0210, -0.4200,  0.9181,  0.0695,  0.0098,  0.2165,
        #   1.0000,  1.0000,  1.0000,  1.0000])

        self.prev_action = action.reshape((self.n_joints_tot,1))

        # action[8:12] = 0.0
        self.get_logger().info(f"obs: {obs}")

        self.get_logger().info(f"Action shape: {action.shape}, Action: {action}")
        # time.sleep(10)
        


        # warmup default pose
        delta = now - self.start_time
        elapsed = delta.nanoseconds * 1e-9
        if elapsed < self._warmup_duration:
            target = self.default_pose
            # l'azione non viene applicata: come al reset in IsaacLab, last_action = 0
            self.prev_action = np.zeros((self.n_joints_tot,1))
        else:
            target = action * self.action_scale.flatten() + self.default_pose

        # pubblicazione
        # Joints
        msg = JointsCommand()
        msg.kp_scale = [1.0]*self.n_joints_pos
        msg.kd_scale = [1.0]*self.n_joints_pos

        msg.header.stamp = now.to_msg()
        msg.name = self.joint_names_pos
        msg.position = target[0:8].tolist()
        # il PD controller usa velocity/effort come feed-forward
        msg.velocity = [0.0]*self.n_joints_pos
        msg.effort = [0.0]*self.n_joints_pos
        self.joint_pub.publish(msg)
        # self.get_logger().info(f"Published target: {target}\n")
        # self.get_logger().info(f"Action: {action}\n")
        # Wheels
        msg = WheelVelocityCommand()
        ''' - LF_WHEEL_JNT
            - LH_WHEEL_JNT
            - RF_WHEEL_JNT
            - RH_WHEEL_JNT
        '''
        # mapping RL -> Gazebo controller (posizione nella lista del controller)
        action_wheels = [target[8], target[9], target[10], target[11]]
        # IsaacLab order: [LF, LH, RF, RH]
        # Gazebo order:   [RF, LF, RH, LH]

        msg.v_rf = action_wheels[2]  # RF_WHEEL_JNT <-- RF (idx 2)
        msg.v_lf = action_wheels[0]  # LF_WHEEL_JNT <-- LF (idx 0)
        msg.v_rb = action_wheels[3]  # RH_WHEEL_JNT <-- RH (idx 3)
        msg.v_lb = action_wheels[1]  # LH_WHEEL_JNT <-- LH (idx 1)

        self.wheels_pub.publish(msg)

        


def main(args=None):
    rclpy.init(args=args)
    node = InferenceController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()