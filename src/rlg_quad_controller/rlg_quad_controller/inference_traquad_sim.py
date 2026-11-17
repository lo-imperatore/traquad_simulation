"""
Inference node for the traquad RSL-RL policies trained in IsaacLab (e.g. traquad_omnistyle), Gazebo version.

Policy interface (read from env.yaml, checked against the checkpoint):
  observations (policy group, in env.yaml order, each term multiplied by its `scale` and clipped by its `clip`):
    base_lin_acc       (3)  base acceleration in the base frame, gravity removed, clipped to +-30 m/s^2
    projected_gravity  (3)
    velocity_commands  (3)  [vx, vy, wz]
    joint_pos          (4)  HFE - default pose
    joint_vel          (8)  HFE + joint_wheel_2_* velocities
    actions            (8)  last raw action
  actions (8): 4 HFE position offsets (scale 0.75 around the default pose) + 1 velocity per track
               (scale 62.67 rad/s, applied to the 4 wheels of the track by wheels_vel_cnt)

Joint order of IsaacLab (PhysX articulation order, NOT the order of the regexes):
  HFE:          LH, LF, RF, RH
  wheel_2 vel:  LEFT_H, LEFT_F, RIGHT_F, RIGHT_H
  track action terms: LF, LH, RF, RH (order of the action terms in env.yaml)
"""
import numpy as np
import rclpy
import yaml
from rclpy.node import Node

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu, JointState
from pi3hat_moteus_int_msgs.msg import JointsCommand
from custom_interfaces.msg import WheelVelocityCommand

from .utils.rlg_utils import build_player, run_inference

GRAVITY = 9.81

# IsaacLab articulation order (verified with robot.joint_names on the training USD)
HFE_NAMES = ['LH_HFE', 'LF_HFE', 'RF_HFE', 'RH_HFE']
WHEEL2_NAMES = ['joint_wheel_2_LEFT_H', 'joint_wheel_2_LEFT_F', 'joint_wheel_2_RIGHT_F', 'joint_wheel_2_RIGHT_H']
TRACK_TERMS = ['joint_vel_LF', 'joint_vel_LH', 'joint_vel_RF', 'joint_vel_RH']


def quat_rotate_inverse(q_wxyz, v):
    """Rotate a world-frame vector into the body frame (q: body -> world)."""
    w, x, y, z = q_wxyz
    q_vec = np.array([x, y, z])
    a = v * (2.0 * w * w - 1.0)
    b = np.cross(q_vec, v) * w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c


class TraquadInference(Node):

    TERM_DIMS = {'base_lin_acc': 3, 'base_ang_vel': 3, 'projected_gravity': 3, 'velocity_commands': 3,
                 'joint_pos': 4, 'joint_vel': 8, 'actions': 8}

    def __init__(self):
        super().__init__('inference_traquad_sim')
        p = self.declare_parameter
        p('model_path', '')
        p('env_cfg_path', '')
        p('agent_cfg_path', '')
        p('joint_state_topic', '/Joint_Feedback')
        p('wheels_state_topic', '/Wheel_Feedback')
        p('joint_target_topic', '/pd_controller/command')
        p('wheels_target_topic', '/wheels_vel_controller/wheels_velocity_cmd')
        p('cmd_vel_topic', '/cmd_vel')
        p('imu_topic', '/mulinex/imu')
        p('warmup_duration', 3.0)
        # first-order low-pass on the accelerometer (1.0 = no filter, as in training)
        p('acc_filter_alpha', 1.0)
        p('log_obs', False)
        # rbt_pd_cnt matches the command by INDEX: order of its 'joint' parameter
        p('pd_joint_order', ['LF_HFE', 'LH_HFE', 'RF_HFE', 'RH_HFE'])
        g = lambda n: self.get_parameter(n).value

        with open(g('env_cfg_path')) as f:
            self.env_cfg = yaml.load(f, Loader=yaml.Loader)

        # --- observation terms, scales and clips from env.yaml ---
        policy_cfg = self.env_cfg['observations']['policy']
        self.terms = [k for k, v in policy_cfg.items() if isinstance(v, dict) and 'func' in v]
        unknown = [t for t in self.terms if t not in self.TERM_DIMS]
        if unknown:
            raise RuntimeError(f'Unsupported observation terms: {unknown}')
        # history (IsaacLab CircularBuffer): each term holds its last N values, oldest first, after clip and scale;
        # the group history_length, if set, applies to every term; the buffer starts filled with the first value
        group_hist = policy_cfg.get('history_length')
        self.term_hist = {t: max(1, (group_hist if group_hist is not None else policy_cfg[t].get('history_length')) or 1)
                          for t in self.terms}
        self.obs_buffers = None
        self.term_scale = {t: policy_cfg[t].get('scale') for t in self.terms}
        self.term_clip = {t: policy_cfg[t].get('clip') for t in self.terms}
        self.acc_clip = policy_cfg.get('base_lin_acc', {}).get('params', {}).get('clip', 30.0)
        self._check_joint_terms(policy_cfg)

        # --- actions ---
        acts = self.env_cfg['actions']
        if list(acts.keys()) != ['joint_pos'] + TRACK_TERMS:
            raise RuntimeError(f'Unexpected action terms: {list(acts.keys())}')
        self.leg_scale = float(acts['joint_pos']['scale'])
        self.track_scale = np.array([float(acts[t]['scale']) for t in TRACK_TERMS])
        init = self.env_cfg['scene']['robot']['init_state']['joint_pos']
        self.default_hfe = np.array([init[n] for n in HFE_NAMES])

        # --- policy ---
        self.rate_hz = 1.0 / (self.env_cfg['sim']['dt'] * self.env_cfg['decimation'])
        self.player = build_player(g('model_path'), g('env_cfg_path'), g('agent_cfg_path'))
        obs_dim = sum(self.TERM_DIMS[t] * self.term_hist[t] for t in self.terms)
        if getattr(self.player, 'obs_dim', obs_dim) != obs_dim or getattr(self.player, 'act_dim', 8) != 8:
            raise RuntimeError(f'Policy expects {self.player.obs_dim} obs / {self.player.act_dim} actions, '
                               f'node builds {obs_dim} / 8')
        self.get_logger().info(f'Loaded {g("model_path")}: obs {self.terms} ({obs_dim}), history {self.term_hist}, '
                               f'rate {self.rate_hz:.1f} Hz, '
                               f'default HFE {self.default_hfe}, leg scale {self.leg_scale}, '
                               f'track scale {self.track_scale}')

        # --- state ---
        self.hfe_pos = dict(zip(HFE_NAMES, self.default_hfe))
        self.hfe_vel = {n: 0.0 for n in HFE_NAMES}
        self.wheel_vel = {n: 0.0 for n in WHEEL2_NAMES}
        self.acc_b = np.zeros(3)
        self.ang_vel = np.zeros(3)
        self.proj_g = np.array([0.0, 0.0, -1.0])
        self.cmd = np.zeros(3)
        self.prev_action = np.zeros(8)
        self.imu_received = False
        self.acc_alpha = float(g('acc_filter_alpha'))
        self.log_obs = bool(g('log_obs'))
        self.warmup = float(g('warmup_duration'))
        self.pd_order = list(g('pd_joint_order'))
        if sorted(self.pd_order) != sorted(HFE_NAMES):
            raise RuntimeError(f'pd_joint_order must be a permutation of {HFE_NAMES}')
        self.pd_idx = [HFE_NAMES.index(n) for n in self.pd_order]

        # --- ROS ---
        self.joint_pub = self.create_publisher(JointsCommand, g('joint_target_topic'), 10)
        self.wheels_pub = self.create_publisher(WheelVelocityCommand, g('wheels_target_topic'), 10)
        self.create_subscription(JointState, g('joint_state_topic'), self.joint_cb, 10)
        self.create_subscription(JointState, g('wheels_state_topic'), self.wheels_cb, 10)
        self.create_subscription(Imu, g('imu_topic'), self.imu_cb, 10)
        self.create_subscription(Twist, g('cmd_vel_topic'), self.cmd_cb, 10)
        self.start_time = None
        self.create_timer(1.0 / self.rate_hz, self.step)

    def _check_joint_terms(self, policy_cfg):
        """The node hardcodes the IsaacLab joint order: make sure the env uses the joints it expects."""
        expected = {'joint_pos': ['.*HFE'], 'joint_vel': ['.*HFE', 'joint_wheel_2_.*']}
        for t, names in expected.items():
            got = policy_cfg[t].get('params', {}).get('asset_cfg', {}).get('joint_names')
            if t in self.terms and got != names:
                raise RuntimeError(f'Observation {t} uses joints {got}, node supports {names}')

    # ---------------- callbacks ----------------
    def joint_cb(self, msg: JointState):
        for n, q, dq in zip(msg.name, msg.position, msg.velocity):
            if n in self.hfe_pos:
                self.hfe_pos[n] = q
                self.hfe_vel[n] = dq

    def wheels_cb(self, msg: JointState):
        for n, dq in zip(msg.name, msg.velocity):
            if n in self.wheel_vel:
                self.wheel_vel[n] = dq

    def imu_cb(self, msg: Imu):
        q = np.array([msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z])
        self.proj_g = quat_rotate_inverse(q / np.linalg.norm(q), np.array([0.0, 0.0, -1.0]))
        self.ang_vel = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
        # the IMU measures the specific force a - g: add gravity back (in the body frame) to get a
        f = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
        acc = np.clip(f + GRAVITY * self.proj_g, -self.acc_clip, self.acc_clip)
        self.acc_b = self.acc_alpha * acc + (1.0 - self.acc_alpha) * self.acc_b
        self.imu_received = True

    def cmd_cb(self, msg: Twist):
        # training ranges: lin_vel_y is always 0
        self.cmd = np.array([msg.linear.x, 0.0, msg.angular.z])

    # ---------------- policy step ----------------
    def build_obs(self):
        values = {
            'base_lin_acc': self.acc_b,
            'base_ang_vel': self.ang_vel,
            'projected_gravity': self.proj_g,
            'velocity_commands': self.cmd,
            'joint_pos': np.array([self.hfe_pos[n] for n in HFE_NAMES]) - self.default_hfe,
            'joint_vel': np.array([self.hfe_vel[n] for n in HFE_NAMES] + [self.wheel_vel[n] for n in WHEEL2_NAMES]),
            'actions': self.prev_action,
        }
        current = {}
        for t in self.terms:
            v = np.asarray(values[t], dtype=float)
            if self.term_clip[t] is not None:
                v = np.clip(v, *self.term_clip[t])
            if self.term_scale[t] is not None:
                v = v * self.term_scale[t]
            current[t] = v
        if self.obs_buffers is None:
            self.obs_buffers = {t: [current[t]] * self.term_hist[t] for t in self.terms}
        else:
            for t in self.terms:
                self.obs_buffers[t] = self.obs_buffers[t][1:] + [current[t]]
        return np.concatenate([np.concatenate(self.obs_buffers[t]) for t in self.terms]).reshape(1, -1)

    def step(self):
        now = self.get_clock().now()
        if self.start_time is None:
            if now.nanoseconds == 0:  # sim clock not received yet
                return
            self.start_time = now
        if not self.imu_received:
            self.get_logger().warn('Waiting for IMU...', throttle_duration_sec=2.0)
            return

        obs = self.build_obs()
        action = run_inference(self.player, obs, det=True).flatten()
        elapsed = (now - self.start_time).nanoseconds * 1e-9

        if elapsed < self.warmup:
            # hold the default pose with the tracks still; as after a reset in IsaacLab, last_action = 0
            hfe_target = self.default_hfe
            track_vel = np.zeros(4)
            self.prev_action = np.zeros(8)
        else:
            hfe_target = self.default_hfe + self.leg_scale * action[:4]
            track_vel = self.track_scale * action[4:]
            self.prev_action = action

        if self.log_obs:
            self.get_logger().info(f'obs {np.round(obs.flatten(), 3)}\nact {np.round(action, 3)}',
                                   throttle_duration_sec=0.5)

        msg = JointsCommand()
        msg.header.stamp = now.to_msg()
        msg.name = self.pd_order
        msg.position = hfe_target[self.pd_idx].tolist()
        msg.velocity = [0.0] * 4
        msg.effort = [0.0] * 4
        msg.kp_scale = [1.0] * 4
        msg.kd_scale = [1.0] * 4
        self.joint_pub.publish(msg)

        w = WheelVelocityCommand()  # tracks: LF, LH, RF, RH
        w.v_lf, w.v_lb, w.v_rf, w.v_rb = (float(x) for x in track_vel)
        self.wheels_pub.publish(w)


def main(args=None):
    rclpy.init(args=args)
    node = TraquadInference()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
