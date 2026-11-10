"""
Bridge between an IsaacLab omniquad policy running on the laptop and the real robot (Pi3Hat).

The node runs on the laptop, on the same ROS_DOMAIN_ID as the robot, and talks to the robot
controllers over the network:
  in : /state_broadcaster/joints_state (pi3hat JointsStates)  legs + wheels position/velocity
       /IMU_Broadcaster/imu            (sensor_msgs Imu)        orientation + angular velocity
       /cmd_vel                        (geometry_msgs Twist)    velocity command (teleop)
  out: /joint_controller/command      (pi3hat JointsCommand)   12 joints: legs position, wheels velocity

Phases (the robot always gets a command except in WAIT and with dry_run):
  WAIT    -> no command until joint states and IMU are arriving
  STANDUP -> interpolation from the measured pose to the policy default pose, wheels stopped
  HOLD    -> default pose, wheels stopped; the policy runs in shadow (only logged)
  POLICY  -> policy actions applied (scaled by leg/wheel_action_gain)
Service ~/start: HOLD -> POLICY.  Service ~/stop: POLICY -> HOLD (default pose, wheels stopped).
Fault (stale data, tilt, non-finite action) -> HOLD.

dry_run: True runs everything (phases included) without publishing anything.
"""
import signal
import time

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import Twist
from pi3hat_moteus_int_msgs.msg import JointsCommand, JointsStates
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu
from std_srvs.srv import Trigger

from .utils.rlg_utils import build_player, run_inference
from .utils.torch_utils import quat_rotate_inverse_numpy

WAIT, STANDUP, HOLD, POLICY = 'WAIT', 'STANDUP', 'HOLD', 'POLICY'

# Order of the policy (IsaacLab articulation order), same as inference_controller_sim
POLICY_LEGS = ['LF_HFE', 'LH_HFE', 'RF_HFE', 'RH_HFE', 'LF_KFE', 'LH_KFE', 'RF_KFE', 'RH_KFE']
POLICY_WHEELS = ['LF', 'LH', 'RF', 'RH']   # IsaacLab *_ANKLE

# Supported observation terms and their size
OBS_TERM_DIMS = {
    'base_ang_vel': 3, 'projected_gravity': 3, 'velocity_commands': 3,
    'joint_pos': 8, 'joint_vel': 12, 'actions': 12,
}


class OmniquadRobotBridge(Node):

    def __init__(self):
        super().__init__('inference_omniquad_robot')

        p = self.declare_parameter
        p('model_path', '')
        p('env_cfg_path', '')
        p('agent_cfg_path', '')
        p('joint_state_topic', '/state_broadcaster/joints_state')
        p('imu_topic', '/IMU_Broadcaster/imu')
        p('cmd_topic', '/joint_controller/command')
        p('cmd_vel_topic', '/cmd_vel')
        # Command order of the robot controller (as getup_omni_1 / bridge_node): keep it, the
        # controller may match the joints by index and not by name.
        p('robot_joint_order', ['LF_HFE', 'LF_KFE', 'LH_HFE', 'LH_KFE', 'RF_HFE', 'RF_KFE',
                                'RH_HFE', 'RH_KFE', 'LF_WHEEL', 'LH_WHEEL', 'RF_WHEEL', 'RH_WHEEL'])
        # robot wheel joint names and signs in policy order LF, LH, RF, RH (robot = sign * policy).
        # Default signs from bridge_node.py (left wheels inverted): CHECK them on the stand.
        p('robot_wheel_names', ['LF_WHEEL', 'LH_WHEEL', 'RF_WHEEL', 'RH_WHEEL'])
        p('wheel_signs', [-1.0, -1.0, 1.0, 1.0])
        p('dry_run', True)
        p('auto_start', False)               # go to POLICY right after STANDUP
        p('standup_time', 5.0)               # s
        p('leg_action_gain', 1.0)            # scale of the leg offset from the default pose
        p('wheel_action_gain', 1.0)          # scale of the wheel velocity (0 = legs only)
        p('max_leg_offset', 1.2)             # rad, clamp of |target - default|
        p('max_wheel_vel', 20.0)             # rad/s
        p('max_tilt_deg', 45.0)              # fault above this tilt
        p('state_timeout', 0.1)              # s, fault if joint states / IMU are older
        p('leg_kp_scale', 1.0)
        p('leg_kd_scale', 1.0)
        p('wheel_kp_scale', 1.0)
        p('wheel_kd_scale', 1.0)
        p('log_path', '')                    # .npz with obs/actions/targets, written at exit
        p('status_period', 1.0)              # s, status print

        g = lambda n: self.get_parameter(n).value
        self.dry_run = g('dry_run')
        self.auto_start = g('auto_start')
        self.standup_time = g('standup_time')
        self.leg_gain = g('leg_action_gain')
        self.wheel_gain = g('wheel_action_gain')
        self.max_leg_offset = g('max_leg_offset')
        self.max_wheel_vel = g('max_wheel_vel')
        self.min_gravity_z = -np.cos(np.deg2rad(g('max_tilt_deg')))
        self.state_timeout = g('state_timeout')
        self.log_path = g('log_path')
        self.robot_order = list(g('robot_joint_order'))
        self.robot_wheels = list(g('robot_wheel_names'))
        self.wheel_signs = np.array(g('wheel_signs'), dtype=float)
        assert len(self.robot_wheels) == 4 and len(self.wheel_signs) == 4
        assert set(self.robot_order) == set(POLICY_LEGS + self.robot_wheels), \
            f'robot_joint_order {self.robot_order} must contain {POLICY_LEGS + self.robot_wheels}'

        # --- config of the training ---
        with open(g('env_cfg_path')) as f:
            env_cfg = yaml.load(f, Loader=yaml.Loader)
        self.dt = env_cfg['sim']['dt'] * env_cfg['decimation']
        self.leg_scale = env_cfg['actions']['joint_pos']['scale']
        self.wheel_scale = env_cfg['actions']['joint_vel']['scale']
        init = env_cfg['scene']['robot']['init_state']['joint_pos']
        self.default_legs = np.array([init[n] for n in POLICY_LEGS])
        ranges = env_cfg['commands']['base_velocity']['ranges']
        self.cmd_lo = np.array([ranges['lin_vel_x'][0], ranges['lin_vel_y'][0], ranges['ang_vel_z'][0]])
        self.cmd_hi = np.array([ranges['lin_vel_x'][1], ranges['lin_vel_y'][1], ranges['ang_vel_z'][1]])

        pol = env_cfg['observations']['policy']
        self.obs_terms = [k for k, v in pol.items() if isinstance(v, dict) and 'func' in v]
        unknown = [t for t in self.obs_terms if t not in OBS_TERM_DIMS]
        if unknown:
            raise RuntimeError(f'Unsupported observation terms (base_lin_vel needs an estimator): {unknown}')
        gh = pol.get('history_length')
        self.obs_hist = {t: max(1, (gh if gh is not None else pol[t].get('history_length')) or 1)
                         for t in self.obs_terms}
        self.obs_buffers = None

        self.player = build_player(g('model_path'), g('env_cfg_path'), g('agent_cfg_path'))
        obs_dim = sum(OBS_TERM_DIMS[t] * self.obs_hist[t] for t in self.obs_terms)
        if getattr(self.player, 'obs_dim', obs_dim) != obs_dim:
            raise RuntimeError(f'Policy expects {self.player.obs_dim} obs, node builds {obs_dim}')
        self.get_logger().info(
            f'Policy {g("model_path")}\n  rate {1 / self.dt:.1f} Hz, obs {obs_dim} {self.obs_hist}\n'
            f'  default legs {self.default_legs}\n  leg scale {self.leg_scale}, wheel scale {self.wheel_scale}'
            f'\n  cmd range {self.cmd_lo} .. {self.cmd_hi}')

        # --- state ---
        self.leg_pos = None                       # policy order
        self.leg_vel = np.zeros(8)
        self.wheel_vel = np.zeros(4)              # policy order and sign
        self.ang_vel = np.zeros(3)
        self.gravity = np.array([0.0, 0.0, -1.0])
        self.t_joints = self.t_imu = None         # monotonic receipt times
        self.cmd = np.zeros(3)
        self.prev_action = np.zeros(12)
        self.last_action = np.zeros(12)
        self.phase = WAIT
        self.phase_t0 = time.monotonic()
        self.standup_from = None
        self.fault = ''
        self.n_clamped = 0
        self.loop_dts, self.t_last_loop = [], None
        self.log = {k: [] for k in ('t', 'phase', 'obs', 'action', 'target_legs', 'target_wheels',
                                    'leg_pos', 'wheel_vel', 'cmd')}

        # --- ROS ---
        # sensor QoS (best effort, depth 5): compatible with any publisher, no queueing over WiFi
        self.create_subscription(JointsStates, g('joint_state_topic'), self.joint_cb, qos_profile_sensor_data)
        self.create_subscription(Imu, g('imu_topic'), self.imu_cb, qos_profile_sensor_data)
        self.create_subscription(Twist, g('cmd_vel_topic'), self.cmd_vel_cb, 10)
        self.cmd_pub = self.create_publisher(JointsCommand, g('cmd_topic'), 10)
        self.create_service(Trigger, '~/start', self.start_srv)
        self.create_service(Trigger, '~/stop', self.stop_srv)
        self.create_timer(self.dt, self.step)
        self.create_timer(g('status_period'), self.status)
        self.get_logger().warn('DRY RUN: nothing is published' if self.dry_run else
                               f'LIVE: commands on {g("cmd_topic")}')

    # ------------------------------------------------------------------ callbacks
    def joint_cb(self, msg: JointsStates):
        idx = {n: i for i, n in enumerate(msg.name)}
        try:
            self.leg_pos = np.array([msg.position[idx[n]] for n in POLICY_LEGS])
            self.leg_vel = np.array([msg.velocity[idx[n]] for n in POLICY_LEGS])
            self.wheel_vel = self.wheel_signs * np.array([msg.velocity[idx[n]] for n in self.robot_wheels])
        except (KeyError, IndexError):
            self.get_logger().error(f'joint state without the expected joints: {list(msg.name)}',
                                    throttle_duration_sec=2.0)
            return
        self.t_joints = time.monotonic()

    def imu_cb(self, msg: Imu):
        q = np.array([msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z])
        n = np.linalg.norm(q)
        if not np.isfinite(n) or n < 0.5:
            self.get_logger().error('IMU without a valid orientation', throttle_duration_sec=2.0)
            return
        self.gravity = quat_rotate_inverse_numpy((q / n).reshape(4, 1),
                                                 np.array([0.0, 0.0, -1.0]).reshape(3, 1)).flatten()
        self.ang_vel = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
        self.t_imu = time.monotonic()

    def cmd_vel_cb(self, msg: Twist):
        self.cmd = np.clip([msg.linear.x, msg.linear.y, msg.angular.z], self.cmd_lo, self.cmd_hi)

    def start_srv(self, req, res):
        if self.phase != HOLD:
            res.success, res.message = False, f'phase is {self.phase}, start only from HOLD'
        else:
            self.set_phase(POLICY)
            res.success, res.message = True, 'policy active'
        return res

    def stop_srv(self, req, res):
        if self.phase == POLICY:
            self.set_phase(HOLD)
        res.success, res.message = True, f'phase {self.phase}'
        return res

    # ------------------------------------------------------------------ main loop
    def set_phase(self, phase, reason=''):
        if phase == self.phase:
            return
        self.get_logger().warn(f'{self.phase} -> {phase} {reason}')
        self.phase, self.phase_t0 = phase, time.monotonic()
        self.prev_action = np.zeros(12)
        if phase == STANDUP:
            self.standup_from = self.leg_pos.copy()

    def check_fault(self, now):
        if self.t_joints is None or self.t_imu is None:
            return 'no data'
        if now - self.t_joints > self.state_timeout:
            return f'joint states {1e3 * (now - self.t_joints):.0f} ms old'
        if now - self.t_imu > self.state_timeout:
            return f'IMU {1e3 * (now - self.t_imu):.0f} ms old'
        if self.gravity[2] > self.min_gravity_z:
            return f'tilt, gravity {np.round(self.gravity, 2)}'
        return ''

    def build_obs(self):
        current = {
            'base_ang_vel': self.ang_vel,
            'projected_gravity': self.gravity,
            'velocity_commands': self.cmd if self.phase == POLICY else np.zeros(3),
            'joint_pos': self.leg_pos - self.default_legs,
            'joint_vel': np.concatenate([self.leg_vel, self.wheel_vel]),
            'actions': self.prev_action,
        }
        if self.obs_buffers is None:   # as at the reset in IsaacLab: history filled with the first obs
            self.obs_buffers = {t: [current[t]] * self.obs_hist[t] for t in self.obs_terms}
        else:
            for t in self.obs_terms:
                self.obs_buffers[t] = self.obs_buffers[t][1:] + [current[t]]
        frame = np.concatenate([current[t] for t in self.obs_terms])
        return np.concatenate([np.concatenate(self.obs_buffers[t]) for t in self.obs_terms])[None, :], frame

    def step(self):
        now = time.monotonic()
        if self.t_last_loop is not None:
            self.loop_dts.append(now - self.t_last_loop)
        self.t_last_loop = now

        fault = self.check_fault(now)
        if self.phase == WAIT:
            if not fault:
                self.set_phase(STANDUP, '(data ok)')
            return
        if fault.split(' ')[:2] != self.fault.split(' ')[:2] and fault:   # log once per kind
            self.get_logger().error(f'FAULT: {fault}')
        self.fault = fault
        if fault and self.phase == POLICY:
            self.set_phase(HOLD, f'(fault: {fault})')
        if self.leg_pos is None:
            return

        # the policy runs in every phase after WAIT (shadow in STANDUP/HOLD) to keep the history full
        obs, frame = self.build_obs()
        action = run_inference(self.player, obs.astype(np.float32), det=True).flatten()
        if not np.all(np.isfinite(action)):
            self.get_logger().error('non-finite action')
            action = np.zeros(12)
            if self.phase == POLICY:
                self.set_phase(HOLD, '(non-finite action)')
        self.last_action = action

        wheels = np.zeros(4)
        if self.phase == STANDUP:
            a = min(1.0, (now - self.phase_t0) / self.standup_time)
            legs = (1 - a) * self.standup_from + a * self.default_legs
            if a >= 1.0:
                self.set_phase(POLICY if self.auto_start and not fault else HOLD, '(standup done)')
        elif self.phase == HOLD:
            legs = self.default_legs.copy()
        else:   # POLICY
            self.prev_action = action      # IsaacLab last_action is the raw policy output
            offset = self.leg_gain * self.leg_scale * action[:8]
            clamped = np.clip(offset, -self.max_leg_offset, self.max_leg_offset)
            self.n_clamped += int(np.any(clamped != offset))
            legs = self.default_legs + clamped
            wheels = np.clip(self.wheel_gain * self.wheel_scale * action[8:], -self.max_wheel_vel, self.max_wheel_vel)

        self.publish(legs, wheels, now)
        if self.log_path:
            for k, v in (('t', now), ('phase', self.phase), ('obs', frame), ('action', action),
                         ('target_legs', legs), ('target_wheels', wheels), ('leg_pos', self.leg_pos),
                         ('wheel_vel', self.wheel_vel), ('cmd', self.cmd)):
                self.log[k].append(v)

    def publish(self, legs, wheels, now):
        """legs: target positions in POLICY_LEGS order, wheels: velocities in policy order and sign."""
        if self.dry_run:
            return
        pos = dict(zip(POLICY_LEGS, legs))
        vel = dict(zip(self.robot_wheels, self.wheel_signs * wheels))
        msg = JointsCommand()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.robot_order
        msg.position = [float(pos[n]) if n in pos else float('nan') for n in self.robot_order]
        msg.velocity = [float(vel.get(n, 0.0)) for n in self.robot_order]
        msg.effort = [0.0] * len(self.robot_order)
        kp = {True: self.get_parameter('leg_kp_scale').value, False: self.get_parameter('wheel_kp_scale').value}
        kd = {True: self.get_parameter('leg_kd_scale').value, False: self.get_parameter('wheel_kd_scale').value}
        msg.kp_scale = [float(kp[n in pos]) for n in self.robot_order]
        msg.kd_scale = [float(kd[n in pos]) for n in self.robot_order]
        self.cmd_pub.publish(msg)

    def status(self):
        now = time.monotonic()
        age = lambda t: 'never' if t is None else f'{1e3 * (now - t):.0f}ms'
        dts = np.array(self.loop_dts) if self.loop_dts else np.array([np.nan])
        self.loop_dts = []
        msg = (f'[{self.phase}{" DRY" if self.dry_run else ""}] loop {1e3 * dts.mean():.1f}/{1e3 * dts.max():.1f}ms '
               f'age joints {age(self.t_joints)} imu {age(self.t_imu)} | cmd {np.round(self.cmd, 2)}')
        if self.leg_pos is not None:
            msg += (f'\n  gravity {np.round(self.gravity, 2)} ang_vel {np.round(self.ang_vel, 2)}'
                    f'\n  leg err {np.round(self.leg_pos - self.default_legs, 2)}'
                    f'\n  wheel vel {np.round(self.wheel_vel, 1)}'
                    f'\n  action {np.round(self.last_action, 2)} (clamped steps {self.n_clamped})')
        if self.fault:
            msg += f'\n  FAULT: {self.fault}'
        self.get_logger().info(msg)

    def shutdown(self):
        """Last command: default pose (or current pose before the standup), wheels stopped."""
        if self.leg_pos is not None and self.phase != WAIT:
            legs = self.default_legs if self.phase in (HOLD, POLICY) else self.leg_pos
            for _ in range(3):
                self.publish(legs, np.zeros(4), time.monotonic())
        if self.log_path and self.log['t']:
            np.savez(self.log_path, **{k: np.array(v) for k, v in self.log.items()},
                     policy_legs=POLICY_LEGS, policy_wheels=POLICY_WHEELS, obs_terms=self.obs_terms)
            self.get_logger().info(f'log saved to {self.log_path}')


def main(args=None):
    # SIGINT/SIGTERM handled here (not by rclpy): the node stops spinning, then publishes the
    # last command (default pose, wheels stopped) and saves the log
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    stop = []
    signal.signal(signal.SIGINT, lambda *_: stop.append(1))
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    node = OmniquadRobotBridge()
    try:
        while not stop and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
