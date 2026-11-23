"""
Rest / stand sequence of the omni_controller (activation hip offset, cosine interpolation, see utils/getup.py) in Gazebo:
the same code of inference_omniquad_robot drives the pd_controller (/pd_controller/command) of the simulated omniquad.

The node holds the measured pose, then:  ros2 service call /getup_omniquad_sim/stand std_srvs/srv/Trigger
(or auto_stand:=true).  ~/rest goes back to rest_pose.  Spawn the robot at rest with
  ros2 launch mulinex_ignition gz_omniquad_sim.launch.py start_pose:=rest control_cfg_file_name:=omniquad_rest_mf.yaml
Wheels are not commanded (locked by the wheel controller default).
"""
import numpy as np
import rclpy
import yaml
from pi3hat_moteus_int_msgs.msg import JointsCommand
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger

from .utils.getup import rest_sequence, stand_sequence

POLICY_LEGS = ['LF_HFE', 'LH_HFE', 'RF_HFE', 'RH_HFE', 'LF_KFE', 'LH_KFE', 'RF_KFE', 'RH_KFE']
HOLD, STANDUP, STOOD, REST, RESTED = 'HOLD', 'STANDUP', 'STOOD', 'REST', 'RESTED'


class GetupSim(Node):
    def __init__(self):
        super().__init__('getup_omniquad_sim')
        p = self.declare_parameter
        p('env_cfg_path', '')          # IsaacLab env.yaml: default pose = stand pose
        p('joint_state_topic', '/Joint_Feedback')
        p('joint_target_topic', '/pd_controller/command')
        p('auto_stand', False)
        p('standup_time', 5.0)
        p('activation_hip_offset_deg', 7.5)
        p('activation_time', 2.0)
        p('rest_time', 5.0)
        p('rest_pose', [2.56, -2.56, -2.56, 2.56, -3.0, 3.0, 3.0, -3.0])   # contracted, POLICY_LEGS order
        g = lambda n: self.get_parameter(n).value
        with open(g('env_cfg_path')) as f:
            env = yaml.load(f, Loader=yaml.Loader)
        init = env['scene']['robot']['init_state']['joint_pos']
        self.stand_pose = np.array([init[n] for n in POLICY_LEGS])
        self.rest_pose = np.array(g('rest_pose'), dtype=float)
        self.hfe = np.array(['HFE' in n for n in POLICY_LEGS])
        self.standup_time, self.act_deg, self.act_time, self.rest_time = (
            g('standup_time'), g('activation_hip_offset_deg'), g('activation_time'), g('rest_time'))
        self.q = None
        self.phase, self.t0, self.seq, self.hold = HOLD, None, None, None
        self.pub = self.create_publisher(JointsCommand, g('joint_target_topic'), 10)
        self.create_subscription(JointState, g('joint_state_topic'), self.state_cb, 10)
        self.create_service(Trigger, '~/stand', self.stand_srv)
        self.create_service(Trigger, '~/rest', self.rest_srv)
        self.create_timer(0.01, self.step)   # 100 Hz
        self.auto = g('auto_stand')
        self.get_logger().info(f'stand pose {np.round(self.stand_pose, 2)}, rest pose {np.round(self.rest_pose, 2)}')

    def state_cb(self, msg):
        idx = {n: i for i, n in enumerate(msg.name)}
        try:
            self.q = np.array([msg.position[idx[n]] for n in POLICY_LEGS])
        except KeyError:
            return
        if self.hold is None:
            self.hold = self.q.copy()        # hold where the legs are until a transition starts

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def set_phase(self, phase):
        self.get_logger().info(f'{self.phase} -> {phase}')
        self.phase, self.t0 = phase, self.now()

    def stand_srv(self, req, res):
        if self.q is None or self.phase in (STANDUP, REST):
            res.success, res.message = False, f'not possible in phase {self.phase} / no joint state'
            return res
        self.seq = stand_sequence(self.q, self.stand_pose, self.hfe, self.act_deg, self.act_time, self.standup_time)
        self.set_phase(STANDUP)
        res.success, res.message = True, 'stand started'
        return res

    def rest_srv(self, req, res):
        if self.q is None or self.phase in (STANDUP, REST):
            res.success, res.message = False, f'not possible in phase {self.phase} / no joint state'
            return res
        self.seq = rest_sequence(self.q, self.rest_pose, self.rest_time)
        self.set_phase(REST)
        res.success, res.message = True, 'rest started'
        return res

    def step(self):
        if self.hold is None:
            return
        if self.auto and self.phase == HOLD:
            self.auto = False
            self.stand_srv(None, Trigger.Response())
        if self.phase in (STANDUP, REST):
            target, seg, done = self.seq.at(self.now() - self.t0)
            if done:
                self.hold = self.stand_pose.copy() if self.phase == STANDUP else self.rest_pose.copy()
                self.set_phase(STOOD if self.phase == STANDUP else RESTED)
        else:
            target = self.hold
        msg = JointsCommand()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = POLICY_LEGS
        msg.position = [float(x) for x in target]
        msg.velocity = [0.0] * 8
        msg.effort = [0.0] * 8
        msg.kp_scale = [1.0] * 8
        msg.kd_scale = [1.0] * 8
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = GetupSim()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
