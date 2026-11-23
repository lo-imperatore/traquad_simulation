#!/usr/bin/env python3
"""
Run the test scenarios of the worlds made by generate_test_worlds.py and print a report.
Also reports, for every obstacle, the traversal time: from the front wheels reaching it to the
rear wheels passing it (at 0.3 m/s on flat ground that is 2 * 0.36 / 0.3 = 2.4 s).

Needs the simulation and the policy already running, e.g.
  ros2 launch mulinex_ignition gz_omniquad_sim.launch.py world_name:=test_steps.sdf spawn_x:=-1.5
  ros2 launch rlg_quad_controller omniquad_inference_sim.launch.py
then
  python3 eval_scenarios.py test_steps            (or test_slopes / test_stairs / test_indoor_course)

For every lane the robot is teleported to x = -1.5 on the lane and driven along +x at
--speed. Like a human operator, the script keeps the heading and the lane centre with a
small yaw / lateral correction (the policy has no heading control). A lane passes if the
robot reaches goal_x without flipping before the timeout (simulation time).
"""
import argparse
import json
import math
import subprocess
import time

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node

# world -> list of (lane name, lane y, goal x, timeout [s sim], [(x, obstacle name), ...][, start x, start z])
COURSE = [(0.5, 'door threshold 2 cm'), (1.5, 'polished tiles mu 0.3'), (4.4, 'cable protector 4 cm'),
          (5.5, 'curb up 6 cm'), (6.5, 'damaged pavement'), (9.5, 'curb down 6 cm'),
          (10.5, 'loading ramp 8 deg'), (12.5, 'dock')]
SCENARIOS = {
    'test_steps': [(f'step {h} cm', i * 3.0, 3.0, 40.0, [(0.0, f'step up {h} cm'), (2.0, f'step down {h} cm')])
                   for i, h in enumerate([4, 6, 8, 10, 12])],
    'test_slopes': [(f'slope {a} deg', i * 3.0, 2 * 2.0 * math.cos(math.radians(a)) + 1.5 + 0.6, 60.0,
                     [(0.0, f'ramp up {a} deg'), (2.0 * math.cos(math.radians(a)), 'plateau'),
                      (2.0 * math.cos(math.radians(a)) + 1.5, f'ramp down {a} deg')])
                    for i, a in enumerate([5, 10, 15, 20])],
    # risers up at x = 0 .. 0.9, landing 0.9 - 2.1, risers down at x = 2.1 .. 3.0 (generate_test_worlds.gen_stairs)
    'test_stairs': [(f'stairs {h} cm', i * 3.0, 3.8, 60.0,
                     [(0.0, 'first step up'), (0.9, 'last step up'), (2.1, 'first step down'), (3.0, 'last step down')])
                    for i, h in enumerate([4, 6, 8, 10, 12])],
    'test_indoor_course': [('indoor course', 0.0, 14.0, 150.0, COURSE),
                           # second half only, starting on the sidewalk after the curb up
                           ('indoor course from sidewalk', 0.0, 14.0, 100.0, COURSE, 5.95, 0.5)],
}


WHEEL_REACH = 0.36   # wheel edge distance ahead of / behind the base [m]


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def roll_pitch_of(q):
    roll = math.atan2(2 * (q.w * q.x + q.y * q.z), 1 - 2 * (q.x * q.x + q.y * q.y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (q.w * q.y - q.z * q.x))))
    return roll, pitch


def clip(v, lim):
    return max(-lim, min(lim, v))


class Evaluator(Node):
    def __init__(self, world, odom_topic, model):
        super().__init__('scenario_evaluator')
        self.world, self.model = world, model
        self.odom = None
        self.create_subscription(Odometry, odom_topic, self._odom_cb, 50)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

    def _odom_cb(self, msg):
        self.odom = msg

    def sim_time(self):
        s = self.odom.header.stamp
        return s.sec + s.nanosec * 1e-9

    def wait_odom(self, timeout=30.0):
        t0 = time.time()
        while self.odom is None and time.time() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
        return self.odom is not None

    def send(self, vx=0.0, vy=0.0, wz=0.0):
        m = Twist()
        m.linear.x, m.linear.y, m.angular.z = float(vx), float(vy), float(wz)
        self.cmd_pub.publish(m)

    def teleport(self, x, y, z):
        req = f'name: "{self.model}", position: {{x: {x}, y: {y}, z: {z}}}, orientation: {{w: 1}}'
        for cmd in ('ign', 'gz'):
            try:
                r = subprocess.run([cmd, 'service', '-s', f'/world/{self.world}/set_pose',
                                    '--reqtype', 'ignition.msgs.Pose', '--reptype', 'ignition.msgs.Boolean',
                                    '--timeout', '3000', '--req', req], capture_output=True, text=True, timeout=10)
                if 'true' in r.stdout:
                    return True
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue
        return False

    def spin_for(self, sim_seconds, **cmd):
        t_end = self.sim_time() + sim_seconds
        while self.sim_time() < t_end:
            self.send(**cmd)
            rclpy.spin_once(self, timeout_sec=0.05)

    def run_lane(self, name, y_lane, goal_x, timeout, obstacles, start_x=-1.5, start_z=0.45, speed=0.3):
        if not self.teleport(start_x, y_lane, start_z):
            return {'lane': name, 'result': 'ERROR: teleport failed'}
        self.spin_for(2.0)                                   # let it land and settle (zero command)
        t0 = self.sim_time()
        max_x, last_progress_t = -1e9, t0
        max_roll = max_pitch = 0.0
        max_dev = 0.0
        result = 'TIMEOUT'
        # traversal of an obstacle at ox: from front wheels reaching it to rear wheels passing it
        t_in, t_out = [None] * len(obstacles), [None] * len(obstacles)
        while True:
            rclpy.spin_once(self, timeout_sec=0.05)
            p, q = self.odom.pose.pose.position, self.odom.pose.pose.orientation
            t = self.sim_time()
            roll, pitch = roll_pitch_of(q)
            max_roll, max_pitch = max(max_roll, abs(roll)), max(max_pitch, abs(pitch))
            max_dev = max(max_dev, abs(p.y - y_lane))
            for i, (ox, _) in enumerate(obstacles):
                if t_in[i] is None and p.x + WHEEL_REACH >= ox:
                    t_in[i] = t
                if t_out[i] is None and p.x - WHEEL_REACH >= ox:
                    t_out[i] = t
            if p.x > max_x + 0.02:
                max_x, last_progress_t = p.x, t
            if abs(roll) > math.radians(60) or abs(pitch) > math.radians(60):
                result = 'FLIPPED'
                break
            if p.x >= goal_x:
                result = 'PASS'
                break
            if t - last_progress_t > 12.0:
                result = 'STUCK'
                break
            if t - t0 > timeout:
                break
            # operator: keep heading 0 and the lane centre
            self.send(vx=speed, vy=clip(-1.0 * (p.y - y_lane), 0.2), wz=clip(-1.5 * yaw_of(q), 0.5))
        self.spin_for(1.0)                                   # stop
        where = 'start'
        for ox, oname in obstacles:
            if max_x + WHEEL_REACH >= ox:
                where = oname
        r = {'lane': name, 'result': result, 'max_x': round(max_x, 2), 'goal_x': round(goal_x, 2),
             'time_s': round(self.sim_time() - t0, 1), 'max_roll_deg': round(math.degrees(max_roll), 1),
             'max_pitch_deg': round(math.degrees(max_pitch), 1), 'max_lane_dev_m': round(max_dev, 2),
             # null = reached but not crossed; obstacles never reached are left out
             'traverse_s': {oname: (round(t_out[i] - t_in[i], 2) if t_out[i] is not None else None)
                            for i, (_, oname) in enumerate(obstacles) if t_in[i] is not None}}
        if result != 'PASS':
            r['failed_at'] = where
            if r['traverse_s'].get(where) is not None:    # edge passed, but the robot got stuck there anyway
                r['traverse_s'][where] = None
        return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('world', choices=sorted(SCENARIOS))
    ap.add_argument('--speed', type=float, default=0.3, help='forward command [m/s]')
    ap.add_argument('--lanes', default='', help='comma separated lane indices (default: all)')
    ap.add_argument('--model', default='mulinex')
    ap.add_argument('--odom', default='/mulinex/gt_odom')
    ap.add_argument('--json', default='', help='also write the results to this file')
    a = ap.parse_args()

    rclpy.init()
    ev = Evaluator(a.world, a.odom, a.model)
    if not ev.wait_odom():
        raise SystemExit(f'no odometry on {a.odom}: is the simulation running?')
    lanes = SCENARIOS[a.world]
    if a.lanes:
        lanes = [lanes[int(i)] for i in a.lanes.split(',')]
    results = []
    for lane in lanes:
        r = ev.run_lane(*lane, speed=a.speed)
        print(json.dumps(r), flush=True)
        results.append(r)
    ev.send()
    if a.json:
        json.dump(results, open(a.json, 'w'), indent=1)
    print(f'\n{a.world}: {sum(r["result"] == "PASS" for r in results)}/{len(results)} passed')
    rclpy.shutdown()


if __name__ == '__main__':
    main()
