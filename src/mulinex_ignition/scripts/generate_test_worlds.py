#!/usr/bin/env python3
"""
Generate the Gazebo (Ignition Fortress) test worlds for the omniquad policy:

  world/test_steps.sdf          5 lanes, single step up + step down: 4, 6, 8, 10, 12 cm
  world/test_slopes.sdf         4 lanes, ramp up / plateau / ramp down: 5, 10, 15, 20 deg
  world/test_stairs.sdf         5 lanes, 4 steps up / landing / 4 steps down (30 cm treads, as in
                                training): 4, 6, 8, 10, 12 cm risers
  world/test_indoor_course.sdf  realistic route: doorway threshold, slippery tiles,
                                cable protector, curb up, damaged pavement, curb down,
                                loading ramp up to a dock

The robot spawns facing +x. Each lane/course starts at x = 0, so spawn it at
x = -1.5 (see LANES below for the y of every lane).
Reference robot dimensions (trained default pose): wheelbase ~0.58 m, track ~0.36 m,
wheel radius ~0.05 m, knee clearance ~0.18 m.

Usage:  python3 generate_test_worlds.py   (writes into ../world)
"""
import math
import os
import random

WORLD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'world')

# lane y positions, used by the worlds and printed for the spawn arguments
LANE_SPACING = 3.0
LANE_WIDTH = 1.6

CONCRETE = (0.55, 0.55, 0.53)
ASPHALT = (0.25, 0.25, 0.27)
TILE = (0.85, 0.85, 0.80)
YELLOW = (0.9, 0.75, 0.1)
WOOD = (0.6, 0.45, 0.25)
WALL = (0.92, 0.91, 0.88)
RUBBER = (0.1, 0.1, 0.1)
STEEL = (0.45, 0.47, 0.5)


def world_header(name):
    return f'''<sdf version="1.6">
  <world name="{name}">
    <physics name="1ms" type="ignored">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-contact-system" name="gz::sim::systems::Contact"/>
    <plugin name="rgl::RGLServerPluginManager" filename="RGLServerPluginManager">
      <do_ignore_entities_in_lidar_link>true</do_ignore_entities_in_lidar_link>
    </plugin>

    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <attenuation><range>1000</range><constant>0.9</constant><linear>0.01</linear><quadratic>0.001</quadratic></attenuation>
      <direction>-0.5 0.1 -0.9</direction>
    </light>

    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry><plane><normal>0 0 1</normal><size>200 200</size></plane></geometry>
        </collision>
        <visual name="visual">
          <geometry><plane><normal>0 0 1</normal><size>200 200</size></plane></geometry>
          <material><ambient>0.8 0.8 0.8 1</ambient><diffuse>0.8 0.8 0.8 1</diffuse><specular>0.1 0.1 0.1 1</specular></material>
        </visual>
      </link>
    </model>
'''


WORLD_FOOTER = '''  </world>
</sdf>
'''


def box(name, center, size, color, rpy=(0.0, 0.0, 0.0), mu=None, collide=True):
    """One static box link (used inside a model)."""
    x, y, z = center
    sx, sy, sz = size
    r, p, yw = rpy
    surface = ''
    if mu is not None:
        surface = (f'<surface><friction><ode><mu>{mu}</mu><mu2>{mu}</mu2></ode></friction></surface>')
    collision = (f'''
        <collision name="{name}_collision">
          <geometry><box><size>{sx:.4f} {sy:.4f} {sz:.4f}</size></box></geometry>
          {surface}
        </collision>''' if collide else '')
    cr, cg, cb = color
    return f'''
      <link name="{name}">
        <pose>{x:.4f} {y:.4f} {z:.4f} {r:.5f} {p:.5f} {yw:.5f}</pose>{collision}
        <visual name="{name}_visual">
          <geometry><box><size>{sx:.4f} {sy:.4f} {sz:.4f}</size></box></geometry>
          <material><ambient>{cr} {cg} {cb} 1</ambient><diffuse>{cr} {cg} {cb} 1</diffuse><specular>0.1 0.1 0.1 1</specular></material>
        </visual>
      </link>'''


def model(name, links):
    return f'''
    <model name="{name}">
      <static>true</static>{''.join(links)}
    </model>
'''


def platform_top(name, x0, x1, y, top_z, width, color, mu=None):
    """Box whose top face is at top_z, spanning x0..x1 (sits on the ground)."""
    return box(name, ((x0 + x1) / 2, y, top_z / 2), (x1 - x0, width, top_z), color, mu=mu)


def ramp(name, x0, y, z0, angle_deg, length, width, color, thickness=0.1, down=False):
    """Inclined plate whose top surface starts at (x0, z0) and rises (or falls) along +x."""
    a = math.radians(angle_deg)
    dz = length * math.sin(a) * (-1 if down else 1)
    dx = length * math.cos(a)
    # center of the top face, then move half thickness along the face normal
    cx, cz = x0 + dx / 2, z0 + dz / 2
    pitch = a if down else -a          # SDF pitch: positive tilts +x downward
    nx, nz = (math.sin(a), math.cos(a)) if not down else (-math.sin(a), math.cos(a))
    cx -= nx * thickness / 2
    cz -= nz * thickness / 2
    return box(name, (cx, y, cz), (length, width, thickness), color, rpy=(0, pitch, 0)), x0 + dx, z0 + dz


def lane_marker(name, y, x0, x1):
    """Thin painted start line (visual only) to make lanes readable in the GUI."""
    return box(name, (x0 - 0.3, y, 0.0005), (0.05, LANE_WIDTH, 0.001), YELLOW, collide=False)


def gen_steps():
    heights = [0.04, 0.06, 0.08, 0.10, 0.12]
    out = world_header('test_steps')
    lanes = []
    for i, h in enumerate(heights):
        y = i * LANE_SPACING
        links = [
            # step up at x = 0, 2 m platform, step down at x = 2
            platform_top(f'step_{int(h*100)}cm', 0.0, 2.0, y, h, LANE_WIDTH, CONCRETE),
            lane_marker(f'start_{i}', y, 0.0, 0.0),
        ]
        out += model(f'lane_step_{int(h*100)}cm', links)
        lanes.append((f'step {int(h*100)} cm', y))
    out += WORLD_FOOTER
    return out, lanes


def gen_slopes():
    angles = [5, 10, 15, 20]
    out = world_header('test_slopes')
    lanes = []
    for i, ang in enumerate(angles):
        y = i * LANE_SPACING
        L = 2.0
        up, x1, z1 = ramp(f'ramp_up_{ang}', 0.0, y, 0.0, ang, L, LANE_WIDTH, ASPHALT)
        plateau = platform_top(f'plateau_{ang}', x1, x1 + 1.5, y, z1, LANE_WIDTH, CONCRETE)
        down, _, _ = ramp(f'ramp_down_{ang}', x1 + 1.5, y, z1, ang, L, LANE_WIDTH, ASPHALT, down=True)
        out += model(f'lane_slope_{ang}deg', [up, plateau, down, lane_marker(f'start_{i}', y, 0.0, 0.0)])
        lanes.append((f'slope {ang} deg (top {z1:.2f} m)', y))
    out += WORLD_FOOTER
    return out, lanes


STAIRS_HEIGHTS = [0.04, 0.06, 0.08, 0.10, 0.12]
STAIRS_N, STAIRS_TREAD, STAIRS_LANDING = 4, 0.30, 1.2


def gen_stairs():
    """Each lane: risers at x = 0, 0.3, 0.6, 0.9 up to a 1.2 m landing, then 4 steps down (same tread)."""
    n, t, landing = STAIRS_N, STAIRS_TREAD, STAIRS_LANDING
    out = world_header('test_stairs')
    lanes = []
    for i, h in enumerate(STAIRS_HEIGHTS):
        y = i * LANE_SPACING
        top_end = (n - 1) * t + landing
        # stacked boxes: step k spans its riser up to the matching riser down
        links = [platform_top(f'stairs_{int(h*100)}cm_{k}', k * t, top_end + (n - 1 - k) * t, y, (k + 1) * h,
                              LANE_WIDTH, CONCRETE) for k in range(n)]
        links.append(lane_marker(f'start_{i}', y, 0.0, 0.0))
        out += model(f'lane_stairs_{int(h*100)}cm', links)
        lanes.append((f'stairs {int(h*100)} cm (top {n * h:.2f} m)', y))
    out += WORLD_FOOTER
    return out, lanes


def gen_indoor_course(seed=3):
    """
    Realistic route, driving along +x from an indoor warehouse to the street and onto a dock:
      x 0.5      doorway (0.9 m clear) with a 2 cm x 8 cm aluminium threshold
      x 1.5-3.5  polished floor tiles, mu 0.3 (3 mm thick)
      x 4.5      cable protector, 4 cm high, 30 cm wide (trapezoid)
      x 5.5      curb up, 6 cm, onto the sidewalk
      x 6.5-8.5  damaged pavement on the sidewalk: 10 cm tiles, 0-2 cm random height
      x 9.5      curb down, 6 cm, onto the asphalt
      x 10.5     loading ramp 8 deg, 2 m, up to a dock (0.28 m), 2 m long
    """
    rnd = random.Random(seed)
    W = 1.4                                   # corridor / path width
    out = world_header('test_indoor_course')
    links = []

    # walls of the warehouse (x < 0.5) with a doorway
    links.append(box('wall_left', (-1.75, W / 2 + 0.6, 1.0), (4.5, 0.1, 2.0), WALL))
    links.append(box('wall_right', (-1.75, -W / 2 - 0.6, 1.0), (4.5, 0.1, 2.0), WALL))
    links.append(box('wall_back', (-4.0, 0.0, 1.0), (0.1, W + 1.3, 2.0), WALL))
    door_clear = 0.9
    post_w = (W + 1.2 - door_clear) / 2
    links.append(box('door_wall_l', (0.5, door_clear / 2 + post_w / 2, 1.0), (0.12, post_w, 2.0), WALL))
    links.append(box('door_wall_r', (0.5, -door_clear / 2 - post_w / 2, 1.0), (0.12, post_w, 2.0), WALL))
    links.append(box('door_lintel', (0.5, 0.0, 2.05), (0.12, door_clear, 0.1), WALL, collide=False))
    links.append(box('threshold', (0.5, 0.0, 0.01), (0.08, door_clear, 0.02), STEEL))

    # polished tiles (slippery)
    links.append(box('polished_tiles', (2.5, 0.0, 0.0015), (2.0, W, 0.003), TILE, mu=0.3))

    # cable protector: flat top + two 25 deg ramps, 4 cm high
    h, top = 0.04, 0.10
    a = math.radians(25)
    rl = h / math.sin(a)
    links.append(box('cable_top', (4.5, 0.0, h / 2), (top, W, h), YELLOW))
    for side, sx in (('in', -1), ('out', 1)):
        cx = 4.5 + sx * (top / 2 + rl * math.cos(a) / 2)
        links.append(box(f'cable_ramp_{side}', (cx, 0.0, h / 2 - 0.004), (rl, W, 0.012), YELLOW,
                         rpy=(0, sx * a, 0)))

    # sidewalk with curbs (6 cm) and damaged pavement patch on it
    curb = 0.06
    links.append(platform_top('sidewalk', 5.5, 9.5, 0.0, curb, W + 0.6, CONCRETE))
    for ix in range(20):
        for iy in range(int(W / 0.1)):
            hz = rnd.uniform(0.0, 0.02)
            if hz < 0.003:
                continue
            x = 6.5 + 0.05 + ix * 0.1
            y = -W / 2 + 0.05 + iy * 0.1
            g = 0.45 + rnd.uniform(-0.08, 0.08)
            links.append(box(f'paver_{ix}_{iy}', (x, y, curb + hz / 2), (0.098, 0.098, hz), (g, g, g * 0.97)))

    # asphalt road and loading ramp up to a dock
    links.append(box('road_paint', (10.0, 0.0, 0.0005), (1.0, W + 0.6, 0.001), ASPHALT, collide=False))
    up, x1, z1 = ramp('dock_ramp', 10.5, 0.0, 0.0, 8, 2.0, W, STEEL)
    links.append(up)
    links.append(platform_top('dock', x1, x1 + 2.0, 0.0, z1, W + 0.6, CONCRETE))

    # some realistic clutter beside the path (pallets, boxes) - not on the path
    for i, (px, py) in enumerate([(-2.5, 0.88), (-1.0, -0.88), (2.5, 1.2), (7.5, -1.4)]):
        links.append(box(f'pallet_{i}', (px, py, 0.07), (1.2, 0.8, 0.14), WOOD))
        links.append(box(f'crate_{i}', (px, py, 0.14 + 0.25), (0.8, 0.6, 0.5), (0.7, 0.55, 0.35)))

    out += model('indoor_course', links)
    out += WORLD_FOOTER
    lanes = [('indoor course (start)', 0.0)]
    return out, lanes, x1 + 2.0


if __name__ == '__main__':
    os.makedirs(WORLD_DIR, exist_ok=True)
    s, steps_lanes = gen_steps()
    open(os.path.join(WORLD_DIR, 'test_steps.sdf'), 'w').write(s)
    s, slope_lanes = gen_slopes()
    open(os.path.join(WORLD_DIR, 'test_slopes.sdf'), 'w').write(s)
    s, course_lanes, end_x = gen_indoor_course()
    open(os.path.join(WORLD_DIR, 'test_indoor_course.sdf'), 'w').write(s)
    s, stairs_lanes = gen_stairs()
    open(os.path.join(WORLD_DIR, 'test_stairs.sdf'), 'w').write(s)

    print('Spawn the robot at x = -1.5 and the lane y below, e.g.')
    print('  ros2 launch mulinex_ignition gz_omniquad_sim.launch.py world_name:=test_steps.sdf spawn_y:=3.0')
    for world, lanes in (('test_steps.sdf', steps_lanes), ('test_slopes.sdf', slope_lanes),
                         ('test_indoor_course.sdf', course_lanes), ('test_stairs.sdf', stairs_lanes)):
        print(world)
        for name, y in lanes:
            print(f'  y = {y:4.1f}  {name}')
    print(f'indoor course ends at x = {end_x:.1f}')
