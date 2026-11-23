# TraQuad simulation

Simulation of the TraQuad (Mulinex on tracks) in Isaac Sim 6.1 / Isaac Lab 3.0, plus the older Gazebo setup below.

- `isaac_sim/`: the robot asset (`assets/traquad`), the replica of the real controllers
  (`traquad_real_controller.py`), the replay of the real runs against the Qualisys ground truth (`real_replay.py`,
  `real_id/`) and the test scripts. Start from `isaac_sim/README.md`: current model, identified parameters, how to run,
  changelog.
- `isaaclab_overlay/`: the TraQuad tasks and robot config for Isaac Lab, linked into `isaaclab_traquad/` by
  `isaaclab_overlay/setup_isaaclab.sh`.
- `open_traquad.py`: the robot in Isaac Lab with joystick-like velocity commands.
- `real_robot/traquad_missions.py`: identification missions for the real robot (run on the NUC).
- `src/mulinex_description`: xacro of the robot (`traquad.xacro`, tracks in `track.xacro`).
- Real runs with the Qualisys: github.com/giacomotambe/traquad_bags; robot code: github.com/giacomotambe/mulsbc_ws
  (SBC) and github.com/giacomotambe/mulinex_app (NUC).

# Dockerized Gazebo Harmonic Simulator 
## introduction
This repository contains a docker compose structure to simulate robots in Gazebo Sim Fortress supported by ROS2 humble on ubuntu 22.04.
The Repository purpose is to provide an easy set up to allow robotic software developer to test their own controller in simulation.
## prerequisite  

Docker Engine: follow the docker engine installation [here](https://docs.docker.com/engine/install/ubuntu/) and then enable the docker usage as non-root user as specified [here](https://docs.docker.com/engine/install/linux-postinstall/)

Install Nvidia Driver: TODO

Nvidia-Container-Toolkit: follow the nvidia-container-toolkit following [this](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

## Repository content
Only sources are in git: `build/`, `install/`, `log/`, rosbags, experiment recordings, videos, logs (`*.npz`) and the policies (`src/rlg_quad_controller/models/`) are ignored (`.gitignore`). After a clone: build the workspace and copy the policies into `src/rlg_quad_controller/models/`.

For the real-robot bridge only (laptop, no simulator): `colcon build --symlink-install --packages-up-to rlg_quad_controller` (builds `pi3hat_moteus_int_msgs` and `rlg_quad_controller`; an RSL-RL policy needs `torch`, `numpy`, `pyyaml`, not `rl-games`/`gym`).

## Build 

``` bash compose_build.bash ```

## Start Compose 

``` docker compose up -d ```

``` docker compose exec ros2_humble_sim bash```

This command launches a docker with ros inside it. The first time you open it, you have to run 

``` colcon build --symlink-install```


## NVIDIA stuff test 

To test the correct isntallation of the nvidia support you have to start the container and call

``` nvidia-smi```

## HOW TO BUILD: 
If, after deleting the folders "launch", "install" and "log", when trying to build, you will receive an error: ...

Do the following:
- ```rm -rf build install log```
- ```colcon build --packages-select pi3hat_moteus_int_msgs --symlink-install```
- ```source install/setup.bash```
- ```colcon build --symlink-install```
- ```source install/setup.bash```

Ignore the warnings and just do whatever you need to.


## LUNCH ROBOT SIMULATION

#### TERMINAL 1:
This terminal is used to run Gazebo

``` source install/setup.bash ```

``` ros2 launch mulinex_ignition gz_harmonic_sim_w_rbt.launch.py ```
#### TERMINAL 2:
This terminal is used to run the node to move the robot

``` source install/setup.bash ```

``` ros2 run mulinex_ignition-py getup ```



## Move the robot
There are three controllers: 
- "omni_control" that, given a reference velocity, solves the inverse kinematics and moves the wheels to target the reference velocity.
    - topic from which it reads: /omni_control/command
    - topic message type: pi3hat_moteus_int_msgs/msg/OmniMulinexCommand 
    - To test this controller, ```ros2 topic pub /omni_control/command pi3hat_moteus_int_msgs/msg/OmniMulinexCommand "{v_x: 0.0, v_y: 10.0, omega: 10.0}"```
    Or, alternatively, run ``` ros2 run mulinex_ignition_py move_wheels ```
- "pd_control" that moves the leg joints using a PD controller.
    - topic from which it reads: /pd_controller/command
    - topic message type:JointState (from Sensor_msgs.msg)
    - To test this controller, run ``` ros2 run mulinex_ignition_py getup ```
- "wheels_vel_cnt", that moves the wheels individually, with a velocity reference.
    - topic from which it reads: /wheels_vel_controller/wheels_velocity_cmd
    - topic message type: WheelVelocityCommand (path: docker_simulator/src/custom_interfaces/msg/WheelVelocityCommand.msg)
    - To test this controller, run ```ros2 run mulinex_ignition_py move_wheels_2```

## RUN THE INFERENCE

### SETUP
- Go to IsaacSim > logs > desired network that you trained
- Copy the files:
    - nn/omniquad_flat.pth
    - params/agent.yaml
    - params/env.yaml
- Paste the 3 files in the Gazebo Folder: src/rlg_quad_controller/models
- The policies are not in git (`src/rlg_quad_controller/models/*` is ignored, see `models/README.md`): after a clone, copy the runs you need there by hand.
- The inference node needs `torch`, `rl-games==1.6.1` and `gym` inside the container. They are installed by the Dockerfile (rebuild with ``` bash compose_build.bash ```); in an already running container you can install them with ``` pip install --user "rl-games==1.6.1" "gym==0.23.1" "numpy<2" ``` (lost if the container is removed).


### RUN
On every terminal you start: 

``` docker compose exec ros2_humble_sim bash```

``` source install/setup.bash ```


#### TERMINAL 1:
This terminal is used to launch Gazebo, the controllers, etc. It spawns the omniquad (`mul_ignition.xacro`) with the controller config `omniquad_mf.yaml` (8 leg joints in the PD controller, 4 omni wheels in the wheel velocity controller).
``` ros2 launch mulinex_ignition gz_omniquad_sim.launch.py ```

(`gz_harmonic_sim_w_rbt_PD_wheels.launch.py` alone still spawns the traquad with `mulinex_mf.yaml`.)

#### TERMINAL 2:
The keyboard to control the robot
``` ros2 run teleop_twist_keyboard teleop_twist_keyboard ```

#### TERMINAL 3:
The inference node (simulation version: uses `/Joint_Feedback`, `/Wheel_Feedback`, `/mulinex/imu` and the sim clock)
``` ros2 launch rlg_quad_controller omniquad_inference_sim.launch.py ```

The leg default pose is read from `env.yaml` (`scene.robot.init_state.joint_pos`); keep `init_pos` in `src/mulinex_ignition/config/omniquad_mf.yaml` and `DEFAULT_POSE` in `gz_omniquad_sim.launch.py` (the legs are spawned in it) consistent with it if you train with a different default pose.

The leg actuator is calibrated against IsaacLab: PD at 1000 Hz (the physics rate), Kp 100.7, Kd 4, joint velocity limit 10 rad/s (`leg_velocity_limit:=`), 5 N m. IsaacLab trains with Kp 100 / Kd 1 solved implicitly; an explicit PD with Kd 1 makes the policy shake 2-3x more than in IsaacLab, Kd 4 matches its joint speeds and action changes (Kd 6 suppresses the jitter of the 2026-10-18 policy entirely).

### TRAQUAD POLICY (IsaacLab RSL-RL, e.g. traquad_omnistyle)
Copy the IsaacLab run into `src/rlg_quad_controller/models/<name>/` (`params/env.yaml`, `params/agent.yaml` and the `model_<iter>.pt` to use; the highest iteration is installed), then `colcon build --symlink-install`.

Terminal 1, Gazebo set up like the training (3 rollers of 14 mm with 0.175 N m dry friction, wheel contact mu 0.75 = training mean, wheel drives capped at 0.694 N m, leg PD 60 / 0.6, legs spawned in the default pose HFE ±1.47 from `config/traquad_policy.yaml`; add `headless:=true` for no GUI):

``` ros2 launch mulinex_ignition gz_traquad_policy.launch.py ```

Terminal 2, the policy (`model:=<name>`, `checkpoint:=model_<iter>.pt`, `log_obs:=true` to print obs/actions):

``` ros2 launch rlg_quad_controller traquad_inference_sim.launch.py ```

Terminal 3: ``` ros2 run teleop_twist_keyboard teleop_twist_keyboard ``` (training ranges: vx ±1 m/s, wz ±1 rad/s, no lateral velocity).

Notes:
- The joint order of IsaacLab is the PhysX articulation order: HFE `LH, LF, RF, RH`, wheel velocities `LEFT_H, LEFT_F, RIGHT_F, RIGHT_H` (hardcoded in `inference_traquad_sim.py`, check it with `env.scene["robot"].joint_names` if the USD changes). `rbt_pd_cnt` takes the command by index, in the order of its `joint` parameter (`pd_joint_order` of the node).
- The roller dry friction sets how easily the tracks slide sideways: without it (`roller_friction:=0.0`) the robot over-turns about 3x, since the policy has no gyro in its observations.
- Not reproduced yet: the wheel friction compensation and the in-air ankle controller of `TrackVelocityAction` (the Gazebo wheel drive is the velocity servo of gz_ros2_control, capped at `wheel_effort`).

### TEST SCENARIOS
Worlds generated by `src/mulinex_ignition/scripts/generate_test_worlds.py` (edit the dimensions there and re-run it). The robot faces +x, every lane starts at x = 0: spawn it at `spawn_x:=-1.5` and the lane `spawn_y`.

| World | Lanes (spawn_y) |
|---|---|
| `test_steps.sdf` | single step up + down: 4 cm (0), 6 cm (3), 8 cm (6), 10 cm (9), 12 cm (12) |
| `test_slopes.sdf` | ramp up / plateau / ramp down: 5° (0), 10° (3), 15° (6), 20° (9) |
| `test_indoor_course.sdf` | realistic route (0): doorway with 2 cm threshold, polished tiles (mu 0.3), 4 cm cable protector, 6 cm curb up, damaged pavement, 6 cm curb down, 8° loading ramp to a dock |

``` ros2 launch mulinex_ignition gz_omniquad_sim.launch.py world_name:=test_steps.sdf spawn_x:=-1.5 spawn_y:=6.0 ```

Automatic evaluation (with Gazebo and the inference node running): it teleports the robot to every lane, drives it forward at 0.3 m/s keeping the lane, and reports PASS / STUCK / FLIPPED and where it failed:

``` python3 src/mulinex_ignition/scripts/eval_scenarios.py test_steps ```

## USEFUL COMMANDS
- ``` ros2 run rqt_graph rqt_graph ``` shows a graph with all the nodes and all the topics and who writes where
- LANCIARE UN CONTROLLER:  ```ros2 run controller_manager spawner pd_controller --param-file src/mulinex_ignition/config/mulinex_mf.yaml```
- Verificare i topic emessi da Gazebo (per poi poter fare il bridge): 
    - per vedere la lista: ```ign topic -l```
    - per fare l'echo: ```ign topic -e -t /mulinex/imu```
- Avvio manuale del bridge Gazebo --> ROS: ```ros2 run ros_gz_bridge parameter_bridge /mulinex/imu@sensor_msgs/msg/Imu[ignition.msgs.IMU```
- convertire il file rosbag .db3 in .mcap: 
    - salvarsi il file convert.yaml, entrarci e modificare il nome dell'output con il nome desiderato
    - lanciare il comando ```ros2 bag convert -i <nome_cartella_rosbag> -o convert.yaml```

## REGISTRARE LA ROSBAG:
Lanciare il comando ``` ros2 launch mulinex_ignition_py rosbag_rec.launch.py ```

## CONNETTERSI AL ROBOT FISICO:
Su un nuovo terminale, ti connetti al robot: 
- apri il docker
- ```ssh mulsbc@100.100.100.3```  --> password: 123456
- ```sudo su``` --> password 123456
- ```cd mulinex_ws/```
- ```source install/setup.bash```
- ```export ROS_DOMAIN_ID=1```
- ```ros2 launch pi3hat_hw_interface start_MJBots_Pi3Hat_hw_Interface.py urdf_file:=omni_mulinex.xacro conf_file:=omniquad_joy.yaml```
- Quando poi vorrai spegnere il robot, digita ```shutdown now```, poi puoi staccare la corrente al robot

Su un nuovo terminale, lancerai i nodi dal tuo PC per controllare il robot:
- apri il docker
- ```export ROS_DOMAIN_ID=1```
- ```ros2 launch mulinex_ignition_py controller_start_launch.py```
- ora lancia tutti i nodi che vuoi:
    - Getup: ```ros2 run mulinex_ignition_py getup_omni_1```
    - run della rosbag: ```ros2 bag play rosbag/run_20250827_140849 --topics /joint_controller/command```
    - INFERENCE:
        - su un terminale, devi lanciare il nodo che fa da bridge tra i topic di ROS e quelli del robot: ```ros2 run mulinex_ignition_py bridge_node```
        - su un altro terminale, lanci il nodo di inference: ```ros2 launch rlg_quad_controller omniquad_inference.launch.py```
        - su un terzo terminale, lancia la teleop_twist_keyboard




## TROUBLESHOOT
- If ``` colcon build ``` does not work the first time you launch it during the installation, try to do: 
    - ``` docker rm -f traquad_ros2_humble``` 
    - ``` docker compose up -d``` 
    - and then, again: ``` docker compose exec ros2_humble_sim bash```
- PROBLEMI INTERCORSI NEL PASSAGGIO GAZEBO --> ROS2:
    - scrittura di tutto il codice per controllers e inference
    - nel GitHub dei PhD di Unipi era presente una inference un po' diversa rispetto a quella di Gazebo. Quindi, a parità di obs dati in input alla rete, si ottenevano actions diverse. Abbiamo sostituito l'inference copiando e incollando la stessa funzione.
    - su IsaacSim i "joints_pos" all'interno del vettore obs contiene angoli di deviazione rispetto alla posizione di default (Knee: 120°, Hip: 60°; entrambi espressi in radianti) e quindi sono valori molto vicini a 0. Invece, su ROS2 le letture sono degli angoli rispetto alla posizione orizzontale, quindi occorre sottrarre il valore di default delle gambe prima di passare il vettore obs alla rete.
    - I valori delle angular velocity e delle projected gravity erano uguali a zero. Questo era dovuto al fatto che il bridge Gazebo->ROS non era settato bene. Gazebo pubblicava su un topic "mulinex/imu" mentre ROS cercava di creare il bridge per il topic "omnicar/imu". Si è quindi sostituito questo nome del topic in:
        - mulinex.xacro
        - file di launch "gz_harmonic_sim_W_rbt_PD_wheels.launch.py"
        - file di inference "inference_controller.py"



## REAL ROBOT: OMNIQUAD POLICY FROM THE LAPTOP
`inference_omniquad_robot` runs the policy on the laptop and drives the robot over the network, through the robot controllers (`state_broadcaster`, `joint_controller`, `IMU_Broadcaster`, as `getup_omni_1.py`):
in `/state_broadcaster/joints_state` (`JointsStates`), `/IMU_Broadcaster/imu`, `/cmd_vel`; out `/joint_controller/command` (`JointsCommand`, 12 joints: legs position, wheels velocity with position NaN).
With `output_mode: omni_controller` (default of the config, robot running `omni_controller` of CentroEPiaggio/omni_controller, domain 10, CycloneDDS): legs out on `/omni_controller/joints_reference`, wheels on `/omni_controller/direct_wheels_cmd`, a `/nuc_heartbeat` is published (the robot damps the legs if it stops for > 1 s; not published in dry run), and at STANDUP the node calls `set_wheel_mode(true)` and `activate_srv`. The IMU comes from `/imu_sensor/imu`: on the robot run `ros2 control load_controller --set-state active imu_sensor` (the launch does not spawn it). Joint states stay on `/state_broadcaster/joints_state`. From the laptop: container with host network (`compose.yaml`), then `source robot_env.sh` in the shell (`ROS_DOMAIN_ID=10`, `rmw_cyclonedds_cpp`; the compose default stays domain 1 / Fast DDS for the simulation, never run Gazebo in a shell with the robot environment). `acc_add_gravity: true` adds the gravity back to the Pi3Hat acceleration (it is reported without gravity).
Topics, joint order, wheel signs, gains and safety limits are in `src/rlg_quad_controller/config/omniquad_robot_config.yaml`.

Phases: `WAIT` (no command until joint states and IMU arrive) → `STANDUP` (measured pose → default pose in 5 s, wheels stopped) → `HOLD` (default pose, policy computed but not applied) → `POLICY` with `ros2 service call /inference_omniquad_robot/start std_srvs/srv/Trigger`; `.../stop` goes back to `HOLD`. Stale joint states/IMU (> 0.1 s), tilt > 45° or a non-finite action switch `POLICY` → `HOLD`. At Ctrl-C the last command is default pose + wheels stopped. A status (loop time, data age, gravity, joint errors, wheel speeds, action) is printed every second.

Rest / stand / activation (replica of the transitions of `omni_controller`, `utils/getup.py`): `STANDUP` = HFE activation offset (`activation_hip_offset_deg` 7.5, `activation_time` 2 s, HFE moved opposite to the stand direction) then cosine interpolation to the policy default pose (`standup_time`); services `~/rest` (HOLD -> REST -> RESTED, cosine to `rest_pose`, default zeros, `rest_time`; `release_at_rest:=true` releases the torque) and `~/stand` (RESTED -> STANDUP -> HOLD). `pose_check_tol` (0.6 rad): before the first STANDUP the measured legs must be near `rest_pose` or the default pose, else the node stays in WAIT (joint zero / sign convention check: the stand pose of the omni_controller yaml, HFE -0.56 / KFE 1.8 for LF, has the opposite signs of the policy default pose, HFE 2.0 / KFE -1.2: verify the zero of the real robot before the first stand). Same sequence in Gazebo: `ros2 launch mulinex_ignition gz_omniquad_sim.launch.py start_pose:=rest control_cfg_file_name:=omniquad_rest_mf.yaml`, then `ros2 launch rlg_quad_controller getup_omniquad_sim.launch.py` and `ros2 service call /getup_omniquad_sim/stand std_srvs/srv/Trigger` (`.../rest` to go back). Note: from the all-zero rest pose the simulated legs (5 N m) lift the body only to HFE 0.35 rad: use a folded rest pose in simulation (`rest_pose`).

Network: laptop and robot on the same network, same `ROS_DOMAIN_ID` (compose.yaml sets 1: export the robot one) and `RMW_IMPLEMENTATION`; check with `ros2 topic hz /state_broadcaster/joints_state` from the laptop before starting.

``` ros2 launch rlg_quad_controller omniquad_robot_bridge.launch.py ```  dry run (default): nothing is published

Arguments: `model:=` (default `2026-11-09_21-26-58_finetune_curbs`), `checkpoint:=`, `dry_run:=false`, `auto_start:=true`, `leg_action_gain:=` / `wheel_action_gain:=` (0..1, scale the policy output), `log_path:=/path/run.npz` (obs, actions, targets, measured state).

Suggested first test:
1. Robot on the stand (wheels off the ground), `dry_run` with the robot controllers running: gravity ≈ (0, 0, -1) when level, nose down → gravity x > 0, left side down → gravity y > 0, turning left → ang_vel z > 0; leg error ≈ 0 in the default pose; turn a wheel forward by hand → its wheel vel > 0 (else fix `wheel_signs`).
2. Stand, `dry_run:=false wheel_action_gain:=0.0`: standup, `HOLD`, then `start` with zero command: legs must stay close to the default pose without oscillating.
3. Stand, `wheel_action_gain:=1.0`, `/cmd_vel` vx 0.3: all four wheels must turn forward.
4. Ground, safety operator on the e-stop, zero command then vx ±0.3, wz ±0.5.
