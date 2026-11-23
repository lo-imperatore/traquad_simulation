# Policies (not in git)

Copy here the IsaacLab runs to use, one folder per run, then `colcon build --symlink-install`:

- RSL-RL run: `<run>/params/env.yaml`, `<run>/params/agent.yaml` and the checkpoint `<run>/model_<iter>.pt`
  (only the highest iteration is installed; tensorboard logs and other checkpoints are not needed).
- rl-games: `<name>/env.yaml`, `<name>/agent.yaml`, `<name>/*.pth`.

The launch files take the folder name with `model:=<run>` (and optionally `checkpoint:=model_<iter>.pt`).
Real-robot bridge default: `2026-11-09_21-26-58_finetune_curbs` (model_7049).
