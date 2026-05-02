# unitree_rl_lab uv runtime instructions

This document is for `uv` environment. Content includes:
- [Common Steps](#common-steps)
- [USD Resources](#usd-resources)
- [Possible Errors](#possible-errors)
- [Mimic Mocap Data Processing + Training](#mimic-mocap-data-processing--training)

## Installation

```bash
# Clone the repository
git clone https://github.com/wty-yy-mini/unitree_rl_lab.git
cd unitree_rl_lab

# Create uv venv and install IsaacLab and dependencies
uv venv
uv pip install --no-cache-dir "torch==2.7.0" "torchvision==0.22.0" --index-url https://download.pytorch.org/whl/cu128; \
uv pip install "isaaclab[isaacsim,all]==2.3.2.post1" "omniverse-kit"

# Activate uv venv
source .venv/bin/activate

# Install the repository
pip install -e ./source/unitree_rl_lab/
```

## USD Resources

Before running tasks, make sure to have the Unitree robot USD resource. (We have modified the code to support USD under root directory [unitree.py](source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py))

```bash
# Download USD resources
git clone https://huggingface.co/datasets/unitreerobotics/unitree_model

# Check file size is 27Mb, if not then need to use lfs to download
ls -alh unitree_model/G1/29dof/usd/g1_29dof_rev_1_0/configuration/g1_29dof_rev_1_0_base.usd

# If no git lfs, install it first
cd unitree_model
sudo apt install git-lfs
git lfs install
git lfs pull
```

## Common Steps

```bash
# List available tasks
python scripts/list_envs.py

# Start training (Velocity task example)
python scripts/rsl_rl/train.py --headless --task Unitree-G1-29dof-Velocity
# For Mimic task see below section

# Start inference
python scripts/rsl_rl/play.py --task Unitree-G1-29dof-Velocity
```

## Possible Errors

- If you encounter an error about missing `libnvidia-ml.so.1`, add the following env variable in bashrc:
  ```bash
  export CUDA_LIB_PATH=$(python -c 'import sysconfig, pathlib; print(pathlib.Path(sysconfig.get_paths()["purelib"]) / "nvidia" / "cu13" / "lib")')
  export LD_LIBRARY_PATH=${CUDA_LIB_PATH}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
  ```

## Mimic Mocap Data Processing + Training

For example, to convert `G1_Take_102.bvh_60hz.csv` to the required `npz` format:

```bash
python scripts/mimic/csv_to_npz.py \
  -i source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/robots/g1_29dof/dance_102/G1_Take_102.bvh_60hz.csv \
  --input_fps 60 \
  --headless
```

To visualize the converted `npz` file:

```bash
python scripts/mimic/replay_npz.py \
  -f source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/robots/g1_29dof/dance_102/G1_Take_102.bvh_60hz.npz
```

Start training with the Mimic task:

```bash
# Specify the motion file in env_cfg
python scripts/rsl_rl/train.py --headless --task Unitree-G1-29dof-Mimic-Dance-102

# Use custom motion task
python scripts/rsl_rl/train.py --headless --task Unitree-G1-29dof-Mimic-Custom \
  --motion_files /path/to/motion.npz \
  --experiment_name unitree_g1_29dof_mimic_{your_motion_name}
```

### Multi motions training

Support training with multiple motions by providing a directory of `npz` files. For example 2 motions under `data/dailylife_data_v1.1`.

```bash
python scripts/mimic/csv_to_npz.py \
  -i data/dailylife_data_v1.1 \
  --input_fps 30 \
  --headless
```

Start training with multiple motions:

```bash
python scripts/rsl_rl/train.py --headless --task Unitree-G1-29dof-Mimic-Custom \
  --motion_files data/dailylife_data_v1.1/load_lift_1_2.npz data/dailylife_data_v1.1/locomotion_jump_1_3.npz \
  --experiment_name unitree_g1_29dof_mimic_dailylife_multi
```

The motion files information will be automatically stored in `logs/rsl_rl/<experiment_name>/<timestamp>/params/motion_files.yaml`
