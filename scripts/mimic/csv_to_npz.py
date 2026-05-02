"""This script replay motions from csv files and output them to npz files

.. code-block:: bash

    # Usage
    python csv_to_npz.py -i path_to_input.csv another_input_dir --input_fps 60

    # Example
    python scripts/mimic/csv_to_npz.py \
        -i data/dailylife_data_v1.1 \
        --input_fps 30 \
        --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse
from pathlib import Path

import numpy as np

from isaaclab.app import AppLauncher


def _resolve_conversion_jobs(
    input_paths_str: list[str],
    output_name: str | None,
) -> tuple[list[Path], list[tuple[Path, Path]]]:
    """Resolve input CSV files and their target NPZ output paths.

    Args:
        input_paths_str: Input CSV file or directory paths from CLI.
        output_name: Optional output path override from CLI.

    Returns:
        A tuple of resolved input paths and the list of `(csv_path, npz_path)` jobs.
    """
    input_paths = [Path(input_path_str).expanduser().resolve() for input_path_str in input_paths_str]
    for input_path in input_paths:
        if not input_path.exists():
            raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if len(input_paths) == 1 and input_paths[0].is_file():
        input_path = input_paths[0]
        if input_path.suffix.lower() != ".csv":
            raise ValueError(f"Input file must be a CSV file: {input_path}")
        output_path = Path(output_name).expanduser().resolve() if output_name else input_path.with_suffix(".npz")
        return input_paths, [(input_path, output_path)]

    jobs: list[tuple[Path, Path]] = []
    output_root = Path(output_name).expanduser().resolve() if output_name else None
    if output_root is not None and output_root.suffix.lower() == ".npz":
        raise ValueError("--output_name must be a directory when --input expands to multiple CSV files.")

    for input_path in input_paths:
        if input_path.is_file():
            if input_path.suffix.lower() != ".csv":
                raise ValueError(f"Input file must be a CSV file: {input_path}")
            output_path = (output_root / input_path.name).with_suffix(".npz") if output_root else input_path.with_suffix(".npz")
            jobs.append((input_path, output_path))
            continue

        csv_files = sorted(p for p in input_path.rglob("*.csv") if p.is_file())
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found under directory: {input_path}")

        if output_root is None:
            jobs.extend((csv_path, csv_path.with_suffix(".npz")) for csv_path in csv_files)
        else:
            jobs.extend(
                (csv_path, (output_root / csv_path.relative_to(input_path)).with_suffix(".npz"))
                for csv_path in csv_files
            )

    return input_paths, jobs


# add argparse arguments
parser = argparse.ArgumentParser(description="Replay motion from csv file and output to npz file.")
parser.add_argument("--input", "-i", type=str, nargs="+", required=True, help="Input motion CSV file or directory.")
parser.add_argument("--input_fps", type=int, default=60, help="The fps of the input motion.")
parser.add_argument(
    "--frame_range",
    nargs=2,
    type=int,
    metavar=("START", "END"),
    help=(
        "frame range: START END (both inclusive). The frame index starts from 1. If not provided, all frames will be"
        " loaded."
    ),
)
parser.add_argument(
    "--output_name",
    type=str,
    help="Output NPZ file path for single-file input, or output directory for directory input.",
)
parser.add_argument("--output_fps", type=int, default=50, help="The fps of the output motion.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
args_cli.input_paths, args_cli.jobs = _resolve_conversion_jobs(args_cli.input, args_cli.output_name)


# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul, quat_slerp

##
# Pre-defined configs
##
from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_CFG as ROBOT_CFG  # Currently only support G1-29dof


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # articulation
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


class MotionLoader:
    def __init__(
        self,
        motion_file: str,
        input_fps: int,
        output_fps: int,
        device: torch.device,
        frame_range: tuple[int, int] | None,
    ):
        self.motion_file = motion_file
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.input_dt = 1.0 / self.input_fps
        self.output_dt = 1.0 / self.output_fps
        self.current_idx = 0
        self.device = device
        self.frame_range = frame_range
        self._load_motion()
        self._interpolate_motion()
        self._compute_velocities()

    def _load_motion(self):
        """Loads the motion from the csv file."""
        if self.frame_range is None:
            motion = torch.from_numpy(np.loadtxt(self.motion_file, delimiter=","))
        else:
            motion = torch.from_numpy(
                np.loadtxt(
                    self.motion_file,
                    delimiter=",",
                    skiprows=self.frame_range[0] - 1,
                    max_rows=self.frame_range[1] - self.frame_range[0] + 1,
                )
            )
        motion = motion.to(torch.float32).to(self.device)
        self.motion_base_poss_input = motion[:, :3]
        self.motion_base_rots_input = motion[:, 3:7]
        self.motion_base_rots_input = self.motion_base_rots_input[:, [3, 0, 1, 2]]  # convert to wxyz
        self.motion_dof_poss_input = motion[:, 7:]

        self.input_frames = motion.shape[0]
        self.duration = (self.input_frames - 1) * self.input_dt
        print(f"Motion loaded ({self.motion_file}), duration: {self.duration} sec, frames: {self.input_frames}")

    def _interpolate_motion(self):
        """Interpolates the motion to the output fps."""
        times = torch.arange(0, self.duration, self.output_dt, device=self.device, dtype=torch.float32)
        self.output_frames = times.shape[0]
        index_0, index_1, blend = self._compute_frame_blend(times)
        self.motion_base_poss = self._lerp(
            self.motion_base_poss_input[index_0],
            self.motion_base_poss_input[index_1],
            blend.unsqueeze(1),
        )
        self.motion_base_rots = self._slerp(
            self.motion_base_rots_input[index_0],
            self.motion_base_rots_input[index_1],
            blend,
        )
        self.motion_dof_poss = self._lerp(
            self.motion_dof_poss_input[index_0],
            self.motion_dof_poss_input[index_1],
            blend.unsqueeze(1),
        )
        print(
            f"Motion interpolated, input frames: {self.input_frames}, input fps: {self.input_fps}, output frames:"
            f" {self.output_frames}, output fps: {self.output_fps}"
        )

    def _lerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        """Linear interpolation between two tensors."""
        return a * (1 - blend) + b * blend

    def _slerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        """Spherical linear interpolation between two quaternions."""
        slerped_quats = torch.zeros_like(a)
        for i in range(a.shape[0]):
            slerped_quats[i] = quat_slerp(a[i], b[i], blend[i])
        return slerped_quats

    def _compute_frame_blend(self, times: torch.Tensor) -> torch.Tensor:
        """Computes the frame blend for the motion."""
        phase = times / self.duration
        index_0 = (phase * (self.input_frames - 1)).floor().long()
        index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1))
        blend = phase * (self.input_frames - 1) - index_0
        return index_0, index_1, blend

    def _compute_velocities(self):
        """Computes the velocities of the motion."""
        self.motion_base_lin_vels = torch.gradient(self.motion_base_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_dof_vels = torch.gradient(self.motion_dof_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_base_ang_vels = self._so3_derivative(self.motion_base_rots, self.output_dt)

    def _so3_derivative(self, rotations: torch.Tensor, dt: float) -> torch.Tensor:
        """Computes the derivative of a sequence of SO3 rotations.

        Args:
            rotations: shape (B, 4).
            dt: time step.
        Returns:
            shape (B, 3).
        """
        q_prev, q_next = rotations[:-2], rotations[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))  # shape (B−2, 4)

        omega = axis_angle_from_quat(q_rel) / (2.0 * dt)  # shape (B−2, 3)
        omega = torch.cat([omega[:1], omega, omega[-1:]], dim=0)  # repeat first and last sample
        return omega

    def get_next_state(
        self,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Gets the next state of the motion."""
        state = (
            self.motion_base_poss[self.current_idx : self.current_idx + 1],
            self.motion_base_rots[self.current_idx : self.current_idx + 1],
            self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
            self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
            self.motion_dof_poss[self.current_idx : self.current_idx + 1],
            self.motion_dof_vels[self.current_idx : self.current_idx + 1],
        )
        self.current_idx += 1
        reset_flag = False
        if self.current_idx >= self.output_frames:
            self.current_idx = 0
            reset_flag = True
        return state, reset_flag


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    robot = scene["robot"]
    robot_joint_indexes = robot.find_joints(scene.cfg.robot.joint_sdk_names, preserve_order=True)[0]

    for input_csv, output_npz in args_cli.jobs:
        motion = MotionLoader(
            motion_file=str(input_csv),
            input_fps=args_cli.input_fps,
            output_fps=args_cli.output_fps,
            device=sim.device,
            frame_range=args_cli.frame_range,
        )
        log = {
            "fps": [args_cli.output_fps],
            "joint_pos": [],
            "joint_vel": [],
            "body_pos_w": [],
            "body_quat_w": [],
            "body_lin_vel_w": [],
            "body_ang_vel_w": [],
        }
        file_saved = False

        while simulation_app.is_running():
            (
                (
                    motion_base_pos,
                    motion_base_rot,
                    motion_base_lin_vel,
                    motion_base_ang_vel,
                    motion_dof_pos,
                    motion_dof_vel,
                ),
                reset_flag,
            ) = motion.get_next_state()

            root_states = robot.data.default_root_state.clone()
            root_states[:, :3] = motion_base_pos
            root_states[:, :2] += scene.env_origins[:, :2]
            root_states[:, 3:7] = motion_base_rot
            root_states[:, 7:10] = motion_base_lin_vel
            root_states[:, 10:] = motion_base_ang_vel
            robot.write_root_state_to_sim(root_states)

            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            joint_pos[:, robot_joint_indexes] = motion_dof_pos
            joint_vel[:, robot_joint_indexes] = motion_dof_vel
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            sim.render()  # We don't want physic (sim.step())
            scene.update(sim.get_physics_dt())

            pos_lookat = root_states[0, :3].cpu().numpy()
            sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)

            if not file_saved:
                log["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())
                log["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())
                log["body_pos_w"].append(robot.data.body_pos_w[0, :].cpu().numpy().copy())
                log["body_quat_w"].append(robot.data.body_quat_w[0, :].cpu().numpy().copy())
                log["body_lin_vel_w"].append(robot.data.body_lin_vel_w[0, :].cpu().numpy().copy())
                log["body_ang_vel_w"].append(robot.data.body_ang_vel_w[0, :].cpu().numpy().copy())

            if reset_flag and not file_saved:
                file_saved = True
                for k in (
                    "joint_pos",
                    "joint_vel",
                    "body_pos_w",
                    "body_quat_w",
                    "body_lin_vel_w",
                    "body_ang_vel_w",
                ):
                    log[k] = np.stack(log[k], axis=0)

                output_npz.parent.mkdir(parents=True, exist_ok=True)
                np.savez(output_npz, **log)
                print("[INFO]: Motion npz file saved to", output_npz)
                break
    print("[INFO]: All done, exiting simulator.")
    exit()


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = SimulationContext(sim_cfg)
    # Design scene
    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
