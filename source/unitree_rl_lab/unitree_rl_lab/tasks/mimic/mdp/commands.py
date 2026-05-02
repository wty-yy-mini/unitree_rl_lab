from __future__ import annotations

import math
import os
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import yaml
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)
from unitree_rl_lab.utils.concat_batch_tensor import ConcatBatchTensor
from unitree_rl_lab.utils.path_utils import resolve_files

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class MotionLoader:
    """Load one or more motion datasets into concatenated variable-length tensors."""

    def __init__(
        self,
        motion_files: str | Sequence[str],
        body_indexes: Sequence[int] | torch.Tensor,
        device: str = "cpu",
        motion_log_path: str | None = None,
    ):
        """Load motion files and pack them into concatenated tensors.

        Args:
            motion_files: One or more motion file or directory paths.
            body_indexes: Body indices to keep from each stored motion.
            device: Target torch device for the loaded motion tensors.
            motion_log_path: Optional YAML path used to save resolved motion metadata.
        """
        self.device = torch.device(device)
        self.motion_files = resolve_files(motion_files, suffix=".npz")
        self.motion_names = [Path(path).stem for path in self.motion_files]
        self.motion_log_path = motion_log_path

        body_indexes_tensor = torch.as_tensor(body_indexes, dtype=torch.long).cpu()
        body_indexes_np = body_indexes_tensor.numpy()

        fps_values: list[float] = []
        joint_pos_tensors: list[torch.Tensor] = []
        joint_vel_tensors: list[torch.Tensor] = []
        body_pos_tensors: list[torch.Tensor] = []
        body_quat_tensors: list[torch.Tensor] = []
        body_lin_vel_tensors: list[torch.Tensor] = []
        body_ang_vel_tensors: list[torch.Tensor] = []

        for motion_file in self.motion_files:
            with np.load(motion_file) as data:
                fps_values.append(float(np.asarray(data["fps"]).reshape(-1)[0]))
                joint_pos_tensors.append(torch.tensor(data["joint_pos"], dtype=torch.float32, device=self.device))
                joint_vel_tensors.append(torch.tensor(data["joint_vel"], dtype=torch.float32, device=self.device))
                body_pos_tensors.append(
                    torch.tensor(data["body_pos_w"][:, body_indexes_np], dtype=torch.float32, device=self.device)
                )
                body_quat_tensors.append(
                    torch.tensor(data["body_quat_w"][:, body_indexes_np], dtype=torch.float32, device=self.device)
                )
                body_lin_vel_tensors.append(
                    torch.tensor(data["body_lin_vel_w"][:, body_indexes_np], dtype=torch.float32, device=self.device)
                )
                body_ang_vel_tensors.append(
                    torch.tensor(data["body_ang_vel_w"][:, body_indexes_np], dtype=torch.float32, device=self.device)
                )

        self.fps = torch.tensor(fps_values, dtype=torch.float32, device=self.device)
        self.joint_pos = ConcatBatchTensor(tensors=joint_pos_tensors, device=self.device)
        self.joint_vel = ConcatBatchTensor(tensors=joint_vel_tensors, device=self.device)
        self.body_pos_w = ConcatBatchTensor(tensors=body_pos_tensors, device=self.device)
        self.body_quat_w = ConcatBatchTensor(tensors=body_quat_tensors, device=self.device)
        self.body_lin_vel_w = ConcatBatchTensor(tensors=body_lin_vel_tensors, device=self.device)
        self.body_ang_vel_w = ConcatBatchTensor(tensors=body_ang_vel_tensors, device=self.device)

        self.motion_lengths = self.joint_pos.batch_sizes.to(dtype=torch.long, device=self.device)
        self.motion_ends = torch.cumsum(self.motion_lengths, dim=0)
        self.motion_starts = self.motion_ends - self.motion_lengths
        self.num_motions = len(self.motion_files)
        self.total_steps = int(self.motion_lengths.sum().item())
        self.log_motion_files()

    def flat_step_indices(self, motion_ids: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:
        """Convert per-motion time steps into flat concatenated step indices.

        Args:
            motion_ids: Motion indices for each sampled environment.
            time_steps: Per-motion step indices for each sampled environment.

        Returns:
            Flat indices into the concatenated motion-step space.
        """
        return self.motion_starts[motion_ids] + time_steps

    def unpack_flat_step_indices(self, flat_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert flat concatenated step indices into motion and time indices.

        Args:
            flat_indices: Flat indices into the concatenated motion-step space.

        Returns:
            A tuple of `(motion_ids, time_steps)` aligned with `flat_indices`.
        """
        motion_ids = torch.bucketize(flat_indices, self.motion_ends, right=True)
        time_steps = flat_indices - self.motion_starts[motion_ids]
        return motion_ids, time_steps

    def log_motion_files(self) -> None:
        """Print and optionally save resolved motion-file metadata."""
        records = []
        for motion_idx, motion_file in enumerate(self.motion_files):
            records.append(
                {
                    "index": motion_idx,
                    "name": self.motion_names[motion_idx],
                    "path": motion_file,
                    "frames": int(self.motion_lengths[motion_idx].item()),
                    "fps": float(self.fps[motion_idx].item()),
                }
            )

        print(f"[INFO] Loaded {len(records)} motion files:")
        for record in records:
            print(
                f"  [{record['index']}] frames={record['frames']} fps={record['fps']:.2f} "
                f"name={record['name']} path={record['path']}"
            )

        if self.motion_log_path is None:
            return

        motion_log_path = Path(self.motion_log_path).expanduser().resolve()
        motion_log_path.parent.mkdir(parents=True, exist_ok=True)
        with motion_log_path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(
                {
                    "num_motions": len(records),
                    "total_steps": self.total_steps,
                    "motions": records,
                },
                file,
                sort_keys=False,
                allow_unicode=True,
            )


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        """Initialize motion commands for single-motion or multi-motion datasets."""
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )

        motion_files = self.cfg.motion_files
        if motion_files is None and self.cfg.motion_file is not None:
            motion_files = [self.cfg.motion_file]
        self.motion = MotionLoader(
            motion_files,
            self.body_indexes,
            device=self.device,
            motion_log_path=self.cfg.motion_log_path,
        )
        self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        self.step_failed_count = torch.zeros(self.motion.total_steps, dtype=torch.float32, device=self.device)
        self._current_step_failed = torch.zeros(self.motion.total_steps, dtype=torch.float32, device=self.device)
        self._failure_term_names = self._resolve_failure_term_names()
        self._failure_term_indices: torch.Tensor | None = None

        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    def _gather_motion_frame(
        self,
        motion_ids: torch.Tensor | None = None,
        time_steps: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Gather all current motion tensors for a batch of motion steps.

        Args:
            motion_ids: Optional motion ids to gather. Defaults to all env motion ids.
            time_steps: Optional in-motion step ids to gather. Defaults to all env time steps.

        Returns:
            Mapping with joint, body, and anchor tensors for the selected motion steps.
        """
        if motion_ids is None:
            motion_ids = self.motion_ids
        if time_steps is None:
            time_steps = self.time_steps

        body_pos_w = self.motion.body_pos_w.gather(motion_ids, time_steps)          # shape (num_envs, num_bodies, 3)
        body_quat_w = self.motion.body_quat_w.gather(motion_ids, time_steps)
        body_lin_vel_w = self.motion.body_lin_vel_w.gather(motion_ids, time_steps)
        body_ang_vel_w = self.motion.body_ang_vel_w.gather(motion_ids, time_steps)
        return {
            "joint_pos": self.motion.joint_pos.gather(motion_ids, time_steps),
            "joint_vel": self.motion.joint_vel.gather(motion_ids, time_steps),
            "body_pos_w": body_pos_w + self._env.scene.env_origins[:, None, :],     # shape (num_envs, num_bodies, 3)
            "body_quat_w": body_quat_w,
            "body_lin_vel_w": body_lin_vel_w,
            "body_ang_vel_w": body_ang_vel_w,
            "anchor_pos_w": body_pos_w[:, self.motion_anchor_body_index] + self._env.scene.env_origins, # shape (num_envs, 3)
            "anchor_quat_w": body_quat_w[:, self.motion_anchor_body_index],
            "anchor_lin_vel_w": body_lin_vel_w[:, self.motion_anchor_body_index],
            "anchor_ang_vel_w": body_ang_vel_w[:, self.motion_anchor_body_index],
        }

    @property
    def flat_time_steps(self) -> torch.Tensor:
        """Return flat concatenated step indices for all environments."""
        return self.motion.flat_step_indices(self.motion_ids, self.time_steps)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion.joint_pos.gather(self.motion_ids, self.time_steps)

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion.joint_vel.gather(self.motion_ids, self.time_steps)

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w.gather(self.motion_ids, self.time_steps) + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w.gather(self.motion_ids, self.time_steps)

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w.gather(self.motion_ids, self.time_steps)

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w.gather(self.motion_ids, self.time_steps)

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w.gather(self.motion_ids, self.time_steps)[:, self.motion_anchor_body_index] + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w.gather(self.motion_ids, self.time_steps)[:, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w.gather(self.motion_ids, self.time_steps)[:, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w.gather(self.motion_ids, self.time_steps)[:, self.motion_anchor_body_index]

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]

    def _update_metrics(self):
        motion_frame = self._gather_motion_frame()
        self.metrics["error_anchor_pos"] = torch.norm(motion_frame["anchor_pos_w"] - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(motion_frame["anchor_quat_w"], self.robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(
            motion_frame["anchor_lin_vel_w"] - self.robot_anchor_lin_vel_w, dim=-1
        )
        self.metrics["error_anchor_ang_vel"] = torch.norm(
            motion_frame["anchor_ang_vel_w"] - self.robot_anchor_ang_vel_w, dim=-1
        )

        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(
            dim=-1
        )

        self.metrics["error_body_lin_vel"] = torch.norm(
            motion_frame["body_lin_vel_w"] - self.robot_body_lin_vel_w, dim=-1
        ).mean(dim=-1)
        self.metrics["error_body_ang_vel"] = torch.norm(
            motion_frame["body_ang_vel_w"] - self.robot_body_ang_vel_w, dim=-1
        ).mean(dim=-1)

        self.metrics["error_joint_pos"] = torch.norm(motion_frame["joint_pos"] - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(motion_frame["joint_vel"] - self.robot_joint_vel, dim=-1)

    def _resolve_failure_term_names(self) -> list[str]:
        """Collect termination term names that should count as true failures.

        Timeout terms are excluded so normal episode exhaustion does not bias
        the adaptive sampler toward late motion steps.

        Returns:
            Names of non-timeout termination terms from the environment config.
        """
        failure_term_names: list[str] = []
        for term_name, term_cfg in self._env.cfg.terminations.__dict__.items():
            if term_cfg is None or not hasattr(term_cfg, "time_out"):
                continue
            if not term_cfg.time_out:
                failure_term_names.append(term_name)
        return failure_term_names

    def _get_failure_term_indices(self) -> torch.Tensor:
        """Resolve and cache failure-term indices after managers are initialized.

        Returns:
            Cached tensor of non-timeout termination term indices.
        """
        if self._failure_term_indices is not None:
            return self._failure_term_indices
        if len(self._failure_term_names) == 0:
            self._failure_term_indices = torch.empty(0, dtype=torch.long, device=self.device)
            return self._failure_term_indices
        term_name_to_idx = self._env.termination_manager._term_name_to_term_idx
        self._failure_term_indices = torch.tensor(
            [term_name_to_idx[term_name] for term_name in self._failure_term_names],
            dtype=torch.long,
            device=self.device,
        )
        return self._failure_term_indices

    def _get_episode_failed_mask(self, env_ids: Sequence[int]) -> torch.Tensor:
        """Return which resampled environments ended due to true failure terms.

        Args:
            env_ids: Environment ids selected for command resampling.

        Returns:
            Boolean mask aligned with `env_ids` that marks true failure resets.
        """
        failure_term_indices = self._get_failure_term_indices()
        if len(env_ids) == 0 or failure_term_indices.numel() == 0:
            return torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)
        return self._env.termination_manager._term_dones[env_ids][:, failure_term_indices].any(dim=1)

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        """Sample new `(motion_id, time_step)` pairs from flat EMA step weights."""
        episode_failed = self._get_episode_failed_mask(env_ids)
        if torch.any(episode_failed):
            safe_time_steps = torch.minimum(self.time_steps, self.motion.motion_lengths[self.motion_ids] - 1)
            failed_steps = self.motion.flat_step_indices(self.motion_ids, safe_time_steps)[env_ids][episode_failed]
            self._current_step_failed += torch.bincount(failed_steps, minlength=self.motion.total_steps).to(
                self._current_step_failed.dtype
            )

        sampling_probabilities = self.step_failed_count + self.cfg.adaptive_uniform_ratio / float(
            self.motion.total_steps
        )
        sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

        sampled_steps = torch.multinomial(sampling_probabilities, len(env_ids), replacement=True)
        sampled_motion_ids, sampled_time_steps = self.motion.unpack_flat_step_indices(sampled_steps)
        self.motion_ids[env_ids] = sampled_motion_ids
        self.time_steps[env_ids] = sampled_time_steps

        H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
        H_norm = H if self.motion.total_steps <= 1 else H / math.log(self.motion.total_steps)
        pmax, imax = sampling_probabilities.max(dim=0)
        self.metrics["sampling_entropy"][:] = H_norm
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = imax.float() / self.motion.total_steps

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample command states for environments that need a new motion start."""
        if len(env_ids) == 0:
            return
        self._adaptive_sampling(env_ids)
        motion_frame = self._gather_motion_frame()
        root_pos = motion_frame["body_pos_w"][env_ids, 0].clone()
        root_ori = motion_frame["body_quat_w"][env_ids, 0].clone()
        root_lin_vel = motion_frame["body_lin_vel_w"][env_ids, 0].clone()
        root_ang_vel = motion_frame["body_ang_vel_w"][env_ids, 0].clone()

        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori = quat_mul(orientations_delta, root_ori)
        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel += rand_samples[:, :3]
        root_ang_vel += rand_samples[:, 3:]

        joint_pos = motion_frame["joint_pos"][env_ids].clone()
        joint_vel = motion_frame["joint_vel"][env_ids].clone()

        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos = torch.clip(joint_pos, soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1])
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos, root_ori, root_lin_vel, root_ang_vel], dim=-1),
            env_ids=env_ids,
        )

    def _update_command(self):
        """Advance motion playback and refresh relative target poses."""
        self.time_steps += 1
        env_ids = torch.where(self.time_steps >= self.motion.motion_lengths[self.motion_ids])[0]
        self._resample_command(env_ids)
        motion_frame = self._gather_motion_frame()

        anchor_pos_w_repeat = motion_frame["anchor_pos_w"][:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = motion_frame["anchor_quat_w"][:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, motion_frame["body_quat_w"])
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, motion_frame["body_pos_w"] - anchor_pos_w_repeat)

        self.step_failed_count = (
            self.cfg.adaptive_alpha * self._current_step_failed + (1 - self.cfg.adaptive_alpha) * self.step_failed_count
        )
        self._current_step_failed.zero_()

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/current/anchor")
                )
                self.goal_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/anchor")
                )

                self.current_body_visualizers = []
                self.goal_body_visualizers = []
                for name in self.cfg.body_names:
                    self.current_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/current/" + name)
                        )
                    )
                    self.goal_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/" + name)
                        )
                    )

            self.current_anchor_visualizer.set_visibility(True)
            self.goal_anchor_visualizer.set_visibility(True)
            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(True)
                self.goal_body_visualizers[i].set_visibility(True)

        else:
            if hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer.set_visibility(False)
                self.goal_anchor_visualizer.set_visibility(False)
                for i in range(len(self.cfg.body_names)):
                    self.current_body_visualizers[i].set_visibility(False)
                    self.goal_body_visualizers[i].set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        self.current_anchor_visualizer.visualize(self.robot_anchor_pos_w, self.robot_anchor_quat_w)
        self.goal_anchor_visualizer.visualize(self.anchor_pos_w, self.anchor_quat_w)

        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i])
            self.goal_body_visualizers[i].visualize(self.body_pos_relative_w[:, i], self.body_quat_relative_w[:, i])


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = MotionCommand

    asset_name: str = MISSING

    motion_files: list[str] | None = None
    motion_file: str | None = None
    motion_log_path: str | None = None
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)

    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
