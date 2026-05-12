# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass

from .rl_cfg import RslRlActorCfg, RslRlBaseRunnerCfg, RslRlRecurrentActorCfg


@configclass
class RslRlDistillationAlgorithmCfg:
    """Configuration for the distillation algorithm."""

    class_name: str = "Distillation"
    """The algorithm class name."""

    num_learning_epochs: int = MISSING
    """Number of optimization epochs per update."""

    learning_rate: float = MISSING
    """Student learning rate."""

    gradient_length: int = MISSING
    """Number of steps between optimizer updates."""

    max_grad_norm: float | None = None
    """Maximum gradient norm."""

    optimizer: Literal["adam", "adamw", "sgd", "rmsprop"] = "adam"
    """Optimizer used for the student."""

    loss_type: Literal["mse", "huber"] = "mse"
    """Behavior-cloning loss type."""


@configclass
class RslRlDistillationRunnerCfg(RslRlBaseRunnerCfg):
    """Configuration of the runner for distillation algorithms."""

    class_name: str = "DistillationRunner"
    """The runner class name."""

    student: RslRlActorCfg | RslRlRecurrentActorCfg = MISSING
    """Student model configuration."""

    teacher: RslRlActorCfg | RslRlRecurrentActorCfg = MISSING
    """Teacher model configuration."""

    algorithm: RslRlDistillationAlgorithmCfg = MISSING
    """Algorithm configuration."""
