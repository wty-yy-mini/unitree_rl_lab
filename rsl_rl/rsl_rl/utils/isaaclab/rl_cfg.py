# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass

from .rnd_cfg import RslRlRndCfg
from .symmetry_cfg import RslRlSymmetryCfg


@configclass
class RslRlGaussianDistributionCfg:
    """Configuration for a Gaussian action distribution."""

    class_name: str = "GaussianDistribution"
    """The distribution class name."""

    init_std: float = 1.0
    """Initial action standard deviation."""

    std_type: Literal["scalar", "log"] = "scalar"
    """The standard-deviation parameterization."""


@configclass
class RslRlHeteroscedasticGaussianDistributionCfg:
    """Configuration for a state-dependent Gaussian action distribution."""

    class_name: str = "HeteroscedasticGaussianDistribution"
    """The distribution class name."""

    init_std: float = 1.0
    """Initial action standard deviation."""

    std_type: Literal["scalar", "log"] = "scalar"
    """The standard-deviation parameterization."""


@configclass
class RslRlActorCfg:
    """Configuration for an actor model in rsl_rl 5.3.0."""

    class_name: str = "MLPModel"
    """The model class name."""

    hidden_dims: list[int] = MISSING
    """Hidden dimensions of the model head."""

    activation: str = MISSING
    """Activation function used by the model."""

    obs_normalization: bool = False
    """Whether to normalize actor observations."""

    distribution_cfg: RslRlGaussianDistributionCfg | RslRlHeteroscedasticGaussianDistributionCfg = (
        RslRlGaussianDistributionCfg()
    )
    """Stochastic output distribution configuration."""


@configclass
class RslRlCriticCfg:
    """Configuration for a critic model in rsl_rl 5.3.0."""

    class_name: str = "MLPModel"
    """The model class name."""

    hidden_dims: list[int] = MISSING
    """Hidden dimensions of the model head."""

    activation: str = MISSING
    """Activation function used by the model."""

    obs_normalization: bool = False
    """Whether to normalize critic observations."""


@configclass
class RslRlRecurrentActorCfg(RslRlActorCfg):
    """Configuration for a recurrent actor model."""

    class_name: str = "RNNModel"
    """The recurrent model class name."""

    rnn_type: Literal["lstm", "gru"] = MISSING
    """The recurrent layer type."""

    rnn_hidden_dim: int = MISSING
    """Hidden dimension of the recurrent layer."""

    rnn_num_layers: int = MISSING
    """Number of recurrent layers."""


@configclass
class RslRlRecurrentCriticCfg(RslRlCriticCfg):
    """Configuration for a recurrent critic model."""

    class_name: str = "RNNModel"
    """The recurrent model class name."""

    rnn_type: Literal["lstm", "gru"] = MISSING
    """The recurrent layer type."""

    rnn_hidden_dim: int = MISSING
    """Hidden dimension of the recurrent layer."""

    rnn_num_layers: int = MISSING
    """Number of recurrent layers."""


@configclass
class RslRlPpoAlgorithmCfg:
    """Configuration for the PPO algorithm."""

    class_name: str = "PPO"
    """The algorithm class name."""

    num_learning_epochs: int = MISSING
    """Number of optimization epochs per update."""

    num_mini_batches: int = MISSING
    """Number of mini-batches per update."""

    learning_rate: float = MISSING
    """Policy learning rate."""

    schedule: str = MISSING
    """Learning-rate schedule."""

    gamma: float = MISSING
    """Discount factor."""

    lam: float = MISSING
    """GAE lambda."""

    entropy_coef: float = MISSING
    """Entropy regularization coefficient."""

    desired_kl: float = MISSING
    """Target KL divergence for adaptive learning rate."""

    max_grad_norm: float = MISSING
    """Maximum gradient norm."""

    value_loss_coef: float = MISSING
    """Value loss coefficient."""

    use_clipped_value_loss: bool = MISSING
    """Whether to use clipped value loss."""

    clip_param: float = MISSING
    """PPO clipping parameter."""

    optimizer: Literal["adam", "adamw", "sgd", "rmsprop"] = "adam"
    """Optimizer used by PPO."""

    normalize_advantage_per_mini_batch: bool = False
    """Whether to normalize advantage per mini-batch."""

    share_cnn_encoders: bool = False
    """Whether actor and critic share CNN encoders."""

    rnd_cfg: RslRlRndCfg | None = None
    """Optional RND configuration."""

    symmetry_cfg: RslRlSymmetryCfg | None = None
    """Optional symmetry configuration."""


@configclass
class RslRlBaseRunnerCfg:
    """Base configuration of the runner."""

    seed: int = 42
    """Random seed."""

    device: str = "cuda:0"
    """Device for the RL agent."""

    num_steps_per_env: int = MISSING
    """Rollout steps per environment per update."""

    max_iterations: int = MISSING
    """Maximum number of training iterations."""

    obs_groups: dict[str, list[str]] = {}
    """Observation-set mapping passed directly to rsl_rl."""

    clip_actions: float | None = None
    """Optional action clipping value."""

    save_interval: int = MISSING
    """Checkpoint interval in iterations."""

    experiment_name: str = MISSING
    """Experiment name."""

    run_name: str = ""
    """Run name suffix."""

    logger: Literal["tensorboard", "neptune", "wandb"] = "tensorboard"
    """Logger backend."""

    neptune_project: str = "isaaclab"
    """Neptune project name."""

    wandb_project: str = "isaaclab"
    """Weights & Biases project name."""

    resume: bool = False
    """Whether to resume training from a checkpoint."""

    load_run: str = ".*"
    """Run directory pattern used when resuming."""

    load_checkpoint: str = "model_.*.pt"
    """Checkpoint filename pattern used when resuming."""

    check_for_nan: bool = True
    """Whether to check environment outputs for NaNs."""

    torch_compile_mode: str | None = None
    """Optional torch.compile mode for actor and critic."""


@configclass
class RslRlOnPolicyRunnerCfg(RslRlBaseRunnerCfg):
    """Configuration of the runner for on-policy algorithms."""

    class_name: str = "OnPolicyRunner"
    """The runner class name."""

    actor: RslRlActorCfg | RslRlRecurrentActorCfg = MISSING
    """Actor model configuration."""

    critic: RslRlCriticCfg | RslRlRecurrentCriticCfg = MISSING
    """Critic model configuration."""

    algorithm: RslRlPpoAlgorithmCfg = MISSING
    """Algorithm configuration."""
