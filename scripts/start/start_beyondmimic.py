#!/usr/bin/env python3
"""Launch one BeyondMimic-style training job for each configured motion folder.

Overview:
This script reads a manually configured list of motion folders under a shared
data root and launches one training process per configured folder with all NPZ
motions under that folder. It also records the launched PID list and a JSON
manifest under the configured log directory.

Quick Start:
    python scripts/start/start_beyondmimic.py

Notes:
    Configure all runtime options in the ``User Config`` block at the top of
    this file before launching.
    Add or remove entries in ``JOBS`` to control which folders are used for
    training and which CUDA device each job should use.
    The ``seed`` field in each job is optional and defaults to ``0``.
    Set ``DRY_RUN = True`` to print planned commands without starting training.
"""

from __future__ import annotations

import csv
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from collections import deque


# =========================
# User Config
# =========================
TASK = "Unitree-G1-29dof-Mimic-Custom"
TRAIN_SCRIPT = "scripts/rsl_rl/train.py"
DATA_ROOT = "data/dailylife_data_v1.1"
LOG_ROOT = "logs/start_jobs"
LAUNCHER_NAME = "beyondmimic_dailylife"
EXPERIMENT_PREFIX = "beyondmimic_dailylife"

HEADLESS = True
DRY_RUN = False
DETACH_PROCESS = True
LOG_TAIL_LINES = 100
MONITOR_INTERVAL_SEC = 2.0

JOBS: list[dict[str, str | int]] = [
    # {"dir": "load/lift",         "cuda": "0"},
    # {"dir": "load/push",         "cuda": "0"},
    # {"dir": "load/throw",        "cuda": "0"},
    # {"dir": "locomotion/jump",   "cuda": "1"},
    # {"dir": "locomotion/run",    "cuda": "1"},
    # {"dir": "locomotion/walk",   "cuda": "1"},
    {"dir": "transition/getup",  "cuda": "3"},
    {"dir": "transition/sit",    "cuda": "4"},
    # {"dir": "transition/squat",  "cuda": "2"},
    # {"dir": "transition/stand",  "cuda": "3"},
    # {"dir": "transition/turn",   "cuda": "3"},
]

COMMON_TRAIN_ARGS: list[str] = [
    "--logger",
    "tensorboard",
    # "--max_iterations", "80000",
    # "--num_envs", "4096",
]

PER_JOB_TRAIN_ARGS: dict[str, list[str]] = {
    # "locomotion/jump": ["--max_iterations", "80000"],
}

HELPER_ARG = "--log-tail-helper"


@dataclass(frozen=True)
class ResolvedTrainingJob:
    """Store one resolved training job with directory and motion metadata."""

    relative_dir: str
    directory: Path
    cuda_device: str
    seed: int
    job_name_override: str | None
    experiment_suffix: str | None
    motion_files: tuple[Path, ...]

    @property
    def normalized_relative_dir(self) -> str:
        """Return the normalized relative folder path used by this job."""
        return self.relative_dir.strip().strip("/").replace("\\", "/")

    @property
    def job_name(self) -> str:
        """Return the job name used for logs and training run names."""
        if self.job_name_override:
            return self.job_name_override
        return self.normalized_relative_dir.replace("/", "_")


@dataclass(frozen=True)
class LaunchedJob:
    """Store runtime handles and paths for one launched training job."""

    job: ResolvedTrainingJob
    process: subprocess.Popen[bytes]
    stdout_helper: subprocess.Popen[str]
    stderr_helper: subprocess.Popen[str]
    stdout_log: Path
    stderr_log: Path
    stdout_meta: Path
    stderr_meta: Path

def resolve_training_jobs(data_root: Path) -> list[ResolvedTrainingJob]:
    """Resolve configured training folders into launchable jobs.

    Args:
        data_root: Root directory containing all candidate motion folders.

    Returns:
        A list of resolved training jobs with discovered NPZ motion files.
    """
    resolved_jobs: list[ResolvedTrainingJob] = []
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root does not exist or is not a directory: {data_root}")
    if not JOBS:
        raise ValueError("JOBS is empty. Please configure at least one training folder.")

    for spec in JOBS:
        relative_dir = str(spec.get("dir", "")).strip().strip("/").replace("\\", "/")
        if not relative_dir:
            raise ValueError("Encountered an empty dir in JOBS.")
        cuda_device = str(spec.get("cuda", "")).strip()
        if not cuda_device:
            raise ValueError(f"Missing cuda for configured training directory: {relative_dir}")
        seed = int(spec.get("seed", 0))
        directory = (data_root / relative_dir).resolve()
        if not directory.is_dir():
            raise FileNotFoundError(f"Configured training directory does not exist: {directory}")
        motion_files = tuple(sorted(directory.glob("*.npz")))
        if not motion_files:
            raise FileNotFoundError(f"No NPZ files found in configured training directory: {directory}")
        resolved_jobs.append(
            ResolvedTrainingJob(
                relative_dir=relative_dir,
                directory=directory,
                cuda_device=cuda_device,
                seed=seed,
                job_name_override=str(spec["name"]) if "name" in spec and spec["name"] is not None else None,
                experiment_suffix=str(spec["exp"]) if "exp" in spec and spec["exp"] is not None else None,
                motion_files=motion_files,
            )
        )
    return resolved_jobs


def build_experiment_name(job: ResolvedTrainingJob) -> str:
    """Build the experiment name for one resolved training job.

    The configured prefix is prepended when present so that all launched jobs
    share a common naming convention under the training logs.

    Args:
        job: Resolved training job used to derive the final experiment suffix.

    Returns:
        The experiment name passed to the training script.
    """
    safe_prefix = EXPERIMENT_PREFIX.strip().strip("_")
    suffix = job.experiment_suffix or job.job_name
    return f"{safe_prefix}_{suffix}" if safe_prefix else suffix


def resolve_device(job: ResolvedTrainingJob) -> str:
    """Resolve the CUDA_VISIBLE_DEVICES value for one training job.

    Args:
        job: Resolved training job whose training device should be selected.

    Returns:
        A device selector string used for ``CUDA_VISIBLE_DEVICES``.
    """
    return str(job.cuda_device).replace("cuda:", "")


def collect_train_args(job: ResolvedTrainingJob) -> list[str]:
    """Combine global and per-job extra training arguments.

    Args:
        job: Resolved training job whose extra arguments should be assembled.

    Returns:
        A flat list of additional CLI arguments for the training script.
    """
    extra_args = list(COMMON_TRAIN_ARGS)
    extra_args.extend(PER_JOB_TRAIN_ARGS.get(job.normalized_relative_dir, []))
    extra_args.extend(PER_JOB_TRAIN_ARGS.get(job.job_name, []))
    return extra_args


def format_command(command: list[str], env_overrides: dict[str, str] | None = None) -> str:
    """Format a command list into a shell-safe log string.

    Args:
        command: Command tokens prepared for subprocess execution.
        env_overrides: Optional environment variables prepended in shell form.

    Returns:
        A shell-escaped command string suitable for logs and dry runs.
    """
    prefix = ""
    if env_overrides:
        prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in env_overrides.items()) + " "
    return prefix + shlex.join(command)


def run_log_tail_helper(fd: int, output_path: str, meta_path: str, max_lines: int) -> int:
    """Read from an inherited file descriptor and keep only the latest lines.

    Args:
        fd: Inherited file descriptor connected to a child-process stream.
        output_path: Log file updated with the latest buffered lines.
        meta_path: JSON file storing helper progress metadata.
        max_lines: Maximum number of recent lines kept in the file.

    Returns:
        Zero when the helper finishes consuming the stream.
    """
    lines = deque(maxlen=max_lines)
    path = Path(output_path)
    meta = Path(meta_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    meta.write_text(json.dumps({"lines": 0, "alive": True}), encoding="utf-8")
    total_lines = 0
    with os.fdopen(fd, "r", encoding="utf-8", errors="replace", buffering=1) as stream:
        for raw_line in iter(stream.readline, ""):
            lines.append(raw_line)
            total_lines += 1
            path.write_text("".join(lines), encoding="utf-8")
            meta.write_text(json.dumps({"lines": total_lines, "alive": True}), encoding="utf-8")
    if lines:
        path.write_text("".join(lines), encoding="utf-8")
    meta.write_text(json.dumps({"lines": total_lines, "alive": False}), encoding="utf-8")
    return 0


def start_log_tail_helper(
    read_fd: int,
    output_path: Path,
    meta_path: Path,
    max_lines: int,
) -> subprocess.Popen[str]:
    """Start a detached helper that tails one inherited stream into a log file.

    Args:
        read_fd: Read end of the pipe inherited by the helper process.
        output_path: Destination log file path.
        meta_path: JSON file storing helper progress metadata.
        max_lines: Maximum number of recent lines kept in the file.

    Returns:
        The spawned helper subprocess handle.
    """
    helper_command = [
        sys.executable,
        __file__,
        HELPER_ARG,
        str(read_fd),
        str(output_path),
        str(meta_path),
        str(max_lines),
    ]
    return subprocess.Popen(
        helper_command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        pass_fds=(read_fd,),
        text=True,
    )

def launch_job(
    command: list[str],
    cwd: Path,
    stdout_path: Path,
    stderr_path: Path,
    env_overrides: dict[str, str],
) -> LaunchedJob:
    """Launch one training process and keep only the latest log lines.

    Args:
        command: Fully assembled subprocess command.
        cwd: Working directory used to launch the training process.
        stdout_path: File path that receives the child process stdout stream.
        stderr_path: File path that receives the child process stderr stream.
        env_overrides: Extra environment variables for the training process.

    Returns:
        Runtime handles and metadata paths for the launched job.
    """
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path.write_text("", encoding="utf-8")
    stderr_path.write_text("", encoding="utf-8")
    stdout_meta = stdout_path.with_suffix(".meta.json")
    stderr_meta = stderr_path.with_suffix(".meta.json")
    stdout_read_fd, stdout_write_fd = os.pipe()
    stderr_read_fd, stderr_write_fd = os.pipe()
    try:
        stdout_helper = start_log_tail_helper(stdout_read_fd, stdout_path, stdout_meta, LOG_TAIL_LINES)
        stderr_helper = start_log_tail_helper(stderr_read_fd, stderr_path, stderr_meta, LOG_TAIL_LINES)
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=stdout_write_fd,
            stderr=stderr_write_fd,
            stdin=subprocess.DEVNULL,
            start_new_session=DETACH_PROCESS,
            env={**os.environ, **env_overrides},
        )
    finally:
        os.close(stdout_read_fd)
        os.close(stderr_read_fd)
        os.close(stdout_write_fd)
        os.close(stderr_write_fd)

    return LaunchedJob(
        job=None,  # placeholder replaced by caller
        process=process,
        stdout_helper=stdout_helper,
        stderr_helper=stderr_helper,
        stdout_log=stdout_path,
        stderr_log=stderr_path,
        stdout_meta=stdout_meta,
        stderr_meta=stderr_meta,
    )


def read_log_meta(meta_path: Path) -> dict[str, object]:
    """Read one helper metadata file if it exists.

    Args:
        meta_path: Helper sidecar JSON path.

    Returns:
        Parsed metadata with fallback defaults on missing or transient errors.
    """
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {"lines": 0, "alive": False}


def print_monitor_snapshot(launched_jobs: list[LaunchedJob]) -> None:
    """Print one status snapshot for all launched jobs.

    Args:
        launched_jobs: All launched jobs being monitored by the main process.
    """
    print("[MONITOR]")
    for launched in launched_jobs:
        stdout_meta = read_log_meta(launched.stdout_meta)
        stderr_meta = read_log_meta(launched.stderr_meta)
        train_status = "running" if launched.process.poll() is None else f"exit={launched.process.returncode}"
        print(
            "  "
            f"{launched.job.job_name}: "
            f"train_pid={launched.process.pid} "
            f"cuda_visible_devices={resolve_device(launched.job)} "
            f"stdout_pid={launched.stdout_helper.pid} "
            f"stderr_pid={launched.stderr_helper.pid} "
            f"status={train_status} "
            f"stdout_lines={stdout_meta.get('lines', 0)} "
            f"stderr_lines={stderr_meta.get('lines', 0)}"
        )


def monitor_jobs(launched_jobs: list[LaunchedJob]) -> None:
    """Block in the foreground and periodically report job and helper status.

    Args:
        launched_jobs: All launched jobs that should remain visible in the terminal.
    """
    try:
        while True:
            print_monitor_snapshot(launched_jobs)
            if all(launched.process.poll() is not None for launched in launched_jobs):
                break
            time.sleep(MONITOR_INTERVAL_SEC)
    except KeyboardInterrupt:
        print("[INFO] Monitor interrupted. Child processes continue running.")


def main() -> int:
    """Launch configured training jobs and write PID artifacts.

    This function resolves repository-relative paths, validates each configured
    training folder, launches each job, and writes both a machine-readable
    manifest and a plain-text PID summary for later inspection.

    Returns:
        Zero when the launcher finishes its orchestration flow.
    """
    repo_root = Path(__file__).resolve().parents[2]
    data_root = (repo_root / DATA_ROOT).resolve() if not Path(DATA_ROOT).is_absolute() else Path(DATA_ROOT).resolve()
    train_script = (
        (repo_root / TRAIN_SCRIPT).resolve() if not Path(TRAIN_SCRIPT).is_absolute() else Path(TRAIN_SCRIPT).resolve()
    )
    # Preserve the interpreter entry path exactly as launched. Resolving the
    # symlink can bypass the Isaac Lab environment wrapper and lose packages.
    python_exec = (
        str((repo_root / sys.executable).absolute())
        if not Path(sys.executable).is_absolute()
        else sys.executable
    )
    log_root = (repo_root / LOG_ROOT).resolve() if not Path(LOG_ROOT).is_absolute() else Path(LOG_ROOT).resolve()

    jobs = resolve_training_jobs(data_root)

    launch_id = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{LAUNCHER_NAME}"
    launch_root = log_root / launch_id
    launch_root.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    launched_jobs: list[LaunchedJob] = []
    print(f"[INFO] Repo root: {repo_root}")
    print(f"[INFO] Data root: {data_root}")
    print(f"[INFO] Launch directory: {launch_root}")
    print(f"[INFO] Training task: {TASK}")
    print(f"[INFO] Found {len(jobs)} configured training jobs.")

    for index, job in enumerate(jobs):
        device = resolve_device(job)
        experiment_name = build_experiment_name(job)
        seed = job.seed
        stdout_path = launch_root / "stdout" / f"{job.job_name}.log"
        stderr_path = launch_root / "stderr" / f"{job.job_name}.log"
        env_overrides = {"CUDA_VISIBLE_DEVICES": device}

        command = [python_exec, str(train_script)]
        if HEADLESS:
            command.append("--headless")
        command.extend(
            [
                "--task",
                TASK,
                "--experiment_name",
                experiment_name,
                "--run_name",
                job.job_name,
                "--seed",
                str(seed),
                "--motion_files",
                *[str(path) for path in job.motion_files],
            ]
        )
        command.extend(collect_train_args(job))

        record: dict[str, object] = {
            "index": index,
            "job_name": job.job_name,
            "relative_dir": job.normalized_relative_dir,
            "directory": str(job.directory),
            "num_motion_files": len(job.motion_files),
            "motion_files": [str(path) for path in job.motion_files],
            "device": device,
            "seed": seed,
            "experiment_name": experiment_name,
            "stdout_log": str(stdout_path),
            "stderr_log": str(stderr_path),
            "command": command,
            "command_str": format_command(command, env_overrides),
            "cuda_visible_devices": device,
            "pid": None,
            "status": "planned" if DRY_RUN else "launched",
        }

        print(
            f"[PLAN] {job.job_name}: cuda_visible_devices={device} motions={len(job.motion_files)} "
            f"experiment={experiment_name}"
        )
        if DRY_RUN:
            print(f"       {record['command_str']}")
            records.append(record)
            continue

        launched = launch_job(
            command=command,
            cwd=repo_root,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            env_overrides=env_overrides,
        )
        launched = LaunchedJob(
            job=job,
            process=launched.process,
            stdout_helper=launched.stdout_helper,
            stderr_helper=launched.stderr_helper,
            stdout_log=launched.stdout_log,
            stderr_log=launched.stderr_log,
            stdout_meta=launched.stdout_meta,
            stderr_meta=launched.stderr_meta,
        )
        record["pid"] = launched.process.pid
        record["stdout_helper_pid"] = launched.stdout_helper.pid
        record["stderr_helper_pid"] = launched.stderr_helper.pid
        record["stdout_meta"] = str(launched.stdout_meta)
        record["stderr_meta"] = str(launched.stderr_meta)
        print(
            f"[LAUNCHED] {job.job_name}: "
            f"pid={launched.process.pid} cuda_visible_devices={device} "
            f"stdout_helper={launched.stdout_helper.pid} stderr_helper={launched.stderr_helper.pid}"
        )
        records.append(record)
        launched_jobs.append(launched)

    manifest = {
        "launcher_name": LAUNCHER_NAME,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "repo_root": str(repo_root),
        "data_root": str(data_root),
        "task": TASK,
        "train_script": str(train_script),
        "python": python_exec,
        "dry_run": DRY_RUN,
        "jobs": records,
    }
    manifest_path = launch_root / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)

    pid_summary_path = launch_root / "pids.csv"
    with pid_summary_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "job_name",
                "cuda_visible_devices",
                "pid",
                "stdout_helper_pid",
                "stderr_helper_pid",
                "experiment_name",
                "stdout_log",
            ]
        )
        for record in records:
            writer.writerow(
                [
                    record["job_name"],
                    record["cuda_visible_devices"],
                    record["pid"],
                    record["stdout_helper_pid"],
                    record["stderr_helper_pid"],
                    record["experiment_name"],
                    record["stdout_log"],
                ]
            )

    print(f"[INFO] Manifest written to: {manifest_path}")
    print(f"[INFO] PID summary written to: {pid_summary_path}")
    if not DRY_RUN:
        print("[INFO] Launched PIDs:")
        for record in records:
            print(
                f"  {record['job_name']}: pid={record['pid']} cuda_visible_devices={record['cuda_visible_devices']} "
                f"stdout_helper={record['stdout_helper_pid']} stderr_helper={record['stderr_helper_pid']} "
                f"motions={record['num_motion_files']}"
            )
        monitor_jobs(launched_jobs)
    return 0


if __name__ == "__main__":
    if os.name != "posix":
        raise SystemExit("This launcher currently expects a POSIX environment.")
    if len(sys.argv) >= 6 and sys.argv[1] == HELPER_ARG:
        raise SystemExit(run_log_tail_helper(int(sys.argv[2]), sys.argv[3], sys.argv[4], int(sys.argv[5])))
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    raise SystemExit(main())
