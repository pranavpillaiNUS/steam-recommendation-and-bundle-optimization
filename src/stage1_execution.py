"""Supervise an owned successor worker with sampled process-tree budgets.

This is a polling guard, not an operating-system hard memory limit. Summed
RSS can count shared pages more than once, and peaks or very short-lived
descendants between samples can be missed. Child output goes directly to a
caller-selected private file, so unread pipes cannot block the worker.
"""

from __future__ import annotations

import math
from numbers import Integral, Real
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Sequence

import psutil


def _positive_seconds(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a positive finite number")
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return seconds


def _remember(process: psutil.Process, known: dict[tuple[int, float], psutil.Process]) -> None:
    try:
        known[(process.pid, process.create_time())] = process
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        pass


def _discover(root: psutil.Process, known: dict[tuple[int, float], psutil.Process]) -> None:
    """Retain observed descendants even if the original parent subsequently exits."""
    for ancestor in [root, *list(known.values())]:
        try:
            if ancestor.is_running():
                for child in ancestor.children(recursive=True):
                    _remember(child, known)
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            pass


def _sample(root: psutil.Process, known: dict[tuple[int, float], psutil.Process]) -> int:
    _discover(root, known)
    total = 0
    for process in known.values():
        try:
            # is_running checks create_time as well as PID, avoiding PID reuse.
            if process.is_running():
                total += process.memory_info().rss
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            pass
    return total


def _stop_tree(worker: subprocess.Popen[bytes], root: psutil.Process, known: dict[tuple[int, float], psutil.Process]) -> list[int]:
    """Stop the owned parent and all still-live observed/discovered descendants."""
    try:
        _discover(root, known)
    except psutil.AccessDenied:
        # Even if observation failed, terminate everything already identified.
        pass
    stopped: list[int] = []
    # Stop the parent first so it cannot deliberately restart terminated children.
    if worker.poll() is None:
        worker.kill()
        stopped.append(worker.pid)
    descendants = [process for process in known.values() if process.pid != worker.pid]
    denied: list[int] = []
    for process in reversed(descendants):
        try:
            if process.is_running():
                process.kill()
                stopped.append(process.pid)
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            pass
        except psutil.AccessDenied:
            denied.append(process.pid)
    if descendants:
        _, survivors = psutil.wait_procs(descendants, timeout=3)
        for process in survivors:
            try:
                process.kill()
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                pass
            except psutil.AccessDenied:
                denied.append(process.pid)
    worker.wait()
    if denied:
        raise RuntimeError(f"permission denied while terminating owned descendants: {sorted(set(denied))}")
    return stopped


def run_supervised(
    command: Sequence[str],
    *,
    cwd: Path,
    log_path: Path,
    maximum_wall_seconds: float,
    maximum_rss_bytes: int,
    poll_interval_seconds: float = 0.1,
) -> dict[str, Any]:
    """Launch, monitor, and reap a worker without a shell or visible window.

    Status is ``complete`` for return code zero, ``failed`` for other natural
    exits, ``timeout`` for the wall budget, or ``memory_limit`` for sampled
    process-tree RSS. All limits are validated before launching. A new log is
    required (exclusive creation); an existing log is never overwritten.
    The caller is responsible for choosing an appropriately private log path.
    """
    if isinstance(command, (str, bytes)) or not isinstance(command, Sequence) or not command:
        raise ValueError("command must be a nonempty sequence of argument strings")
    arguments = list(command)
    if any(not isinstance(argument, str) or "\0" in argument for argument in arguments) or not arguments[0]:
        raise ValueError("command arguments must be strings without NUL characters")
    wall_limit = _positive_seconds(maximum_wall_seconds, "maximum_wall_seconds")
    interval = _positive_seconds(poll_interval_seconds, "poll_interval_seconds")
    if isinstance(maximum_rss_bytes, bool) or not isinstance(maximum_rss_bytes, Integral) or maximum_rss_bytes <= 0:
        raise ValueError("maximum_rss_bytes must be a positive integer")
    memory_limit = int(maximum_rss_bytes)
    directory = Path(cwd).resolve()
    if not directory.is_dir():
        raise ValueError("cwd must be an existing directory")
    destination = Path(log_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    known: dict[tuple[int, float], psutil.Process] = {}
    peak = samples = 0
    stopped: list[int] = []
    status = "failed"
    with destination.open("xb") as log:
        worker = subprocess.Popen(
            arguments,
            cwd=directory,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            shell=False,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        try:
            root = psutil.Process(worker.pid)
        except psutil.NoSuchProcess:
            returncode = worker.wait()
            return {
                "status": "complete" if returncode == 0 else "failed",
                "returncode": returncode,
                "pid": worker.pid,
                "elapsed_seconds": time.monotonic() - started,
                "peak_process_tree_rss_bytes": 0,
                "sample_count": 0,
                "observed_process_count": 0,
                "terminated_process_ids": [],
                "log_path": destination.as_posix(),
                "resource_contract": _contract(wall_limit, memory_limit, interval),
            }
        except BaseException:
            if worker.poll() is None:
                worker.kill()
            worker.wait()
            raise
        try:
            _remember(root, known)
            while True:
                peak = max(peak, _sample(root, known))
                samples += 1
                elapsed = time.monotonic() - started
                if peak > memory_limit:
                    status = "memory_limit"
                    break
                returncode = worker.poll()
                if returncode is not None:
                    status = "complete" if returncode == 0 else "failed"
                    break
                if elapsed >= wall_limit:
                    status = "timeout"
                    break
                time.sleep(min(interval, max(0.0, wall_limit - elapsed)))
        finally:
            # Also reap observed descendants left behind by a normally exiting
            # worker; the launched job's lifetime ends when run_supervised ends.
            stopped = _stop_tree(worker, root, known)
    return {
        "status": status,
        "returncode": worker.returncode,
        "pid": worker.pid,
        "elapsed_seconds": time.monotonic() - started,
        "peak_process_tree_rss_bytes": int(peak),
        "sample_count": samples,
        "observed_process_count": len(known),
        "terminated_process_ids": stopped,
        "log_path": destination.as_posix(),
        "resource_contract": _contract(wall_limit, memory_limit, interval),
    }


def _contract(wall_seconds: float, rss_bytes: int, interval_seconds: float) -> dict[str, Any]:
    return {
        "maximum_wall_seconds": wall_seconds,
        "maximum_rss_bytes": rss_bytes,
        "poll_interval_seconds": interval_seconds,
        "rss_measurement": "sampled_sum_of_parent_and_observed_descendant_rss",
        "hard_os_memory_limit": False,
        "between_sample_overshoot_possible": True,
        "short_lived_descendants_may_be_missed": True,
        "shared_pages_may_be_counted_more_than_once": True,
        "shell": False,
    }
