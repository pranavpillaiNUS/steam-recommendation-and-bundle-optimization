"""Sample a running Stage 1 process into a compact reproducible resource trace."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Sequence

import psutil


def monitor_process(pid: int, output: Path, interval_seconds: float = 5.0) -> None:
    process = psutil.Process(int(pid))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "observed_at_utc",
                "pid",
                "rss_bytes",
                "vms_bytes",
                "thread_count",
                "cpu_user_seconds",
                "cpu_system_seconds",
            ),
        )
        writer.writeheader()
        while True:
            try:
                memory = process.memory_info()
                cpu = process.cpu_times()
                writer.writerow(
                    {
                        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
                        "pid": process.pid,
                        "rss_bytes": memory.rss,
                        "vms_bytes": memory.vms,
                        "thread_count": process.num_threads(),
                        "cpu_user_seconds": cpu.user,
                        "cpu_system_seconds": cpu.system,
                    }
                )
                handle.flush()
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                break
            time.sleep(float(interval_seconds))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Monitor one Stage 1 process")
    parser.add_argument("pid", type=int)
    parser.add_argument("output", type=Path)
    parser.add_argument("--interval", type=float, default=5.0)
    args = parser.parse_args(argv)
    monitor_process(args.pid, args.output, args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
