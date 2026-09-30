import json
import subprocess
import sys

import psutil
import pytest

from src.stage1_execution import run_supervised


def _run(tmp_path, script, **limits):
    return run_supervised(
        [sys.executable, "-u", "-c", script],
        cwd=tmp_path,
        log_path=tmp_path / "private-worker.log",
        maximum_wall_seconds=limits.get("maximum_wall_seconds", 10),
        maximum_rss_bytes=limits.get("maximum_rss_bytes", 512 * 1024 * 1024),
        poll_interval_seconds=limits.get("poll_interval_seconds", 0.02),
    )


def test_supervisor_captures_stdout_stderr_and_reaps_clean_exit(tmp_path):
    result = _run(tmp_path, "import sys, time; print('out'); print('err', file=sys.stderr); time.sleep(.15)")
    assert result["status"] == "complete"
    assert result["returncode"] == 0
    assert result["peak_process_tree_rss_bytes"] > 0
    assert result["sample_count"] > 0
    assert not psutil.pid_exists(result["pid"])
    assert {"out", "err"}.issubset((tmp_path / "private-worker.log").read_text().splitlines())
    assert result["resource_contract"]["hard_os_memory_limit"] is False


def test_supervisor_reports_natural_nonzero_exit(tmp_path):
    result = _run(tmp_path, "raise SystemExit(7)")
    assert result["status"] == "failed"
    assert result["returncode"] == 7


def test_supervisor_times_out_and_reaps_owned_worker(tmp_path):
    result = _run(tmp_path, "import time; time.sleep(60)", maximum_wall_seconds=0.2)
    assert result["status"] == "timeout"
    assert result["returncode"] != 0
    assert result["pid"] in result["terminated_process_ids"]
    assert not psutil.pid_exists(result["pid"])


def test_supervisor_stops_worker_when_sampled_memory_exceeds_budget(tmp_path):
    result = _run(tmp_path, "import time; data=bytearray(64*1024*1024); time.sleep(60)", maximum_rss_bytes=48 * 1024 * 1024)
    assert result["status"] == "memory_limit"
    assert result["peak_process_tree_rss_bytes"] > 48 * 1024 * 1024
    assert not psutil.pid_exists(result["pid"])


def test_supervisor_counts_descendant_memory_and_terminates_child(tmp_path):
    child_script = "import time; data=bytearray(96*1024*1024); time.sleep(60)"
    child_command = repr([sys.executable, "-u", "-c", child_script])
    script = f"import subprocess, time; child=subprocess.Popen({child_command}, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)); print(child.pid, flush=True); time.sleep(60)"
    result = _run(tmp_path, script, maximum_rss_bytes=80 * 1024 * 1024)
    child_pid = int((tmp_path / "private-worker.log").read_text().strip())
    assert result["status"] == "memory_limit"
    assert result["observed_process_count"] >= 2
    assert child_pid in result["terminated_process_ids"]
    assert not psutil.pid_exists(child_pid)
    assert not psutil.pid_exists(result["pid"])


@pytest.mark.parametrize("exit_naturally", [True, False])
def test_supervisor_cleans_observed_children_on_natural_exit_and_timeout(tmp_path, exit_naturally):
    child_command = repr([sys.executable, "-c", "import time; time.sleep(60)"])
    sleep_seconds = 0.25 if exit_naturally else 60
    script = f"import subprocess, time; child=subprocess.Popen({child_command}, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)); print(child.pid, flush=True); time.sleep({sleep_seconds})"
    result = _run(tmp_path, script, maximum_wall_seconds=10 if exit_naturally else 0.5)
    assert result["status"] == ("complete" if exit_naturally else "timeout")
    child_pid = int((tmp_path / "private-worker.log").read_text().strip())
    assert child_pid in result["terminated_process_ids"]
    assert not psutil.pid_exists(child_pid)


def test_supervisor_reaps_worker_after_monitor_exception(tmp_path, monkeypatch):
    import src.stage1_execution as execution
    actual_popen = subprocess.Popen
    workers = []
    def recording_popen(*args, **kwargs):
        worker = actual_popen(*args, **kwargs)
        workers.append(worker)
        return worker
    def fail_monitor(*args, **kwargs):
        raise RuntimeError("injected monitoring failure")
    monkeypatch.setattr(subprocess, "Popen", recording_popen)
    monkeypatch.setattr(execution, "_sample", fail_monitor)
    with pytest.raises(RuntimeError, match="injected"):
        _run(tmp_path, "import time; time.sleep(60)")
    assert len(workers) == 1
    assert workers[0].returncode is not None
    assert not psutil.pid_exists(workers[0].pid)


@pytest.mark.parametrize("setting,value", [
    ("maximum_wall_seconds", float("nan")),
    ("maximum_wall_seconds", float("inf")),
    ("maximum_wall_seconds", 0),
    ("maximum_wall_seconds", True),
    ("maximum_rss_bytes", 0),
    ("maximum_rss_bytes", 12.5),
    ("maximum_rss_bytes", True),
    ("poll_interval_seconds", -1),
    ("poll_interval_seconds", float("nan")),
])
def test_invalid_limits_are_rejected_before_process_launch(tmp_path, monkeypatch, setting, value):
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid settings must not launch a process")
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    with pytest.raises(ValueError):
        _run(tmp_path, "pass", **{setting: value})
    assert not (tmp_path / "private-worker.log").exists()


@pytest.mark.parametrize("command", [[], "python command", [1], ["python", "bad\0value"]])
def test_invalid_command_is_rejected_before_launch(tmp_path, command):
    with pytest.raises(ValueError):
        run_supervised(command, cwd=tmp_path, log_path=tmp_path / "worker.log", maximum_wall_seconds=1, maximum_rss_bytes=100)


def test_existing_private_log_is_preserved(tmp_path):
    path = tmp_path / "private-worker.log"
    path.write_text("existing evidence")
    with pytest.raises(FileExistsError):
        _run(tmp_path, "pass")
    assert path.read_text() == "existing evidence"


def test_supervisor_uses_literal_arguments_without_shell_expansion(tmp_path):
    argument = "$HOME; literal `value` & text"
    result = run_supervised(
        [sys.executable, "-c", "import sys, json; print(json.dumps(sys.argv[1]))", argument],
        cwd=tmp_path,
        log_path=tmp_path / "worker.log",
        maximum_wall_seconds=10,
        maximum_rss_bytes=512 * 1024 * 1024,
    )
    assert result["status"] == "complete"
    assert json.loads((tmp_path / "worker.log").read_text()) == argument
