import json
from pathlib import Path
import shutil

import pytest

from src.stage1_public_verify import canonical_json_bytes, semantic_sha256
from src.stage1_successor import (
    CycleContext, OpaqueValidationMask, VerifiedRunStore, _inside,
    design_input_inventory, execute_job, main, run_development_job, run_smoke,
    supervised_development_job, synthetic_design_data,
)


def _store(tmp_path):
    dependency = tmp_path / "config.json"
    dependency.write_text('{"version":1}', encoding="utf-8")
    store = VerifiedRunStore(CycleContext(tmp_path, "s1-test"), "run1")
    store.directory.mkdir(parents=True)
    output = store.directory / "model.bin"
    output.write_bytes(b"model")
    specification = {"algorithm": "test", "required_outputs": ["model"]}
    dependencies = {"config": dependency}
    saved = store.save(specification=specification, dependencies=dependencies,
                       outputs={"model": output}, summary={"score": 0.5})
    return store, specification, dependencies, saved


def test_cache_rechecks_dependencies_outputs_and_specification(tmp_path):
    store, specification, dependencies, saved = _store(tmp_path)
    assert store.load(specification=specification, dependencies=dependencies) == saved
    with pytest.raises(ValueError, match="specification"):
        store.load(specification={"algorithm": "different"}, dependencies=dependencies)
    dependencies["config"].write_text('{"version":2}', encoding="utf-8")
    with pytest.raises(ValueError, match="dependencies"):
        store.load(specification=specification, dependencies=dependencies)
    dependencies["config"].write_text('{"version":1}', encoding="utf-8")
    (store.directory / "model.bin").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash changed"):
        store.load(specification=specification, dependencies=dependencies)
    (store.directory / "model.bin").unlink()
    with pytest.raises(FileNotFoundError, match="missing"):
        store.load(specification=specification, dependencies=dependencies)


def test_cache_rejects_changed_inventory_and_self_hash(tmp_path):
    store, specification, dependencies, saved = _store(tmp_path)
    with pytest.raises(ValueError, match="dependencies"):
        store.load(specification=specification, dependencies={})
    saved["summary"]["score"] = 1.0
    store.path.write_bytes(canonical_json_bytes(saved))
    with pytest.raises(ValueError, match="manifest hash"):
        store.load(specification=specification, dependencies=dependencies)
    saved.pop("manifest_id")
    saved["outputs"] = {}
    saved["manifest_id"] = semantic_sha256(saved)
    store.path.write_bytes(canonical_json_bytes(saved))
    with pytest.raises(ValueError, match="output inventory"):
        store.load(specification=specification, dependencies=dependencies)


def test_completed_run_cannot_be_overwritten(tmp_path):
    store, specification, dependencies, _ = _store(tmp_path)
    with pytest.raises(FileExistsError, match="immutable"):
        store.save(specification=specification, dependencies=dependencies,
                   outputs={"model": store.directory / "model.bin"}, summary={})


@pytest.mark.parametrize("cycle", ["s1-v2-20260814", "s1-v1-anything", "../escape", "C:/data", "", "a/b"])
def test_context_protects_frozen_cycles_and_path_boundaries(tmp_path, cycle):
    with pytest.raises(ValueError):
        CycleContext(tmp_path, cycle)


@pytest.mark.parametrize("path", ["../outside", "C:/outside", "/outside", "a\\b"])
def test_artifact_paths_cannot_escape_root(tmp_path, path):
    with pytest.raises(ValueError):
        _inside(tmp_path, path)


def test_inventory_has_only_design_training_validation_and_feature_inputs():
    root = Path(__file__).resolve().parents[1]
    inventory = design_input_inventory(root)
    assert len(inventory) == 10
    assert not any("assessment" in key or "design_test" in key for key in inventory)
    assert "validation_other_holdout_mask" in inventory


def test_full_synthetic_pipeline_roundtrip_cache_and_corruption(tmp_path, monkeypatch):
    # The temporary dependency substitutes only filesystem placement, not the algorithms.
    (tmp_path / "requirements-frozen.txt").write_text("synthetic fixture", encoding="utf-8")
    first = run_smoke(tmp_path, "s1-smoke-test")
    assert first["status"] == "ok"
    assert first["data"] == "synthetic_only"
    assert set(first["models"]) == {"popularity", "als", "bpr", "bpr_genre"}
    for result in first["models"].values():
        assert not result["cache_reused"]
        assert result["summary"]["evaluation_users"] == 36
        assert 0 <= result["summary"]["mean_ndcg_at_20"] <= 1
        assert set(result["summary"]["timing_seconds"]) == {
            "fit", "serialization", "ranking", "scenario_diagnostics", "total"}
        assert result["summary"]["runtime_environment"]["logical_cpus"] >= 1
        assert result["summary"]["environment_export"]
    from src import stage1_backend, stage1_training

    def forbidden_fit(*args, **kwargs):
        raise AssertionError("cache should skip fitting")

    monkeypatch.setattr(stage1_backend, "fit_implicit_als", forbidden_fit)
    monkeypatch.setattr(stage1_training, "fit_minibatch_bpr", forbidden_fit)
    second = run_smoke(tmp_path, "s1-smoke-test")
    assert all(value["cache_reused"] for value in second["models"].values())
    assert [r["manifest_id"] for r in first["models"].values()] == [
        r["manifest_id"] for r in second["models"].values()]
    synthetic = tmp_path / "outputs/modeling/protected/s1-smoke-test"
    scenario = json.loads((synthetic / "synthetic-als/synthetic_scenarios.json").read_text())
    assert len(scenario["scenarios"]) == 4
    assert not scenario["real_stage1_interface_frozen"]
    assert not list((synthetic / "synthetic-als/scratch").glob("*.f64"))
    (synthetic / "synthetic-popularity/parameters.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash changed"):
        run_smoke(tmp_path, "s1-smoke-test")


def test_cli_requires_explicit_job(tmp_path, capsys):
    assert main(["validate", "--root", str(tmp_path)]) == 1
    report = json.loads(capsys.readouterr().err)
    assert report["status"] == "error"
    assert "--job" in report["error"]


def test_preflight_reports_missing_private_inputs_readably(tmp_path, monkeypatch, capsys):
    from src import stage1_public_verify
    source = Path(__file__).resolve().parents[1] / "outputs/modeling/cycles/s1-v2-20260814"
    cycle = tmp_path / "outputs/modeling/cycles/s1-v2-20260814"
    cycle.mkdir(parents=True)
    for name in ("stage1_split_manifest.json", "item_feature_manifest.json"):
        shutil.copyfile(source / name, cycle / name)
    monkeypatch.setattr(stage1_public_verify, "verify_public_stage1", lambda *args, **kwargs: {"status": "ok"})
    assert main(["preflight", "--root", str(tmp_path)]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "private_inputs_missing"
    assert report["frozen_public_evidence"] == {"status": "ok"}
    assert len(report["missing_design_inputs"]) == report["required_design_inputs"] == 10
    assert report["assessment_or_bundle_outcomes_used"] is False
    assert "public verifier" in report["next_action"]
    assert not (tmp_path / "outputs/modeling/protected").exists()


def test_validate_exits_readably_before_launch_without_private_inputs(tmp_path, monkeypatch, capsys):
    from src import stage1_execution, stage1_successor
    monkeypatch.setattr(stage1_successor, "PROJECT_ROOT", tmp_path)
    (tmp_path / "dev.json").write_text(json.dumps({"jobs": {"popularity": {"family": "popularity"}}}),
                                       encoding="utf-8")
    message = "Restore the listed original hash-matching private files under this root."
    monkeypatch.setattr(stage1_successor, "preflight",
                        lambda root: {"missing_design_inputs": ["private.npz"], "next_action": message})

    def forbidden(*args, **kwargs):
        raise AssertionError("the worker must not launch without its private inputs")

    monkeypatch.setattr(stage1_execution, "run_supervised", forbidden)
    arguments = ["validate", "--root", str(tmp_path), "--config", "dev.json", "--job", "popularity"]
    assert main(arguments) == 1
    report = json.loads(capsys.readouterr().err)
    assert report == {"status": "error", "error_type": "FileNotFoundError", "error": message}
    assert not (tmp_path / "outputs").exists()


@pytest.mark.parametrize("status", ["complete", "timeout"])
def test_supervised_job_records_supervision_and_rejects_unfinished_workers(tmp_path, monkeypatch, status):
    from src import stage1_execution, stage1_successor
    from src.stage1_public_verify import file_sha256
    monkeypatch.setattr(stage1_successor, "PROJECT_ROOT", tmp_path)
    config = tmp_path / "dev.json"
    config.write_text(json.dumps({
        "jobs": {"popularity": {"family": "popularity"}},
        "resource_budget": {"maximum_job_wall_seconds": 10, "maximum_process_tree_rss_bytes": 2 ** 30,
                            "poll_interval_seconds": 0.1},
    }), encoding="utf-8")
    monkeypatch.setattr(stage1_successor, "preflight",
                        lambda root: {"missing_design_inputs": [], "next_action": ""})
    commands = []

    def supervisor(command, *, cwd, log_path, maximum_wall_seconds, maximum_rss_bytes, poll_interval_seconds):
        commands.append(command)
        assert (maximum_wall_seconds, maximum_rss_bytes, poll_interval_seconds) == (10, 2 ** 30, 0.1)
        log_path.write_text("progress\n" + json.dumps({"status": "ok", "job": "popularity"}) + "\n",
                            encoding="utf-8")
        return {"status": status, "peak_process_tree_rss_bytes": 1024}

    monkeypatch.setattr(stage1_execution, "run_supervised", supervisor)
    if status == "complete":
        result = supervised_development_job(tmp_path, "s1-supervision-test", "dev.json", "popularity")
        assert result["job"] == "popularity"
        assert result["supervision"]["configuration_sha256"] == file_sha256(config)
    else:
        with pytest.raises(RuntimeError, match="timeout"):
            supervised_development_job(tmp_path, "s1-supervision-test", "dev.json", "popularity")
    assert commands[0][1:4] == ["-m", "src.stage1_successor", "_worker"]
    records = list((tmp_path / "outputs/modeling/protected/s1-supervision-test/supervision/popularity").glob("*.json"))
    assert len(records) == 1
    record = json.loads(records[0].read_text(encoding="utf-8"))
    assert record["status"] == status
    assert record["peak_process_tree_rss_bytes"] == 1024


def test_worker_appends_access_ledger_before_reading_protected_inputs(tmp_path, monkeypatch):
    from src import stage1_successor
    source = Path(__file__).resolve().parents[1] / "outputs/modeling/cycles/s1-v2-20260814"
    cycle = tmp_path / "outputs/modeling/cycles/s1-v2-20260814"
    cycle.mkdir(parents=True)
    for name in ("stage1_split_manifest.json", "item_feature_manifest.json"):
        shutil.copyfile(source / name, cycle / name)
    (tmp_path / "dev.json").write_text(json.dumps({
        "schema_version": 1, "status": "exploratory_development", "source_cycle": "s1-v2-20260814",
        "assessment_access": False, "jobs": {"popularity": {"family": "popularity"}},
    }), encoding="utf-8")
    monkeypatch.setattr(stage1_successor, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(stage1_successor, "preflight",
                        lambda root: {"missing_design_inputs": [], "next_action": ""})
    ledger = tmp_path / "outputs/modeling/protected/s1-ledger-test/access_ledger.jsonl"

    def load(root):
        assert ledger.is_file(), "the ledger must be written before protected inputs are parsed for modeling"
        raise RuntimeError("stop after the ledger check")

    monkeypatch.setattr(stage1_successor, "load_design_data", load)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="stop after"):
            run_development_job(tmp_path, "s1-ledger-test", "dev.json", "popularity")
    lines = ledger.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    entry = json.loads(lines[0])
    assert (entry["cycle_id"], entry["source_cycle"], entry["job"]) == (
        "s1-ledger-test", "s1-v2-20260814", "popularity")
    assert len(entry["inputs"]) == 10
    assert entry["event"] == "before_design_input_load"
    assert entry["assessment_analytical_access"] is False
    assert entry["design_test_outcomes_scored"] is False
    assert entry["other_holdout_mask_access"] == "opaque_api"
    assert entry["preflight_integrity_hashing"] == "may_include_other_present_frozen_private_artifacts"
    assert not any("assessment" in value["access"] or "design_test" in value["access"]
                   for value in entry["inputs"].values())


def test_run_environment_records_software_export_and_hardware_without_paths():
    import hashlib
    from src.stage1_successor import _environment, _installed_distributions, _runtime_environment
    environment = _environment()
    assert environment == _environment()
    assert {"python", "operating_system", "numpy", "scipy", "pandas", "scikit-learn", "implicit",
            "threadpoolctl", "psutil", "pytest"} <= set(environment)
    export = _installed_distributions()
    assert environment["installed_distribution_count"] == len(export) > 0
    assert environment["installed_distributions_sha256"] == hashlib.sha256(
        "\n".join(export).encode("utf-8")).hexdigest()
    runtime = _runtime_environment()
    assert runtime["logical_cpus"] >= 1 and runtime["memory_bytes"] > 0
    assert runtime["blas_libraries"]
    assert all("filepath" not in entry for entry in runtime["blas_libraries"])


def test_software_environment_change_prevents_completed_run(tmp_path, monkeypatch):
    from src import stage1_successor
    dependency = tmp_path / "fixture.txt"
    dependency.write_text("input", encoding="utf-8")
    initial_export = stage1_successor._installed_distributions()
    exports = iter((initial_export, sorted([*initial_export, "changed-during-run==1.0"])))
    monkeypatch.setattr(stage1_successor, "_installed_distributions", lambda: next(exports))
    context = CycleContext(tmp_path, "s1-environment-change")
    with pytest.raises(ValueError, match="software environment changed"):
        execute_job(context, "popularity", {"family": "popularity"}, synthetic_design_data(),
                    {"input": dependency}, scoring={"user_block_size": 2, "item_block_size": 3})
    directory = context.directory / "popularity"
    assert not (directory / "run.json").exists()
    attempt = json.loads((directory / "attempt.json").read_text())
    assert attempt["status"] == "failed"
    assert attempt["specification"]["environment"] == stage1_successor._environment(initial_export)


def test_mask_provider_uses_opaque_api_and_caches_only_one_bounded_batch(tmp_path, monkeypatch):
    import numpy as np
    from src import stage1_split_artifacts
    calls = []

    def opaque(mask, users, items, **kwargs):
        calls.append(users.copy())
        mask[:, -1] = False
        return mask

    monkeypatch.setattr(stage1_split_artifacts, "mask_validation_other_holdouts", opaque)
    provider = OpaqueValidationMask(tmp_path, tmp_path / "manifest.json",
                                   np.arange(4) + 100, np.arange(7), maximum_mask_bytes=14)
    rows = np.array([1, 3])
    np.testing.assert_array_equal(provider(rows, 0, 3), np.ones((2, 3), dtype=bool))
    np.testing.assert_array_equal(provider(rows, 5, 7), [[True, False], [True, False]])
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0], [101, 103])
    with pytest.raises(MemoryError, match="mask"):
        provider(np.array([0, 1, 2]), 0, 3)


def test_streaming_ranking_applies_mask_callback_and_rejects_masked_target():
    import numpy as np
    from src.stage1_runtime import FactorScoreSource, evaluate_factor_ranking
    source = FactorScoreSource(np.ones((2, 1)), np.arange(5.0).reshape(-1, 1),
                               user_block_size=1, item_block_size=2)
    rows = np.array([0, 1])
    targets = np.array([3, 2])
    exclusions = [np.array([0]), np.array([0])]

    def mask(rows, start, stop):
        return np.broadcast_to(np.arange(start, stop) != 4, (len(rows), stop - start))

    callback = evaluate_factor_ranking(source, rows, targets, exclusions, ks=(1, 2),
                                      candidate_mask_provider=mask)
    reference = evaluate_factor_ranking(source, rows, targets, [[0, 4], [0, 4]], ks=(1, 2))
    assert callback.aggregate == reference.aggregate
    with pytest.raises(ValueError, match="target was excluded"):
        evaluate_factor_ranking(source, rows, np.array([4, 2]), exclusions,
                                candidate_mask_provider=mask)
    with pytest.raises(ValueError, match="invalid block"):
        evaluate_factor_ranking(source, rows, targets, exclusions,
                                candidate_mask_provider=lambda *_: np.ones((1, 1), dtype=int))


def test_failed_attempt_is_recorded_and_cannot_be_silently_retried(tmp_path):
    dependency = tmp_path / "fixture.txt"
    dependency.write_text("input", encoding="utf-8")
    context = CycleContext(tmp_path, "s1-failure-test")
    arguments = dict(context=context, name="job", job={"family": "popularity"},
                     data=synthetic_design_data(), dependencies={"input": dependency},
                     scoring={"user_block_size": 2, "item_block_size": 3},
                     maximum_saved_model_bytes=1)
    with pytest.raises(MemoryError, match="saved model"):
        execute_job(**arguments)
    directory = context.directory / "job"
    assert not (directory / "run.json").exists()
    assert json.loads((directory / "attempt.json").read_text())["status"] == "failed"
    with pytest.raises(FileExistsError, match="attempt already exists"):
        execute_job(**arguments)


def test_empty_genre_job_fails_before_claiming_attempt(tmp_path):
    import scipy.sparse as sp
    data = synthetic_design_data()
    data["genres"] = sp.csr_matrix(data["genres"].shape)
    context = CycleContext(tmp_path, "s1-empty-genre")
    with pytest.raises(ValueError, match="nonempty genre"):
        execute_job(context, "genre", {"family": "feature_sum_bpr_identity_genre"},
                    data, {}, scoring={})
    assert not context.directory.exists()


def test_real_runner_cannot_bind_a_different_checkout_as_executing_code(tmp_path):
    with pytest.raises(ValueError, match="executing repository root"):
        run_development_job(tmp_path, "s1-test", "config.json", "job")
