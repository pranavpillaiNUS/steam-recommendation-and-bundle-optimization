from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.stage1_public_verify import (
    MANIFEST_FILES,
    REQUIRED_OUTPUT_LABELS,
    Stage1VerificationError,
    canonical_json_bytes,
    main,
    semantic_sha256,
    verify_public_stage1,
)


CYCLE_ID = "s1-v2-20260814"


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def _seal(value: dict, field: str = "manifest_id") -> dict:
    result = dict(value)
    result[field] = semantic_sha256(result)
    return result


def _file_record(root: Path, path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _build_public_graph(root: Path, *, bad_training_protocol: bool = False) -> dict:
    cycle = root / "outputs" / "modeling" / "cycles" / CYCLE_ID
    cycle.mkdir(parents=True)

    public_artifact = root / "artifacts" / "public.bin"
    public_artifact.parent.mkdir(parents=True)
    public_artifact.write_bytes(b"public-evidence\n")
    public_record = _file_record(root, public_artifact)
    absent_raw = {
        "path": "data/raw/private_source.json",
        "size_bytes": 7,
        "sha256": "a" * 64,
    }
    absent_protected = {
        "path": f"outputs/modeling/protected/{CYCLE_ID}/model.npz",
        "size_bytes": 11,
        "sha256": "b" * 64,
    }

    preference_hash = "1" * 64
    ranking_hash = "2" * 64
    protocol_id = semantic_sha256(
        {
            "model_config_sha256": preference_hash,
            "ranking_config_sha256": ranking_hash,
        }
    )
    interaction_set_id = "3" * 64
    split_set_id = "4" * 64
    feature_set_id = "5" * 64
    specification_id = "6" * 64
    family = "implicit_als"
    configuration_id = "als_test_configuration"
    training_seed = 104729

    manifests: dict[str, dict] = {}
    manifests["source"] = _seal(
        {
            "schema_version": 1,
            "artifact": "source",
            "cycle_id": CYCLE_ID,
            "source_set_id": "7" * 64,
            "inputs": {"public": public_record, "raw": absent_raw},
        }
    )
    manifests["protocol"] = {
        "schema_version": 1,
        "artifact": "protocol",
        "cycle_id": CYCLE_ID,
        "configs": {
            "preference_models": {"semantic_sha256": preference_hash},
            "ranking_evaluation": {"semantic_sha256": ranking_hash},
        },
        "protocol_id": protocol_id,
    }
    manifests["interactions"] = _seal(
        {
            "schema_version": 1,
            "artifact": "interactions",
            "cycle_id": CYCLE_ID,
            "protocol_id": protocol_id,
            "interaction_set_id": interaction_set_id,
        }
    )
    manifests["splits"] = _seal(
        {
            "schema_version": 1,
            "artifact": "splits",
            "cycle_id": CYCLE_ID,
            "protocol_id": protocol_id,
            "interaction_set_id": interaction_set_id,
            "split_set_id": split_set_id,
        }
    )
    manifests["features"] = _seal(
        {
            "schema_version": 1,
            "artifact": "features",
            "cycle_id": CYCLE_ID,
            "protocol_id": protocol_id,
            "interaction_set_id": interaction_set_id,
            "split_set_id": split_set_id,
            "feature_set_id": feature_set_id,
        }
    )
    manifests["estimator"] = _seal(
        {
            "schema_version": 1,
            "artifact": "estimator",
            "cycle_id": CYCLE_ID,
            "protocol_id": protocol_id,
            "interaction_set_id": interaction_set_id,
            "split_set_id": split_set_id,
            "feature_set_id": feature_set_id,
            "specification_id": specification_id,
        }
    )
    manifests["backend_spike"] = _seal(
        {
            "schema_version": 1,
            "artifact": "backend_spike",
            "cycle_id": CYCLE_ID,
            "protocol_id": protocol_id,
            "split_set_id": split_set_id,
            "feature_set_id": feature_set_id,
            "status": "pass",
        }
    )

    validation_log = _seal(
        {
            "schema_version": 1,
            "cycle_id": CYCLE_ID,
            "protocol_id": protocol_id,
            "family": family,
            "configuration_id": configuration_id,
            "training_seed": training_seed,
            "status": "complete",
            "model_artifact": absent_protected,
        },
        "run_id",
    )
    validation_path = cycle / "validation_runs" / "single_run.json"
    _write_json(validation_path, validation_log)

    manifests["training"] = _seal(
        {
            "schema_version": 1,
            "artifact": "training",
            "cycle_id": CYCLE_ID,
            "protocol_id": "f" * 64 if bad_training_protocol else protocol_id,
            "interaction_set_id": interaction_set_id,
            "split_set_id": split_set_id,
            "feature_set_id": feature_set_id,
            "estimator_specification_id": specification_id,
            "backend_spike_manifest_id": manifests["backend_spike"]["manifest_id"],
            "selection": {family: configuration_id},
            "runs": [validation_log],
        }
    )
    manifests["admission"] = _seal(
        {
            "schema_version": 1,
            "artifact": "admission",
            "cycle_id": CYCLE_ID,
            "protocol_id": protocol_id,
            "training_manifest_id": manifests["training"]["manifest_id"],
            "selection": {family: configuration_id},
            "admitted_families": [family],
        },
        "admission_id",
    )
    manifests["gate1"] = _seal(
        {
            "schema_version": 1,
            "artifact": "gate1",
            "cycle_id": CYCLE_ID,
            "admission_id": manifests["admission"]["admission_id"],
            "admitted_families": [family],
            "status": "pass",
            "inputs": {
                "training_manifest_id": manifests["training"]["manifest_id"],
                "split_set_id": split_set_id,
                "feature_set_id": feature_set_id,
            },
        }
    )

    production_log = _seal(
        {
            "schema_version": 1,
            "cycle_id": CYCLE_ID,
            "family": family,
            "configuration_id": configuration_id,
            "training_seed": training_seed,
            "artifacts": {"model": absent_protected},
        },
        "run_id",
    )
    production_path = cycle / "production_runs" / "single_run.json"
    _write_json(production_path, production_log)
    manifests["production"] = _seal(
        {
            "schema_version": 1,
            "artifact": "production",
            "cycle_id": CYCLE_ID,
            "admission_id": manifests["admission"]["admission_id"],
            "gate1_manifest_id": manifests["gate1"]["manifest_id"],
            "admitted_families": [family],
            "status": "complete",
            "inputs": {
                "gate1_manifest_id": manifests["gate1"]["manifest_id"],
                "training_manifest_id": manifests["training"]["manifest_id"],
                "interaction_set_id": interaction_set_id,
                "split_set_id": split_set_id,
            },
            "runs": [production_log],
        }
    )
    manifests["pseudo_utility_gate2"] = _seal(
        {
            "schema_version": 1,
            "artifact": "pseudo_utility_gate2",
            "cycle_id": CYCLE_ID,
            "production_manifest_id": manifests["production"]["manifest_id"],
            "gate2": {"status": "pass"},
            "run_parameters": [
                {
                    "production_run_id": production_log["run_id"],
                    "family": family,
                    "training_seed": training_seed,
                }
            ],
        }
    )

    manifest_entries: dict[str, dict] = {}
    for label, filename in MANIFEST_FILES.items():
        path = cycle / filename
        _write_json(path, manifests[label])
        entry = _file_record(root, path)
        entry["semantic_id"] = (
            manifests[label].get("manifest_id")
            or manifests[label].get("admission_id")
            or manifests[label]["protocol_id"]
        )
        manifest_entries[label] = entry

    output_entries: dict[str, dict] = {}
    for label in sorted(REQUIRED_OUTPUT_LABELS):
        path = root / "outputs" / "public" / f"{label}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{label}\n", encoding="utf-8", newline="\n")
        output_entries[label] = _file_record(root, path)

    evidence = _seal(
        {
            "schema_version": 1,
            "artifact": "stage1_complete_evidence_package",
            "cycle_id": CYCLE_ID,
            "status": "complete",
            "gate1_status": "pass",
            "gate2_status": "pass",
            "admitted_families": [family],
            "manifests": manifest_entries,
            "outputs": output_entries,
            "validation_run_logs": [
                {
                    **_file_record(root, validation_path),
                    "run_id": validation_log["run_id"],
                }
            ],
        }
    )
    evidence_path = cycle / "stage1_evidence_manifest.json"
    _write_json(evidence_path, evidence)
    return {
        "cycle": cycle,
        "evidence": evidence_path,
        "public_artifact": public_artifact,
        "production_log": production_path,
    }


def _rewrite_evidence(path: Path, transform) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    value.pop("manifest_id")
    transform(value)
    _write_json(path, _seal(value))


def test_verifies_complete_graph_and_allows_only_private_absences(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _build_public_graph(tmp_path)

    report = verify_public_stage1(tmp_path, cycle_id=CYCLE_ID)

    assert report["status"] == "ok"
    assert report["manifests_verified"] == 12
    assert report["validation_run_logs_verified"] == 1
    assert report["production_run_logs_verified"] == 1
    assert report["allowed_absent_private_references"] >= 2

    assert main(["--root", str(tmp_path), "--cycle-id", CYCLE_ID]) == 0
    cli_report = json.loads(capsys.readouterr().out)
    assert cli_report["status"] == "ok"


def test_rejects_evidence_manifest_self_hash_tampering(tmp_path: Path) -> None:
    graph = _build_public_graph(tmp_path)
    evidence = json.loads(graph["evidence"].read_text(encoding="utf-8"))
    evidence["status"] = "tampered"
    _write_json(graph["evidence"], evidence)

    with pytest.raises(Stage1VerificationError, match="semantic hash mismatch"):
        verify_public_stage1(tmp_path, cycle_id=CYCLE_ID)


def test_rejects_top_level_output_hash_or_size_mismatch(tmp_path: Path) -> None:
    graph = _build_public_graph(tmp_path)
    output = tmp_path / "outputs" / "public" / "summary.txt"
    output.write_bytes(b"changed after evidence freeze\n")

    with pytest.raises(Stage1VerificationError, match="artifact hash mismatch"):
        verify_public_stage1(tmp_path, cycle_id=CYCLE_ID)


def test_rejects_path_escape_even_when_evidence_is_resealed(tmp_path: Path) -> None:
    graph = _build_public_graph(tmp_path)

    def escape(value: dict) -> None:
        value["outputs"]["summary"]["path"] = "../outside.txt"

    _rewrite_evidence(graph["evidence"], escape)

    with pytest.raises(Stage1VerificationError, match="safe repository-relative path"):
        verify_public_stage1(tmp_path, cycle_id=CYCLE_ID)


def test_rejects_missing_public_recursive_reference(tmp_path: Path) -> None:
    graph = _build_public_graph(tmp_path)
    graph["public_artifact"].unlink()

    with pytest.raises(Stage1VerificationError, match="required recorded artifact is missing"):
        verify_public_stage1(tmp_path, cycle_id=CYCLE_ID)


def test_rejects_cross_manifest_protocol_mismatch(tmp_path: Path) -> None:
    _build_public_graph(tmp_path, bad_training_protocol=True)

    with pytest.raises(Stage1VerificationError, match="training.protocol_id mismatch"):
        verify_public_stage1(tmp_path, cycle_id=CYCLE_ID)


def test_rejects_missing_production_run_log(tmp_path: Path) -> None:
    graph = _build_public_graph(tmp_path)
    graph["production_log"].unlink()

    with pytest.raises(Stage1VerificationError, match="production run-log ID inventory"):
        verify_public_stage1(tmp_path, cycle_id=CYCLE_ID)
