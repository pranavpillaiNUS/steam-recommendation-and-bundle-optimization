"""Fit and freeze S1.11 pseudo-utility scenarios without Stage 2 outcomes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from src.features import load_feature_artifacts
from src.pseudo_utility import fit_global_parameters, transform
from src.stage1_backend import load_parameter_archive
from src.stage1_protocol import canonical_json_bytes, file_sha256, load_json, semantic_sha256
from src.stage1_split_artifacts import load_design_training_artifacts, load_evaluation_user_sample


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CYCLE_ID = "s1-v2-20260814"
CONFIG = PROJECT_ROOT / "configs" / "cycles" / CYCLE_ID / "pseudo_utility_scenarios.json"
CYCLE_DIR = PROJECT_ROOT / "outputs" / "modeling" / "cycles" / CYCLE_ID
PROTECTED_DIR = PROJECT_ROOT / "outputs" / "modeling" / "protected" / CYCLE_ID
SPLIT_MANIFEST = CYCLE_DIR / "stage1_split_manifest.json"
FEATURE_DIR = PROTECTED_DIR / "stage1_features"
PRODUCTION_MANIFEST = CYCLE_DIR / "stage1_production_manifest.json"
DIAGNOSTIC_TABLE = CYCLE_DIR / "stage1_pseudo_utility_diagnostics.csv"
OUTPUT_MANIFEST = CYCLE_DIR / "pseudo_utility_scenarios_manifest.json"


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_bytes(canonical_json_bytes(value))
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _verify_self_hash(value: Mapping[str, Any], field: str) -> None:
    unsigned = dict(value)
    claimed = unsigned.pop(field, None)
    if claimed != semantic_sha256(unsigned):
        raise ValueError(f"{field} mismatch")


def _item_vectors(family: str, shared: Mapping[str, np.ndarray], genres: Any) -> tuple[np.ndarray, np.ndarray]:
    if family == "implicit_als":
        return np.asarray(shared["item_factors"], dtype=np.float64), np.zeros(shared["item_factors"].shape[0], dtype=np.float64)
    vectors = np.asarray(shared["identity_factors"], dtype=np.float64).copy()
    if family == "feature_sum_bpr_identity_genre":
        vectors += genres @ np.asarray(shared["feature_factors"], dtype=np.float64)
    return vectors, np.asarray(shared["item_bias"], dtype=np.float64)


def _score(user_factors: np.ndarray, item_vectors: np.ndarray, bias: np.ndarray, batch_size: int = 128) -> np.ndarray:
    result = np.empty((user_factors.shape[0], item_vectors.shape[0]), dtype=np.float64)
    for start in range(0, user_factors.shape[0], batch_size):
        stop = min(start + batch_size, user_factors.shape[0])
        result[start:stop] = np.asarray(user_factors[start:stop], dtype=np.float64) @ item_vectors.T + bias
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("production scores are nonfinite")
    return result


def freeze_pseudo_utilities(
    *, output_path: str | Path = OUTPUT_MANIFEST
) -> dict[str, Any]:
    destination = Path(output_path).resolve()
    if destination.exists():
        saved = load_json(destination)
        _verify_self_hash(saved, "manifest_id")
        return saved
    config = load_json(CONFIG)
    production = load_json(PRODUCTION_MANIFEST)
    _verify_self_hash(production, "manifest_id")
    if production.get("status") != "complete" or production.get("access_boundary", {}).get("stage2_objectives_or_bundle_outcomes") != "not_accessed":
        raise ValueError("production is not eligible for Gate 2")
    training = load_design_training_artifacts(manifest_path=SPLIT_MANIFEST)
    sample = load_evaluation_user_sample(manifest_path=SPLIT_MANIFEST)
    sample_rows = np.searchsorted(training.user_ids, sample["user_ids"])
    if not np.array_equal(training.user_ids[sample_rows], sample["user_ids"]):
        raise ValueError("pseudo-utility parameter users are misaligned")
    features = load_feature_artifacts(FEATURE_DIR)
    item_rows = np.searchsorted(features.item_ids, training.item_ids)
    if not np.array_equal(features.item_ids[item_rows], training.item_ids):
        raise ValueError("pseudo-utility item features are misaligned")
    genres = features.genre[item_rows]
    scenario_ids = [row["scenario_id"] for row in config["scenarios"]]
    run_parameters: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for run in production["runs"]:
        shared_entry = run["artifacts"]["shared_parameters"]
        assessment_entry = run["artifacts"]["assessment_user_factors"]
        shared = load_parameter_archive(PROJECT_ROOT / shared_entry["path"], expected_sha256=shared_entry["sha256"])
        assessment = load_parameter_archive(PROJECT_ROOT / assessment_entry["path"], expected_sha256=assessment_entry["sha256"])
        item_vectors, bias = _item_vectors(str(run["family"]), shared, genres)
        design_scores = _score(shared["design_user_factors"][sample_rows], item_vectors, bias)
        parameters = fit_global_parameters(design_scores)
        score_sample_hash = hashlib.sha256(np.ascontiguousarray(design_scores, dtype="<f8").tobytes()).hexdigest()
        assessment_rows = min(128, assessment["user_factors"].shape[0])
        diagnostic_scores = _score(assessment["user_factors"][:assessment_rows], item_vectors, bias)
        for scenario_id in scenario_ids:
            values = transform(scenario_id, diagnostic_scores, parameters)
            diagnostics.append(
                {
                    "production_run_id": run["run_id"],
                    "family": run["family"],
                    "training_seed": run["training_seed"],
                    "scenario_id": scenario_id,
                    "diagnostic_users": assessment_rows,
                    "diagnostic_values": int(values.size),
                    "minimum": float(values.min()),
                    "maximum": float(values.max()),
                    "mean": float(values.mean(dtype=np.float64)),
                    "nonfinite_count": int(np.count_nonzero(~np.isfinite(values))),
                    "negative_count": int(np.count_nonzero(values < 0.0)),
                }
            )
        run_parameters.append(
            {
                "production_run_id": run["run_id"],
                "family": run["family"],
                "training_seed": run["training_seed"],
                "global_parameters": parameters,
                "parameter_score_sample_shape": list(design_scores.shape),
                "parameter_score_sample_sha256": score_sample_hash,
                "eligible_item_count": int(item_vectors.shape[0]),
                "eligible_item_ids_sha256": hashlib.sha256(np.ascontiguousarray(training.item_ids, dtype="<i8").tobytes()).hexdigest(),
                "source_artifact_sha256": {"shared_parameters": shared_entry["sha256"], "assessment_user_factors": assessment_entry["sha256"]},
            }
        )
    fields = ["production_run_id", "family", "training_seed", "scenario_id", "diagnostic_users", "diagnostic_values", "minimum", "maximum", "mean", "nonfinite_count", "negative_count"]
    with DIAGNOSTIC_TABLE.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(diagnostics)
    if any(row["nonfinite_count"] or row["negative_count"] for row in diagnostics):
        raise AssertionError("Gate 2 transformation diagnostic failed")
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "artifact": "stage1_pseudo_utility_scenarios_and_gate2",
        "cycle_id": CYCLE_ID,
        "production_manifest_id": production["manifest_id"],
        "scenario_config": config,
        "scenario_config_sha256": file_sha256(CONFIG),
        "transformation_code_sha256": file_sha256(PROJECT_ROOT / "src" / "pseudo_utility.py"),
        "runner_sha256": file_sha256(Path(__file__)),
        "run_parameters": run_parameters,
        "diagnostics": {
            "path": DIAGNOSTIC_TABLE.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix(),
            "size_bytes": DIAGNOSTIC_TABLE.stat().st_size,
            "sha256": file_sha256(DIAGNOSTIC_TABLE),
            "rows": len(diagnostics),
        },
        "gate2": {
            "status": "pass",
            "all_scenarios_deterministic": True,
            "all_diagnostics_finite_and_nonnegative": True,
            "full_dense_catalogue_matrix_saved": False,
            "frozen_before_bundle_objective_access": True,
            "true_utility_or_money_claimed": False,
        },
        "access_boundary": {
            "assessment_latent_scores": "bounded_blocks_used_for_diagnostics",
            "stage2_objectives_or_bundle_outcomes": "not_accessed",
        },
    }
    manifest["manifest_id"] = semantic_sha256(manifest)
    _atomic_json(destination, manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Freeze pseudo-utility scenarios and Gate 2")
    parser.parse_args(argv)
    result = freeze_pseudo_utilities()
    print(json.dumps({"gate2": result["gate2"]["status"], "manifest_id": result["manifest_id"], "runs": len(result["run_parameters"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
