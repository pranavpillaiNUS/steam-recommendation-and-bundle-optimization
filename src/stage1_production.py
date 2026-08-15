"""Run S1.10 production refits and assessment-user fold-in after Gate 1."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np
import scipy.sparse as sp

from src.features import load_feature_artifacts
from src.interactions import SparseInteractionData, load_sparse_interactions
from src.stage1_backend import (
    fit_feature_sum_bpr,
    fit_implicit_als,
    fold_in_als,
    fold_in_bpr_user,
    load_parameter_archive,
    save_parameter_archive,
)
from src.stage1_gate1 import _load_split_artifact
from src.stage1_protocol import (
    canonical_json_bytes,
    enumerate_als_configurations,
    enumerate_bpr_configurations,
    file_sha256,
    load_json,
    semantic_sha256,
)
from src.stage1_split_artifacts import load_design_training_artifacts, load_validation_targets


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CYCLE_ID = "s1-v2-20260814"
CONFIG_DIR = PROJECT_ROOT / "configs" / "cycles" / CYCLE_ID
CYCLE_DIR = PROJECT_ROOT / "outputs" / "modeling" / "cycles" / CYCLE_ID
PROTECTED_DIR = PROJECT_ROOT / "outputs" / "modeling" / "protected" / CYCLE_ID
MODEL_CONFIG = CONFIG_DIR / "preference_models.json"
INTERACTION_MANIFEST = CYCLE_DIR / "stage1_interaction_manifest.json"
SPLIT_MANIFEST = CYCLE_DIR / "stage1_split_manifest.json"
FEATURE_DIR = PROTECTED_DIR / "stage1_features"
TRAINING_MANIFEST = CYCLE_DIR / "stage1_training_manifest.json"
ADMISSION_MANIFEST = CYCLE_DIR / "stage1_validation_admission_manifest.json"
GATE1_MANIFEST = CYCLE_DIR / "stage1_gate1_manifest.json"
INTERACTION_DIR = PROTECTED_DIR / "stage1_interactions"
OUTPUT_DIR = PROTECTED_DIR / "production"
RUN_DIR = CYCLE_DIR / "production_runs"
OUTPUT_MANIFEST = CYCLE_DIR / "stage1_production_manifest.json"


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


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


def _restore_matrix(
    base: sp.csr_matrix,
    user_ids: np.ndarray,
    item_ids: np.ndarray,
    target_users: np.ndarray,
    target_items: np.ndarray,
    values: np.ndarray,
) -> sp.csr_matrix:
    rows = np.searchsorted(user_ids, target_users)
    columns = np.searchsorted(item_ids, target_items)
    if not np.array_equal(user_ids[rows], target_users) or not np.array_equal(item_ids[columns], target_items):
        raise ValueError("production restoration alignment failed")
    additions = sp.csr_matrix((np.asarray(values, dtype=np.float32), (rows, columns)), shape=base.shape)
    result = (base + additions).tocsr()
    result.sum_duplicates()
    result.sort_indices()
    result.eliminate_zeros()
    return result


def _aligned_restored_playtime(
    base: sp.csr_matrix,
    ownership: sp.csr_matrix,
    user_ids: np.ndarray,
    item_ids: np.ndarray,
    target_users: np.ndarray,
    target_items: np.ndarray,
    target_values: np.ndarray,
) -> sp.csr_matrix:
    result = sp.csr_matrix(
        (
            np.zeros(ownership.nnz, dtype=np.float32),
            ownership.indices.copy(),
            ownership.indptr.copy(),
        ),
        shape=ownership.shape,
    )
    for row in range(base.shape[0]):
        base_start, base_stop = base.indptr[row : row + 2]
        result_start, result_stop = result.indptr[row : row + 2]
        locations = np.searchsorted(
            result.indices[result_start:result_stop],
            base.indices[base_start:base_stop],
        )
        result.data[result_start + locations] = base.data[base_start:base_stop]
    rows = np.searchsorted(user_ids, target_users)
    columns = np.searchsorted(item_ids, target_items)
    for row, column, value in zip(rows, columns, target_values):
        start, stop = result.indptr[row : row + 2]
        location = int(np.searchsorted(result.indices[start:stop], column))
        if location >= stop - start or result.indices[start + location] != column:
            raise ValueError("restored playtime edge is absent from ownership")
        result.data[start + location] = float(value)
    return result


def _production_design(training: SparseInteractionData, validation: Mapping[str, np.ndarray], validation_diagnostics: Mapping[str, np.ndarray], test: Mapping[str, np.ndarray]) -> SparseInteractionData:
    users = np.concatenate((validation["user_ids"], test["user_ids"]))
    items = np.concatenate((validation["item_ids"], test["item_ids"]))
    ownership = _restore_matrix(training.ownership, training.user_ids, training.item_ids, users, items, np.ones(users.size))
    ownership.data[:] = 1.0
    forever = _aligned_restored_playtime(
        training.playtime_forever, ownership, training.user_ids, training.item_ids,
        users, items,
        np.concatenate((validation_diagnostics["playtime_forever"], test["playtime_forever"])),
    )
    recent = _aligned_restored_playtime(
        training.playtime_2weeks, ownership, training.user_ids, training.item_ids,
        users, items,
        np.concatenate((validation_diagnostics["playtime_2weeks"], test["playtime_2weeks"])),
    )
    return SparseInteractionData(ownership=ownership, playtime_forever=forever, playtime_2weeks=recent, user_ids=training.user_ids, item_ids=training.item_ids)


def _assessment_data(root: Path, interaction_manifest: Mapping[str, Any], split: Mapping[str, Any], warm_item_ids: np.ndarray) -> SparseInteractionData:
    hashes = {name: entry["sha256"] for name, entry in interaction_manifest["artifacts"].items()}
    canonical = load_sparse_interactions(INTERACTION_DIR, prefix="canonical", expected_file_hashes=hashes)
    outer = _load_split_artifact(root, split, "outer_user_split")
    assessment_ids = outer["user_ids"][outer["split_code"] == 2]
    user_rows = np.searchsorted(canonical.user_ids, assessment_ids)
    item_columns = np.searchsorted(canonical.item_ids, warm_item_ids)
    if not np.array_equal(canonical.user_ids[user_rows], assessment_ids) or not np.array_equal(canonical.item_ids[item_columns], warm_item_ids):
        raise ValueError("assessment/canonical interaction alignment changed")
    return SparseInteractionData(
        ownership=canonical.ownership[user_rows][:, item_columns].tocsr(),
        playtime_forever=canonical.playtime_forever[user_rows][:, item_columns].tocsr(),
        playtime_2weeks=canonical.playtime_2weeks[user_rows][:, item_columns].tocsr(),
        user_ids=assessment_ids,
        item_ids=warm_item_ids,
    )


def _warm_genres(item_ids: np.ndarray) -> sp.csr_matrix:
    features = load_feature_artifacts(FEATURE_DIR)
    columns = np.searchsorted(features.item_ids, item_ids)
    if not np.array_equal(features.item_ids[columns], item_ids):
        raise ValueError("production feature projection changed")
    return features.genre[columns].tocsr()


def run_production(
    *, project_root: str | Path = PROJECT_ROOT, output_path: str | Path = OUTPUT_MANIFEST
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    destination = Path(output_path).resolve()
    if destination.exists():
        saved = load_json(destination)
        _verify_self_hash(saved, "manifest_id")
        return saved
    model_config = load_json(MODEL_CONFIG)
    interaction_manifest = load_json(INTERACTION_MANIFEST)
    split = load_json(SPLIT_MANIFEST)
    training_manifest = load_json(TRAINING_MANIFEST)
    admission = load_json(ADMISSION_MANIFEST)
    gate1 = load_json(GATE1_MANIFEST)
    _verify_self_hash(training_manifest, "manifest_id")
    _verify_self_hash(admission, "admission_id")
    _verify_self_hash(gate1, "manifest_id")
    if gate1.get("status") != "pass" or gate1.get("admission_id") != admission["admission_id"]:
        raise ValueError("Gate 1 is not closed")
    # Assessment identifiers and histories are first opened below, after Gate 1.
    training = load_design_training_artifacts(project_root=root, manifest_path=SPLIT_MANIFEST)
    validation = load_validation_targets(manifest_path=SPLIT_MANIFEST)
    validation_diagnostics = _load_split_artifact(root, split, "validation_target_diagnostics")
    test = _load_split_artifact(root, split, "design_test_targets")
    production = _production_design(training, validation, validation_diagnostics, test)
    assessment = _assessment_data(root, interaction_manifest, split, production.item_ids)
    genres = _warm_genres(production.item_ids)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    seeds = [int(seed) for seed in model_config["training_seeds"]]
    selected = training_manifest["selection"]
    admitted = list(admission["admitted_families"])
    results: list[dict[str, Any]] = []

    for family in admitted:
        if family == "implicit_als":
            configuration = next(value for value in enumerate_als_configurations(model_config) if value["configuration_id"] == selected[family])
        elif family in ("feature_sum_bpr_identity", "feature_sum_bpr_identity_genre"):
            identity_id = selected["feature_sum_bpr_identity"]
            configuration = next(value for value in enumerate_bpr_configurations(model_config) if value["configuration_id"] == identity_id)
        else:
            raise ValueError(f"unknown admitted family: {family}")
        for seed in seeds:
            run_name = f"{family}__seed{seed}"
            run_path = RUN_DIR / f"{run_name}.json"
            shared_path = OUTPUT_DIR / f"{run_name}__shared.npz"
            assessment_path = OUTPUT_DIR / f"{run_name}__assessment_users.npz"
            diagnostic_path = OUTPUT_DIR / f"{run_name}__fold_in_diagnostics.npz"
            if run_path.exists():
                saved = load_json(run_path)
                _verify_self_hash(saved, "run_id")
                for entry in saved["artifacts"].values():
                    if file_sha256(root / entry["path"]) != entry["sha256"]:
                        raise ValueError("production checkpoint artifact changed")
                results.append(saved)
                continue
            started = time.perf_counter()
            if family == "implicit_als":
                fitted = fit_implicit_als(
                    production.ownership, production.playtime_forever,
                    factors=configuration["factors"], regularization=configuration["regularization"], alpha_o=configuration["alpha_o"], alpha_p=configuration["alpha_p"], tau=configuration["tau"], iterations=configuration["iterations"], training_seed=seed, num_threads=model_config["implicit_als"]["num_threads"],
                )
                shared_arrays = {"design_user_factors": fitted.user_factors, "item_factors": fitted.item_factors}
                item_before = fitted.item_factors.copy()
                assessment_factors = fold_in_als(
                    assessment.ownership, assessment.playtime_forever, fitted.item_factors,
                    regularization=configuration["regularization"], alpha_o=configuration["alpha_o"], alpha_p=configuration["alpha_p"], tau=configuration["tau"],
                )
                if not np.array_equal(item_before, fitted.item_factors):
                    raise AssertionError("ALS fold-in mutated shared item factors")
                statuses = np.where(np.diff(assessment.ownership.indptr) > 0, 1, 0).astype(np.int8)
                iterations = np.zeros(assessment.user_ids.size, dtype=np.int16)
                objectives = np.full(assessment.user_ids.size, np.nan, dtype=np.float64)
                fit_diagnostics = fitted.diagnostics
            else:
                include_genre = family == "feature_sum_bpr_identity_genre"
                fitted = fit_feature_sum_bpr(
                    production.ownership, genres, cycle_id=CYCLE_ID, training_seed=seed,
                    factors=configuration["factors"], regularization=configuration["regularization"], learning_rate=configuration["learning_rate"], epochs=configuration["epochs"], samples_per_epoch=configuration["samples_per_epoch"], include_genre=include_genre,
                )
                p = fitted.parameters
                shared_arrays = {"design_user_factors": p.user_factors, "identity_factors": p.identity_factors, "feature_factors": p.feature_factors, "item_bias": p.item_bias}
                shared_before = tuple(np.asarray(value).copy() for value in (p.identity_factors, p.feature_factors, p.item_bias))
                assessment_factors = np.empty((assessment.user_ids.size, configuration["factors"]), dtype=np.float32)
                statuses = np.zeros(assessment.user_ids.size, dtype=np.int8)
                iterations = np.zeros(assessment.user_ids.size, dtype=np.int16)
                objectives = np.full(assessment.user_ids.size, np.nan, dtype=np.float64)
                for row, user_id in enumerate(assessment.user_ids):
                    begin, end = assessment.ownership.indptr[row : row + 2]
                    vector, diagnostic = fold_in_bpr_user(
                        assessment.ownership.indices[begin:end], user_id=int(user_id), cycle_id=CYCLE_ID,
                        identity_factors=p.identity_factors, item_bias=p.item_bias, genre_features=genres, feature_factors=p.feature_factors,
                        regularization=configuration["regularization"], tolerance=model_config["fold_in"]["pairwise_tolerance"], max_iterations=model_config["fold_in"]["pairwise_max_iterations"],
                    )
                    assessment_factors[row] = vector
                    statuses[row] = 1 if diagnostic["status"] == "complete" else (0 if diagnostic["status"] == "insufficient_history" else 2)
                    iterations[row] = int(diagnostic.get("iterations", 0))
                    objectives[row] = float(diagnostic.get("objective", np.nan))
                if not all(np.array_equal(before, after) for before, after in zip(shared_before, (p.identity_factors, p.feature_factors, p.item_bias))):
                    raise AssertionError("BPR fold-in mutated shared parameters")
                if np.any(statuses == 2):
                    raise RuntimeError(
                        "one or more pairwise assessment fold-ins did not satisfy the frozen solver termination rule"
                    )
                fit_diagnostics = fitted.diagnostics
            shared_entry = save_parameter_archive(shared_path, **shared_arrays)
            assessment_entry = save_parameter_archive(assessment_path, user_factors=assessment_factors, user_ids=assessment.user_ids)
            np.savez_compressed(diagnostic_path, user_ids=assessment.user_ids, status_code=statuses, solver_iterations=iterations, objective=objectives)
            loaded_shared = load_parameter_archive(shared_path, expected_sha256=shared_entry["sha256"])
            loaded_assessment = load_parameter_archive(assessment_path, expected_sha256=assessment_entry["sha256"])
            if family == "implicit_als":
                smoke = np.asarray(loaded_assessment["user_factors"][:5], dtype=np.float64) @ np.asarray(loaded_shared["item_factors"][:7], dtype=np.float64).T
            else:
                item_vectors = np.asarray(loaded_shared["identity_factors"][:7], dtype=np.float64)
                if family.endswith("genre"):
                    item_vectors += genres[:7] @ np.asarray(loaded_shared["feature_factors"], dtype=np.float64)
                smoke = np.asarray(loaded_assessment["user_factors"][:5], dtype=np.float64) @ item_vectors.T + np.asarray(loaded_shared["item_bias"][:7], dtype=np.float64)
            if not np.all(np.isfinite(smoke)):
                raise AssertionError("clean-reload production smoke scores are nonfinite")
            artifacts = {
                "shared_parameters": {"path": _relative(shared_path, root), "size_bytes": shared_path.stat().st_size, "sha256": file_sha256(shared_path)},
                "assessment_user_factors": {"path": _relative(assessment_path, root), "size_bytes": assessment_path.stat().st_size, "sha256": file_sha256(assessment_path)},
                "fold_in_diagnostics": {"path": _relative(diagnostic_path, root), "size_bytes": diagnostic_path.stat().st_size, "sha256": file_sha256(diagnostic_path)},
            }
            run: dict[str, Any] = {
                "schema_version": 1, "cycle_id": CYCLE_ID, "family": family, "configuration_id": selected[family], "training_seed": seed,
                "production_design_edges": int(production.ownership.nnz), "assessment_users": int(assessment.user_ids.size), "assessment_warm_edges": int(assessment.ownership.nnz),
                "fold_in_status_counts": {"complete": int(np.count_nonzero(statuses == 1)), "insufficient_history": int(np.count_nonzero(statuses == 0)), "solver_stopped": int(np.count_nonzero(statuses == 2))},
                "shared_parameters_unchanged_by_fold_in": True, "clean_reload_smoke_sha256": hashlib.sha256(np.ascontiguousarray(smoke, dtype="<f8").tobytes()).hexdigest(),
                "fit_diagnostics": fit_diagnostics, "artifacts": artifacts, "runtime_seconds": time.perf_counter() - started,
            }
            run["run_id"] = semantic_sha256(run)
            _atomic_json(run_path, run)
            results.append(run)

    manifest: dict[str, Any] = {
        "schema_version": 1, "artifact": "stage1_production_refit_and_assessment_fold_in", "cycle_id": CYCLE_ID,
        "gate1_manifest_id": gate1["manifest_id"], "admission_id": admission["admission_id"], "admitted_families": admitted,
        "production_contract": {"warm_items": int(production.item_ids.size), "design_users": int(production.user_ids.size), "restored_validation_edges": int(validation["user_ids"].size), "restored_design_test_edges": int(test["user_ids"].size), "production_edges": int(production.ownership.nnz), "pseudo_cold_removal_carried_forward": False},
        "assessment_contract": {"users": int(assessment.user_ids.size), "warm_history_edges": int(assessment.ownership.nnz), "reserved_ranking_positive": None, "shared_parameters_frozen_before_fold_in": True},
        "runs": results,
        "access_boundary": {"assessment_ids_and_histories": "accessed_only_after_gate1", "assessment_bundle_objectives": "not_accessed", "stage2_objectives_or_bundle_outcomes": "not_accessed"},
        "inputs": {"gate1_manifest_id": gate1["manifest_id"], "training_manifest_id": training_manifest["manifest_id"], "split_set_id": split["split_set_id"], "interaction_set_id": interaction_manifest["interaction_set_id"], "runner_sha256": file_sha256(Path(__file__))},
        "status": "complete",
    }
    manifest["manifest_id"] = semantic_sha256(manifest)
    _atomic_json(destination, manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Stage 1 production refit and fold-in")
    parser.parse_args(argv)
    result = run_production()
    print(json.dumps({"status": result["status"], "manifest_id": result["manifest_id"], "runs": len(result["runs"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
