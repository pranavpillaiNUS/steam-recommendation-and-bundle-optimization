"""Execute and publish the outcome-free S1.5 backend/fold-in spike."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np
import scipy.sparse as sp

from src.features import load_feature_artifacts
from src.preference_model import (
    FeatureSumParameters,
    feature_sum_item_factors,
    feature_sum_scores,
    score_factor_block,
)
from src.stage1_backend import (
    fit_feature_sum_bpr,
    fit_implicit_als,
    fold_in_als,
    fold_in_bpr_user,
    load_parameter_archive,
    save_parameter_archive,
)
from src.stage1_protocol import (
    canonical_json_bytes,
    file_sha256,
    load_json,
    semantic_sha256,
)
from src.stage1_split_artifacts import load_design_training_artifacts


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CYCLE_ID = "s1-v2-20260814"
DEFAULT_CYCLE_DIR = PROJECT_ROOT / "outputs" / "modeling" / "cycles" / DEFAULT_CYCLE_ID
DEFAULT_PROTECTED_DIR = PROJECT_ROOT / "outputs" / "modeling" / "protected" / DEFAULT_CYCLE_ID
DEFAULT_MODEL_CONFIG = PROJECT_ROOT / "configs" / "cycles" / DEFAULT_CYCLE_ID / "preference_models.json"
DEFAULT_PROTOCOL = DEFAULT_CYCLE_DIR / "stage1_protocol_manifest.json"
DEFAULT_SPLIT = DEFAULT_CYCLE_DIR / "stage1_split_manifest.json"
DEFAULT_FEATURE_MANIFEST = DEFAULT_CYCLE_DIR / "item_feature_manifest.json"
DEFAULT_FEATURE_DIR = DEFAULT_PROTECTED_DIR / "stage1_features"
DEFAULT_FIXTURE_DIR = DEFAULT_PROTECTED_DIR / "stage1_backend_spike"
DEFAULT_TABLE = DEFAULT_CYCLE_DIR / "stage1_backend_equivalence.csv"
DEFAULT_MANIFEST = DEFAULT_CYCLE_DIR / "stage1_backend_spike_manifest.json"


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _check(rows: list[dict[str, str]], check_id: str, passed: bool, detail: str) -> None:
    rows.append(
        {
            "check_id": check_id,
            "status": "pass" if passed else "fail",
            "detail": detail,
        }
    )
    if not passed:
        raise AssertionError(f"backend spike failed: {check_id}: {detail}")


def _write_table(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("check_id", "status", "detail"))
        writer.writeheader()
        writer.writerows(rows)


def run_backend_spike(
    *,
    project_root: str | Path = PROJECT_ROOT,
    model_config_path: str | Path = DEFAULT_MODEL_CONFIG,
    protocol_manifest_path: str | Path = DEFAULT_PROTOCOL,
    split_manifest_path: str | Path = DEFAULT_SPLIT,
    feature_manifest_path: str | Path = DEFAULT_FEATURE_MANIFEST,
    feature_dir: str | Path = DEFAULT_FEATURE_DIR,
    fixture_dir: str | Path = DEFAULT_FIXTURE_DIR,
    table_path: str | Path = DEFAULT_TABLE,
    manifest_path: str | Path = DEFAULT_MANIFEST,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config_path = Path(model_config_path).resolve()
    protocol_path = Path(protocol_manifest_path).resolve()
    split_path = Path(split_manifest_path).resolve()
    feature_manifest = Path(feature_manifest_path).resolve()
    features_path = Path(feature_dir).resolve()
    fixtures = Path(fixture_dir).resolve()
    table = Path(table_path).resolve()
    destination = Path(manifest_path).resolve()
    if destination.exists():
        saved = load_json(destination)
        unsigned = dict(saved)
        claimed = unsigned.pop("manifest_id", None)
        if claimed != semantic_sha256(unsigned):
            raise ValueError("backend spike manifest hash mismatch")
        for entry in saved["artifacts"].values():
            path = root / entry["path"]
            if file_sha256(path) != entry["sha256"]:
                raise ValueError("backend spike artifact hash mismatch")
        return saved

    model_config = load_json(config_path)
    protocol = load_json(protocol_path)
    split = load_json(split_path)
    feature_public = load_json(feature_manifest)
    cycle_id = str(model_config["cycle_id"])
    if any(
        manifest.get("cycle_id") != cycle_id
        for manifest in (protocol, split, feature_public)
    ):
        raise ValueError("backend spike dependency cycle mismatch")
    training = load_design_training_artifacts(
        project_root=root, manifest_path=split_path
    )
    features = load_feature_artifacts(features_path)
    positions = np.searchsorted(features.item_ids, training.item_ids)
    if not np.array_equal(features.item_ids[positions], training.item_ids):
        raise ValueError("warm feature projection is misaligned")
    warm_genres = features.genre[positions].tocsr()

    rows: list[dict[str, str]] = []
    synthetic_ownership = sp.csr_matrix(
        np.asarray([[1, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 1]], dtype=np.float32)
    )
    synthetic_playtime = sp.csr_matrix(
        np.asarray([[2, 0, 0, 0], [0, 5, 0, 0], [1, 0, 0, 3]], dtype=np.float32)
    )
    synthetic_genres = sp.csr_matrix(
        np.asarray([[1, 0], [1, 0], [0, 1], [0.5, 0.5]], dtype=np.float32)
    )

    als = fit_implicit_als(
        synthetic_ownership,
        synthetic_playtime,
        factors=3,
        regularization=0.1,
        alpha_o=20.0,
        alpha_p=2.0,
        tau=5.0,
        iterations=2,
        training_seed=104729,
        num_threads=1,
    )
    als_repeat = fit_implicit_als(
        synthetic_ownership,
        synthetic_playtime,
        factors=3,
        regularization=0.1,
        alpha_o=20.0,
        alpha_p=2.0,
        tau=5.0,
        iterations=2,
        training_seed=104729,
        num_threads=1,
    )
    _check(rows, "als_orientation", als.user_factors.shape == (3, 3) and als.item_factors.shape == (4, 3), "users_by_k and items_by_k")
    _check(rows, "als_seed_repeatability", np.array_equal(als.user_factors, als_repeat.user_factors) and np.array_equal(als.item_factors, als_repeat.item_factors), "byte-equal repeated synthetic fit")
    direct = np.asarray(als.user_factors, dtype=np.float64) @ np.asarray(
        als.item_factors, dtype=np.float64
    ).T
    bounded = score_factor_block(als.user_factors, als.item_factors, np.arange(3), np.arange(4))
    _check(rows, "als_direct_batched_scores", np.allclose(direct, bounded, rtol=0, atol=1e-7), "bounded scorer agrees")
    folded_als = fold_in_als(
        synthetic_ownership[:1], synthetic_playtime[:1], als.item_factors,
        regularization=0.1, alpha_o=20.0, alpha_p=2.0, tau=5.0,
    )
    _check(rows, "als_fold_in", folded_als.shape == (1, 3) and np.all(np.isfinite(folded_als)), "exact frozen-item ridge solve")

    bpr_kwargs = dict(
        cycle_id=cycle_id,
        training_seed=104729,
        factors=3,
        regularization=0.001,
        learning_rate=0.05,
        epochs=2,
        samples_per_epoch=200,
    )
    identity = fit_feature_sum_bpr(synthetic_ownership, synthetic_genres, include_genre=False, **bpr_kwargs)
    identity_repeat = fit_feature_sum_bpr(synthetic_ownership, synthetic_genres, include_genre=False, **bpr_kwargs)
    genre = fit_feature_sum_bpr(synthetic_ownership, synthetic_genres, include_genre=True, **bpr_kwargs)
    identity_equal = all(
        np.array_equal(getattr(identity.parameters, name), getattr(identity_repeat.parameters, name))
        for name in ("user_factors", "identity_factors", "item_bias")
    )
    _check(rows, "bpr_seed_repeatability", identity_equal, "byte-equal repeated fallback fit")
    identity_hashes = [x["triple_stream_sha256"] for x in identity.diagnostics["epochs"]]
    genre_hashes = [x["triple_stream_sha256"] for x in genre.diagnostics["epochs"]]
    _check(rows, "bpr_controlled_triple_stream", identity_hashes == genre_hashes, "identity and genre consume identical triples")
    item_vectors = feature_sum_item_factors(genre.parameters, synthetic_genres, rho=1.0)
    cold = 2
    content_only = synthetic_genres[cold] @ genre.parameters.feature_factors
    _check(rows, "pseudo_cold_identity_suppression", np.allclose(item_vectors[cold] - genre.parameters.identity_factors[cold], np.asarray(content_only).reshape(-1), rtol=0, atol=1e-7), "content block remains after explicit identity subtraction")
    shared_before = (
        genre.parameters.identity_factors.copy(), genre.parameters.feature_factors.copy(), genre.parameters.item_bias.copy()
    )
    folded_bpr, fold_diagnostic = fold_in_bpr_user(
        [0, 2], user_id=76561198000000000, cycle_id=cycle_id,
        identity_factors=genre.parameters.identity_factors,
        item_bias=genre.parameters.item_bias,
        genre_features=synthetic_genres,
        feature_factors=genre.parameters.feature_factors,
        regularization=0.001,
    )
    _check(rows, "bpr_fold_in", np.all(np.isfinite(folded_bpr)) and fold_diagnostic["triple_count"] == 2, "deterministic convex user-only solve")
    _check(rows, "fold_in_shared_parameter_immutability", all(np.array_equal(before, after) for before, after in zip(shared_before, (genre.parameters.identity_factors, genre.parameters.feature_factors, genre.parameters.item_bias))), "shared arrays unchanged")
    zero_user, zero_diagnostic = fold_in_bpr_user(
        [], user_id=1, cycle_id=cycle_id,
        identity_factors=genre.parameters.identity_factors,
        item_bias=genre.parameters.item_bias,
        genre_features=synthetic_genres,
        feature_factors=genre.parameters.feature_factors,
        regularization=0.001,
    )
    _check(rows, "insufficient_history_fallback", np.count_nonzero(zero_user) == 0 and zero_diagnostic["status"] == "insufficient_history", "reported zero vector")

    fixtures.mkdir(parents=True, exist_ok=False)
    als_entry = save_parameter_archive(fixtures / "synthetic_als.npz", user_factors=als.user_factors, item_factors=als.item_factors)
    bpr_entry = save_parameter_archive(
        fixtures / "synthetic_bpr_genre.npz",
        user_factors=genre.parameters.user_factors,
        identity_factors=genre.parameters.identity_factors,
        feature_factors=genre.parameters.feature_factors,
        item_bias=genre.parameters.item_bias,
    )
    reloaded = load_parameter_archive(fixtures / "synthetic_bpr_genre.npz", expected_sha256=bpr_entry["sha256"])
    reloaded_parameters = FeatureSumParameters(
        user_factors=reloaded["user_factors"],
        identity_factors=reloaded["identity_factors"],
        feature_factors=reloaded["feature_factors"],
        item_bias=reloaded["item_bias"],
    )
    original_scores = feature_sum_scores(
        genre.parameters, synthetic_genres, rho=1.0
    )
    reloaded_scores = feature_sum_scores(
        reloaded_parameters, synthetic_genres, rho=1.0
    )
    _check(rows, "clean_reload_score_reproduction", np.array_equal(original_scores, reloaded_scores), "serialized arrays reproduce scores exactly")

    # A permitted real-shape smoke fit catches dimensional and memory failures
    # without consuming any held-out coordinate.
    smoke_started = time.perf_counter()
    smoke = fit_feature_sum_bpr(
        training.ownership,
        warm_genres,
        cycle_id=cycle_id,
        training_seed=int(model_config["training_seeds"][0]),
        factors=32,
        regularization=0.001,
        learning_rate=0.05,
        epochs=1,
        samples_per_epoch=10_000,
        include_genre=True,
    )
    planned_parameter_bytes = sum(
        np.asarray(getattr(smoke.parameters, name)).nbytes
        for name in ("user_factors", "identity_factors", "feature_factors", "item_bias")
    )
    _check(rows, "planned_dimension_smoke", planned_parameter_bytes < int(model_config["resource_budget"]["maximum_saved_model_bytes_per_seed"]), f"{planned_parameter_bytes} parameter bytes; {time.perf_counter() - smoke_started:.3f}s")

    _write_table(table, rows)
    artifacts = {
        "equivalence_table": table,
        "synthetic_als": fixtures / "synthetic_als.npz",
        "synthetic_bpr_genre": fixtures / "synthetic_bpr_genre.npz",
    }
    artifact_entries = {
        label: {
            "path": _relative(path, root),
            "size_bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for label, path in artifacts.items()
    }
    inputs = {
        "model_config": config_path,
        "protocol_manifest": protocol_path,
        "split_manifest": split_path,
        "feature_manifest": feature_manifest,
        "backend_code": Path(__file__).resolve().with_name("stage1_backend.py"),
        "spike_code": Path(__file__).resolve(),
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "artifact": "stage1_backend_and_fold_in_feasibility_spike",
        "cycle_id": cycle_id,
        "protocol_id": protocol["protocol_id"],
        "split_set_id": split["split_set_id"],
        "feature_set_id": feature_public["feature_set_id"],
        "status": "pass",
        "checks_passed": len(rows),
        "checks_failed": 0,
        "backend_amendment": {
            "timing": "prospective_before_validation_target_access",
            "implicit_als": "implicit_0.7.2_native_exact_least_squares",
            "lightfm_1_17": "unavailable_on_windows_python_3_10_setup_attribute_error_after_isolated_and_nonisolated_attempts",
            "pairwise_fallback": "tested_numpy_feature_sum_bpr",
            "fallback_update_rule": "one_full_sampled_epoch_gradient_then_one_coordinatewise_adagrad_step",
            "parameter_accumulation": "float64_gradient_and_adagrad_accumulator_float32_parameter_cast_after_each_epoch",
            "adagrad_epsilon": 1e-8,
            "initialization": "namespaced_pcg64_normal_mean_0_sd_0.01",
            "unlogged_retry_allowed": False,
        },
        "fold_in_contract_status": "fully_specified_in_frozen_v2_model_config_and_exercised_here",
        "real_shape_smoke": {
            "users": training.ownership.shape[0],
            "items": training.ownership.shape[1],
            "edges": training.ownership.nnz,
            "factors": 32,
            "samples": 10_000,
            "parameter_bytes": planned_parameter_bytes,
        },
        "artifacts": artifact_entries,
        "inputs": {
            label: {
                "path": _relative(path, root),
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for label, path in inputs.items()
        },
        "environment": {
            "python": platform.python_version(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "scipy", "implicit")
            },
        },
        "access_boundary": {
            "design_training_only": True,
            "validation_targets_or_metrics": "not_accessed",
            "design_test_targets": "sealed_not_accessed",
            "assessment_ids_or_histories": "sealed_not_accessed",
            "pseudo_cold_identifiers": "not_accessed",
            "stage2_objectives_or_bundle_outcomes": "not_accessed",
        },
    }
    manifest["manifest_id"] = semantic_sha256(manifest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(canonical_json_bytes(manifest))
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Stage 1 backend spike")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args(argv)
    result = run_backend_spike(manifest_path=args.manifest)
    print(json.dumps({"status": result["status"], "checks_passed": result["checks_passed"], "manifest_id": result["manifest_id"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
