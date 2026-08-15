"""Open S1.7/S1.8 only after the hashed admission decision and close Gate 1."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import scipy.sparse as sp

from src.features import load_feature_artifacts
from src.ranking import (
    ExpectedCoverageAccumulator,
    evaluate_score_block,
    paired_bootstrap_mean_difference,
    topk_inclusion_probabilities,
)
from src.stage1_backend import (
    fit_feature_sum_bpr,
    load_parameter_archive,
    save_parameter_archive,
)
from src.stage1_protocol import (
    canonical_json_bytes,
    enumerate_bpr_configurations,
    file_sha256,
    load_json,
    semantic_sha256,
)
from src.stage1_split_artifacts import (
    load_design_training_artifacts,
    load_evaluation_user_sample,
    load_validation_targets,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CYCLE_ID = "s1-v2-20260814"
CONFIG_DIR = PROJECT_ROOT / "configs" / "cycles" / CYCLE_ID
CYCLE_DIR = PROJECT_ROOT / "outputs" / "modeling" / "cycles" / CYCLE_ID
PROTECTED_DIR = PROJECT_ROOT / "outputs" / "modeling" / "protected" / CYCLE_ID
RANKING_CONFIG = CONFIG_DIR / "ranking_evaluation.json"
MODEL_CONFIG = CONFIG_DIR / "preference_models.json"
SPLIT_MANIFEST = CYCLE_DIR / "stage1_split_manifest.json"
FEATURE_MANIFEST = CYCLE_DIR / "item_feature_manifest.json"
FEATURE_DIR = PROTECTED_DIR / "stage1_features"
TRAINING_MANIFEST = CYCLE_DIR / "stage1_training_manifest.json"
ADMISSION_MANIFEST = CYCLE_DIR / "stage1_validation_admission_manifest.json"
OUTPUT_DIR = PROTECTED_DIR / "gate1"
DESIGN_TEST_TABLE = CYCLE_DIR / "stage1_design_test_leaderboard.csv"
PSEUDO_COLD_TABLE = CYCLE_DIR / "stage1_pseudo_cold_results.csv"
SEGMENT_TABLE = CYCLE_DIR / "stage1_design_test_segments.csv"
GATE1_MANIFEST = CYCLE_DIR / "stage1_gate1_manifest.json"


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


def _verify_self_hash(manifest: Mapping[str, Any], field: str) -> None:
    unsigned = dict(manifest)
    claimed = unsigned.pop(field, None)
    if claimed != semantic_sha256(unsigned):
        raise ValueError(f"{field} mismatch")


def _load_split_artifact(root: Path, split: Mapping[str, Any], name: str) -> dict[str, np.ndarray]:
    entry = split["artifacts"][name]
    path = (root / entry["path"]).resolve()
    protected_cycle = (root / "outputs" / "modeling" / "protected" / str(split["cycle_id"])).resolve()
    path.relative_to(protected_cycle)
    if file_sha256(path) != entry["sha256"]:
        raise ValueError(f"split artifact hash mismatch: {name}")
    with np.load(path, allow_pickle=False) as payload:
        result = {key: np.asarray(payload[key]) for key in payload.files}
    expected_fields = set(split["artifact_semantics"][name]["fields"])
    if set(result) != expected_fields:
        raise ValueError(f"split artifact schema mismatch: {name}")
    return result


def _warm_genres(feature_dir: Path, warm_item_ids: np.ndarray) -> sp.csr_matrix:
    features = load_feature_artifacts(feature_dir)
    locations = np.searchsorted(features.item_ids, warm_item_ids)
    if not np.array_equal(features.item_ids[locations], warm_item_ids):
        raise ValueError("warm feature projection changed")
    return features.genre[locations].tocsr()


def _selected_runs(training_manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    selected = training_manifest["selection"]
    result = [run for run in training_manifest["runs"] if run["family"] == "popularity"]
    for family, configuration_id in selected.items():
        result.extend(
            run for run in training_manifest["runs"]
            if run["family"] == family and run["configuration_id"] == configuration_id
        )
    expected = 1 + 3 * len(selected)
    if len(result) != expected:
        raise ValueError("selected validation run inventory is incomplete")
    return result


def _score_function(
    run: Mapping[str, Any], arrays: Mapping[str, np.ndarray], genres: sp.csr_matrix
) -> Callable[[np.ndarray], np.ndarray]:
    family = str(run["family"])
    if family == "popularity":
        return lambda rows: np.broadcast_to(np.asarray(arrays["item_counts"], dtype=np.float64), (rows.size, arrays["item_counts"].size))
    if family == "implicit_als":
        return lambda rows: np.asarray(arrays["user_factors"][rows], dtype=np.float64) @ np.asarray(arrays["item_factors"], dtype=np.float64).T
    if family == "feature_sum_bpr_identity":
        items = np.asarray(arrays["identity_factors"], dtype=np.float64)
    elif family == "feature_sum_bpr_identity_genre":
        items = np.asarray(arrays["identity_factors"], dtype=np.float64) + genres @ np.asarray(arrays["feature_factors"], dtype=np.float64)
    else:
        raise ValueError(f"unknown selected model family: {family}")
    bias = np.asarray(arrays["item_bias"], dtype=np.float64)
    return lambda rows: np.asarray(arrays["user_factors"][rows], dtype=np.float64) @ items.T + bias


def _sample_targets(
    sample_ids: np.ndarray,
    targets: Mapping[str, np.ndarray],
    training_user_ids: np.ndarray,
    training_item_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    user_rows = np.searchsorted(training_user_ids, sample_ids)
    order = np.argsort(targets["user_ids"])
    sorted_users = targets["user_ids"][order]
    locations = np.searchsorted(sorted_users, sample_ids)
    if not np.array_equal(sorted_users[locations], sample_ids):
        raise ValueError("sample/target user alignment changed")
    target_rows = order[locations]
    target_ids = targets["item_ids"][target_rows]
    target_columns = np.searchsorted(training_item_ids, target_ids)
    if not np.array_equal(training_item_ids[target_columns], target_ids):
        raise ValueError("target item is outside warm catalogue")
    return user_rows, target_rows, target_ids, target_columns, locations


def _evaluate_test_run(
    *, run: Mapping[str, Any], score: Callable[[np.ndarray], np.ndarray], training: Any,
    sample_ids: np.ndarray, test: Mapping[str, np.ndarray], validation: Mapping[str, np.ndarray],
    ranking_config: Mapping[str, Any], output_path: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    user_rows, target_rows, target_ids, target_columns, _ = _sample_targets(sample_ids, test, training.user_ids, training.item_ids)
    _, validation_rows, validation_ids, validation_columns, _ = _sample_targets(sample_ids, validation, training.user_ids, training.item_ids)
    if not np.array_equal(test["user_ids"][target_rows], validation["user_ids"][validation_rows]):
        raise ValueError("validation/test row pairing changed")
    config = ranking_config["ranking"]
    ks = tuple(config["ks"])
    metrics = {"strictly_above": np.empty(sample_ids.size, dtype=np.int64), "tied_block_size": np.empty(sample_ids.size, dtype=np.int64), "expected_rank": np.empty(sample_ids.size)}
    for k in ks:
        metrics[f"recall_at_{k}"] = np.empty(sample_ids.size)
        metrics[f"ndcg_at_{k}"] = np.empty(sample_ids.size)
    coverage = ExpectedCoverageAccumulator(training.item_ids.size)
    support = np.asarray(training.ownership.sum(axis=0)).reshape(-1).astype(np.int64)
    top_count = int(math.ceil(0.01 * training.item_ids.size))
    top_order = np.lexsort((training.item_ids, -support))
    top_mask = np.zeros(training.item_ids.size, dtype=bool)
    top_mask[top_order[:top_count]] = True
    top_exposure = total_exposure = 0.0
    batch_size = int(config["score_user_batch_size"])
    started = time.perf_counter()
    for start in range(0, sample_ids.size, batch_size):
        stop = min(start + batch_size, sample_ids.size)
        rows = user_rows[start:stop]
        scores = np.asarray(score(rows), dtype=np.float64)
        masks = np.ones(scores.shape, dtype=bool)
        for local, row in enumerate(rows):
            begin, end = training.ownership.indptr[row : row + 2]
            masks[local, training.ownership.indices[begin:end]] = False
        masks[np.arange(stop - start), validation_columns[start:stop]] = False
        if not np.all(masks[np.arange(stop - start), target_columns[start:stop]]):
            raise ValueError("design-test target was excluded")
        result = evaluate_score_block(scores, target_columns[start:stop], masks, ks=ks, maximum_score_block_bytes=config["maximum_score_block_bytes"])
        for name, values in result.items():
            metrics[name][start:stop] = values
        for local in range(stop - start):
            probabilities = topk_inclusion_probabilities(scores[local], max(ks), masks[local])
            coverage.update(probabilities)
            top_exposure += float(probabilities[top_mask].sum())
            total_exposure += float(probabilities.sum())
    activity = np.diff(training.ownership.indptr)[user_rows].astype(np.int32)
    target_support = support[target_columns].astype(np.int32)
    played = (np.asarray(test["playtime_forever"])[target_rows] > 0).astype(np.int8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, user_ids=sample_ids, target_item_ids=target_ids, user_activity=activity, target_training_support=target_support, target_played=played, **metrics)
    aggregate = {f"mean_{name}": float(np.mean(values)) for name, values in metrics.items()}
    aggregate.update({"expected_catalogue_coverage_at_20": coverage.expected_fraction, "expected_top_one_percent_concentration_at_20": top_exposure / total_exposure, "evaluation_users": int(sample_ids.size), "runtime_seconds": time.perf_counter() - started})
    return aggregate, metrics


def _restore_full_design(training: Any, validation: Mapping[str, np.ndarray], test: Mapping[str, np.ndarray]) -> sp.csr_matrix:
    rows = []
    columns = []
    for targets in (validation, test):
        user_rows = np.searchsorted(training.user_ids, targets["user_ids"])
        item_columns = np.searchsorted(training.item_ids, targets["item_ids"])
        if not np.array_equal(training.user_ids[user_rows], targets["user_ids"]) or not np.array_equal(training.item_ids[item_columns], targets["item_ids"]):
            raise ValueError("full-design restoration alignment failed")
        rows.append(user_rows)
        columns.append(item_columns)
    restored = sp.csr_matrix((np.ones(sum(x.size for x in rows), dtype=np.float32), (np.concatenate(rows), np.concatenate(columns))), shape=training.ownership.shape)
    full = (training.ownership + restored).tocsr()
    full.data[:] = 1.0
    full.sum_duplicates()
    full.sort_indices()
    return full


def _run_pseudo_cold(
    *, root: Path, split: Mapping[str, Any], training: Any, validation: Mapping[str, np.ndarray], test: Mapping[str, np.ndarray],
    genres: sp.csr_matrix, model_config: Mapping[str, Any], ranking_config: Mapping[str, Any], training_manifest: Mapping[str, Any], output_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cold = _load_split_artifact(root, split, "pseudo_cold_items")
    cold_columns = np.searchsorted(training.item_ids, cold["item_ids"])
    if not np.array_equal(training.item_ids[cold_columns], cold["item_ids"]):
        raise ValueError("pseudo-cold cohort is not in warm catalogue")
    full = _restore_full_design(training, validation, test)
    cold_positive = full[:, cold_columns].tocoo()
    keep_users = np.diff(full.indptr) - np.asarray(full[:, cold_columns].sum(axis=1)).reshape(-1) > 0
    target_keep = keep_users[cold_positive.row]
    target_users = cold_positive.row[target_keep].astype(np.int64)
    target_cold_columns = cold_positive.col[target_keep].astype(np.int64)
    cold_removed = full.tocsc(copy=True)
    for column in cold_columns:
        cold_removed.data[cold_removed.indptr[column] : cold_removed.indptr[column + 1]] = 0.0
    cold_removed.eliminate_zeros()
    cold_removed = cold_removed.tocsr()
    selected_id = training_manifest["selection"]["feature_sum_bpr_identity"]
    configuration = next(
        dict(value)
        for value in enumerate_bpr_configurations(model_config)
        if value["configuration_id"] == selected_id
    )
    results: list[dict[str, Any]] = []

    popularity = np.asarray(training.ownership.sum(axis=0)).reshape(-1)[cold_columns]
    masks = np.ones((target_users.size, cold_columns.size), dtype=bool)
    known_cold = full[:, cold_columns].tocsr()
    for row, user in enumerate(target_users):
        begin, end = known_cold.indptr[user : user + 2]
        masks[row, known_cold.indices[begin:end]] = False
        masks[row, target_cold_columns[row]] = True

    def evaluate_cold(score_batch: Callable[[int, int], np.ndarray]) -> dict[str, np.ndarray]:
        ks = tuple(ranking_config["ranking"]["ks"])
        result = {
            "strictly_above": np.empty(target_users.size, dtype=np.int64),
            "tied_block_size": np.empty(target_users.size, dtype=np.int64),
            "expected_rank": np.empty(target_users.size, dtype=np.float64),
        }
        for k in ks:
            result[f"recall_at_{k}"] = np.empty(target_users.size, dtype=np.float64)
            result[f"ndcg_at_{k}"] = np.empty(target_users.size, dtype=np.float64)
        batch_size = int(ranking_config["ranking"]["score_user_batch_size"])
        for start in range(0, target_users.size, batch_size):
            stop = min(start + batch_size, target_users.size)
            block = evaluate_score_block(
                np.asarray(score_batch(start, stop), dtype=np.float64),
                target_cold_columns[start:stop],
                masks[start:stop],
                ks=ks,
                maximum_score_block_bytes=ranking_config["ranking"]["maximum_score_block_bytes"],
            )
            for name, values in block.items():
                result[name][start:stop] = values
        return result

    pop_metrics = evaluate_cold(
        lambda start, stop: np.broadcast_to(popularity, (stop - start, popularity.size))
    )
    results.append({"family": "popularity", "training_seed": None, "target_edges": int(target_users.size), **{f"mean_{key}": float(np.mean(value)) for key, value in pop_metrics.items()}})

    cold_genres = genres[cold_columns]
    for seed in model_config["training_seeds"]:
        fitted = fit_feature_sum_bpr(
            cold_removed, genres, cycle_id=str(model_config["cycle_id"]), training_seed=int(seed),
            factors=configuration["factors"], regularization=configuration["regularization"], learning_rate=configuration["learning_rate"],
            epochs=configuration["epochs"], samples_per_epoch=configuration["samples_per_epoch"], include_genre=True,
        )
        parameters = fitted.parameters
        path = output_dir / f"pseudo_cold_genre_seed{int(seed)}.npz"
        artifact = save_parameter_archive(path, user_factors=parameters.user_factors, identity_factors=parameters.identity_factors, feature_factors=parameters.feature_factors, item_bias=parameters.item_bias)
        content_only = cold_genres @ np.asarray(parameters.feature_factors, dtype=np.float64)
        metrics = evaluate_cold(
            lambda start, stop: np.asarray(
                parameters.user_factors[target_users[start:stop]], dtype=np.float64
            )
            @ content_only.T
        )
        result = {"family": "feature_sum_bpr_identity_genre_content_only", "training_seed": int(seed), "target_edges": int(target_users.size), "model_artifact": {"path": _relative(path, root), "size_bytes": path.stat().st_size, "sha256": artifact["sha256"]}, "identity_and_bias_suppressed": True, "identity_only_result": "unavailable", **{f"mean_{key}": float(np.mean(value)) for key, value in metrics.items()}}
        results.append(result)
    summary = {"cohort_items": int(cold_columns.size), "evaluable_positive_edges": int(target_users.size), "users": int(np.unique(target_users).size), "cold_columns_sha256": hashlib.sha256(np.ascontiguousarray(cold_columns, dtype="<i8").tobytes()).hexdigest(), "temporary_cold_training_edges": int(cold_removed.nnz), "normal_full_design_edges": int(full.nnz)}
    return results, summary


def run_gate1(
    *, project_root: str | Path = PROJECT_ROOT, output_path: str | Path = GATE1_MANIFEST
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    destination = Path(output_path).resolve()
    if destination.exists():
        saved = load_json(destination)
        _verify_self_hash(saved, "manifest_id")
        return saved
    ranking_config = load_json(RANKING_CONFIG)
    model_config = load_json(MODEL_CONFIG)
    split = load_json(SPLIT_MANIFEST)
    features_public = load_json(FEATURE_MANIFEST)
    training_manifest = load_json(TRAINING_MANIFEST)
    admission = load_json(ADMISSION_MANIFEST)
    _verify_self_hash(training_manifest, "manifest_id")
    _verify_self_hash(admission, "admission_id")
    if admission.get("timing") != "hashed_before_any_design_test_target_access" or admission.get("training_manifest_id") != training_manifest["manifest_id"]:
        raise ValueError("design-test gate is not open")
    training = load_design_training_artifacts(project_root=root, manifest_path=SPLIT_MANIFEST)
    genres = _warm_genres(FEATURE_DIR, training.item_ids)
    sample = load_evaluation_user_sample(manifest_path=SPLIT_MANIFEST)
    validation = load_validation_targets(manifest_path=SPLIT_MANIFEST)
    # The first access to these sealed coordinates occurs only after the checks above.
    test = _load_split_artifact(root, split, "design_test_targets")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    selected = _selected_runs(training_manifest)
    test_rows: list[dict[str, Any]] = []
    per_family_metrics: dict[str, list[dict[str, np.ndarray]]] = {}
    for run in selected:
        arrays = load_parameter_archive(root / run["model_artifact"]["path"], expected_sha256=run["model_artifact"]["sha256"])
        score = _score_function(run, arrays, genres)
        metric_path = OUTPUT_DIR / f"design_test__{run['run_id']}.npz"
        aggregate, metric_arrays = _evaluate_test_run(run=run, score=score, training=training, sample_ids=sample["user_ids"], test=test, validation=validation, ranking_config=ranking_config, output_path=metric_path)
        row = {"family": run["family"], "configuration_id": run["configuration_id"], "training_seed": run["training_seed"], "run_id": run["run_id"], **aggregate, "metric_artifact": {"path": _relative(metric_path, root), "size_bytes": metric_path.stat().st_size, "sha256": file_sha256(metric_path)}}
        test_rows.append(row)
        per_family_metrics.setdefault(str(run["family"]), []).append(metric_arrays)
    fields = ["family", "configuration_id", "training_seed", "mean_recall_at_10", "mean_recall_at_20", "mean_ndcg_at_10", "mean_ndcg_at_20", "mean_expected_rank", "expected_catalogue_coverage_at_20", "expected_top_one_percent_concentration_at_20", "runtime_seconds", "run_id"]
    with DESIGN_TEST_TABLE.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in test_rows:
            writer.writerow({key: row[key] for key in fields})

    segment_rows: list[dict[str, Any]] = []
    activity_bounds = np.asarray(ranking_config["outer_user_split"]["activity_bands"], dtype=np.int64)
    support_bounds = np.asarray([0, 5, 20, 100, 500], dtype=np.int64)
    for row in test_rows:
        with np.load(root / row["metric_artifact"]["path"], allow_pickle=False) as payload:
            activity = np.asarray(payload["user_activity"])
            support = np.asarray(payload["target_training_support"])
            played = np.asarray(payload["target_played"])
            target_ids = np.asarray(payload["target_item_ids"])
            columns = np.searchsorted(training.item_ids, target_ids)
            metadata = (np.diff(genres.indptr)[columns] > 0).astype(np.int8)
            dimensions = {
                "user_activity_band": np.searchsorted(activity_bounds, activity, side="right") - 1,
                "item_support_band": np.searchsorted(support_bounds, support, side="right") - 1,
                "target_played": played,
                "metadata_covered": metadata,
            }
            for dimension, labels in dimensions.items():
                for label in np.unique(labels):
                    mask = labels == label
                    segment_rows.append(
                        {
                            "family": row["family"],
                            "configuration_id": row["configuration_id"],
                            "training_seed": row["training_seed"],
                            "dimension": dimension,
                            "segment_code": int(label),
                            "users": int(np.count_nonzero(mask)),
                            "mean_recall_at_20": float(np.mean(np.asarray(payload["recall_at_20"])[mask])),
                            "mean_ndcg_at_20": float(np.mean(np.asarray(payload["ndcg_at_20"])[mask])),
                            "mean_expected_rank": float(np.mean(np.asarray(payload["expected_rank"])[mask])),
                        }
                    )
    segment_fields = ["family", "configuration_id", "training_seed", "dimension", "segment_code", "users", "mean_recall_at_20", "mean_ndcg_at_20", "mean_expected_rank"]
    with SEGMENT_TABLE.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=segment_fields)
        writer.writeheader()
        writer.writerows(segment_rows)

    pseudo_rows, pseudo_summary = _run_pseudo_cold(root=root, split=split, training=training, validation=validation, test=test, genres=genres, model_config=model_config, ranking_config=ranking_config, training_manifest=training_manifest, output_dir=OUTPUT_DIR)
    pseudo_fields = sorted({key for row in pseudo_rows for key in row if key != "model_artifact"})
    with PSEUDO_COLD_TABLE.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=pseudo_fields)
        writer.writeheader()
        writer.writerows([{key: row.get(key) for key in pseudo_fields} for row in pseudo_rows])

    pop = per_family_metrics["popularity"][0]
    contrasts: dict[str, Any] = {}
    for family in ("implicit_als", "feature_sum_bpr_identity", "feature_sum_bpr_identity_genre"):
        family_metrics = per_family_metrics[family]
        family_rows = sorted(
            (row for row in test_rows if row["family"] == family),
            key=lambda row: int(row["training_seed"]),
        )
        differences = np.mean(np.vstack([x["ndcg_at_20"] - pop["ndcg_at_20"] for x in family_metrics]), axis=0)
        interval = paired_bootstrap_mean_difference(differences, np.zeros_like(differences), replicates=ranking_config["uncertainty"]["paired_user_bootstrap_replicates"], seed=ranking_config["uncertainty"]["bootstrap_seed"], confidence_level=ranking_config["uncertainty"]["confidence_level"])
        seed_specific = [
            {
                "training_seed": int(row["training_seed"]),
                "mean_ndcg_at_20_difference": float(np.mean(metric["ndcg_at_20"] - pop["ndcg_at_20"])),
                "mean_recall_at_20_difference": float(np.mean(metric["recall_at_20"] - pop["recall_at_20"])),
            }
            for row, metric in zip(family_rows, family_metrics)
        ]
        contrasts[family] = {"paired_ndcg_at_20": asdict(interval), "mean_recall_at_20_difference": float(np.mean([np.mean(x["recall_at_20"] - pop["recall_at_20"]) for x in family_metrics])), "seed_specific": seed_specific, "seed_range_ndcg_at_20_difference": [min(x["mean_ndcg_at_20_difference"] for x in seed_specific), max(x["mean_ndcg_at_20_difference"] for x in seed_specific)]}
    genre_metrics = per_family_metrics["feature_sum_bpr_identity_genre"]
    identity_metrics = per_family_metrics["feature_sum_bpr_identity"]
    genre_difference = np.mean(np.vstack([a["ndcg_at_20"] - b["ndcg_at_20"] for a, b in zip(genre_metrics, identity_metrics)]), axis=0)
    genre_seed_specific = [
        {
            "training_seed": int(seed),
            "mean_ndcg_at_20_difference": float(np.mean(a["ndcg_at_20"] - b["ndcg_at_20"])),
            "mean_recall_at_20_difference": float(np.mean(a["recall_at_20"] - b["recall_at_20"])),
        }
        for seed, a, b in zip(model_config["training_seeds"], genre_metrics, identity_metrics)
    ]
    contrasts["genre_versus_identity"] = {"paired_ndcg_at_20": asdict(paired_bootstrap_mean_difference(genre_difference, np.zeros_like(genre_difference), replicates=ranking_config["uncertainty"]["paired_user_bootstrap_replicates"], seed=ranking_config["uncertainty"]["bootstrap_seed"], confidence_level=ranking_config["uncertainty"]["confidence_level"])), "mean_recall_at_20_difference": float(np.mean([np.mean(a["recall_at_20"] - b["recall_at_20"]) for a, b in zip(genre_metrics, identity_metrics)])), "seed_specific": genre_seed_specific, "seed_range_ndcg_at_20_difference": [min(x["mean_ndcg_at_20_difference"] for x in genre_seed_specific), max(x["mean_ndcg_at_20_difference"] for x in genre_seed_specific)]}

    manifest: dict[str, Any] = {
        "schema_version": 1, "artifact": "stage1_gate1_closeout", "cycle_id": CYCLE_ID,
        "admission_id": admission["admission_id"], "admitted_families": admission["admitted_families"], "selection_unchanged_after_design_test": True,
        "design_test_runs": test_rows, "design_test_contrasts": contrasts,
        "pseudo_cold": {"summary": pseudo_summary, "runs": pseudo_rows, "identity_only_result": "unavailable", "candidate_catalogue_items": 300},
        "tables": {"design_test": {"path": _relative(DESIGN_TEST_TABLE, root), "size_bytes": DESIGN_TEST_TABLE.stat().st_size, "sha256": file_sha256(DESIGN_TEST_TABLE)}, "segments": {"path": _relative(SEGMENT_TABLE, root), "size_bytes": SEGMENT_TABLE.stat().st_size, "sha256": file_sha256(SEGMENT_TABLE), "rows": len(segment_rows)}, "pseudo_cold": {"path": _relative(PSEUDO_COLD_TABLE, root), "size_bytes": PSEUDO_COLD_TABLE.stat().st_size, "sha256": file_sha256(PSEUDO_COLD_TABLE)}},
        "segment_boundaries": {"user_activity_lower_bounds": ranking_config["outer_user_split"]["activity_bands"], "item_support_lower_bounds": [0, 5, 20, 100, 500], "played_rule": "heldout_playtime_forever_greater_than_zero", "metadata_coverage_rule": "target_genre_row_has_nonzero"},
        "claim_ledger": {"ranking": "held-out ownership reconstruction only", "genre": "controlled predictive ablation_not_causal", "pseudo_cold": "content-only cohort evidence_not_general_unseen-item_proof", "utilities": "not_identified_by_gate1"},
        "access_boundary": {"validation_and_design_test": "accessed", "pseudo_cold_cohort": "accessed_for_s1_8", "assessment_ids_or_histories": "sealed_not_accessed", "stage2_objectives_or_bundle_outcomes": "not_accessed"},
        "inputs": {"training_manifest_id": training_manifest["manifest_id"], "split_set_id": split["split_set_id"], "feature_set_id": features_public["feature_set_id"], "runner_sha256": file_sha256(Path(__file__))},
        "status": "pass",
    }
    manifest["manifest_id"] = semantic_sha256(manifest)
    _atomic_json(destination, manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run S1.7-S1.9 and close Gate 1")
    parser.parse_args(argv)
    result = run_gate1()
    print(json.dumps({"status": result["status"], "manifest_id": result["manifest_id"], "admitted_families": result["admitted_families"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
