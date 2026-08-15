"""Run S1.6 validation tuning and freeze admission before design-test access."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import scipy.sparse as sp

from src.features import load_feature_artifacts
from src.preference_model import popularity_scores
from src.ranking import (
    ExpectedCoverageAccumulator,
    evaluate_score_block,
    expected_top_item_concentration,
    paired_bootstrap_mean_difference,
    topk_inclusion_probabilities,
)
from src.stage1_backend import (
    fit_feature_sum_bpr,
    fit_implicit_als,
    load_parameter_archive,
    save_parameter_archive,
)
from src.stage1_protocol import (
    canonical_json_bytes,
    enumerate_als_configurations,
    enumerate_bpr_configurations,
    file_sha256,
    load_json,
    semantic_sha256,
)
from src.stage1_split_artifacts import (
    load_design_training_artifacts,
    load_evaluation_user_sample,
    load_validation_targets,
    mask_validation_other_holdouts,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CYCLE_ID = "s1-v2-20260814"
CONFIG_DIR = PROJECT_ROOT / "configs" / "cycles" / DEFAULT_CYCLE_ID
CYCLE_DIR = PROJECT_ROOT / "outputs" / "modeling" / "cycles" / DEFAULT_CYCLE_ID
PROTECTED_DIR = PROJECT_ROOT / "outputs" / "modeling" / "protected" / DEFAULT_CYCLE_ID
DEFAULT_MODEL_CONFIG = CONFIG_DIR / "preference_models.json"
DEFAULT_RANKING_CONFIG = CONFIG_DIR / "ranking_evaluation.json"
DEFAULT_PROTOCOL = CYCLE_DIR / "stage1_protocol_manifest.json"
DEFAULT_INTERACTION = CYCLE_DIR / "stage1_interaction_manifest.json"
DEFAULT_SPLIT = CYCLE_DIR / "stage1_split_manifest.json"
DEFAULT_FEATURE_MANIFEST = CYCLE_DIR / "item_feature_manifest.json"
DEFAULT_ESTIMATOR = CYCLE_DIR / "stage1_estimator_spec_manifest.json"
DEFAULT_SPIKE = CYCLE_DIR / "stage1_backend_spike_manifest.json"
DEFAULT_FEATURE_DIR = PROTECTED_DIR / "stage1_features"
DEFAULT_MODEL_DIR = PROTECTED_DIR / "validation_models"
DEFAULT_METRIC_DIR = PROTECTED_DIR / "validation_metrics"
DEFAULT_RUN_DIR = CYCLE_DIR / "validation_runs"
DEFAULT_LEADERBOARD = CYCLE_DIR / "stage1_validation_leaderboard.csv"
DEFAULT_TRAINING_MANIFEST = CYCLE_DIR / "stage1_training_manifest.json"
DEFAULT_ADMISSION = CYCLE_DIR / "stage1_validation_admission_manifest.json"


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_bytes(canonical_json_bytes(value))
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _array_hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _project_warm_genres(feature_dir: Path, warm_item_ids: np.ndarray) -> sp.csr_matrix:
    features = load_feature_artifacts(feature_dir)
    positions = np.searchsorted(features.item_ids, warm_item_ids)
    if (
        np.any(positions >= features.item_ids.size)
        or not np.array_equal(features.item_ids[positions], warm_item_ids)
    ):
        raise ValueError("warm catalogue and feature rows are misaligned")
    return features.genre[positions].tocsr()


def _evaluation_coordinates(
    training_user_ids: np.ndarray,
    training_item_ids: np.ndarray,
    *,
    split_manifest_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sample = load_evaluation_user_sample(manifest_path=split_manifest_path)
    validation = load_validation_targets(manifest_path=split_manifest_path)
    sample_ids = sample["user_ids"]
    user_rows = np.searchsorted(training_user_ids, sample_ids)
    if (
        np.any(user_rows >= training_user_ids.size)
        or not np.array_equal(training_user_ids[user_rows], sample_ids)
    ):
        raise ValueError("evaluation users are not aligned to training rows")
    order = np.argsort(validation["user_ids"])
    validation_users = validation["user_ids"][order]
    validation_items = validation["item_ids"][order]
    locations = np.searchsorted(validation_users, sample_ids)
    if (
        np.any(locations >= validation_users.size)
        or not np.array_equal(validation_users[locations], sample_ids)
    ):
        raise ValueError("evaluation users lack validation targets")
    target_ids = validation_items[locations]
    target_columns = np.searchsorted(training_item_ids, target_ids)
    if (
        np.any(target_columns >= training_item_ids.size)
        or not np.array_equal(training_item_ids[target_columns], target_ids)
    ):
        raise ValueError("validation target is outside the warm catalogue")
    return sample_ids, user_rows.astype(np.int64), target_ids, target_columns.astype(np.int64)


def evaluate_validation_model(
    *,
    score_batch: Callable[[np.ndarray], np.ndarray],
    training: Any,
    split_manifest_path: Path,
    ranking_config: Mapping[str, Any],
    metric_path: Path,
) -> dict[str, Any]:
    sample_ids, user_rows, target_ids, target_columns = _evaluation_coordinates(
        training.user_ids, training.item_ids, split_manifest_path=split_manifest_path
    )
    ranking = ranking_config["ranking"]
    ks = tuple(ranking["ks"])
    maximum_bytes = int(ranking["maximum_score_block_bytes"])
    batch_size = int(ranking["score_user_batch_size"])
    metrics = {
        "strictly_above": np.empty(sample_ids.size, dtype=np.int64),
        "tied_block_size": np.empty(sample_ids.size, dtype=np.int64),
        "expected_rank": np.empty(sample_ids.size, dtype=np.float64),
    }
    for k in ks:
        metrics[f"recall_at_{k}"] = np.empty(sample_ids.size, dtype=np.float64)
        metrics[f"ndcg_at_{k}"] = np.empty(sample_ids.size, dtype=np.float64)
    coverage = ExpectedCoverageAccumulator(training.item_ids.size)
    support = np.asarray(training.ownership.sum(axis=0)).reshape(-1).astype(np.int64)
    top_count = int(math.ceil(0.01 * training.item_ids.size))
    top_order = np.lexsort((training.item_ids, -support))
    top_mask = np.zeros(training.item_ids.size, dtype=bool)
    top_mask[top_order[:top_count]] = True
    top_exposure = 0.0
    total_exposure = 0.0
    started = time.perf_counter()
    for start in range(0, sample_ids.size, batch_size):
        stop = min(start + batch_size, sample_ids.size)
        rows = user_rows[start:stop]
        scores = np.asarray(score_batch(rows), dtype=np.float64)
        expected_shape = (stop - start, training.item_ids.size)
        if scores.shape != expected_shape or not np.all(np.isfinite(scores)):
            raise ValueError("scorer returned invalid validation block")
        masks = np.ones(expected_shape, dtype=bool)
        for local, training_row in enumerate(rows):
            edge_start = int(training.ownership.indptr[training_row])
            edge_stop = int(training.ownership.indptr[training_row + 1])
            masks[local, training.ownership.indices[edge_start:edge_stop]] = False
        masks = mask_validation_other_holdouts(
            masks,
            sample_ids[start:stop],
            training.item_ids,
            manifest_path=split_manifest_path,
        )
        if not np.all(masks[np.arange(stop - start), target_columns[start:stop]]):
            raise ValueError("validation target was excluded")
        result = evaluate_score_block(
            scores,
            target_columns[start:stop],
            masks,
            ks=ks,
            maximum_score_block_bytes=maximum_bytes,
        )
        for name, values in result.items():
            metrics[name][start:stop] = values
        for local in range(stop - start):
            probabilities = topk_inclusion_probabilities(
                scores[local], max(ks), masks[local]
            )
            coverage.update(probabilities)
            top_exposure += float(probabilities[top_mask].sum())
            total_exposure += float(probabilities.sum())
    metric_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        metric_path,
        user_ids=sample_ids,
        target_item_ids=target_ids,
        user_activity=np.diff(training.ownership.indptr)[user_rows].astype(np.int32),
        target_training_support=support[target_columns].astype(np.int32),
        **metrics,
    )
    aggregate = {
        f"mean_{name}": float(np.mean(values, dtype=np.float64))
        for name, values in metrics.items()
    }
    aggregate.update(
        {
            "expected_catalogue_coverage_at_20": coverage.expected_fraction,
            "expected_top_one_percent_concentration_at_20": top_exposure / total_exposure,
            "top_one_percent_item_count": top_count,
            "evaluation_users": int(sample_ids.size),
            "runtime_seconds": time.perf_counter() - started,
        }
    )
    return aggregate


def _load_metric(path: Path, expected_sha256: str) -> dict[str, np.ndarray]:
    if file_sha256(path) != expected_sha256:
        raise ValueError("validation metric artifact hash mismatch")
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def _run_id(family: str, configuration_id: str, seed: int | None) -> str:
    suffix = "deterministic" if seed is None else f"seed{int(seed)}"
    return f"{family}__{configuration_id}__{suffix}"


def run_validation(
    *,
    project_root: str | Path = PROJECT_ROOT,
    model_config_path: str | Path = DEFAULT_MODEL_CONFIG,
    ranking_config_path: str | Path = DEFAULT_RANKING_CONFIG,
    protocol_manifest_path: str | Path = DEFAULT_PROTOCOL,
    interaction_manifest_path: str | Path = DEFAULT_INTERACTION,
    split_manifest_path: str | Path = DEFAULT_SPLIT,
    feature_manifest_path: str | Path = DEFAULT_FEATURE_MANIFEST,
    estimator_manifest_path: str | Path = DEFAULT_ESTIMATOR,
    spike_manifest_path: str | Path = DEFAULT_SPIKE,
    feature_dir: str | Path = DEFAULT_FEATURE_DIR,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    metric_dir: str | Path = DEFAULT_METRIC_DIR,
    run_dir: str | Path = DEFAULT_RUN_DIR,
    leaderboard_path: str | Path = DEFAULT_LEADERBOARD,
    training_manifest_path: str | Path = DEFAULT_TRAINING_MANIFEST,
    admission_path: str | Path = DEFAULT_ADMISSION,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    paths = {
        "model_config": Path(model_config_path).resolve(),
        "ranking_config": Path(ranking_config_path).resolve(),
        "protocol_manifest": Path(protocol_manifest_path).resolve(),
        "interaction_manifest": Path(interaction_manifest_path).resolve(),
        "split_manifest": Path(split_manifest_path).resolve(),
        "feature_manifest": Path(feature_manifest_path).resolve(),
        "estimator_manifest": Path(estimator_manifest_path).resolve(),
        "backend_spike_manifest": Path(spike_manifest_path).resolve(),
        "backend_code": (root / "src" / "stage1_backend.py").resolve(),
        "ranking_code": (root / "src" / "ranking.py").resolve(),
        "runner_code": Path(__file__).resolve(),
    }
    model_config = load_json(paths["model_config"])
    ranking_config = load_json(paths["ranking_config"])
    protocol = load_json(paths["protocol_manifest"])
    interaction = load_json(paths["interaction_manifest"])
    split = load_json(paths["split_manifest"])
    feature_public = load_json(paths["feature_manifest"])
    estimator = load_json(paths["estimator_manifest"])
    spike = load_json(paths["backend_spike_manifest"])
    cycle_id = str(model_config["cycle_id"])
    if spike.get("status") != "pass" or spike.get("access_boundary", {}).get("validation_targets_or_metrics") != "not_accessed":
        raise ValueError("S1.5 did not pass before validation")
    protocol_id = protocol["protocol_id"]
    if any(x.get("cycle_id") != cycle_id for x in (interaction, split, feature_public, estimator, spike)):
        raise ValueError("validation dependency cycle mismatch")
    training = load_design_training_artifacts(project_root=root, manifest_path=paths["split_manifest"])
    warm_genres = _project_warm_genres(Path(feature_dir).resolve(), training.item_ids)
    models = Path(model_dir).resolve()
    metric_root = Path(metric_dir).resolve()
    runs_root = Path(run_dir).resolve()
    models.mkdir(parents=True, exist_ok=True)
    metric_root.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)
    dependency_hashes = {label: file_sha256(path) for label, path in paths.items()}

    entries: list[dict[str, Any]] = []

    def execute(
        *, family: str, configuration: Mapping[str, Any], seed: int | None,
        fit: Callable[[], tuple[dict[str, np.ndarray], Mapping[str, Any]]],
        scorer: Callable[[Mapping[str, np.ndarray], np.ndarray], np.ndarray],
    ) -> dict[str, Any]:
        configuration_id = str(configuration["configuration_id"])
        identifier = _run_id(family, configuration_id, seed)
        run_path = runs_root / f"{identifier}.json"
        model_path = models / f"{identifier}.npz"
        metric_path = metric_root / f"{identifier}.npz"
        if run_path.exists():
            saved = load_json(run_path)
            unsigned = dict(saved)
            claimed = unsigned.pop("run_id", None)
            if claimed != semantic_sha256(unsigned):
                raise ValueError(f"saved validation run hash mismatch: {identifier}")
            if saved.get("dependency_sha256") != dependency_hashes:
                raise ValueError(f"validation run code/dependency drift: {identifier}")
            load_parameter_archive(model_path, expected_sha256=saved["model_artifact"]["sha256"])
            _load_metric(metric_path, saved["metric_artifact"]["sha256"])
            return saved
        started = time.perf_counter()
        arrays, diagnostics = fit()
        model_metadata = save_parameter_archive(model_path, **arrays)
        loaded = load_parameter_archive(model_path, expected_sha256=model_metadata["sha256"])
        aggregate = evaluate_validation_model(
            score_batch=lambda rows: scorer(loaded, rows),
            training=training,
            split_manifest_path=paths["split_manifest"],
            ranking_config=ranking_config,
            metric_path=metric_path,
        )
        parameter_count = int(sum(np.asarray(value).size for value in arrays.values()))
        run: dict[str, Any] = {
            "schema_version": 1,
            "cycle_id": cycle_id,
            "protocol_id": protocol_id,
            "family": family,
            "configuration": dict(configuration),
            "configuration_id": configuration_id,
            "training_seed": seed,
            "status": "complete",
            "parameter_count": parameter_count,
            "diagnostics": diagnostics,
            "aggregate_validation_metrics": aggregate,
            "model_artifact": {
                "path": _relative(model_path, root),
                "size_bytes": model_path.stat().st_size,
                "sha256": file_sha256(model_path),
            },
            "metric_artifact": {
                "path": _relative(metric_path, root),
                "size_bytes": metric_path.stat().st_size,
                "sha256": file_sha256(metric_path),
            },
            "dependency_sha256": dependency_hashes,
            "runtime_seconds_total": time.perf_counter() - started,
        }
        run["run_id"] = semantic_sha256(run)
        _atomic_json(run_path, run)
        return run

    counts = popularity_scores(training.ownership)
    pop_config = model_config["popularity"]
    entries.append(
        execute(
            family="popularity",
            configuration=pop_config,
            seed=None,
            fit=lambda: ({"item_counts": counts}, {"backend": "exact_integer_count"}),
            scorer=lambda arrays, rows: np.broadcast_to(
                np.asarray(arrays["item_counts"], dtype=np.float64),
                (rows.size, training.item_ids.size),
            ),
        )
    )

    seeds = [int(x) for x in model_config["training_seeds"]]
    als_configurations = enumerate_als_configurations(model_config)
    for configuration in als_configurations:
        for seed in seeds:
            def fit_als(configuration=configuration, seed=seed):
                fitted = fit_implicit_als(
                    training.ownership,
                    training.playtime_forever,
                    factors=configuration["factors"],
                    regularization=configuration["regularization"],
                    alpha_o=configuration["alpha_o"],
                    alpha_p=configuration["alpha_p"],
                    tau=configuration["tau"],
                    iterations=configuration["iterations"],
                    training_seed=seed,
                    num_threads=model_config["implicit_als"]["num_threads"],
                )
                return {
                    "user_factors": fitted.user_factors,
                    "item_factors": fitted.item_factors,
                }, fitted.diagnostics
            entries.append(
                execute(
                    family="implicit_als",
                    configuration=configuration,
                    seed=seed,
                    fit=fit_als,
                    scorer=lambda arrays, rows: np.asarray(arrays["user_factors"][rows], dtype=np.float64) @ np.asarray(arrays["item_factors"], dtype=np.float64).T,
                )
            )

    bpr_configurations = enumerate_bpr_configurations(model_config)
    identity_entries: list[dict[str, Any]] = []
    for configuration in bpr_configurations:
        for seed in seeds:
            def fit_bpr(configuration=configuration, seed=seed):
                fitted = fit_feature_sum_bpr(
                    training.ownership,
                    warm_genres,
                    cycle_id=cycle_id,
                    training_seed=seed,
                    factors=configuration["factors"],
                    regularization=configuration["regularization"],
                    learning_rate=configuration["learning_rate"],
                    epochs=configuration["epochs"],
                    samples_per_epoch=configuration["samples_per_epoch"],
                    include_genre=False,
                )
                p = fitted.parameters
                return {"user_factors": p.user_factors, "identity_factors": p.identity_factors, "feature_factors": p.feature_factors, "item_bias": p.item_bias}, fitted.diagnostics
            entry = execute(
                family="feature_sum_bpr_identity",
                configuration=configuration,
                seed=seed,
                fit=fit_bpr,
                scorer=lambda arrays, rows: np.asarray(arrays["user_factors"][rows], dtype=np.float64) @ np.asarray(arrays["identity_factors"], dtype=np.float64).T + np.asarray(arrays["item_bias"], dtype=np.float64),
            )
            entries.append(entry)
            identity_entries.append(entry)

    def select_configuration(family_entries: Sequence[Mapping[str, Any]]) -> str:
        grouped: dict[str, list[Mapping[str, Any]]] = {}
        for entry in family_entries:
            grouped.setdefault(str(entry["configuration_id"]), []).append(entry)
        rows = []
        for configuration_id, values in grouped.items():
            if len(values) != len(seeds):
                raise ValueError("a stochastic validation configuration lacks seeds")
            mean_ndcg = float(np.mean([x["aggregate_validation_metrics"]["mean_ndcg_at_20"] for x in values]))
            mean_recall = float(np.mean([x["aggregate_validation_metrics"]["mean_recall_at_20"] for x in values]))
            rows.append((configuration_id, mean_ndcg, mean_recall, int(values[0]["parameter_count"]), int(values[0]["configuration"].get("factors", 0))))
        return sorted(rows, key=lambda x: (-x[1], -x[2], x[3], x[4], x[0]))[0][0]

    selected_als = select_configuration([x for x in entries if x["family"] == "implicit_als"])
    selected_identity = select_configuration(identity_entries)
    selected_identity_config = next(x for x in bpr_configurations if x["configuration_id"] == selected_identity)
    genre_entries: list[dict[str, Any]] = []
    for seed in seeds:
        def fit_genre(seed=seed):
            fitted = fit_feature_sum_bpr(
                training.ownership,
                warm_genres,
                cycle_id=cycle_id,
                training_seed=seed,
                factors=selected_identity_config["factors"],
                regularization=selected_identity_config["regularization"],
                learning_rate=selected_identity_config["learning_rate"],
                epochs=selected_identity_config["epochs"],
                samples_per_epoch=selected_identity_config["samples_per_epoch"],
                include_genre=True,
            )
            p = fitted.parameters
            return {"user_factors": p.user_factors, "identity_factors": p.identity_factors, "feature_factors": p.feature_factors, "item_bias": p.item_bias}, fitted.diagnostics
        genre_configuration = {**selected_identity_config, "configuration_id": f"genre__{selected_identity}"}
        entry = execute(
            family="feature_sum_bpr_identity_genre",
            configuration=genre_configuration,
            seed=seed,
            fit=fit_genre,
            scorer=lambda arrays, rows: np.asarray(arrays["user_factors"][rows], dtype=np.float64) @ (np.asarray(arrays["identity_factors"], dtype=np.float64) + warm_genres @ np.asarray(arrays["feature_factors"], dtype=np.float64)).T + np.asarray(arrays["item_bias"], dtype=np.float64),
        )
        entries.append(entry)
        genre_entries.append(entry)

    leaderboard = Path(leaderboard_path).resolve()
    leaderboard.parent.mkdir(parents=True, exist_ok=True)
    fields = ["family", "configuration_id", "training_seed", "parameter_count", "mean_recall_at_10", "mean_recall_at_20", "mean_ndcg_at_10", "mean_ndcg_at_20", "mean_expected_rank", "expected_catalogue_coverage_at_20", "expected_top_one_percent_concentration_at_20", "runtime_seconds_total", "run_id"]
    with leaderboard.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for entry in entries:
            metric = entry["aggregate_validation_metrics"]
            writer.writerow({
                "family": entry["family"], "configuration_id": entry["configuration_id"], "training_seed": entry["training_seed"], "parameter_count": entry["parameter_count"],
                "mean_recall_at_10": metric["mean_recall_at_10"], "mean_recall_at_20": metric["mean_recall_at_20"], "mean_ndcg_at_10": metric["mean_ndcg_at_10"], "mean_ndcg_at_20": metric["mean_ndcg_at_20"], "mean_expected_rank": metric["mean_expected_rank"],
                "expected_catalogue_coverage_at_20": metric["expected_catalogue_coverage_at_20"], "expected_top_one_percent_concentration_at_20": metric["expected_top_one_percent_concentration_at_20"], "runtime_seconds_total": entry["runtime_seconds_total"], "run_id": entry["run_id"],
            })

    training_manifest: dict[str, Any] = {
        "schema_version": 1, "artifact": "stage1_complete_validation_training_and_ranking", "cycle_id": cycle_id, "protocol_id": protocol_id,
        "interaction_set_id": interaction["interaction_set_id"], "split_set_id": split["split_set_id"], "feature_set_id": feature_public["feature_set_id"], "estimator_specification_id": estimator["specification_id"], "backend_spike_manifest_id": spike["manifest_id"],
        "runs": [{key: entry[key] for key in ("run_id", "family", "configuration_id", "training_seed", "status", "parameter_count", "model_artifact", "metric_artifact", "aggregate_validation_metrics")} for entry in entries],
        "selection": {"implicit_als": selected_als, "feature_sum_bpr_identity": selected_identity, "feature_sum_bpr_identity_genre": f"genre__{selected_identity}"},
        "leaderboard": {"path": _relative(leaderboard, root), "size_bytes": leaderboard.stat().st_size, "sha256": file_sha256(leaderboard)},
        "access_boundary": {"validation_targets_and_metrics": "accessed_for_s1_6", "design_test_targets": "sealed_not_accessed", "assessment_ids_or_histories": "sealed_not_accessed", "pseudo_cold_identifiers": "not_accessed", "stage2_objectives_or_bundle_outcomes": "not_accessed"},
    }
    training_manifest["manifest_id"] = semantic_sha256(training_manifest)
    _atomic_json(Path(training_manifest_path).resolve(), training_manifest)

    pop_entry = next(x for x in entries if x["family"] == "popularity")
    pop_metric = _load_metric(root / pop_entry["metric_artifact"]["path"], pop_entry["metric_artifact"]["sha256"])
    winners = {
        "implicit_als": [x for x in entries if x["family"] == "implicit_als" and x["configuration_id"] == selected_als],
        "feature_sum_bpr_identity": [x for x in entries if x["family"] == "feature_sum_bpr_identity" and x["configuration_id"] == selected_identity],
        "feature_sum_bpr_identity_genre": genre_entries,
    }
    criteria = ranking_config["stage2_admission"]
    decisions: dict[str, Any] = {}
    admitted: list[str] = []
    for family, family_entries in winners.items():
        family_entries = sorted(family_entries, key=lambda x: int(x["training_seed"]))
        loaded_metrics = [_load_metric(root / x["metric_artifact"]["path"], x["metric_artifact"]["sha256"]) for x in family_entries]
        ndcg_differences = np.mean(np.vstack([x["ndcg_at_20"] - pop_metric["ndcg_at_20"] for x in loaded_metrics]), axis=0)
        recall_difference = float(np.mean([np.mean(x["recall_at_20"] - pop_metric["recall_at_20"]) for x in loaded_metrics]))
        interval = paired_bootstrap_mean_difference(ndcg_differences, np.zeros_like(ndcg_differences), replicates=ranking_config["uncertainty"]["paired_user_bootstrap_replicates"], seed=ranking_config["uncertainty"]["bootstrap_seed"], confidence_level=ranking_config["uncertainty"]["confidence_level"])
        personalized_pass = interval.lower >= criteria["personalized_noninferiority_vs_popularity"]["paired_ndcg_at_20_lower_95_bound_minimum"] and recall_difference >= criteria["personalized_noninferiority_vs_popularity"]["mean_recall_at_20_difference_minimum"]
        genre_pass = True
        genre_contrast = None
        if family == "feature_sum_bpr_identity_genre":
            identity_winners = sorted(winners["feature_sum_bpr_identity"], key=lambda x: int(x["training_seed"]))
            identity_metrics = [_load_metric(root / x["metric_artifact"]["path"], x["metric_artifact"]["sha256"]) for x in identity_winners]
            genre_ndcg = np.mean(np.vstack([a["ndcg_at_20"] - b["ndcg_at_20"] for a, b in zip(loaded_metrics, identity_metrics)]), axis=0)
            genre_recall = float(np.mean([np.mean(a["recall_at_20"] - b["recall_at_20"]) for a, b in zip(loaded_metrics, identity_metrics)]))
            genre_interval = paired_bootstrap_mean_difference(genre_ndcg, np.zeros_like(genre_ndcg), replicates=ranking_config["uncertainty"]["paired_user_bootstrap_replicates"], seed=ranking_config["uncertainty"]["bootstrap_seed"], confidence_level=ranking_config["uncertainty"]["confidence_level"])
            genre_pass = genre_interval.lower >= criteria["genre_noninferiority_vs_identity"]["paired_ndcg_at_20_lower_95_bound_minimum"] and genre_recall >= criteria["genre_noninferiority_vs_identity"]["mean_recall_at_20_difference_minimum"]
            genre_contrast = {"paired_ndcg_at_20": asdict(genre_interval), "mean_recall_at_20_difference": genre_recall, "passes": genre_pass}
        passes = bool(personalized_pass and genre_pass)
        decisions[family] = {"configuration_id": family_entries[0]["configuration_id"], "seed_run_ids": [x["run_id"] for x in family_entries], "versus_popularity": {"paired_ndcg_at_20": asdict(interval), "mean_recall_at_20_difference": recall_difference, "passes": personalized_pass}, "genre_versus_identity": genre_contrast, "admitted": passes}
        if passes:
            admitted.append(family)
    narrowed_label = False
    if not admitted:
        best_family = max(winners, key=lambda family: float(np.mean([x["aggregate_validation_metrics"]["mean_ndcg_at_20"] for x in winners[family]])))
        admitted = [best_family]
        decisions[best_family]["admitted"] = True
        decisions[best_family]["admission_override"] = criteria["if_none_qualify"]
        narrowed_label = True
    admitted = admitted[: int(criteria["maximum_personalized_families"])]
    admission: dict[str, Any] = {
        "schema_version": 1, "artifact": "stage1_validation_selected_pretest_admission", "cycle_id": cycle_id, "protocol_id": protocol_id,
        "training_manifest_id": training_manifest["manifest_id"], "selection": training_manifest["selection"], "criteria": criteria, "decisions": decisions, "admitted_families": admitted, "narrowed_methodological_label": narrowed_label,
        "timing": "hashed_before_any_design_test_target_access", "access_boundary": {"validation_metrics": "accessed", "design_test_targets": "sealed_not_accessed", "assessment_ids_or_histories": "sealed_not_accessed", "pseudo_cold_identifiers": "not_accessed", "stage2_objectives_or_bundle_outcomes": "not_accessed"},
    }
    admission["admission_id"] = semantic_sha256(admission)
    _atomic_json(Path(admission_path).resolve(), admission)
    return admission


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Stage 1 validation tuning")
    parser.parse_args(argv)
    result = run_validation()
    print(json.dumps({"admission_id": result["admission_id"], "admitted_families": result["admitted_families"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
