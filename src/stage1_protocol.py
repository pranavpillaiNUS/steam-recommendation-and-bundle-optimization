"""Frozen Stage 1 protocol helpers.

This module exists to make S1.0 executable rather than leaving the split and
model-selection rules as notebook prose.  It contains no fitting code and must
not inspect validation, design-test, assessment, or bundle outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_CONFIG = PROJECT_ROOT / "configs" / "preference_models.json"
DEFAULT_RANKING_CONFIG = PROJECT_ROOT / "configs" / "ranking_evaluation.json"
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "outputs" / "modeling" / "stage1_protocol_manifest.json"
)


def canonical_json_bytes(value: Any) -> bytes:
    """Return the unique UTF-8 JSON representation used for content IDs."""

    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def semantic_sha256(value: Any) -> str:
    """Hash JSON semantics independently of key order or whitespace."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path, *, chunk_bytes: int = 1024 * 1024) -> str:
    """Hash a file without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON object and reject non-object roots."""

    with Path(path).open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    if not isinstance(result, dict):
        raise ValueError(f"configuration root must be an object: {path}")
    return result


def _hash_field(digest: "hashlib._Hash", value: Any) -> None:
    payload = canonical_json_bytes(value)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def stable_hash_uint64(namespace: str, *values: Any) -> int:
    """Return a stable unsigned hash with unambiguous typed field boundaries."""

    digest = hashlib.sha256()
    _hash_field(digest, namespace)
    for value in values:
        _hash_field(digest, value)
    return int.from_bytes(digest.digest()[:8], "big", signed=False)


def activity_band(activity: int, lower_bounds: Sequence[int]) -> int:
    """Map activity to the index of its largest applicable lower bound."""

    bounds = np.asarray(lower_bounds, dtype=np.int64)
    if bounds.ndim != 1 or bounds.size == 0 or np.any(np.diff(bounds) <= 0):
        raise ValueError("activity lower bounds must be a nonempty increasing vector")
    index = int(np.searchsorted(bounds, int(activity), side="right") - 1)
    if index < 0:
        raise ValueError(f"activity {activity} is below the first eligible bound")
    return index


def stratified_hash_partition(
    ids: Sequence[Any],
    strata: Sequence[Any],
    *,
    assessment_fraction: float,
    namespace: str,
) -> dict[Any, str]:
    """Assign exact per-stratum assessment counts independent of input order.

    The first ``floor(fraction * stratum_size)`` IDs in stable hash order enter
    assessment.  The returned mapping is keyed by the original scalar IDs so a
    caller can restore any desired row order explicitly.
    """

    if len(ids) != len(strata):
        raise ValueError("ids and strata must have equal length")
    if not 0.0 < assessment_fraction < 1.0:
        raise ValueError("assessment_fraction must lie strictly between zero and one")
    if len(set(ids)) != len(ids):
        raise ValueError("ids must be unique")

    groups: dict[Any, list[Any]] = {}
    for identifier, stratum in zip(ids, strata):
        groups.setdefault(stratum, []).append(identifier)

    assignment: dict[Any, str] = {}
    for stratum in sorted(groups, key=lambda value: canonical_json_bytes(value)):
        ordered = sorted(
            groups[stratum],
            key=lambda identifier: (
                stable_hash_uint64(namespace, stratum, identifier),
                canonical_json_bytes(identifier),
            ),
        )
        n_assessment = int(np.floor(assessment_fraction * len(ordered)))
        assessment = set(ordered[:n_assessment])
        for identifier in ordered:
            assignment[identifier] = (
                "assessment" if identifier in assessment else "design"
            )
    return assignment


def _float_slug(value: float) -> str:
    text = format(float(value), ".12g")
    return text.replace("-", "m").replace(".", "p").replace("+", "")


def enumerate_als_configurations(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Expand the frozen ALS Cartesian grid in deterministic ID order."""

    als = config["implicit_als"]
    result: list[dict[str, Any]] = []
    for factors in als["factors"]:
        for regularization in als["regularization"]:
            for alpha_o in als["alpha_o"]:
                for scheme in als["confidence_schemes"]:
                    configuration_id = (
                        f"als__k{int(factors):03d}"
                        f"__reg{_float_slug(regularization)}"
                        f"__ao{_float_slug(alpha_o)}"
                        f"__{scheme['id']}"
                    )
                    result.append(
                        {
                            "configuration_id": configuration_id,
                            "factors": int(factors),
                            "regularization": float(regularization),
                            "alpha_o": float(alpha_o),
                            "alpha_p": float(scheme["alpha_p"]),
                            "tau": float(scheme["tau"]),
                            "confidence_scheme": scheme["id"],
                            "iterations": int(als["iterations"]),
                        }
                    )
    return sorted(result, key=lambda row: row["configuration_id"])


def enumerate_bpr_configurations(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Expand the frozen pairwise grid in deterministic ID order."""

    bpr = config["pairwise_feature_sum"]
    result: list[dict[str, Any]] = []
    for factors in bpr["factors"]:
        for regularization in bpr["regularization"]:
            for learning_rate in bpr["learning_rate"]:
                configuration_id = (
                    f"bpr__k{int(factors):03d}"
                    f"__reg{_float_slug(regularization)}"
                    f"__lr{_float_slug(learning_rate)}"
                )
                result.append(
                    {
                        "configuration_id": configuration_id,
                        "factors": int(factors),
                        "regularization": float(regularization),
                        "learning_rate": float(learning_rate),
                        "epochs": int(bpr["epochs"]),
                        "samples_per_epoch": int(bpr["samples_per_epoch"]),
                    }
                )
    return sorted(result, key=lambda row: row["configuration_id"])


def validate_protocol_configs(
    model_config: Mapping[str, Any], ranking_config: Mapping[str, Any]
) -> None:
    """Reject protocol files that violate binding S1.0 invariants."""

    for config in (model_config, ranking_config):
        if config.get("schema_version") != 1:
            raise ValueError("unsupported Stage 1 configuration schema")
    common = ("cycle_id", "frozen_at_utc", "repository_baseline_commit")
    for key in common:
        if model_config.get(key) != ranking_config.get(key):
            raise ValueError(f"configuration mismatch for {key}")

    ladder = model_config.get("model_ladder")
    required_ladder = [
        "popularity",
        "implicit_als",
        "feature_sum_bpr_identity",
        "feature_sum_bpr_identity_genre",
    ]
    if ladder != required_ladder:
        raise ValueError("the required four-rung model ladder changed")
    seeds = model_config.get("training_seeds", [])
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError("at least three distinct training seeds are required")
    if model_config["genre_features"].get("tags_enabled") is not False:
        raise ValueError("tags cannot be enabled in the frozen core")
    if model_config["pairwise_feature_sum"]["early_stopping"] != (
        "disabled_fixed_epochs"
    ):
        raise ValueError("the controlled BPR ablation requires fixed epochs")
    if model_config["pairwise_feature_sum"]["controlled_ablation"].get(
        "only_toggle"
    ) != "genre_feature_block":
        raise ValueError("genre must be the only controlled-ablation toggle")

    outer = ranking_config["outer_user_split"]
    if outer["design_fraction"] != 0.8 or outer["assessment_fraction"] != 0.2:
        raise ValueError("outer user split must remain 80/20")
    if outer.get("assessment_ids_available_to_tuning") is not False:
        raise ValueError("assessment IDs must be unavailable to tuning")
    if ranking_config["ranking"]["primary_metric"] != "mean_ndcg_at_20":
        raise ValueError("mean NDCG@20 is the frozen primary selection metric")
    if ranking_config["ranking"]["tie_policy"] != (
        "expected_uniform_random_order_within_exact_score_tied_block"
    ):
        raise ValueError("the common expected tie policy changed")
    if ranking_config["ranking"].get("save_dense_score_matrix") is not False:
        raise ValueError("dense catalogue score matrices are prohibited")
    protected = ranking_config["protected_outcomes"]
    if protected.get("stage2_objectives_available") is not False:
        raise ValueError("Stage 2 objectives cannot be available during Stage 1")
    if protected.get("bundle_outcomes_available") is not False:
        raise ValueError("bundle outcomes cannot be available during Stage 1")

    if len(enumerate_als_configurations(model_config)) != 24:
        raise ValueError("unexpected ALS grid size")
    if len(enumerate_bpr_configurations(model_config)) != 4:
        raise ValueError("unexpected BPR grid size")


def _package_versions(names: Iterable[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def build_protocol_manifest(
    model_config: Mapping[str, Any],
    ranking_config: Mapping[str, Any],
    *,
    verify_inputs: bool = True,
    model_config_path: str | Path = DEFAULT_MODEL_CONFIG,
    ranking_config_path: str | Path = DEFAULT_RANKING_CONFIG,
) -> dict[str, Any]:
    """Build the deterministic S1.0 manifest without reading protected outcomes."""

    validate_protocol_configs(model_config, ranking_config)
    inputs = ranking_config["inputs"]
    verified_inputs: dict[str, dict[str, Any]] = {}
    for label, path_key, hash_key in (
        ("interactions", "interactions_path", "interactions_sha256"),
        ("features", "features_path", "features_sha256"),
    ):
        expected = str(inputs[hash_key]).lower()
        path = PROJECT_ROOT / inputs[path_key]
        actual = file_sha256(path) if verify_inputs else expected
        if actual != expected:
            raise ValueError(f"{label} input hash mismatch: {actual} != {expected}")
        verified_inputs[label] = {
            "path": inputs[path_key],
            "sha256": actual,
            "size_bytes": path.stat().st_size if path.exists() else None,
        }

    model_id = semantic_sha256(model_config)
    ranking_id = semantic_sha256(ranking_config)
    protocol_id = semantic_sha256(
        {
            "model_config_sha256": model_id,
            "ranking_config_sha256": ranking_id,
        }
    )
    return {
        "schema_version": 1,
        "cycle_id": model_config["cycle_id"],
        "frozen_at_utc": model_config["frozen_at_utc"],
        "timestamp_policy": "fixed_in_preregistered_configs",
        "repository_baseline_commit": model_config["repository_baseline_commit"],
        "protocol_id": protocol_id,
        "configs": {
            "preference_models": {
                "path": Path(model_config_path).resolve().relative_to(
                    PROJECT_ROOT.resolve()
                ).as_posix(),
                "semantic_sha256": model_id,
            },
            "ranking_evaluation": {
                "path": Path(ranking_config_path).resolve().relative_to(
                    PROJECT_ROOT.resolve()
                ).as_posix(),
                "semantic_sha256": ranking_id,
            },
        },
        "inputs": verified_inputs,
        "grid": {
            "training_seeds": list(model_config["training_seeds"]),
            "als_configurations": enumerate_als_configurations(model_config),
            "bpr_configurations": enumerate_bpr_configurations(model_config),
            "tags_enabled": False,
        },
        "environment_at_freeze": {
            "python": platform.python_version(),
            "packages": _package_versions(
                ["numpy", "scipy", "pandas", "scikit-learn", "implicit", "lightfm"]
            ),
        },
        "protected_outcomes": dict(ranking_config["protected_outcomes"]),
        "first_permitted_next_action": "S1.1 sparse ID and interaction contract",
    }


def write_manifest(manifest: Mapping[str, Any], path: str | Path) -> None:
    """Write one stable, human-readable protocol manifest."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.write_text(payload, encoding="utf-8", newline="\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate and freeze the Stage 1 protocol")
    parser.add_argument("--models", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--ranking", type=Path, default=DEFAULT_RANKING_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)

    model_config = load_json(args.models)
    ranking_config = load_json(args.ranking)
    manifest = build_protocol_manifest(
        model_config,
        ranking_config,
        model_config_path=args.models,
        ranking_config_path=args.ranking,
    )
    if not args.check_only:
        write_manifest(manifest, args.manifest)
    print(
        json.dumps(
            {
                "protocol_id": manifest["protocol_id"],
                "als_configurations": len(manifest["grid"]["als_configurations"]),
                "bpr_configurations": len(manifest["grid"]["bpr_configurations"]),
                "training_seeds": len(manifest["grid"]["training_seeds"]),
                "wrote_manifest": not args.check_only,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
