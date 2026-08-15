"""Publish the cycle-scoped backend-neutral S1.4 estimator contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from src.stage1_protocol import (
    canonical_json_bytes,
    file_sha256,
    load_json,
    semantic_sha256,
    validate_protocol_configs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CYCLE_ID = "s1-v2-20260814"
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "configs" / "cycles" / DEFAULT_CYCLE_ID
DEFAULT_CYCLE_DIR = PROJECT_ROOT / "outputs" / "modeling" / "cycles" / DEFAULT_CYCLE_ID
DEFAULT_OUTPUT = DEFAULT_CYCLE_DIR / "stage1_estimator_spec_manifest.json"


def _entry(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def build_estimator_manifest(
    *,
    project_root: str | Path = PROJECT_ROOT,
    model_config_path: str | Path = DEFAULT_CONFIG_DIR / "preference_models.json",
    ranking_config_path: str | Path = DEFAULT_CONFIG_DIR / "ranking_evaluation.json",
    protocol_manifest_path: str | Path = DEFAULT_CYCLE_DIR / "stage1_protocol_manifest.json",
    interaction_manifest_path: str | Path = DEFAULT_CYCLE_DIR / "stage1_interaction_manifest.json",
    split_manifest_path: str | Path = DEFAULT_CYCLE_DIR / "stage1_split_manifest.json",
    feature_manifest_path: str | Path = DEFAULT_CYCLE_DIR / "item_feature_manifest.json",
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    paths = {
        "model_config": Path(model_config_path).resolve(),
        "ranking_config": Path(ranking_config_path).resolve(),
        "protocol_manifest": Path(protocol_manifest_path).resolve(),
        "interaction_manifest": Path(interaction_manifest_path).resolve(),
        "split_manifest": Path(split_manifest_path).resolve(),
        "feature_manifest": Path(feature_manifest_path).resolve(),
        "reference_equations": (root / "src" / "preference_model.py").resolve(),
        "production_backend": (root / "src" / "stage1_backend.py").resolve(),
    }
    model = load_json(paths["model_config"])
    ranking = load_json(paths["ranking_config"])
    protocol = load_json(paths["protocol_manifest"])
    interaction = load_json(paths["interaction_manifest"])
    split = load_json(paths["split_manifest"])
    features = load_json(paths["feature_manifest"])
    validate_protocol_configs(model, ranking)
    cycle_id = str(model["cycle_id"])
    model_hash = semantic_sha256(model)
    ranking_hash = semantic_sha256(ranking)
    protocol_id = semantic_sha256(
        {"model_config_sha256": model_hash, "ranking_config_sha256": ranking_hash}
    )
    if protocol.get("protocol_id") != protocol_id:
        raise ValueError("protocol/config binding changed")
    for dependency in (interaction, split, features):
        if dependency.get("cycle_id") != cycle_id or dependency.get("protocol_id") != protocol_id:
            raise ValueError("estimator dependency identity changed")
    inputs = {label: _entry(path, root) for label, path in sorted(paths.items())}
    conventions = {
        "interpretation": model["interpretation"],
        "popularity": "design-training binary ownership count",
        "implicit_als": {
            "target": "binary ownership",
            "confidence": model["implicit_als"]["confidence_equation"],
            "unobserved_confidence": 1.0,
            "objective": "sum_ui c_ui*(o_ui-x_u_dot_q_i)^2 + lambda*(squared_frobenius_X+squared_frobenius_Q)",
            "iterations": model["implicit_als"]["iterations"],
            "score": "x_u dot q_i",
        },
        "feature_sum_bpr": {
            "score": "b_i + x_u dot (eta_i + rho*F_i*G)",
            "rho_identity": 0.0,
            "rho_genre": 1.0,
            "loss": "sum_triples logaddexp(0,-(s_ui-s_uj)) + lambda*squared_l2_active_parameters",
            "sampler_namespace": f"{cycle_id}:bpr:<training_seed>:triple-sampler",
            "sampler": model["pairwise_feature_sum"]["negative_sampling"],
            "epochs": model["pairwise_feature_sum"]["epochs"],
            "samples_per_epoch": model["pairwise_feature_sum"]["samples_per_epoch"],
        },
        "fold_in": model["fold_in"],
        "score_materialization": {
            "full_dense_user_catalogue_saved": False,
            "bounded_blocks_only": True,
        },
    }
    dependencies = {
        "protocol_id": protocol_id,
        "interaction_set_id": interaction["interaction_set_id"],
        "split_set_id": split["split_set_id"],
        "feature_set_id": features["feature_set_id"],
    }
    identity = {
        "cycle_id": cycle_id,
        "dependencies": dependencies,
        "conventions": conventions,
        "input_sha256": {label: value["sha256"] for label, value in inputs.items()},
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "artifact": "stage1_backend_neutral_estimator_specification",
        "cycle_id": cycle_id,
        **dependencies,
        "specification_id": semantic_sha256(identity),
        "conventions": conventions,
        "parameter_schemas": {
            "popularity": {"item_counts": ["n_items", "int64"]},
            "implicit_als": {"user_factors": ["n_users", "k", "float32"], "item_factors": ["n_items", "k", "float32"]},
            "feature_sum_bpr_identity": {"user_factors": ["n_users", "k", "float32"], "identity_factors": ["n_items", "k", "float32"], "item_bias": ["n_items", "float32"]},
            "feature_sum_bpr_identity_genre": {"user_factors": ["n_users", "k", "float32"], "identity_factors": ["n_items", "k", "float32"], "feature_factors": ["n_genres", "k", "float32"], "item_bias": ["n_items", "float32"]},
        },
        "inputs": inputs,
        "backend_status": {"implicit_0_7_2": "to_be_exercised_in_s1_5", "lightfm_1_17": "to_be_exercised_in_s1_5", "fallback": "requires_prospective_s1_5_amendment"},
        "access_boundary": {
            "real_model_fit": False,
            "validation_targets_or_metrics": "not_accessed",
            "design_test_targets": "sealed_not_accessed",
            "assessment_ids_or_histories": "sealed_not_accessed",
            "stage2_objectives_or_bundle_outcomes": "not_accessed",
        },
    }
    manifest["manifest_id"] = semantic_sha256(manifest)
    return manifest


def generate_estimator_manifest(*, output_path: str | Path = DEFAULT_OUTPUT, **kwargs: Any) -> dict[str, Any]:
    destination = Path(output_path)
    expected = build_estimator_manifest(**kwargs)
    if destination.exists():
        saved = load_json(destination)
        if saved != expected:
            raise FileExistsError("refusing to overwrite a changed estimator specification")
        return saved
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(canonical_json_bytes(expected))
    return expected


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Freeze the cycle-scoped S1.4 estimator contract")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    result = generate_estimator_manifest(output_path=args.output)
    print(json.dumps({"specification_id": result["specification_id"], "manifest_id": result["manifest_id"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
