"""Freeze and verify the backend-neutral S1.4 estimator specification.

This publication binds written equations, reference implementations, numerical
policies, and artifact schemas. It does not fit a model, load held-out targets,
exercise production backends, or create downstream scores.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import tempfile
from typing import Any, Mapping, Sequence

from src.stage1_interaction_artifacts import EXPECTED_PROTOCOL_ID
from src.stage1_protocol import (
    file_sha256,
    load_json,
    semantic_sha256,
    validate_protocol_configs,
    write_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_CYCLE_ID = "s1-v1-20260718"
DEFAULT_MODEL_CONFIG = PROJECT_ROOT / "configs" / "preference_models.json"
DEFAULT_RANKING_CONFIG = PROJECT_ROOT / "configs" / "ranking_evaluation.json"
DEFAULT_PROTOCOL_MANIFEST = (
    PROJECT_ROOT / "outputs" / "modeling" / "stage1_protocol_manifest.json"
)
DEFAULT_INTERACTION_MANIFEST = (
    PROJECT_ROOT / "outputs" / "modeling" / "stage1_interaction_manifest.json"
)
DEFAULT_SPLIT_MANIFEST = (
    PROJECT_ROOT / "outputs" / "modeling" / "stage1_split_manifest.json"
)
DEFAULT_FEATURE_MANIFEST = (
    PROJECT_ROOT / "outputs" / "modeling" / "item_feature_manifest.json"
)
DEFAULT_SPECIFICATION_NOTE = (
    PROJECT_ROOT / "notes" / "preference_model_specification.md"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "modeling"
    / "stage1_estimator_spec_manifest.json"
)

SCHEMA_VERSION = 1


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"estimator specification input escaped root: {path}") from exc


def _input_entry(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": _relative_path(path, root),
        "size_bytes": int(path.stat().st_size),
        "sha256": file_sha256(path),
    }


def _source_text_sha256(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _validate_self_hash(manifest: Mapping[str, Any], *, label: str) -> None:
    unsigned = dict(manifest)
    manifest_id = unsigned.pop("manifest_id", None)
    if manifest_id != semantic_sha256(unsigned):
        raise ValueError(f"{label} manifest semantic hash mismatch")


def _validate_dependencies(
    *,
    model_config: Mapping[str, Any],
    ranking_config: Mapping[str, Any],
    protocol: Mapping[str, Any],
    interaction: Mapping[str, Any],
    split: Mapping[str, Any],
    features: Mapping[str, Any],
) -> None:
    validate_protocol_configs(model_config, ranking_config)
    if model_config.get("cycle_id") != EXPECTED_CYCLE_ID:
        raise ValueError("unexpected estimator specification cycle")
    model_hash = semantic_sha256(model_config)
    ranking_hash = semantic_sha256(ranking_config)
    expected_protocol = semantic_sha256(
        {
            "model_config_sha256": model_hash,
            "ranking_config_sha256": ranking_hash,
        }
    )
    if expected_protocol != EXPECTED_PROTOCOL_ID:
        raise ValueError("frozen configuration protocol changed")
    if protocol.get("protocol_id") != expected_protocol:
        raise ValueError("protocol manifest differs from frozen configurations")
    if protocol.get("configs", {}).get("preference_models", {}).get(
        "semantic_sha256"
    ) != model_hash:
        raise ValueError("preference-model semantic hash changed")
    if protocol.get("configs", {}).get("ranking_evaluation", {}).get(
        "semantic_sha256"
    ) != ranking_hash:
        raise ValueError("ranking semantic hash changed")

    for label, manifest in (
        ("interaction", interaction),
        ("split", split),
        ("feature", features),
    ):
        _validate_self_hash(manifest, label=label)
        if manifest.get("cycle_id") != EXPECTED_CYCLE_ID:
            raise ValueError(f"{label} manifest cycle changed")
        if manifest.get("protocol_id") != EXPECTED_PROTOCOL_ID:
            raise ValueError(f"{label} manifest protocol changed")
    if split.get("interaction_set_id") != interaction.get("interaction_set_id"):
        raise ValueError("split/interaction dependency changed")
    if features.get("interaction_set_id") != interaction.get(
        "interaction_set_id"
    ):
        raise ValueError("feature/interaction dependency changed")
    if features.get("split_set_id") != split.get("split_set_id"):
        raise ValueError("feature/split dependency changed")


def _conventions(model_config: Mapping[str, Any]) -> dict[str, Any]:
    als = model_config["implicit_als"]
    pairwise = model_config["pairwise_feature_sum"]
    numerical = model_config["numerical_policy"]
    return {
        "interpretation": model_config["interpretation"],
        "popularity": {
            "score": "sum_of_design_training_binary_ownership_by_item",
            "tie_source": "exact_equal_integer_counts",
            "parameters": 0,
        },
        "implicit_als": {
            "target": "binary_ownership",
            "playtime_field": "playtime_forever",
            "playtime_field_status": (
                "prospective_s1_4_clarification_before_any_model_fit"
            ),
            "confidence": als["confidence_equation"],
            "unobserved_confidence": 1.0,
            "owned_unplayed_confidence": "1 + alpha_o",
            "objective": (
                "sum_ui c_ui*(o_ui-x_u_dot_q_i)^2"
                "+lambda_x*||X||_F^2+lambda_q*||Q||_F^2"
            ),
            "core_regularization": "lambda_x_equals_lambda_q",
            "normal_equations": (
                "baseline_gram_plus_sparse_observed_c_minus_one_correction"
            ),
            "right_hand_side": "observed_item_factors_transpose_times_full_c",
            "linear_solver": "float64_cholesky_without_explicit_inverse",
            "jitter_sequence": list(numerical["linear_solve_jitter"]),
            "stored_factor_dtype": numerical["float_dtype"],
            "accumulator_dtype": numerical["accumulator_dtype"],
            "optimization_claim": (
                "each_fixed_block_is_convex_ridge_joint_factorization_is_"
                "nonconvex_no_global_optimum_claim"
            ),
            "iterations": als["iterations"],
            "early_stopping": als["early_stopping"],
        },
        "pairwise_feature_sum": {
            "score": "b_i + x_u dot (eta_i + rho * F_i G)",
            "genre_matrix_role": (
                "genre_block_only_identity_is_eta_and_is_not_duplicated_in_F"
            ),
            "loss": (
                "sum_triples logaddexp(0,-(s_ui-s_uj))"
                "+lambda*||active_parameters||_2^2"
            ),
            "loss_reduction": "sum",
            "regularization_gradient": "2*lambda*parameter",
            "item_bias": pairwise["item_bias"],
            "user_bias": pairwise["user_bias"],
            "identity_rho": 0.0,
            "identity_plus_genre_rho": 1.0,
            "rho_status": "fixed_block_weight_not_an_optimized_parameter",
            "inactive_content_regularization": "excluded",
            "optimizer": pairwise["optimizer"],
            "epochs": pairwise["epochs"],
            "early_stopping": pairwise["early_stopping"],
            "sampler": {
                "positive": "uniform_training_edges_with_replacement",
                "negative": (
                    "uniform_warm_items_with_replacement_reject_training_"
                    "positives_only"
                ),
                "validation_and_test_positive_status": (
                    "eligible_as_unobserved_negatives_because_not_in_training"
                ),
                "rng": (
                    "numpy_generator_pcg64_seeded_by_first_eight_sha256_"
                    "bytes_as_unsigned_big_endian"
                ),
                "rng_namespace": (
                    "s1-v1-20260718:bpr:<training_seed>:triple-sampler"
                ),
                "epoch_policy": "one_continuing_stream_not_reset_per_epoch",
                "draw_order": (
                    "scalar_positive_edge_then_scalar_negative_proposals_"
                    "including_rejections"
                ),
                "full_catalogue_user": "fail_before_sampling",
                "stream_sharing": (
                    "same_saved_or_hash_verified_triples_for_identity_and_genre"
                ),
                "stream_hash_encoding": (
                    "ordered_little_endian_int64_u_i_j_rows"
                ),
            },
            "optimization_claim": (
                "joint_training_is_nonconvex_multiple_seeds_and_heldout_"
                "ranking_support_selection_no_global_optimum_claim"
            ),
        },
        "score_materialization": {
            "dense_full_user_item_matrix": False,
            "bounded_pair_or_block_scoring": True,
        },
    }


def _parameter_schemas() -> dict[str, Any]:
    return {
        "popularity": {
            "item_counts": ["n_items", "int64"],
        },
        "implicit_als": {
            "user_factors": ["n_users", "k", "float32"],
            "item_factors": ["n_items", "k", "float32"],
        },
        "feature_sum_bpr_identity": {
            "user_factors": ["n_users", "k", "float32"],
            "identity_factors": ["n_items", "k", "float32"],
            "item_bias": ["n_items", "float32"],
        },
        "feature_sum_bpr_identity_genre": {
            "user_factors": ["n_users", "k", "float32"],
            "identity_factors": ["n_items", "k", "float32"],
            "feature_factors": ["n_genres", "k", "float32"],
            "item_bias": ["n_items", "float32"],
        },
    }


def _diagnostic_schema() -> dict[str, Any]:
    return {
        "common": [
            "configuration_id",
            "training_seed",
            "status",
            "runtime_seconds",
            "nonfinite_failure",
            "package_versions",
            "user_item_map_hashes",
        ],
        "implicit_als": [
            "initial_objective",
            "post_user_block_objective_by_iteration",
            "post_item_block_objective_by_iteration",
            "final_objective_after_float32_cast",
            "maximum_absolute_solve_residual",
            "maximum_scaled_solve_residual",
            "jitter_counts",
        ],
        "pairwise": [
            "triple_stream_sha256_by_epoch",
            "loss_trace",
            "gradient_nonfinite_failure",
        ],
        "failure_policy": "invalidate_configuration_without_unlogged_retry",
    }


def _artifact_schema() -> dict[str, Any]:
    return {
        "later_model_artifact": {
            "arrays": "only_parameter_arrays_declared_by_model_family",
            "metadata": (
                "configuration_seed_ids_map_hashes_feature_set_id_and_array_"
                "sha256_values"
            ),
            "dense_score_matrix": "prohibited",
        },
        "score_reproduction": (
            "equation_plus_configuration_plus_hash_verified_parameter_arrays"
        ),
        "fold_in": "not_implemented_or_claimed_until_s1_5_and_s1_10",
    }


def build_estimator_spec_manifest(
    *,
    project_root: str | Path = PROJECT_ROOT,
    model_config_path: str | Path = DEFAULT_MODEL_CONFIG,
    ranking_config_path: str | Path = DEFAULT_RANKING_CONFIG,
    protocol_manifest_path: str | Path = DEFAULT_PROTOCOL_MANIFEST,
    interaction_manifest_path: str | Path = DEFAULT_INTERACTION_MANIFEST,
    split_manifest_path: str | Path = DEFAULT_SPLIT_MANIFEST,
    feature_manifest_path: str | Path = DEFAULT_FEATURE_MANIFEST,
    specification_note_path: str | Path = DEFAULT_SPECIFICATION_NOTE,
) -> dict[str, Any]:
    """Build the deterministic public S1.4 specification manifest."""

    root = Path(project_root).resolve()
    paths = {
        "preference_models_config": Path(model_config_path).resolve(),
        "ranking_config": Path(ranking_config_path).resolve(),
        "protocol_manifest": Path(protocol_manifest_path).resolve(),
        "interaction_manifest": Path(interaction_manifest_path).resolve(),
        "split_manifest": Path(split_manifest_path).resolve(),
        "feature_manifest": Path(feature_manifest_path).resolve(),
        "mathematical_specification": Path(specification_note_path).resolve(),
        "reference_implementation": (
            root / "src" / "preference_model.py"
        ).resolve(),
    }
    model_config = load_json(paths["preference_models_config"])
    ranking_config = load_json(paths["ranking_config"])
    protocol = load_json(paths["protocol_manifest"])
    interaction = load_json(paths["interaction_manifest"])
    split = load_json(paths["split_manifest"])
    features = load_json(paths["feature_manifest"])
    _validate_dependencies(
        model_config=model_config,
        ranking_config=ranking_config,
        protocol=protocol,
        interaction=interaction,
        split=split,
        features=features,
    )

    conventions = _conventions(model_config)
    parameter_schemas = _parameter_schemas()
    diagnostic_schema = _diagnostic_schema()
    artifact_schema = _artifact_schema()
    inputs = {
        label: _input_entry(path, root)
        for label, path in sorted(paths.items())
    }
    source_hashes = {
        "preference_model_text_sha256": _source_text_sha256(
            paths["reference_implementation"]
        ),
        "specification_note_text_sha256": _source_text_sha256(
            paths["mathematical_specification"]
        ),
        "generator_text_sha256": _source_text_sha256(Path(__file__)),
    }
    dependencies = {
        "protocol_id": protocol["protocol_id"],
        "interaction_set_id": interaction["interaction_set_id"],
        "split_set_id": split["split_set_id"],
        "feature_set_id": features["feature_set_id"],
        "preference_models_semantic_sha256": semantic_sha256(model_config),
    }
    specification_identity = {
        "cycle_id": EXPECTED_CYCLE_ID,
        "dependencies": dependencies,
        "conventions": conventions,
        "parameter_schemas": parameter_schemas,
        "diagnostic_schema": diagnostic_schema,
        "artifact_schema": artifact_schema,
        "source_hashes": source_hashes,
    }
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact": "stage1_backend_neutral_estimator_specification",
        "cycle_id": EXPECTED_CYCLE_ID,
        **dependencies,
        "specification_id": semantic_sha256(specification_identity),
        "conventions": conventions,
        "parameter_schemas": parameter_schemas,
        "diagnostic_schema": diagnostic_schema,
        "artifact_schema": artifact_schema,
        "inputs": inputs,
        "provenance": {
            "command": "python -m src.stage1_estimator_spec",
            "module_hash_policy": "utf8_text_with_lf_newlines",
            "modules": source_hashes,
            "python": platform.python_version(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "scipy")
            },
        },
        "backend_status": {
            "implicit": {
                "intended_version": model_config["implicit_als"][
                    "preferred_version"
                ],
                "equivalence": "not_tested_until_s1_5",
            },
            "lightfm": {
                "intended_version": model_config["pairwise_feature_sum"][
                    "preferred_version"
                ],
                "equivalence": "not_tested_until_s1_5",
            },
            "fallback_activation": (
                "requires_documented_prospective_amendment_after_s1_5_failure"
            ),
        },
        "access_boundary": {
            "real_model_fit": False,
            "validation_targets_or_metrics": "not_accessed",
            "design_test_targets": "sealed_not_accessed",
            "assessment_ids_or_histories": "sealed_not_accessed",
            "pseudo_cold_items": "reserved_for_s1_8_not_accessed",
            "stage2_objectives_or_bundle_outcomes": "not_accessed",
        },
    }
    manifest["manifest_id"] = semantic_sha256(manifest)
    return manifest


def generate_estimator_spec_manifest(
    *,
    output_path: str | Path = DEFAULT_OUTPUT,
    **kwargs: Any,
) -> dict[str, Any]:
    """Publish the immutable public specification without overwriting."""

    output = Path(output_path).resolve()
    expected = build_estimator_spec_manifest(**kwargs)
    if output.exists():
        saved = load_json(output)
        if saved != expected:
            raise FileExistsError(
                "refusing to overwrite a nonidentical estimator specification"
            )
        return saved

    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".stage1-estimator-spec-",
        suffix=".json",
        dir=output.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        write_manifest(expected, temporary)
        os.link(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return expected


def verify_estimator_spec_manifest(
    *,
    output_path: str | Path = DEFAULT_OUTPUT,
    **kwargs: Any,
) -> dict[str, Any]:
    """Rebuild and compare the S1.4 specification without writes."""

    output = Path(output_path).resolve()
    saved = load_json(output)
    _validate_self_hash(saved, label="estimator specification")
    expected = build_estimator_spec_manifest(**kwargs)
    if saved != expected:
        raise ValueError("saved estimator specification changed")
    return saved


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate or verify the S1.4 estimator specification"
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--models", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--ranking", type=Path, default=DEFAULT_RANKING_CONFIG)
    parser.add_argument(
        "--protocol-manifest",
        type=Path,
        default=DEFAULT_PROTOCOL_MANIFEST,
    )
    parser.add_argument(
        "--interaction-manifest",
        type=Path,
        default=DEFAULT_INTERACTION_MANIFEST,
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=DEFAULT_SPLIT_MANIFEST,
    )
    parser.add_argument(
        "--feature-manifest",
        type=Path,
        default=DEFAULT_FEATURE_MANIFEST,
    )
    parser.add_argument(
        "--specification-note",
        type=Path,
        default=DEFAULT_SPECIFICATION_NOTE,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    common = {
        "project_root": args.project_root,
        "model_config_path": args.models,
        "ranking_config_path": args.ranking,
        "protocol_manifest_path": args.protocol_manifest,
        "interaction_manifest_path": args.interaction_manifest,
        "split_manifest_path": args.split_manifest,
        "feature_manifest_path": args.feature_manifest,
        "specification_note_path": args.specification_note,
    }
    if args.check_only:
        manifest = verify_estimator_spec_manifest(
            output_path=args.output,
            **common,
        )
    else:
        manifest = generate_estimator_spec_manifest(
            output_path=args.output,
            **common,
        )
    print(
        json.dumps(
            {
                "specification_id": manifest["specification_id"],
                "manifest_id": manifest["manifest_id"],
                "check_only": bool(args.check_only),
                "real_model_fit": manifest["access_boundary"]["real_model_fit"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
