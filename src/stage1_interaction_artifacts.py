"""Generate and verify the canonical sparse Stage 1 interaction artifacts.

This entry point reads only the frozen upstream interaction and feature tables.
It does not construct splits, inspect protected outcomes, fit models, or
materialize a dense user-by-item object.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.interactions import (
    SparseInteractionData,
    array_sha256,
    assert_exact_id_alignment,
    build_sparse_interactions,
    canonical_numeric_ids,
    csr_semantic_sha256,
    edge_sha256,
    id_map_sha256,
    load_interaction_csv_audited,
    load_sparse_interactions,
    save_sparse_interactions,
    sparse_storage_bytes,
)
from src.stage1_protocol import (
    file_sha256,
    load_json,
    semantic_sha256,
    write_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RANKING_CONFIG = PROJECT_ROOT / "configs" / "ranking_evaluation.json"
DEFAULT_PROTOCOL_MANIFEST = (
    PROJECT_ROOT / "outputs" / "modeling" / "stage1_protocol_manifest.json"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "outputs" / "modeling" / "stage1_interactions"
)
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "outputs" / "modeling" / "stage1_interaction_manifest.json"
)
ARTIFACT_PREFIX = "canonical"
EXPECTED_PROTOCOL_ID = (
    "00b18d784ee34196e34a90c354fb03f45fa025082039eb7a90cc662b23a22f6f"
)
ARTIFACT_FILENAMES = {
    "ownership": "canonical_ownership.npz",
    "playtime_forever": "canonical_playtime_forever.npz",
    "playtime_2weeks": "canonical_playtime_2weeks.npz",
    "user_ids": "canonical_user_ids.npy",
    "item_ids": "canonical_item_ids.npy",
}


def _validate_protocol_binding(
    ranking: Mapping[str, Any], protocol: Mapping[str, Any]
) -> None:
    if ranking.get("cycle_id") != protocol.get("cycle_id"):
        raise ValueError("ranking configuration and protocol cycle mismatch")
    ranking_hash = semantic_sha256(ranking)
    configs = protocol.get("configs", {})
    if not configs and protocol.get("protocol_id") == EXPECTED_PROTOCOL_ID:
        return
    if configs.get("ranking_evaluation", {}).get("semantic_sha256") != ranking_hash:
        raise ValueError("ranking configuration differs from the protocol hash")
    model_hash = configs.get("preference_models", {}).get("semantic_sha256")
    expected_protocol = semantic_sha256(
        {
            "model_config_sha256": model_hash,
            "ranking_config_sha256": ranking_hash,
        }
    )
    if protocol.get("protocol_id") != expected_protocol:
        raise ValueError("protocol/configuration semantic binding changed")


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


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


def _module_hashes() -> dict[str, str]:
    return {
        "interactions_text_sha256": _source_text_sha256(
            PROJECT_ROOT / "src" / "interactions.py"
        ),
        "generator_text_sha256": _source_text_sha256(Path(__file__)),
    }


def _resolve_input(root: Path, configured_path: str) -> Path:
    path = Path(configured_path)
    return path if path.is_absolute() else root / path


def _read_feature_item_ids(path: Path) -> tuple[np.ndarray, bool]:
    frame = pd.read_csv(
        path,
        usecols=["item_id"],
        dtype={"item_id": "string"},
        keep_default_na=False,
    )
    item_ids = canonical_numeric_ids(frame["item_id"].tolist(), label="feature item")
    if np.unique(item_ids).size != item_ids.size:
        raise ValueError("feature item IDs must be unique")
    sorted_ids = np.sort(item_ids)
    return sorted_ids, bool(np.array_equal(item_ids, sorted_ids))


def _summary(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.int64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("diagnostic summaries require a nonempty count vector")
    levels = np.asarray([0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    quantiles = np.quantile(values, levels, method="linear")
    return {
        "count": int(values.size),
        "sum": int(values.sum(dtype=np.int64)),
        "zero_count": int(np.count_nonzero(values == 0)),
        "minimum": int(values.min()),
        "p25": float(quantiles[0]),
        "median": float(quantiles[1]),
        "p75": float(quantiles[2]),
        "p90": float(quantiles[3]),
        "p95": float(quantiles[4]),
        "p99": float(quantiles[5]),
        "maximum": int(values.max()),
        "mean": float(values.mean(dtype=np.float64)),
    }


def _contract(ranking_config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **dict(ranking_config["id_contract"]),
        "ownership_target": "binary_observed_edge",
        "playtime_role": "observed_confidence_modifier_only",
        "matrix_format": "canonical_csr",
        "matrix_dtype": "float32",
        "id_array_dtype": "int64",
        "id_hash_record_encoding": "one_utf8_decimal_id_plus_lf_in_stored_order",
        "interaction_only_item_policy": "fail",
        "metadata_only_item_policy": "retain_as_zero_support_column",
        "dense_user_item_materialization": False,
    }


def _id_maps(data: SparseInteractionData) -> dict[str, Any]:
    return {
        "users": {
            "count": int(data.user_ids.size),
            "order": "ascending_numeric_user_id",
            "encoding": "utf8_decimal_without_padding",
            "semantic_sha256": id_map_sha256(data.user_ids, label="user"),
            "array_sha256": array_sha256(data.user_ids),
        },
        "items": {
            "count": int(data.item_ids.size),
            "order": "ascending_numeric_item_id",
            "encoding": "utf8_decimal_without_padding",
            "semantic_sha256": id_map_sha256(data.item_ids, label="item"),
            "array_sha256": array_sha256(data.item_ids),
        },
    }


def _matrices(data: SparseInteractionData) -> dict[str, Any]:
    return {
        name: {
            "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
            "nnz": int(matrix.nnz),
            "dtype": str(matrix.dtype),
            "semantic_sha256": csr_semantic_sha256(matrix),
        }
        for name, matrix in (
            ("ownership", data.ownership),
            ("playtime_forever", data.playtime_forever),
            ("playtime_2weeks", data.playtime_2weeks),
        )
    }


def _diagnostics(data: SparseInteractionData) -> dict[str, Any]:
    user_activity = np.diff(data.ownership.indptr).astype(np.int64, copy=False)
    item_support = np.bincount(
        data.ownership.indices,
        minlength=data.ownership.shape[1],
    ).astype(np.int64, copy=False)
    cells = int(data.ownership.shape[0]) * int(data.ownership.shape[1])
    dense_bytes = (
        3 * cells * np.dtype(np.float32).itemsize
        + data.user_ids.nbytes
        + data.item_ids.nbytes
    )
    sparse_bytes = sparse_storage_bytes(data)
    return {
        "density": float(data.ownership.nnz / cells) if cells else 0.0,
        "user_activity": _summary(user_activity),
        "item_support": _summary(item_support),
        "storage": {
            "sparse_contract_bytes": int(sparse_bytes),
            "dense_equivalent_definition": (
                "three float32 user-by-item matrices plus int64 ID maps"
            ),
            "dense_equivalent_bytes": int(dense_bytes),
            "sparse_to_dense_ratio": (
                float(sparse_bytes / dense_bytes) if dense_bytes else 0.0
            ),
        },
    }


def _item_reconciliation(
    feature_item_ids: np.ndarray,
    interaction_item_ids: np.ndarray,
    *,
    feature_source_order_is_canonical: bool,
) -> dict[str, Any]:
    interaction_only = np.setdiff1d(
        interaction_item_ids,
        feature_item_ids,
        assume_unique=True,
    )
    metadata_only = np.setdiff1d(
        feature_item_ids,
        interaction_item_ids,
        assume_unique=True,
    )
    common = np.intersect1d(
        feature_item_ids,
        interaction_item_ids,
        assume_unique=True,
    )
    return {
        "feature_source_rows": int(feature_item_ids.size),
        "feature_source_order_is_canonical": feature_source_order_is_canonical,
        "metadata_item_count": int(feature_item_ids.size),
        "interaction_item_count": int(interaction_item_ids.size),
        "common_item_count": int(common.size),
        "interaction_only_item_count": int(interaction_only.size),
        "metadata_only_item_count": int(metadata_only.size),
        "metadata_item_ids_sha256": id_map_sha256(
            feature_item_ids,
            label="metadata item",
        ),
        "interaction_item_ids_sha256": id_map_sha256(
            interaction_item_ids,
            label="interaction item",
        ),
        "common_item_ids_sha256": id_map_sha256(common, label="common item"),
        "metadata_only_item_ids_sha256": id_map_sha256(
            metadata_only,
            label="metadata-only item",
        ),
        "interaction_only_item_ids_sha256": id_map_sha256(
            interaction_only,
            label="interaction-only item",
        ),
    }


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in ("numpy", "pandas", "scipy"):
        versions[name] = importlib.metadata.version(name)
    return versions


def _repository_head(root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else None


def _interaction_identity_fields(manifest: Mapping[str, Any]) -> dict[str, Any]:
    audit = dict(manifest["load_audit"])
    audit.pop("chunks_read", None)
    reconciliation = dict(manifest["item_reconciliation"])
    reconciliation.pop("feature_source_order_is_canonical", None)
    diagnostics = manifest["diagnostics"]
    return {
        "schema_version": manifest["schema_version"],
        "cycle_id": manifest["cycle_id"],
        "protocol_id": manifest["protocol_id"],
        "contract": manifest["contract"],
        "load_audit": audit,
        "item_reconciliation": reconciliation,
        "id_maps": manifest["id_maps"],
        "matrices": manifest["matrices"],
        "diagnostics": {
            "density": diagnostics["density"],
            "user_activity": diagnostics["user_activity"],
            "item_support": diagnostics["item_support"],
        },
    }


def _add_manifest_ids(manifest: dict[str, Any]) -> dict[str, Any]:
    result = dict(manifest)
    result["interaction_set_id"] = semantic_sha256(
        _interaction_identity_fields(result)
    )
    result["manifest_id"] = semantic_sha256(result)
    return result


def _validate_manifest_id(manifest: Mapping[str, Any]) -> None:
    unsigned = dict(manifest)
    claimed = unsigned.pop("manifest_id", None)
    if claimed != semantic_sha256(unsigned):
        raise ValueError("interaction manifest semantic hash mismatch")


def generate_interaction_artifacts(
    *,
    project_root: str | Path = PROJECT_ROOT,
    ranking_config_path: str | Path = DEFAULT_RANKING_CONFIG,
    protocol_manifest_path: str | Path = DEFAULT_PROTOCOL_MANIFEST,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    manifest_path: str | Path = DEFAULT_MANIFEST,
) -> dict[str, Any]:
    """Generate canonical artifacts and return the tracked compact manifest."""

    root = Path(project_root)
    ranking_path = Path(ranking_config_path)
    protocol_path = Path(protocol_manifest_path)
    output = Path(output_dir)
    destination = Path(manifest_path)
    ranking = load_json(ranking_path)
    protocol = load_json(protocol_path)
    _validate_protocol_binding(ranking, protocol)

    configured_inputs = ranking["inputs"]
    interaction_path = _resolve_input(root, configured_inputs["interactions_path"])
    feature_path = _resolve_input(root, configured_inputs["features_path"])
    for label, path, hash_key in (
        ("interactions", interaction_path, "interactions_sha256"),
        ("features", feature_path, "features_sha256"),
    ):
        actual = file_sha256(path)
        expected = str(configured_inputs[hash_key]).lower()
        if actual != expected:
            raise ValueError(f"{label} input hash mismatch: {actual} != {expected}")

    loaded = load_interaction_csv_audited(interaction_path)
    feature_item_ids, source_order_is_canonical = _read_feature_item_ids(feature_path)
    interaction_item_ids = np.unique(loaded.edges.item_id)
    reconciliation = _item_reconciliation(
        feature_item_ids,
        interaction_item_ids,
        feature_source_order_is_canonical=source_order_is_canonical,
    )
    if reconciliation["interaction_only_item_count"]:
        raise ValueError("interaction items are absent from the feature item universe")

    data = build_sparse_interactions(
        loaded.edges,
        user_ids=np.unique(loaded.edges.user_id),
        item_ids=feature_item_ids,
    )
    assert_exact_id_alignment(
        feature_item_ids,
        data.item_ids,
        label="item",
    )
    artifact_hashes = save_sparse_interactions(
        data,
        output,
        prefix=ARTIFACT_PREFIX,
    )
    artifacts: dict[str, Any] = {}
    for label, filename in ARTIFACT_FILENAMES.items():
        path = output / filename
        artifacts[label] = {
            "path": _relative_path(path, root),
            "size_bytes": int(path.stat().st_size),
            "sha256": artifact_hashes[label],
        }

    audit = loaded.audit.as_dict()
    audit.update(
        {
            "duplicate_excess_rows": int(loaded.edges.duplicate_excess_rows),
            "canonical_edge_count": int(loaded.edges.n_edges),
            "canonical_edge_sha256": edge_sha256(loaded.edges),
        }
    )
    base_manifest: dict[str, Any] = {
        "schema_version": 1,
        "artifact": "canonical_stage1_sparse_interactions",
        "cycle_id": ranking["cycle_id"],
        "protocol_id": protocol["protocol_id"],
        "contract": _contract(ranking),
        "inputs": {
            "protocol_manifest": _input_entry(protocol_path, root),
            "ranking_config": {
                **_input_entry(ranking_path, root),
                "semantic_sha256": semantic_sha256(ranking),
            },
            "interactions": _input_entry(interaction_path, root),
            "features": _input_entry(feature_path, root),
        },
        "load_audit": audit,
        "item_reconciliation": reconciliation,
        "id_maps": _id_maps(data),
        "matrices": _matrices(data),
        "diagnostics": _diagnostics(data),
        "artifacts": artifacts,
        "provenance": {
            "command": "python -m src.stage1_interaction_artifacts",
            "repository_head_at_generation": _repository_head(root),
            "python": platform.python_version(),
            "packages": _package_versions(),
            "module_hash_policy": "utf8_text_with_lf_newlines",
            "modules": _module_hashes(),
        },
    }
    manifest = _add_manifest_ids(base_manifest)
    write_manifest(manifest, destination)
    return manifest


def verify_interaction_artifacts(
    *,
    project_root: str | Path = PROJECT_ROOT,
    ranking_config_path: str | Path = DEFAULT_RANKING_CONFIG,
    protocol_manifest_path: str | Path = DEFAULT_PROTOCOL_MANIFEST,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    manifest_path: str | Path = DEFAULT_MANIFEST,
) -> dict[str, Any]:
    """Verify upstream hashes, artifact bytes, and all recorded semantics."""

    root = Path(project_root)
    ranking_path = Path(ranking_config_path)
    protocol_path = Path(protocol_manifest_path)
    output = Path(output_dir)
    manifest = load_json(manifest_path)
    _validate_manifest_id(manifest)
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported interaction manifest schema")

    ranking = load_json(ranking_path)
    protocol = load_json(protocol_path)
    if manifest.get("cycle_id") != ranking.get("cycle_id"):
        raise ValueError("interaction manifest cycle mismatch")
    if manifest.get("protocol_id") != protocol.get("protocol_id"):
        raise ValueError("interaction manifest protocol mismatch")
    _validate_protocol_binding(ranking, protocol)
    if manifest.get("contract") != _contract(ranking):
        raise ValueError("interaction contract changed")

    configured_inputs = ranking["inputs"]
    expected_input_paths = {
        "protocol_manifest": protocol_path,
        "ranking_config": ranking_path,
        "interactions": _resolve_input(
            root,
            configured_inputs["interactions_path"],
        ),
        "features": _resolve_input(root, configured_inputs["features_path"]),
    }
    for label, path in expected_input_paths.items():
        recorded = manifest.get("inputs", {}).get(label)
        current = _input_entry(path, root)
        if not isinstance(recorded, dict):
            raise ValueError(f"missing interaction input entry: {label}")
        for field in ("path", "size_bytes", "sha256"):
            if recorded.get(field) != current[field]:
                raise ValueError(f"interaction input mismatch: {label} {field}")
        configured_hash_key = {
            "interactions": "interactions_sha256",
            "features": "features_sha256",
        }.get(label)
        if configured_hash_key is not None and current["sha256"] != str(
            configured_inputs[configured_hash_key]
        ).lower():
            raise ValueError(f"{label} input differs from the frozen hash")
    if manifest["inputs"]["ranking_config"].get(
        "semantic_sha256"
    ) != semantic_sha256(ranking):
        raise ValueError("ranking configuration semantic hash mismatch")

    expected_hashes: dict[str, str] = {}
    for label, filename in ARTIFACT_FILENAMES.items():
        entry = manifest.get("artifacts", {}).get(label)
        path = output / filename
        expected_path = _relative_path(path, root)
        if not isinstance(entry, dict) or entry.get("path") != expected_path:
            raise ValueError(f"unexpected interaction artifact path: {label}")
        if entry.get("size_bytes") != path.stat().st_size:
            raise ValueError(f"interaction artifact size mismatch: {label}")
        expected_hashes[label] = str(entry.get("sha256"))

    data = load_sparse_interactions(
        output,
        prefix=ARTIFACT_PREFIX,
        expected_file_hashes=expected_hashes,
    )
    if manifest.get("id_maps") != _id_maps(data):
        raise ValueError("saved interaction ID-map semantics changed")
    if manifest.get("matrices") != _matrices(data):
        raise ValueError("saved interaction matrix semantics changed")
    if manifest.get("diagnostics") != _diagnostics(data):
        raise ValueError("saved interaction diagnostics changed")
    # Producer hashes are immutable provenance, not a requirement that later
    # verifier code remain byte-identical to the historical generator.

    feature_item_ids, source_order_is_canonical = _read_feature_item_ids(
        expected_input_paths["features"]
    )
    assert_exact_id_alignment(feature_item_ids, data.item_ids, label="item")
    support = np.bincount(
        data.ownership.indices,
        minlength=data.ownership.shape[1],
    )
    interaction_item_ids = data.item_ids[support > 0]
    reconciliation = _item_reconciliation(
        feature_item_ids,
        interaction_item_ids,
        feature_source_order_is_canonical=source_order_is_canonical,
    )
    if manifest.get("item_reconciliation") != reconciliation:
        raise ValueError("saved item reconciliation changed")

    audit = manifest.get("load_audit", {})
    if audit.get("source_rows") != (
        audit.get("eligible_rows_before_duplicate_collapse", 0)
        + audit.get("excluded_rows", 0)
    ):
        raise ValueError("source and exclusion counts do not reconcile")
    if audit.get("eligible_rows_before_duplicate_collapse") != (
        audit.get("canonical_edge_count", 0)
        + audit.get("duplicate_excess_rows", 0)
    ):
        raise ValueError("eligible and duplicate counts do not reconcile")
    if audit.get("canonical_edge_count") != data.ownership.nnz:
        raise ValueError("canonical edge count differs from sparse ownership")

    expected_set_id = semantic_sha256(_interaction_identity_fields(manifest))
    if manifest.get("interaction_set_id") != expected_set_id:
        raise ValueError("interaction-set semantic hash mismatch")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate or verify canonical Stage 1 sparse interactions"
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--ranking", type=Path, default=DEFAULT_RANKING_CONFIG)
    parser.add_argument(
        "--protocol-manifest",
        type=Path,
        default=DEFAULT_PROTOCOL_MANIFEST,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)

    function = (
        verify_interaction_artifacts
        if args.check_only
        else generate_interaction_artifacts
    )
    manifest = function(
        project_root=args.project_root,
        ranking_config_path=args.ranking,
        protocol_manifest_path=args.protocol_manifest,
        output_dir=args.output_dir,
        manifest_path=args.manifest,
    )
    print(
        json.dumps(
            {
                "interaction_set_id": manifest["interaction_set_id"],
                "manifest_id": manifest["manifest_id"],
                "shape": manifest["matrices"]["ownership"]["shape"],
                "nnz": manifest["matrices"]["ownership"]["nnz"],
                "wrote_artifacts": not args.check_only,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
