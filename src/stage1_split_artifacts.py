"""Generate and verify the frozen S1.2 split artifacts.

The public manifest contains rules, counts, and hashes only. User and item
assignments are stored under the ignored protected directory. The generator
uses pre-model identifiers, support counts, playtime, and metadata only. It
does not inspect model scores, validation metrics, design-test results,
assessment outcomes, or Stage 2 information.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.interactions import (
    SparseInteractionData,
    array_sha256,
    assert_exact_id_alignment,
    canonical_numeric_ids,
    csr_semantic_sha256,
    load_sparse_interactions,
    remove_observed_pairs,
    save_sparse_interactions,
    sparse_storage_bytes,
)
from src.splits import (
    EXCLUDED_LOW_ACTIVITY,
    EXCLUDED_NONWARM_ITEM,
    activity_band_outer_split,
    capacity_aware_edge_split,
    primary_genre_from_csv,
    proportional_evaluation_user_sample,
    select_pseudo_cold_items,
)
from src.stage1_interaction_artifacts import (
    DEFAULT_MANIFEST as DEFAULT_INTERACTION_MANIFEST,
    EXPECTED_PROTOCOL_ID,
    verify_interaction_artifacts,
)
from src.stage1_protocol import (
    file_sha256,
    load_json,
    semantic_sha256,
    write_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_CYCLE_ID = "s1-v1-20260718"
DEFAULT_RANKING_CONFIG = PROJECT_ROOT / "configs" / "ranking_evaluation.json"
DEFAULT_PROTOCOL_MANIFEST = (
    PROJECT_ROOT / "outputs" / "modeling" / "stage1_protocol_manifest.json"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "modeling"
    / "protected"
    / EXPECTED_CYCLE_ID
    / "stage1_splits"
)
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "outputs" / "modeling" / "stage1_split_manifest.json"
)
TRAINING_PREFIX = "design_training"

OUTER_FILENAME = "outer_user_split.npz"
NESTED_FILENAME = "nested_interaction_split.npz"
VALIDATION_FILENAME = "validation_targets.npz"
VALIDATION_DIAGNOSTICS_FILENAME = "validation_target_diagnostics.npz"
VALIDATION_MASK_FILENAME = "validation_other_holdout_mask.npz"
TEST_FILENAME = "design_test_targets.npz"
EVALUATION_FILENAME = "evaluation_user_sample.npz"
PSEUDO_COLD_FILENAME = "pseudo_cold_items.npz"

NPZ_RELATIVE_PATHS = {
    "outer_user_split": Path("sealed") / "audit" / OUTER_FILENAME,
    "nested_interaction_split": Path("sealed") / "audit" / NESTED_FILENAME,
    "validation_targets": (
        Path("permitted") / "validation" / VALIDATION_FILENAME
    ),
    "validation_target_diagnostics": (
        Path("reserved") / "s1_7" / VALIDATION_DIAGNOSTICS_FILENAME
    ),
    "validation_other_holdout_mask": (
        Path("mask_only") / VALIDATION_MASK_FILENAME
    ),
    "design_test_targets": (
        Path("sealed") / "design_test" / TEST_FILENAME
    ),
    "evaluation_user_sample": (
        Path("permitted") / "validation" / EVALUATION_FILENAME
    ),
    "pseudo_cold_items": (
        Path("reserved") / "s1_8" / PSEUDO_COLD_FILENAME
    ),
}
ACCESS_CLASSES = {
    "outer_user_split": "audit_only_contains_sealed_assessment_ids",
    "nested_interaction_split": "audit_only_contains_sealed_design_test_ids",
    "validation_targets": "validation_tuning_permitted",
    "validation_target_diagnostics": "reserved_for_s1_7_diagnostics",
    "validation_other_holdout_mask": "opaque_validation_mask_only",
    "design_test_targets": "sealed_until_validation_admission_is_hashed",
    "evaluation_user_sample": "validation_tuning_permitted",
    "pseudo_cold_items": "reserved_for_s1_8",
}
TRAINING_RELATIVE_DIR = Path("permitted") / "design_training"
TRAINING_ARTIFACT_NAMES = (
    "ownership",
    "playtime_forever",
    "playtime_2weeks",
    "user_ids",
    "item_ids",
)

OUTER_SPLIT_CODES = {
    EXCLUDED_LOW_ACTIVITY: 0,
    "design": 1,
    "assessment": 2,
}
ROLE_CODES = {
    EXCLUDED_NONWARM_ITEM: 0,
    "training": 1,
    "validation": 2,
    "test": 3,
}
USER_STATUS_CODES = {
    "insufficient_warm_history": 0,
    "capacity_exhausted": 1,
    "evaluable": 2,
}


@dataclass(frozen=True)
class SplitState:
    """In-memory deterministic split state before physical serialization."""

    arrays: Mapping[str, Mapping[str, np.ndarray]]
    training: SparseInteractionData
    summary: Mapping[str, Any]


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
        "splits_text_sha256": _source_text_sha256(
            PROJECT_ROOT / "src" / "splits.py"
        ),
        "stage1_protocol_text_sha256": _source_text_sha256(
            PROJECT_ROOT / "src" / "stage1_protocol.py"
        ),
        "generator_text_sha256": _source_text_sha256(Path(__file__)),
    }


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


def _package_versions() -> dict[str, str]:
    return {
        name: importlib.metadata.version(name)
        for name in ("numpy", "pandas", "scipy")
    }


def _fixed_unicode(values: Sequence[str]) -> np.ndarray:
    width = max((len(value) for value in values), default=1)
    return np.asarray(values, dtype=f"<U{width}")


def _hex_to_uint64(values: Sequence[str]) -> np.ndarray:
    return np.asarray(
        [int(value, 16) if value else 0 for value in values],
        dtype=np.uint64,
    )


def _array_semantics(
    arrays: Mapping[str, np.ndarray],
) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "dtype": array.dtype.str,
            "shape": [int(value) for value in array.shape],
            "semantic_sha256": array_sha256(array),
        }
        for name, array in sorted(arrays.items())
    }


def _npz_semantics(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    fields = _array_semantics(arrays)
    return {
        "fields": fields,
        "semantic_sha256": semantic_sha256(fields),
    }


def _training_semantics(data: SparseInteractionData) -> dict[str, Any]:
    return {
        "user_count": int(data.user_ids.size),
        "item_count": int(data.item_ids.size),
        "shape": [int(data.ownership.shape[0]), int(data.ownership.shape[1])],
        "nnz": int(data.ownership.nnz),
        "sparse_storage_bytes": sparse_storage_bytes(data),
        "user_ids_sha256": array_sha256(data.user_ids),
        "item_ids_sha256": array_sha256(data.item_ids),
        "ownership_sha256": csr_semantic_sha256(data.ownership),
        "playtime_forever_sha256": csr_semantic_sha256(
            data.playtime_forever
        ),
        "playtime_2weeks_sha256": csr_semantic_sha256(
            data.playtime_2weeks
        ),
    }


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _assert_arrays_equal(
    expected: Mapping[str, np.ndarray],
    actual: Mapping[str, np.ndarray],
    *,
    label: str,
) -> None:
    if set(expected) != set(actual):
        raise ValueError(f"{label} field names changed")
    for name in expected:
        left = expected[name]
        right = actual[name]
        if left.dtype != right.dtype or left.shape != right.shape:
            raise ValueError(f"{label} field contract changed: {name}")
        if not np.array_equal(left, right):
            raise ValueError(f"{label} field values changed: {name}")


def _load_interactions(
    *,
    root: Path,
    ranking_path: Path,
    protocol_path: Path,
    interaction_manifest_path: Path,
) -> tuple[SparseInteractionData, dict[str, Any]]:
    saved_interaction_manifest = load_json(interaction_manifest_path)
    artifact_dir = (
        root
        / saved_interaction_manifest["artifacts"]["ownership"]["path"]
    ).parent
    interaction_manifest = verify_interaction_artifacts(
        project_root=root,
        ranking_config_path=ranking_path,
        protocol_manifest_path=protocol_path,
        output_dir=artifact_dir,
        manifest_path=interaction_manifest_path,
    )
    artifact_entries = interaction_manifest["artifacts"]
    hashes = {
        name: artifact_entries[name]["sha256"]
        for name in TRAINING_ARTIFACT_NAMES
    }
    data = load_sparse_interactions(
        artifact_dir,
        prefix="canonical",
        expected_file_hashes=hashes,
    )
    return data, interaction_manifest


def _slice_sparse_contract(
    data: SparseInteractionData,
    *,
    user_ids: np.ndarray,
    item_ids: np.ndarray,
) -> SparseInteractionData:
    user_positions = np.searchsorted(data.user_ids, user_ids)
    item_positions = np.searchsorted(data.item_ids, item_ids)
    assert_exact_id_alignment(
        user_ids,
        data.user_ids[user_positions],
        label="user",
    )
    assert_exact_id_alignment(
        item_ids,
        data.item_ids[item_positions],
        label="item",
    )

    def sliced(matrix):
        return matrix[user_positions, :][:, item_positions].tocsr()

    return SparseInteractionData(
        ownership=sliced(data.ownership),
        playtime_forever=sliced(data.playtime_forever),
        playtime_2weeks=sliced(data.playtime_2weeks),
        user_ids=user_ids,
        item_ids=item_ids,
    )


def _pair_values(
    data: SparseInteractionData,
    user_ids: np.ndarray,
    item_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rows = np.searchsorted(data.user_ids, user_ids)
    cols = np.searchsorted(data.item_ids, item_ids)
    if user_ids.size:
        assert_exact_id_alignment(
            user_ids,
            data.user_ids[rows],
            label="target user",
        )
        assert_exact_id_alignment(
            item_ids,
            data.item_ids[cols],
            label="target item",
        )
    owned = np.asarray(data.ownership[rows, cols]).reshape(-1)
    if not np.array_equal(owned, np.ones(user_ids.size, dtype=np.float32)):
        raise ValueError("a split target is not an observed ownership edge")
    forever = np.asarray(data.playtime_forever[rows, cols]).reshape(-1)
    recent = np.asarray(data.playtime_2weeks[rows, cols]).reshape(-1)
    return (
        forever.astype(np.float32, copy=True),
        recent.astype(np.float32, copy=True),
    )


def _target_arrays(
    nested: pd.DataFrame,
    base_data: SparseInteractionData,
    *,
    role: str,
) -> dict[str, np.ndarray]:
    targets = nested.loc[nested["role"].eq(role), ["user_id", "item_id"]]
    users = targets["user_id"].to_numpy(dtype=np.int64, copy=True)
    items = targets["item_id"].to_numpy(dtype=np.int64, copy=True)
    forever, recent = _pair_values(base_data, users, items)
    return {
        "user_ids": users,
        "item_ids": items,
        "playtime_forever": forever,
        "playtime_2weeks": recent,
    }


def _outer_arrays(outer: pd.DataFrame) -> dict[str, np.ndarray]:
    split_codes = np.asarray(
        [OUTER_SPLIT_CODES[value] for value in outer["split"]],
        dtype=np.uint8,
    )
    return {
        "user_ids": outer["user_id"].to_numpy(dtype=np.int64, copy=True),
        "raw_ownership_count": outer["raw_ownership_count"].to_numpy(
            dtype=np.int32,
            copy=True,
        ),
        "activity_band": outer["activity_band"].to_numpy(
            dtype=np.int16,
            copy=True,
        ),
        "split_code": split_codes,
        "assignment_hash": _hex_to_uint64(outer["assignment_hash"]),
    }


def _nested_arrays(nested: pd.DataFrame) -> dict[str, np.ndarray]:
    role_codes = np.asarray(
        [ROLE_CODES[str(value)] for value in nested["role"]],
        dtype=np.uint8,
    )
    status_codes = np.asarray(
        [
            USER_STATUS_CODES[str(value)]
            for value in nested["user_split_status"]
        ],
        dtype=np.uint8,
    )
    support_after = nested["item_training_support_after"].to_numpy(
        dtype=np.int32,
        na_value=-1,
    )
    return {
        "user_ids": nested["user_id"].to_numpy(dtype=np.int64, copy=True),
        "item_ids": nested["item_id"].to_numpy(dtype=np.int64, copy=True),
        "role_code": role_codes,
        "is_warm_item": nested["is_warm_item"].to_numpy(
            dtype=bool,
            copy=True,
        ),
        "evaluable_user": nested["evaluable_user"].to_numpy(
            dtype=bool,
            copy=True,
        ),
        "user_status_code": status_codes,
        "item_support_before": nested["item_support_before"].to_numpy(
            dtype=np.int32,
            copy=True,
        ),
        "item_training_support_after": support_after,
    }


def _evaluation_arrays(sample: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "user_ids": sample["user_id"].to_numpy(dtype=np.int64, copy=True),
        "activity_band": sample["activity_band"].to_numpy(
            dtype=np.int16,
            copy=True,
        ),
        "selection_hash": _hex_to_uint64(sample["selection_hash"]),
        "sample_stratum_quota": sample["sample_stratum_quota"].to_numpy(
            dtype=np.int32,
            copy=True,
        ),
    }


def _pseudo_cold_arrays(items: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "item_ids": items["item_id"].to_numpy(dtype=np.int64, copy=True),
        "design_training_support": items[
            "design_training_support"
        ].to_numpy(dtype=np.int32, copy=True),
        "primary_genre": _fixed_unicode(
            items["primary_genre"].astype(str).tolist()
        ),
        "support_band_index": items["support_band_index"].to_numpy(
            dtype=np.int16,
            copy=True,
        ),
        "support_band_lower": items["support_band_lower"].to_numpy(
            dtype=np.int32,
            copy=True,
        ),
        "support_band_upper_exclusive": items[
            "support_band_upper_exclusive"
        ].to_numpy(dtype=np.int32, copy=True),
        "genre_stratum_quota": items["genre_stratum_quota"].to_numpy(
            dtype=np.int32,
            copy=True,
        ),
        "selection_hash": _hex_to_uint64(items["selection_hash"]),
    }


def _outer_summary(outer: pd.DataFrame) -> dict[str, Any]:
    eligible = outer.loc[outer["split"].ne(EXCLUDED_LOW_ACTIVITY)]
    by_band: dict[str, dict[str, int]] = {}
    for band in sorted(eligible["activity_band"].unique()):
        rows = eligible.loc[eligible["activity_band"].eq(band)]
        by_band[str(int(band))] = {
            "total": int(len(rows)),
            "design": int(rows["split"].eq("design").sum()),
            "assessment": int(rows["split"].eq("assessment").sum()),
        }
    return {
        "source_users": int(len(outer)),
        "source_edges": int(outer["raw_ownership_count"].sum()),
        "excluded_low_activity_users": int(
            outer["split"].eq(EXCLUDED_LOW_ACTIVITY).sum()
        ),
        "excluded_low_activity_edges": int(
            outer.loc[
                outer["split"].eq(EXCLUDED_LOW_ACTIVITY),
                "raw_ownership_count",
            ].sum()
        ),
        "eligible_users": int(len(eligible)),
        "eligible_edges": int(eligible["raw_ownership_count"].sum()),
        "design_users": int(outer["split"].eq("design").sum()),
        "design_edges": int(
            outer.loc[
                outer["split"].eq("design"),
                "raw_ownership_count",
            ].sum()
        ),
        "assessment_users": int(outer["split"].eq("assessment").sum()),
        "assessment_edges": int(
            outer.loc[
                outer["split"].eq("assessment"),
                "raw_ownership_count",
            ].sum()
        ),
        "by_activity_band": by_band,
    }


def _nested_summary(
    nested: pd.DataFrame,
    training: SparseInteractionData,
) -> dict[str, Any]:
    role_counts = nested["role"].value_counts().to_dict()
    user_status = (
        nested[["user_id", "user_split_status"]]
        .drop_duplicates("user_id")
        ["user_split_status"]
        .value_counts()
        .to_dict()
    )
    warm_rows = nested.loc[nested["is_warm_item"]]
    warm_support_after = (
        warm_rows[["item_id", "item_training_support_after"]]
        .drop_duplicates("item_id")
        ["item_training_support_after"]
        .astype(np.int64)
    )
    return {
        "design_edges": int(len(nested)),
        "nonwarm_excluded_edges": int(
            role_counts.get(EXCLUDED_NONWARM_ITEM, 0)
        ),
        "training_edges": int(role_counts.get("training", 0)),
        "validation_edges": int(role_counts.get("validation", 0)),
        "design_test_edges": int(role_counts.get("test", 0)),
        "warm_items": int(warm_rows["item_id"].nunique()),
        "minimum_warm_support_before_holdout": int(
            warm_rows["item_support_before"].min()
        ),
        "minimum_warm_training_support_after_holdout": int(
            warm_support_after.min()
        ),
        "evaluable_users": int(user_status.get("evaluable", 0)),
        "capacity_exhausted_users": int(
            user_status.get("capacity_exhausted", 0)
        ),
        "insufficient_warm_history_users": int(
            user_status.get("insufficient_warm_history", 0)
        ),
        "capacity_rule_claim": (
            "deterministic greedy feasible scan; no maximum-cardinality claim"
        ),
        "training_matrix_nnz": int(training.ownership.nnz),
    }


def _evaluation_summary(sample: pd.DataFrame) -> dict[str, Any]:
    counts = sample.groupby("activity_band", sort=True).size()
    return {
        "sample_users": int(len(sample)),
        "by_activity_band": {
            str(int(band)): int(count) for band, count in counts.items()
        },
    }


def _pseudo_cold_summary(items: pd.DataFrame) -> dict[str, Any]:
    by_band = items.groupby("support_band_index", sort=True).size()
    by_genre = items.groupby("primary_genre", sort=True).size()
    return {
        "item_count": int(len(items)),
        "by_support_band": {
            str(int(band)): int(count) for band, count in by_band.items()
        },
        "by_primary_genre": {
            str(genre): int(count) for genre, count in by_genre.items()
        },
    }


def build_split_state(
    ranking: Mapping[str, Any],
    interaction_data: SparseInteractionData,
    *,
    feature_path: Path,
) -> SplitState:
    """Recreate every frozen assignment without reading model outcomes."""

    activity = np.diff(interaction_data.ownership.indptr).astype(
        np.int64,
        copy=False,
    )
    outer_input = pd.DataFrame(
        {
            "user_id": interaction_data.user_ids,
            "raw_ownership_count": activity,
        }
    )
    outer = activity_band_outer_split(
        outer_input,
        ranking["outer_user_split"],
    )
    outer_summary = _outer_summary(outer)
    if outer_summary["source_users"] != interaction_data.user_ids.size:
        raise AssertionError("outer split did not cover every canonical user")
    if outer_summary["source_users"] != (
        outer_summary["excluded_low_activity_users"]
        + outer_summary["design_users"]
        + outer_summary["assessment_users"]
    ):
        raise AssertionError("outer user cohorts do not reconcile")
    if outer_summary["source_edges"] != interaction_data.ownership.nnz:
        raise AssertionError("outer activity does not reconcile to canonical edges")
    if outer_summary["source_edges"] != (
        outer_summary["excluded_low_activity_edges"]
        + outer_summary["design_edges"]
        + outer_summary["assessment_edges"]
    ):
        raise AssertionError("outer edge cohorts do not reconcile")

    design_user_ids = outer.loc[
        outer["split"].eq("design"),
        "user_id",
    ].to_numpy(dtype=np.int64)
    design_positions = np.searchsorted(
        interaction_data.user_ids,
        design_user_ids,
    )
    assert_exact_id_alignment(
        design_user_ids,
        interaction_data.user_ids[design_positions],
        label="design user",
    )
    design_ownership = interaction_data.ownership[design_positions, :].tocsr()
    design_coo = design_ownership.tocoo(copy=False)
    design_edges = pd.DataFrame(
        {
            "user_id": design_user_ids[design_coo.row],
            "item_id": interaction_data.item_ids[design_coo.col],
            "outer_split": "design",
        }
    )
    nested = capacity_aware_edge_split(
        design_edges,
        ranking["warm_catalogue"],
        ranking["nested_interaction_split"],
    )
    if not nested["user_id"].isin(design_user_ids).all():
        raise AssertionError("assessment user entered the nested split")
    if outer_summary["design_edges"] != len(nested):
        raise AssertionError("design activity does not reconcile to nested edges")

    warm_item_ids = np.sort(
        nested.loc[nested["is_warm_item"], "item_id"].unique().astype(
            np.int64
        )
    )
    base_design_data = _slice_sparse_contract(
        interaction_data,
        user_ids=design_user_ids,
        item_ids=warm_item_ids,
    )
    validation_full = _target_arrays(
        nested,
        base_design_data,
        role="validation",
    )
    design_test = _target_arrays(
        nested,
        base_design_data,
        role="test",
    )
    if not np.array_equal(
        validation_full["user_ids"],
        design_test["user_ids"],
    ):
        raise AssertionError(
            "validation and design-test user cohorts are misaligned"
        )
    if np.any(
        validation_full["item_ids"] == design_test["item_ids"]
    ):
        raise AssertionError(
            "validation and design-test positives are not distinct"
        )
    heldout_users = np.concatenate(
        (validation_full["user_ids"], design_test["user_ids"])
    )
    heldout_items = np.concatenate(
        (validation_full["item_ids"], design_test["item_ids"])
    )
    training = remove_observed_pairs(
        base_design_data,
        heldout_users,
        heldout_items,
    )
    expected_training_edges = int(nested["role"].eq("training").sum())
    if training.ownership.nnz != expected_training_edges:
        raise AssertionError("training matrix edge count differs from split roles")
    if base_design_data.ownership.nnz != (
        training.ownership.nnz + heldout_users.size
    ):
        raise AssertionError("held-out removal count does not reconcile")

    user_status = nested[
        ["user_id", "evaluable_user", "user_split_status"]
    ]
    status_counts = user_status.groupby("user_id", sort=False).agg(
        evaluable_values=("evaluable_user", "nunique"),
        status_values=("user_split_status", "nunique"),
    )
    if (
        status_counts["evaluable_values"].gt(1).any()
        or status_counts["status_values"].gt(1).any()
    ):
        raise AssertionError("nested user status is inconsistent")
    user_status = user_status.drop_duplicates("user_id")
    evaluation_candidates = outer.loc[
        outer["user_id"].isin(
            user_status.loc[
                user_status["evaluable_user"],
                "user_id",
            ]
        )
    ].copy()
    evaluation_candidates["evaluable_user"] = True
    evaluation = proportional_evaluation_user_sample(
        evaluation_candidates,
        ranking["evaluation_users"],
    )

    feature_frame = pd.read_csv(
        feature_path,
        usecols=["item_id", "genres"],
        dtype={"item_id": "string", "genres": "string"},
        keep_default_na=False,
    )
    feature_item_ids = canonical_numeric_ids(
        feature_frame["item_id"].to_numpy(),
        label="feature item",
    )
    if np.unique(feature_item_ids).size != feature_item_ids.size:
        raise ValueError("feature table contains duplicate item IDs")
    feature_order = np.argsort(feature_item_ids, kind="mergesort")
    assert_exact_id_alignment(
        interaction_data.item_ids,
        feature_item_ids[feature_order],
        label="feature item",
    )
    genre_by_item = {
        int(item_id): primary_genre_from_csv(genres)
        for item_id, genres in zip(
            feature_item_ids,
            feature_frame["genres"],
        )
    }
    training_support = np.bincount(
        training.ownership.indices,
        minlength=training.ownership.shape[1],
    ).astype(np.int64)
    pseudo_input = pd.DataFrame(
        {
            "item_id": training.item_ids,
            "design_training_support": training_support,
            "primary_genre": [
                genre_by_item[int(item_id)]
                for item_id in training.item_ids
            ],
        }
    )
    pseudo_cold = select_pseudo_cold_items(
        pseudo_input,
        ranking["pseudo_cold"],
    )

    nested_summary = _nested_summary(nested, training)
    if nested_summary["design_edges"] != (
        nested_summary["nonwarm_excluded_edges"]
        + nested_summary["training_edges"]
        + nested_summary["validation_edges"]
        + nested_summary["design_test_edges"]
    ):
        raise AssertionError("nested edge roles do not reconcile")
    if nested_summary["evaluable_users"] != (
        nested_summary["validation_edges"]
    ) or nested_summary["evaluable_users"] != (
        nested_summary["design_test_edges"]
    ):
        raise AssertionError("evaluable users do not match held-out targets")
    if outer_summary["design_users"] != (
        nested_summary["evaluable_users"]
        + nested_summary["capacity_exhausted_users"]
        + nested_summary["insufficient_warm_history_users"]
    ):
        raise AssertionError("design-user nested statuses do not reconcile")

    validation = {
        "user_ids": validation_full["user_ids"],
        "item_ids": validation_full["item_ids"],
    }
    validation_diagnostics = {
        "playtime_forever": validation_full["playtime_forever"],
        "playtime_2weeks": validation_full["playtime_2weeks"],
    }
    validation_other_holdout_mask = {
        "user_ids": design_test["user_ids"].copy(),
        "item_ids": design_test["item_ids"].copy(),
    }
    arrays = {
        "outer_user_split": _outer_arrays(outer),
        "nested_interaction_split": _nested_arrays(nested),
        "validation_targets": validation,
        "validation_target_diagnostics": validation_diagnostics,
        "validation_other_holdout_mask": validation_other_holdout_mask,
        "design_test_targets": design_test,
        "evaluation_user_sample": _evaluation_arrays(evaluation),
        "pseudo_cold_items": _pseudo_cold_arrays(pseudo_cold),
    }
    summary = {
        "outer_users": outer_summary,
        "nested_interactions": nested_summary,
        "evaluation_users": _evaluation_summary(evaluation),
        "pseudo_cold_items": _pseudo_cold_summary(pseudo_cold),
    }
    return SplitState(arrays=arrays, training=training, summary=summary)


def _contract(ranking: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "outer_user_split": dict(ranking["outer_user_split"]),
        "warm_catalogue": dict(ranking["warm_catalogue"]),
        "nested_interaction_split": dict(
            ranking["nested_interaction_split"]
        ),
        "evaluation_users": dict(ranking["evaluation_users"]),
        "pseudo_cold": {
            **dict(ranking["pseudo_cold"]),
            "primary_genre_rule": (
                "lexicographically_first_unique_label_after_html_unescape_"
                "unicode_nfkc_and_whitespace_collapse"
            ),
            "genre_quota_reconciliation": (
                "hamilton_largest_remainder_with_canonical_lexical_ties"
            ),
        },
        "artifact_id_dtype": "int64",
        "pair_order": "ascending_numeric_user_then_item",
        "artifact_schema": {
            "outer_user_split": {
                "split_code_label_to_value": dict(OUTER_SPLIT_CODES),
                "excluded_assignment_hash_uint64_sentinel": 0,
            },
            "nested_interaction_split": {
                "role_code_label_to_value": dict(ROLE_CODES),
                "user_status_code_label_to_value": dict(
                    USER_STATUS_CODES
                ),
                "nonwarm_training_support_int32_sentinel": -1,
            },
            "validation_targets": {
                "fields": ["user_ids", "item_ids"],
                "purpose": "validation_tuning_targets",
            },
            "validation_target_diagnostics": {
                "fields": [
                    "playtime_forever",
                    "playtime_2weeks",
                ],
                "row_alignment": "validation_targets",
                "purpose": "diagnostic_only_not_model_selection",
            },
            "validation_other_holdout_mask": {
                "fields": ["user_ids", "item_ids"],
                "purpose": (
                    "opaque_masking_of_each_validation_users_test_positive"
                ),
                "raw_coordinates_returned_by_public_api": False,
            },
        },
        "ranking_masks": {
            "validation": (
                "mask_design_training_positives_and_test_positive"
            ),
            "design_test": (
                "mask_design_training_positives_and_validation_positive"
            ),
        },
        "publication_policy": (
            "cycle_scoped_staged_publish_manifest_last_"
            "refuse_nonidentical_overwrite"
        ),
        "stable_hash_projection": (
            "first_64_bits_of_sha256_as_unsigned_integer"
        ),
        "protected_identifiers_in_public_manifest": False,
    }


def _access_boundary() -> dict[str, Any]:
    return {
        "assessment_ids": "sealed_until_s1_10",
        "assessment_item_histories_saved": False,
        "assessment_activity_count_saved_in_audit_only_split": True,
        "design_test_targets": (
            "sealed_until_validation_admission_manifest_is_hashed"
        ),
        "validation_targets": "available_to_validation_tuning",
        "validation_target_diagnostics": "reserved_for_s1_7",
        "validation_other_holdout_mask": (
            "available_only_through_opaque_masking_api"
        ),
        "evaluation_user_sample": "available_to_validation_tuning",
        "outer_audit_loaded_by_tuning_api": False,
        "nested_audit_loaded_by_tuning_api": False,
        "pseudo_cold_items": "reserved_for_s1_8",
        "stage2_objectives_available": False,
        "bundle_outcomes_available": False,
        "public_manifest_contains_identifiers": False,
    }


def _artifact_identity(
    state: SplitState,
) -> dict[str, Any]:
    return {
        name: _npz_semantics(arrays)
        for name, arrays in sorted(state.arrays.items())
    }


def _split_identity_fields(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": manifest["schema_version"],
        "cycle_id": manifest["cycle_id"],
        "protocol_id": manifest["protocol_id"],
        "interaction_set_id": manifest["interaction_set_id"],
        "contract": manifest["contract"],
        "summary": manifest["summary"],
        "artifact_semantics": manifest["artifact_semantics"],
        "training_semantics": manifest["training_semantics"],
        "access_boundary": manifest["access_boundary"],
    }


def _add_manifest_ids(manifest: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(manifest)
    result["split_set_id"] = semantic_sha256(_split_identity_fields(result))
    result["manifest_id"] = semantic_sha256(result)
    return result


def _validate_manifest_id(manifest: Mapping[str, Any]) -> None:
    unsigned = dict(manifest)
    claimed = unsigned.pop("manifest_id", None)
    if claimed != semantic_sha256(unsigned):
        raise ValueError("split manifest semantic hash mismatch")


def _validate_split_set_id(manifest: Mapping[str, Any]) -> None:
    expected = semantic_sha256(_split_identity_fields(manifest))
    if manifest.get("split_set_id") != expected:
        raise ValueError("split-set semantic hash mismatch")


def _validate_public_manifest_redaction(
    manifest: Mapping[str, Any],
) -> None:
    boundary = manifest.get("access_boundary")
    if boundary != _access_boundary():
        raise ValueError("split access boundary changed")
    if manifest.get("contract", {}).get(
        "protected_identifiers_in_public_manifest"
    ) is not False:
        raise ValueError("public split manifest identifier policy changed")

    artifacts = manifest.get("artifacts")
    semantics = manifest.get("artifact_semantics")
    if not isinstance(artifacts, Mapping) or not isinstance(
        semantics,
        Mapping,
    ):
        raise ValueError("split artifact metadata is missing")
    if set(semantics) != set(NPZ_RELATIVE_PATHS):
        raise ValueError("split artifact semantic inventory changed")

    for name, expected_access in ACCESS_CLASSES.items():
        entry = artifacts.get(name)
        identity = semantics.get(name)
        if not isinstance(entry, Mapping) or not isinstance(
            identity,
            Mapping,
        ):
            raise ValueError(f"public artifact metadata is missing: {name}")
        if entry.get("access") != expected_access:
            raise ValueError(f"split artifact access changed: {name}")
        if set(identity) != {"fields", "semantic_sha256"}:
            raise ValueError(f"split artifact identity schema changed: {name}")
        fields = identity.get("fields")
        if not isinstance(fields, Mapping):
            raise ValueError(f"split artifact field schema changed: {name}")
        for metadata in fields.values():
            if not isinstance(metadata, Mapping) or set(metadata) != {
                "dtype",
                "shape",
                "semantic_sha256",
            }:
                raise ValueError(
                    f"public artifact field metadata changed: {name}"
                )
            if not isinstance(metadata.get("shape"), list) or not all(
                isinstance(value, int) for value in metadata["shape"]
            ):
                raise ValueError(
                    f"public artifact shape metadata changed: {name}"
                )
        if entry.get("fields") != fields or entry.get(
            "semantic_sha256"
        ) != identity.get("semantic_sha256"):
            raise ValueError(f"public artifact identity mismatch: {name}")

    for name in TRAINING_ARTIFACT_NAMES:
        artifact_name = f"design_training_{name}"
        entry = artifacts.get(artifact_name)
        if not isinstance(entry, Mapping) or entry.get(
            "access"
        ) != "design_training_permitted":
            raise ValueError(
                f"design-training artifact access changed: {artifact_name}"
            )

    identifier_keys = {"user_ids", "item_ids", "user_id", "item_id"}

    def reject_nested_identifiers(
        value: Any,
        path: tuple[str, ...] = (),
    ) -> None:
        if isinstance(value, Mapping):
            for raw_key, nested in value.items():
                key = str(raw_key)
                field_metadata = (
                    len(path) >= 3
                    and path[0] in {"artifacts", "artifact_semantics"}
                    and path[-1] == "fields"
                )
                if key in identifier_keys and not field_metadata:
                    raise ValueError(
                        "public split manifest contains identifier fields"
                    )
                if (
                    key.endswith("_ids")
                    and isinstance(nested, (list, tuple, np.ndarray))
                    and not field_metadata
                ):
                    raise ValueError(
                        "public split manifest contains identifier arrays"
                    )
                reject_nested_identifiers(nested, path + (key,))
        elif isinstance(value, (list, tuple)):
            for nested in value:
                reject_nested_identifiers(nested, path)

    reject_nested_identifiers(manifest)


def _load_consumer_manifest(
    *,
    manifest_path: str | Path,
) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    _validate_manifest_id(manifest)
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported split manifest schema")
    if not manifest.get("cycle_id") or not manifest.get("protocol_id"):
        raise ValueError("split manifest lacks cycle/protocol identity")
    _validate_split_set_id(manifest)
    _validate_public_manifest_redaction(manifest)
    return manifest


def _save_artifacts(
    state: SplitState,
    *,
    root: Path,
    output_dir: Path,
    published_output_dir: Path | None = None,
) -> dict[str, Any]:
    published = (
        output_dir if published_output_dir is None else published_output_dir
    )
    entries: dict[str, Any] = {}
    for name, relative_path in NPZ_RELATIVE_PATHS.items():
        path = output_dir / relative_path
        published_path = published / relative_path
        arrays = state.arrays[name]
        _write_npz(path, arrays)
        entries[name] = {
            "path": _relative_path(published_path, root),
            "access": ACCESS_CLASSES[name],
            "size_bytes": int(path.stat().st_size),
            "sha256": file_sha256(path),
            **_npz_semantics(arrays),
        }

    training_dir = output_dir / TRAINING_RELATIVE_DIR
    published_training_dir = published / TRAINING_RELATIVE_DIR
    hashes = save_sparse_interactions(
        state.training,
        training_dir,
        prefix=TRAINING_PREFIX,
    )
    for name in TRAINING_ARTIFACT_NAMES:
        suffix = "npz" if name in {
            "ownership",
            "playtime_forever",
            "playtime_2weeks",
        } else "npy"
        filename = f"{TRAINING_PREFIX}_{name}.{suffix}"
        path = training_dir / filename
        published_path = published_training_dir / filename
        entries[f"design_training_{name}"] = {
            "path": _relative_path(published_path, root),
            "access": "design_training_permitted",
            "size_bytes": int(path.stat().st_size),
            "sha256": hashes[name],
        }
    return entries


def _current_artifact_entries(
    state: SplitState,
    *,
    root: Path,
    output_dir: Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    entries: dict[str, Any] = {}
    for name, relative_path in NPZ_RELATIVE_PATHS.items():
        path = output_dir / relative_path
        expected_arrays = state.arrays[name]
        actual_arrays = _read_npz(path)
        _assert_arrays_equal(expected_arrays, actual_arrays, label=name)
        entries[name] = {
            "path": _relative_path(path, root),
            "access": ACCESS_CLASSES[name],
            "size_bytes": int(path.stat().st_size),
            "sha256": file_sha256(path),
            **_npz_semantics(actual_arrays),
        }

    training_hashes: dict[str, str] = {}
    for name in TRAINING_ARTIFACT_NAMES:
        entry = manifest.get("artifacts", {}).get(
            f"design_training_{name}",
            {},
        )
        training_hashes[name] = str(entry.get("sha256"))
    training_dir = output_dir / TRAINING_RELATIVE_DIR
    saved_training = load_sparse_interactions(
        training_dir,
        prefix=TRAINING_PREFIX,
        expected_file_hashes=training_hashes,
    )
    if _training_semantics(saved_training) != _training_semantics(
        state.training
    ):
        raise ValueError("saved design-training sparse semantics changed")
    for name in TRAINING_ARTIFACT_NAMES:
        suffix = "npz" if name in {
            "ownership",
            "playtime_forever",
            "playtime_2weeks",
        } else "npy"
        path = training_dir / f"{TRAINING_PREFIX}_{name}.{suffix}"
        entries[f"design_training_{name}"] = {
            "path": _relative_path(path, root),
            "access": "design_training_permitted",
            "size_bytes": int(path.stat().st_size),
            "sha256": file_sha256(path),
        }
    return entries


def _input_entries(
    *,
    root: Path,
    ranking_path: Path,
    protocol_path: Path,
    interaction_manifest_path: Path,
    feature_path: Path,
    ranking: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "ranking_config": {
            **_input_entry(ranking_path, root),
            "semantic_sha256": semantic_sha256(ranking),
        },
        "protocol_manifest": _input_entry(protocol_path, root),
        "interaction_manifest": _input_entry(
            interaction_manifest_path,
            root,
        ),
        "features": _input_entry(feature_path, root),
    }


def _validate_ranking_protocol_binding(
    ranking: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> None:
    if ranking.get("cycle_id") != protocol.get("cycle_id"):
        raise ValueError("ranking configuration and protocol cycle mismatch")
    expected = (
        protocol.get("configs", {})
        .get("ranking_evaluation", {})
        .get("semantic_sha256")
    )
    actual = semantic_sha256(ranking)
    if expected != actual:
        raise ValueError(
            "ranking configuration differs from the S1.0 protocol hash"
        )
    model_hash = (
        protocol.get("configs", {})
        .get("preference_models", {})
        .get("semantic_sha256")
    )
    expected_protocol = semantic_sha256(
        {
            "model_config_sha256": model_hash,
            "ranking_config_sha256": actual,
        }
    )
    if protocol.get("protocol_id") != expected_protocol:
        raise ValueError("protocol/configuration semantic binding changed")


def _validate_generation_paths(
    *,
    root: Path,
    output_dir: Path,
    manifest_path: Path,
    cycle_id: str,
) -> None:
    protected_root = (
        root / "outputs" / "modeling" / "protected"
    ).resolve()
    protected_cycle_root = (protected_root / cycle_id).resolve()
    resolved_output = output_dir.resolve()
    try:
        resolved_output.relative_to(protected_cycle_root)
    except ValueError as exc:
        raise ValueError(
            "protected split artifacts must remain under the protected cycle"
        ) from exc

    modeling_root = (root / "outputs" / "modeling").resolve()
    resolved_manifest = manifest_path.resolve()
    try:
        resolved_manifest.relative_to(modeling_root)
    except ValueError as exc:
        raise ValueError(
            "public split manifest must remain under outputs/modeling"
        ) from exc
    try:
        resolved_manifest.relative_to(protected_root)
    except ValueError:
        pass
    else:
        raise ValueError("public split manifest cannot be protected")


def _publish_staged_publication(
    *,
    staging_dir: Path,
    output_dir: Path,
    temporary_manifest: Path,
    manifest_path: Path,
) -> None:
    """Publish a complete directory first and a no-clobber manifest last."""

    if output_dir.exists() or manifest_path.exists():
        raise FileExistsError(
            "refusing to overwrite a split publication during publish"
        )
    output_published = False
    manifest_published = False
    try:
        staging_dir.rename(output_dir)
        output_published = True
        os.link(temporary_manifest, manifest_path)
        manifest_published = True
        temporary_manifest.unlink()
    except BaseException:
        if output_published and not manifest_published and output_dir.exists():
            shutil.rmtree(output_dir)
        if temporary_manifest.exists():
            try:
                temporary_manifest.unlink()
            except OSError:
                pass
        raise


def generate_split_artifacts(
    *,
    project_root: str | Path = PROJECT_ROOT,
    ranking_config_path: str | Path = DEFAULT_RANKING_CONFIG,
    protocol_manifest_path: str | Path = DEFAULT_PROTOCOL_MANIFEST,
    interaction_manifest_path: str | Path = DEFAULT_INTERACTION_MANIFEST,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    manifest_path: str | Path = DEFAULT_MANIFEST,
) -> dict[str, Any]:
    """Materialize protected assignments and one compact public manifest."""

    root = Path(project_root)
    ranking_path = Path(ranking_config_path)
    protocol_path = Path(protocol_manifest_path)
    interaction_path = Path(interaction_manifest_path)
    output = Path(output_dir)
    destination = Path(manifest_path)
    ranking = load_json(ranking_path)
    protocol = load_json(protocol_path)
    _validate_ranking_protocol_binding(ranking, protocol)
    _validate_generation_paths(
        root=root,
        output_dir=output,
        manifest_path=destination,
        cycle_id=str(ranking["cycle_id"]),
    )
    if output.exists() or destination.exists():
        if output.exists() and destination.exists():
            try:
                return verify_split_artifacts(
                    project_root=root,
                    ranking_config_path=ranking_path,
                    protocol_manifest_path=protocol_path,
                    interaction_manifest_path=interaction_path,
                    output_dir=output,
                    manifest_path=destination,
                )
            except Exception as exc:
                raise FileExistsError(
                    "refusing to overwrite a nonidentical frozen split cohort"
                ) from exc
        raise FileExistsError(
            "refusing generation with a partial frozen split publication"
        )
    feature_path = root / ranking["inputs"]["features_path"]
    if file_sha256(feature_path) != ranking["inputs"]["features_sha256"]:
        raise ValueError("feature input differs from the frozen hash")

    interaction_data, interaction_manifest = _load_interactions(
        root=root,
        ranking_path=ranking_path,
        protocol_path=protocol_path,
        interaction_manifest_path=interaction_path,
    )
    state = build_split_state(
        ranking,
        interaction_data,
        feature_path=feature_path,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=".stage1_splits-",
            dir=output.parent,
        )
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_manifest_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary_manifest = Path(temporary_manifest_name)
    try:
        artifacts = _save_artifacts(
            state,
            root=root,
            output_dir=staging,
            published_output_dir=output,
        )
        base_manifest = {
            "schema_version": 1,
            "artifact": "frozen_stage1_outer_and_nested_splits",
            "cycle_id": ranking["cycle_id"],
            "protocol_id": protocol["protocol_id"],
            "interaction_set_id": interaction_manifest["interaction_set_id"],
            "contract": _contract(ranking),
            "inputs": _input_entries(
                root=root,
                ranking_path=ranking_path,
                protocol_path=protocol_path,
                interaction_manifest_path=interaction_path,
                feature_path=feature_path,
                ranking=ranking,
            ),
            "summary": dict(state.summary),
            "artifact_semantics": _artifact_identity(state),
            "training_semantics": _training_semantics(state.training),
            "artifacts": artifacts,
            "access_boundary": _access_boundary(),
            "provenance": {
                "command": "python -m src.stage1_split_artifacts",
                "generated_at_utc": datetime.now(timezone.utc)
                .isoformat(timespec="seconds")
                .replace("+00:00", "Z"),
                "repository_head_before_generation_commit": (
                    _repository_head(root)
                ),
                "python": platform.python_version(),
                "packages": _package_versions(),
                "module_hash_policy": "utf8_text_with_lf_newlines",
                "modules": _module_hashes(),
                "publication_policy": (
                    "cycle_scoped_staged_publish_manifest_last_"
                    "refuse_nonidentical_overwrite"
                ),
            },
        }
        manifest = _add_manifest_ids(base_manifest)
        _validate_manifest_id(manifest)
        _validate_split_set_id(manifest)
        _validate_public_manifest_redaction(manifest)
        write_manifest(manifest, temporary_manifest)
        _publish_staged_publication(
            staging_dir=staging,
            output_dir=output,
            temporary_manifest=temporary_manifest,
            manifest_path=destination,
        )
        return manifest
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        if temporary_manifest.exists():
            temporary_manifest.unlink()
        raise


def verify_split_artifacts(
    *,
    project_root: str | Path = PROJECT_ROOT,
    ranking_config_path: str | Path = DEFAULT_RANKING_CONFIG,
    protocol_manifest_path: str | Path = DEFAULT_PROTOCOL_MANIFEST,
    interaction_manifest_path: str | Path = DEFAULT_INTERACTION_MANIFEST,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    manifest_path: str | Path = DEFAULT_MANIFEST,
) -> dict[str, Any]:
    """Regenerate assignments in memory and verify every saved split field."""

    root = Path(project_root)
    ranking_path = Path(ranking_config_path)
    protocol_path = Path(protocol_manifest_path)
    interaction_path = Path(interaction_manifest_path)
    output = Path(output_dir)
    manifest = load_json(manifest_path)
    _validate_manifest_id(manifest)
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported split manifest schema")
    _validate_split_set_id(manifest)
    _validate_public_manifest_redaction(manifest)

    ranking = load_json(ranking_path)
    protocol = load_json(protocol_path)
    _validate_ranking_protocol_binding(ranking, protocol)
    _validate_generation_paths(
        root=root,
        output_dir=output,
        manifest_path=Path(manifest_path),
        cycle_id=str(ranking["cycle_id"]),
    )
    if manifest.get("protocol_id") != protocol.get("protocol_id"):
        raise ValueError("split manifest protocol mismatch")
    if manifest.get("cycle_id") != ranking.get("cycle_id"):
        raise ValueError("split manifest cycle mismatch")
    if manifest.get("contract") != _contract(ranking):
        raise ValueError("split contract changed")
    # Recorded producer hashes identify the historical generator. Verification
    # below regenerates every assignment and semantic hash with current code.

    feature_path = root / ranking["inputs"]["features_path"]
    expected_inputs = _input_entries(
        root=root,
        ranking_path=ranking_path,
        protocol_path=protocol_path,
        interaction_manifest_path=interaction_path,
        feature_path=feature_path,
        ranking=ranking,
    )
    if manifest.get("inputs") != expected_inputs:
        raise ValueError("split upstream inputs changed")
    if expected_inputs["features"]["sha256"] != ranking["inputs"][
        "features_sha256"
    ]:
        raise ValueError("feature input differs from the frozen hash")

    interaction_data, interaction_manifest = _load_interactions(
        root=root,
        ranking_path=ranking_path,
        protocol_path=protocol_path,
        interaction_manifest_path=interaction_path,
    )
    if manifest.get("interaction_set_id") != interaction_manifest.get(
        "interaction_set_id"
    ):
        raise ValueError("split interaction-set dependency changed")
    state = build_split_state(
        ranking,
        interaction_data,
        feature_path=feature_path,
    )
    if manifest.get("summary") != state.summary:
        raise ValueError("split summary changed under exact regeneration")
    if manifest.get("artifact_semantics") != _artifact_identity(state):
        raise ValueError("split array semantics changed under regeneration")
    if manifest.get("training_semantics") != _training_semantics(
        state.training
    ):
        raise ValueError("split training semantics changed under regeneration")

    current_entries = _current_artifact_entries(
        state,
        root=root,
        output_dir=output,
        manifest=manifest,
    )
    if manifest.get("artifacts") != current_entries:
        raise ValueError("split artifact bytes or metadata changed")
    _validate_split_set_id(manifest)
    return manifest


def _verified_artifact_path(
    *,
    root: Path,
    manifest: Mapping[str, Any],
    artifact_name: str,
    expected_access: str,
    expected_suffix: Path,
) -> Path:
    entry = manifest.get("artifacts", {}).get(artifact_name)
    if not isinstance(entry, Mapping):
        raise ValueError(f"split artifact entry is missing: {artifact_name}")
    if entry.get("access") != expected_access:
        raise ValueError(f"split artifact access changed: {artifact_name}")

    relative = Path(str(entry.get("path", "")))
    if relative.is_absolute() or not relative.parts:
        raise ValueError(f"split artifact path is not relative: {artifact_name}")
    suffix_parts = expected_suffix.parts
    if tuple(relative.parts[-len(suffix_parts) :]) != suffix_parts:
        raise ValueError(f"split artifact path scope changed: {artifact_name}")
    path = (root / relative).resolve()
    protected_root = (
        root / "outputs" / "modeling" / "protected" / str(manifest["cycle_id"])
    ).resolve()
    try:
        path.relative_to(protected_root)
    except ValueError as exc:
        raise ValueError(
            f"split artifact escaped the protected cycle: {artifact_name}"
        ) from exc
    if not path.is_file():
        raise ValueError(f"split artifact is missing: {artifact_name}")
    if entry.get("size_bytes") != int(path.stat().st_size):
        raise ValueError(f"split artifact size changed: {artifact_name}")
    if entry.get("sha256") != file_sha256(path):
        raise ValueError(f"split artifact hash changed: {artifact_name}")
    return path


def _load_verified_npz(
    *,
    root: Path,
    manifest: Mapping[str, Any],
    artifact_name: str,
) -> dict[str, np.ndarray]:
    path = _verified_artifact_path(
        root=root,
        manifest=manifest,
        artifact_name=artifact_name,
        expected_access=ACCESS_CLASSES[artifact_name],
        expected_suffix=NPZ_RELATIVE_PATHS[artifact_name],
    )
    arrays = _read_npz(path)
    actual = _npz_semantics(arrays)
    entry = manifest["artifacts"][artifact_name]
    expected = manifest["artifact_semantics"][artifact_name]
    if actual != expected:
        raise ValueError(
            f"split artifact semantics changed: {artifact_name}"
        )
    if entry.get("fields") != actual["fields"] or entry.get(
        "semantic_sha256"
    ) != actual["semantic_sha256"]:
        raise ValueError(
            f"split artifact entry semantics changed: {artifact_name}"
        )
    return arrays


def load_design_training_artifacts(
    *,
    project_root: str | Path = PROJECT_ROOT,
    manifest_path: str | Path = DEFAULT_MANIFEST,
) -> SparseInteractionData:
    """Load the hash-bound design-training sparse contract only."""

    root = Path(project_root)
    manifest = _load_consumer_manifest(manifest_path=manifest_path)
    hashes: dict[str, str] = {}
    training_dir: Path | None = None
    for name in TRAINING_ARTIFACT_NAMES:
        artifact_name = f"design_training_{name}"
        suffix = "npz" if name in {
            "ownership",
            "playtime_forever",
            "playtime_2weeks",
        } else "npy"
        filename = f"{TRAINING_PREFIX}_{name}.{suffix}"
        path = _verified_artifact_path(
            root=root,
            manifest=manifest,
            artifact_name=artifact_name,
            expected_access="design_training_permitted",
            expected_suffix=TRAINING_RELATIVE_DIR / filename,
        )
        if training_dir is None:
            training_dir = path.parent
        elif path.parent != training_dir:
            raise ValueError("design-training artifact directory changed")
        hashes[name] = str(
            manifest["artifacts"][artifact_name]["sha256"]
        )
    if training_dir is None:
        raise ValueError("design-training artifact inventory is empty")
    data = load_sparse_interactions(
        training_dir,
        prefix=TRAINING_PREFIX,
        expected_file_hashes=hashes,
    )
    if _training_semantics(data) != manifest.get("training_semantics"):
        raise ValueError("design-training sparse semantics changed")
    return data


def load_validation_targets(
    *,
    project_root: str | Path = PROJECT_ROOT,
    manifest_path: str | Path = DEFAULT_MANIFEST,
) -> dict[str, np.ndarray]:
    """Load only tuning-permitted validation target coordinates."""

    root = Path(project_root)
    manifest = _load_consumer_manifest(manifest_path=manifest_path)
    arrays = _load_verified_npz(
        root=root,
        manifest=manifest,
        artifact_name="validation_targets",
    )
    if set(arrays) != {"user_ids", "item_ids"}:
        raise ValueError("validation target fields changed")
    return arrays


def load_evaluation_user_sample(
    *,
    project_root: str | Path = PROJECT_ROOT,
    manifest_path: str | Path = DEFAULT_MANIFEST,
) -> dict[str, np.ndarray]:
    """Load the hash-bound validation evaluation-user sample."""

    root = Path(project_root)
    manifest = _load_consumer_manifest(manifest_path=manifest_path)
    return _load_verified_npz(
        root=root,
        manifest=manifest,
        artifact_name="evaluation_user_sample",
    )


def mask_validation_other_holdouts(
    candidate_mask: np.ndarray,
    batch_user_ids: Sequence[Any],
    catalogue_item_ids: Sequence[Any],
    *,
    project_root: str | Path = PROJECT_ROOT,
    manifest_path: str | Path = DEFAULT_MANIFEST,
) -> np.ndarray:
    """Mask test positives during validation without returning their IDs."""

    root = Path(project_root)
    manifest = _load_consumer_manifest(manifest_path=manifest_path)
    exclusions = _load_verified_npz(
        root=root,
        manifest=manifest,
        artifact_name="validation_other_holdout_mask",
    )
    if set(exclusions) != {"user_ids", "item_ids"}:
        raise ValueError("validation other-holdout mask fields changed")
    excluded_users = exclusions["user_ids"]
    excluded_items = exclusions["item_ids"]
    if (
        excluded_users.ndim != 1
        or excluded_items.ndim != 1
        or excluded_users.size != excluded_items.size
        or np.unique(excluded_users).size != excluded_users.size
    ):
        raise ValueError("validation other-holdout mask contract changed")

    users = canonical_numeric_ids(batch_user_ids, label="batch user")
    items = canonical_numeric_ids(
        catalogue_item_ids,
        label="catalogue item",
    )
    if np.unique(users).size != users.size:
        raise ValueError("validation batch user IDs must be unique")
    if items.size > 1 and not np.all(items[1:] > items[:-1]):
        raise ValueError(
            "validation catalogue item IDs must be unique and ascending"
        )
    training_semantics = manifest.get("training_semantics", {})
    if (
        items.size != training_semantics.get("item_count")
        or array_sha256(items)
        != training_semantics.get("item_ids_sha256")
    ):
        raise ValueError(
            "validation catalogue differs from the frozen warm item map"
        )
    excluded_user_set = {int(value) for value in excluded_users}
    if any(int(user_id) not in excluded_user_set for user_id in users):
        raise ValueError(
            "every validation batch user requires one other holdout"
        )
    mask = np.asarray(candidate_mask)
    if mask.dtype != np.dtype(bool):
        raise ValueError("validation candidate mask must have boolean dtype")
    if mask.ndim != 2 or mask.shape != (users.size, items.size):
        raise ValueError("validation candidate mask shape is misaligned")

    result = mask.copy()
    row_by_user = {
        int(user_id): row for row, user_id in enumerate(users)
    }
    for user_id, item_id in zip(excluded_users, excluded_items):
        row = row_by_user.get(int(user_id))
        if row is None:
            continue
        column = int(np.searchsorted(items, item_id))
        if column >= items.size or items[column] != item_id:
            raise ValueError(
                "a validation other holdout is absent from the catalogue"
            )
        result[row, column] = False
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate or verify frozen Stage 1 split artifacts"
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
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
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)

    publication_existed = args.output_dir.exists() or args.manifest.exists()
    function = (
        verify_split_artifacts
        if args.check_only
        else generate_split_artifacts
    )
    manifest = function(
        project_root=args.project_root,
        ranking_config_path=args.ranking,
        protocol_manifest_path=args.protocol_manifest,
        interaction_manifest_path=args.interaction_manifest,
        output_dir=args.output_dir,
        manifest_path=args.manifest,
    )
    summary = manifest["summary"]
    print(
        json.dumps(
            {
                "split_set_id": manifest["split_set_id"],
                "manifest_id": manifest["manifest_id"],
                "design_users": summary["outer_users"]["design_users"],
                "assessment_users": summary["outer_users"][
                    "assessment_users"
                ],
                "evaluable_users": summary["nested_interactions"][
                    "evaluable_users"
                ],
                "training_edges": summary["nested_interactions"][
                    "training_edges"
                ],
                "validation_edges": summary["nested_interactions"][
                    "validation_edges"
                ],
                "design_test_edges": summary["nested_interactions"][
                    "design_test_edges"
                ],
                "evaluation_users": summary["evaluation_users"][
                    "sample_users"
                ],
                "pseudo_cold_items": summary["pseudo_cold_items"][
                    "item_count"
                ],
                "wrote_artifacts": (
                    not args.check_only and not publication_existed
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
