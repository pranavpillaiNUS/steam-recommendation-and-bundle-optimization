"""Generate and verify the frozen S1.3 item-feature publication.

Only pre-model catalogue metadata, the S1.1 canonical item map, and the
permitted S1.2 design-training item map are consumed. Reserved pseudo-cold
identifiers, validation and test targets, assessment data, model scores, and
Stage 2 outcomes are outside this module's access boundary.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.features import (
    GENRE_FILENAME,
    GENRE_NAMES_FILENAME,
    IDENTITY_FILENAME,
    IDENTITY_NAMES_FILENAME,
    ITEM_IDS_FILENAME,
    MANIFEST_FILENAME as SERIALIZATION_MANIFEST_FILENAME,
    ItemFeatureArtifacts,
    assert_exact_item_alignment,
    build_feature_manifest,
    build_item_features,
    file_sha256,
    load_feature_artifacts,
    model_feature_view,
    save_feature_artifacts,
    semantic_sha256,
)
from src.interactions import array_sha256, canonical_numeric_ids, id_map_sha256
from src.stage1_interaction_artifacts import (
    DEFAULT_MANIFEST as DEFAULT_INTERACTION_MANIFEST,
    DEFAULT_OUTPUT_DIR as DEFAULT_INTERACTION_OUTPUT_DIR,
    EXPECTED_PROTOCOL_ID,
    verify_interaction_artifacts,
)
from src.stage1_protocol import (
    load_json,
    validate_protocol_configs,
    write_manifest,
)
from src.stage1_split_artifacts import (
    DEFAULT_MANIFEST as DEFAULT_SPLIT_MANIFEST,
    load_design_training_artifacts,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_CYCLE_ID = "s1-v1-20260718"
DEFAULT_MODEL_CONFIG = PROJECT_ROOT / "configs" / "preference_models.json"
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
    / "stage1_features"
)
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "outputs" / "modeling" / "item_feature_manifest.json"
)
DEFAULT_COVERAGE_TABLE = (
    PROJECT_ROOT / "outputs" / "tables" / "stage1_feature_coverage.csv"
)

PUBLIC_SCHEMA_VERSION = 1
PUBLIC_ARTIFACT_FILENAMES = {
    "item_ids": ITEM_IDS_FILENAME,
    "identity": IDENTITY_FILENAME,
    "genre": GENRE_FILENAME,
    "identity_feature_names": IDENTITY_NAMES_FILENAME,
    "genre_feature_names": GENRE_NAMES_FILENAME,
    "serialization_manifest": SERIALIZATION_MANIFEST_FILENAME,
}
COVERAGE_COLUMNS = (
    "row_kind",
    "feature_block",
    "feature_name",
    "full_catalogue_item_count",
    "warm_training_item_count",
    "nonwarm_catalogue_item_count",
    "learnable_from_warm",
)


@dataclass(frozen=True)
class FeatureFreezeState:
    """Deterministic S1.3 state before physical publication."""

    artifacts: ItemFeatureArtifacts
    warm_item_ids: np.ndarray
    coverage_rows: tuple[Mapping[str, Any], ...]
    coverage_bytes: bytes
    summary: Mapping[str, Any]


@dataclass(frozen=True)
class UpstreamState:
    """Verified public dependencies and permitted item maps."""

    model_config: Mapping[str, Any]
    ranking_config: Mapping[str, Any]
    protocol_manifest: Mapping[str, Any]
    interaction_manifest: Mapping[str, Any]
    split_manifest: Mapping[str, Any]
    canonical_item_ids: np.ndarray
    warm_item_ids: np.ndarray


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"path is outside project root: {path}") from exc


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
        "features_text_sha256": _source_text_sha256(
            PROJECT_ROOT / "src" / "features.py"
        ),
        "generator_text_sha256": _source_text_sha256(Path(__file__)),
    }


def _package_versions() -> dict[str, str]:
    return {
        name: importlib.metadata.version(name)
        for name in ("numpy", "pandas", "scipy")
    }


def _validate_public_manifest_id(manifest: Mapping[str, Any]) -> None:
    unsigned = dict(manifest)
    manifest_id = unsigned.pop("manifest_id", None)
    if manifest_id != semantic_sha256(unsigned):
        raise ValueError("item-feature public manifest semantic hash mismatch")


def _validate_protocol_binding(
    *,
    model_config: Mapping[str, Any],
    ranking_config: Mapping[str, Any],
    protocol_manifest: Mapping[str, Any],
) -> None:
    validate_protocol_configs(model_config, ranking_config)
    cycle_id = model_config.get("cycle_id")
    if ranking_config.get("cycle_id") != cycle_id:
        raise ValueError("model and ranking configuration cycles differ")
    if protocol_manifest.get("cycle_id") != cycle_id:
        raise ValueError("protocol and configuration cycles differ")
    expected_model_hash = semantic_sha256(model_config)
    expected_ranking_hash = semantic_sha256(ranking_config)
    configs = protocol_manifest.get("configs", {})
    if configs.get("preference_models", {}).get(
        "semantic_sha256"
    ) != expected_model_hash:
        raise ValueError("preference-model configuration changed")
    if configs.get("ranking_evaluation", {}).get(
        "semantic_sha256"
    ) != expected_ranking_hash:
        raise ValueError("ranking configuration changed")
    expected_protocol_id = semantic_sha256(
        {
            "model_config_sha256": expected_model_hash,
            "ranking_config_sha256": expected_ranking_hash,
        }
    )
    if expected_protocol_id != protocol_manifest.get("protocol_id"):
        raise ValueError("protocol/configuration semantic binding changed")


def _load_verified_upstreams(
    *,
    root: Path,
    model_config_path: Path,
    ranking_config_path: Path,
    protocol_manifest_path: Path,
    interaction_manifest_path: Path,
    interaction_output_dir: Path,
    split_manifest_path: Path,
) -> UpstreamState:
    model_config = load_json(model_config_path)
    ranking_config = load_json(ranking_config_path)
    protocol_manifest = load_json(protocol_manifest_path)
    _validate_protocol_binding(
        model_config=model_config,
        ranking_config=ranking_config,
        protocol_manifest=protocol_manifest,
    )

    interaction_manifest = verify_interaction_artifacts(
        project_root=root,
        ranking_config_path=ranking_config_path,
        protocol_manifest_path=protocol_manifest_path,
        output_dir=interaction_output_dir,
        manifest_path=interaction_manifest_path,
    )
    cycle_id = model_config["cycle_id"]
    protocol_id = protocol_manifest["protocol_id"]
    if interaction_manifest.get("cycle_id") != cycle_id:
        raise ValueError("interaction cycle differs from the feature cycle")
    if interaction_manifest.get("protocol_id") != protocol_id:
        raise ValueError("interaction protocol differs from the feature protocol")

    item_entry = interaction_manifest["artifacts"]["item_ids"]
    canonical_item_path = root / item_entry["path"]
    canonical_item_ids = np.load(canonical_item_path, allow_pickle=False)
    if array_sha256(canonical_item_ids) != interaction_manifest["id_maps"][
        "items"
    ]["array_sha256"]:
        raise ValueError("S1.1 canonical item-array hash changed")
    if id_map_sha256(
        canonical_item_ids, label="item"
    ) != interaction_manifest["id_maps"]["items"]["semantic_sha256"]:
        raise ValueError("S1.1 canonical item-map hash changed")

    split_manifest = load_json(split_manifest_path)
    _validate_public_manifest_id(split_manifest)
    if split_manifest.get("cycle_id") != cycle_id:
        raise ValueError("split cycle differs from the feature cycle")
    if split_manifest.get("protocol_id") != protocol_id:
        raise ValueError("split protocol differs from the feature protocol")
    if split_manifest.get("interaction_set_id") != interaction_manifest.get(
        "interaction_set_id"
    ):
        raise ValueError("split interaction dependency changed")

    training = load_design_training_artifacts(
        project_root=root,
        manifest_path=split_manifest_path,
    )
    warm_item_ids = training.item_ids
    training_semantics = split_manifest.get("training_semantics", {})
    if array_sha256(warm_item_ids) != training_semantics.get(
        "item_ids_sha256"
    ):
        raise ValueError("S1.2 warm item-map hash changed")
    positions = np.searchsorted(canonical_item_ids, warm_item_ids)
    if (
        np.any(positions >= canonical_item_ids.size)
        or not np.array_equal(canonical_item_ids[positions], warm_item_ids)
    ):
        raise ValueError("S1.2 warm items are not an ordered S1.1 item-map subset")

    return UpstreamState(
        model_config=model_config,
        ranking_config=ranking_config,
        protocol_manifest=protocol_manifest,
        interaction_manifest=interaction_manifest,
        split_manifest=split_manifest,
        canonical_item_ids=canonical_item_ids,
        warm_item_ids=warm_item_ids,
    )


def _read_catalogue_genres(
    path: Path,
) -> list[tuple[int, tuple[str, ...]]]:
    frame = pd.read_csv(
        path,
        usecols=["item_id", "genres"],
        dtype={"item_id": "string", "genres": "string"},
        keep_default_na=False,
    )
    numeric_ids = canonical_numeric_ids(
        frame["item_id"].tolist(),
        label="game_features item",
    )
    if np.unique(numeric_ids).size != numeric_ids.size:
        raise ValueError("game_features contains duplicate item rows")
    records: list[tuple[int, tuple[str, ...]]] = []
    for item_id, raw_genres in zip(
        numeric_ids.tolist(),
        frame["genres"].tolist(),
        strict=True,
    ):
        genres = (
            ()
            if not raw_genres
            else tuple(str(raw_genres).split(", "))
        )
        records.append((item_id, genres))
    return records


def _coverage_rows(
    artifacts: ItemFeatureArtifacts,
    warm_item_ids: np.ndarray,
) -> tuple[Mapping[str, Any], ...]:
    positions = np.searchsorted(artifacts.item_ids, warm_item_ids)
    warm_genre = artifacts.genre[positions, :].tocsr()
    full_row_covered = np.diff(artifacts.genre.indptr) > 0
    warm_row_covered = np.diff(warm_genre.indptr) > 0
    full_label_counts = np.asarray(
        (artifacts.genre > 0).sum(axis=0)
    ).ravel()
    warm_label_counts = np.asarray((warm_genre > 0).sum(axis=0)).ravel()

    rows: list[Mapping[str, Any]] = [
        {
            "row_kind": "coverage_summary",
            "feature_block": "identity",
            "feature_name": "__all_items__",
            "full_catalogue_item_count": int(artifacts.item_ids.size),
            "warm_training_item_count": int(warm_item_ids.size),
            "nonwarm_catalogue_item_count": int(
                artifacts.item_ids.size - warm_item_ids.size
            ),
            "learnable_from_warm": "true",
        },
        {
            "row_kind": "coverage_summary",
            "feature_block": "genre",
            "feature_name": "__any_genre__",
            "full_catalogue_item_count": int(np.count_nonzero(full_row_covered)),
            "warm_training_item_count": int(np.count_nonzero(warm_row_covered)),
            "nonwarm_catalogue_item_count": int(
                np.count_nonzero(full_row_covered)
                - np.count_nonzero(warm_row_covered)
            ),
            "learnable_from_warm": "true",
        },
        {
            "row_kind": "coverage_summary",
            "feature_block": "genre",
            "feature_name": "__zero_content__",
            "full_catalogue_item_count": int(
                np.count_nonzero(~full_row_covered)
            ),
            "warm_training_item_count": int(
                np.count_nonzero(~warm_row_covered)
            ),
            "nonwarm_catalogue_item_count": int(
                np.count_nonzero(~full_row_covered)
                - np.count_nonzero(~warm_row_covered)
            ),
            "learnable_from_warm": "",
        },
    ]
    for feature_name, full_count, warm_count in zip(
        artifacts.genre_feature_names.tolist(),
        full_label_counts.tolist(),
        warm_label_counts.tolist(),
        strict=True,
    ):
        rows.append(
            {
                "row_kind": "feature",
                "feature_block": "genre",
                "feature_name": feature_name,
                "full_catalogue_item_count": int(full_count),
                "warm_training_item_count": int(warm_count),
                "nonwarm_catalogue_item_count": int(full_count - warm_count),
                "learnable_from_warm": (
                    "true" if int(warm_count) > 0 else "false"
                ),
            }
        )
    return tuple(rows)


def _coverage_csv_bytes(
    rows: Sequence[Mapping[str, Any]],
) -> bytes:
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(
        handle,
        fieldnames=list(COVERAGE_COLUMNS),
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue().encode("utf-8")


def build_feature_freeze_state(
    *,
    catalogue_path: Path,
    canonical_item_ids: np.ndarray,
    warm_item_ids: np.ndarray,
) -> FeatureFreezeState:
    """Build the aligned full and warm feature state without protected cohorts."""

    records = _read_catalogue_genres(catalogue_path)
    source_item_ids = np.sort(
        np.asarray([item_id for item_id, _ in records], dtype=np.int64)
    )
    assert_exact_item_alignment(canonical_item_ids, source_item_ids)
    artifacts = build_item_features(canonical_item_ids, records)
    model_feature_view(
        artifacts,
        include_genre=False,
        requested_item_ids=warm_item_ids,
    )
    warm_view = model_feature_view(
        artifacts,
        include_genre=True,
        requested_item_ids=warm_item_ids,
    )
    genre_offset = artifacts.identity.shape[1]
    warm_genre = warm_view.matrix[:, genre_offset:].tocsr()
    warm_row_covered = np.diff(warm_genre.indptr) > 0
    warm_label_counts = np.asarray((warm_genre > 0).sum(axis=0)).ravel()
    rows = _coverage_rows(artifacts, warm_item_ids)
    summary = {
        "full_catalogue_items": int(artifacts.item_ids.size),
        "warm_training_items": int(warm_item_ids.size),
        "identity_features": int(artifacts.identity.shape[1]),
        "genre_features": int(artifacts.genre.shape[1]),
        "genre_nonzeros": int(artifacts.genre.nnz),
        "full_genre_covered_items": int(
            np.count_nonzero(np.diff(artifacts.genre.indptr))
        ),
        "full_zero_content_items": int(
            np.count_nonzero(np.diff(artifacts.genre.indptr) == 0)
        ),
        "warm_genre_covered_items": int(np.count_nonzero(warm_row_covered)),
        "warm_zero_content_items": int(np.count_nonzero(~warm_row_covered)),
        "genre_features_absent_from_warm": int(
            np.count_nonzero(warm_label_counts == 0)
        ),
        "minimum_warm_items_per_genre": int(warm_label_counts.min())
        if warm_label_counts.size
        else 0,
    }
    return FeatureFreezeState(
        artifacts=artifacts,
        warm_item_ids=warm_item_ids.copy(),
        coverage_rows=rows,
        coverage_bytes=_coverage_csv_bytes(rows),
        summary=summary,
    )


def _contract(
    state: FeatureFreezeState,
    model_config: Mapping[str, Any],
) -> dict[str, Any]:
    base = build_feature_manifest(state.artifacts)["contract"]
    configured = model_config["genre_features"]
    expected = {
        "genre_weighting": configured["weighting"],
        "missing_genre_policy": configured["missing_policy"],
        "identity_weight": configured["identity_weight"],
        "genre_block_weight": configured["genre_block_weight"],
        "combined_row_normalization": configured[
            "combined_row_normalization"
        ],
    }
    for key, value in expected.items():
        if base.get(key) != value:
            raise ValueError(f"feature implementation/config mismatch: {key}")
    if configured.get("tags_enabled") is not False:
        raise ValueError("tags cannot enter the S1.3 predictive feature core")
    return {
        **base,
        "predictive_blocks": ["identity", "genre"],
        "model_views": {
            "identity_only": ["identity"],
            "identity_plus_genre": ["identity", "genre"],
        },
        "controlled_ablation_only_toggle": "genre_feature_block",
        "feature_columns_retained_under_warm_row_projection": True,
        "tags_enabled": False,
        "excluded_predictive_fields": [
            "price",
            "bundle_membership",
            "ownership_popularity",
            "playtime",
            "publisher",
            "developer",
            "tags",
        ],
        "publisher_developer_role": "candidate_pool_feasibility_only",
    }


def _input_entries(
    *,
    root: Path,
    model_config_path: Path,
    ranking_config_path: Path,
    protocol_manifest_path: Path,
    interaction_manifest_path: Path,
    split_manifest_path: Path,
    catalogue_path: Path,
) -> dict[str, Any]:
    return {
        label: _input_entry(path, root)
        for label, path in (
            ("preference_models_config", model_config_path),
            ("ranking_config", ranking_config_path),
            ("protocol_manifest", protocol_manifest_path),
            ("interaction_manifest", interaction_manifest_path),
            ("split_manifest", split_manifest_path),
            ("game_features", catalogue_path),
        )
    }


def _semantic_sections(
    *,
    state: FeatureFreezeState,
    upstream: UpstreamState,
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    base = build_feature_manifest(state.artifacts)
    warm_projection = {
        "item_count": int(state.warm_item_ids.size),
        "array_sha256": array_sha256(state.warm_item_ids),
        "semantic_sha256": id_map_sha256(
            state.warm_item_ids,
            label="warm item",
        ),
        "row_order": "ascending_numeric_item_id",
        "full_feature_columns_retained": True,
        "genre_features_absent_from_warm": state.summary[
            "genre_features_absent_from_warm"
        ],
    }
    coverage_semantics = {
        "columns": list(COVERAGE_COLUMNS),
        "row_count": len(state.coverage_rows),
        "semantic_sha256": semantic_sha256(
            [dict(row) for row in state.coverage_rows]
        ),
    }
    contract = _contract(state, upstream.model_config)
    identity = {
        "cycle_id": upstream.model_config["cycle_id"],
        "protocol_id": upstream.protocol_manifest["protocol_id"],
        "interaction_set_id": upstream.interaction_manifest[
            "interaction_set_id"
        ],
        "split_set_id": upstream.split_manifest["split_set_id"],
        "contract": contract,
        "item_map": base["item_map"],
        "blocks": base["blocks"],
        "warm_projection": warm_projection,
        "coverage_semantics": coverage_semantics,
        "input_sha256": {
            label: entry["sha256"]
            for label, entry in sorted(inputs.items())
        },
    }
    return {
        "contract": contract,
        "item_map": base["item_map"],
        "blocks": base["blocks"],
        "warm_projection": warm_projection,
        "coverage_semantics": coverage_semantics,
        "feature_set_id": semantic_sha256(identity),
    }


def _artifact_entries(
    *,
    source_dir: Path,
    published_dir: Path,
    root: Path,
) -> dict[str, Any]:
    return {
        label: {
            "path": _relative_path(published_dir / filename, root),
            "access": "stage1_modeling_input_permitted",
            "size_bytes": int((source_dir / filename).stat().st_size),
            "sha256": file_sha256(source_dir / filename),
        }
        for label, filename in sorted(PUBLIC_ARTIFACT_FILENAMES.items())
    }


def _build_public_manifest(
    *,
    state: FeatureFreezeState,
    upstream: UpstreamState,
    inputs: Mapping[str, Any],
    artifact_entries: Mapping[str, Any],
    coverage_path: Path,
    root: Path,
) -> dict[str, Any]:
    semantic = _semantic_sections(
        state=state,
        upstream=upstream,
        inputs=inputs,
    )
    manifest: dict[str, Any] = {
        "schema_version": PUBLIC_SCHEMA_VERSION,
        "artifact": "frozen_stage1_item_features",
        "cycle_id": upstream.model_config["cycle_id"],
        "protocol_id": upstream.protocol_manifest["protocol_id"],
        "interaction_set_id": upstream.interaction_manifest[
            "interaction_set_id"
        ],
        "split_set_id": upstream.split_manifest["split_set_id"],
        **semantic,
        "summary": dict(state.summary),
        "inputs": dict(inputs),
        "artifacts": dict(artifact_entries),
        "coverage_table": {
            "path": _relative_path(coverage_path, root),
            "size_bytes": len(state.coverage_bytes),
            "sha256": hashlib.sha256(state.coverage_bytes).hexdigest(),
            **semantic["coverage_semantics"],
        },
        "access_boundary": {
            "public_manifest_contains_identifiers": False,
            "design_training_item_map": "permitted_and_consumed",
            "validation_targets": "not_accessed",
            "design_test_targets": "sealed_not_accessed",
            "assessment_ids_or_histories": "sealed_not_accessed",
            "pseudo_cold_items": "reserved_for_s1_8_not_accessed",
            "pseudo_cold_only_genre_count": 0,
            "pseudo_cold_only_genre_count_basis": (
                "every_frozen_genre_occurs_on_warm_training_items"
            ),
            "stage2_objectives_or_bundle_outcomes": "not_accessed",
        },
        "provenance": {
            "command": "python -m src.stage1_feature_artifacts",
            "module_hash_policy": "utf8_text_with_lf_newlines",
            "modules": _module_hashes(),
            "python": platform.python_version(),
            "packages": _package_versions(),
            "publication_policy": (
                "cycle_scoped_staged_publish_manifest_last_"
                "refuse_nonidentical_overwrite"
            ),
        },
    }
    manifest["manifest_id"] = semantic_sha256(manifest)
    return manifest


def _publication_exists(
    *,
    output_dir: Path,
    manifest_path: Path,
    coverage_path: Path,
) -> tuple[bool, bool]:
    flags = (
        output_dir.exists(),
        manifest_path.exists(),
        coverage_path.exists(),
    )
    return any(flags), all(flags)


def _publish_staged(
    *,
    staged_output: Path,
    output_dir: Path,
    staged_coverage: Path,
    coverage_path: Path,
    staged_manifest: Path,
    manifest_path: Path,
) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    coverage_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    created_output = False
    created_coverage = False
    try:
        os.rename(staged_output, output_dir)
        created_output = True
        os.link(staged_coverage, coverage_path)
        created_coverage = True
        os.link(staged_manifest, manifest_path)
    except Exception:
        if created_coverage and coverage_path.exists():
            coverage_path.unlink()
        if created_output and output_dir.exists():
            shutil.rmtree(output_dir)
        raise


def _generation_inputs(
    *,
    root: Path,
    model_config_path: Path,
    ranking_config_path: Path,
    protocol_manifest_path: Path,
    interaction_manifest_path: Path,
    interaction_output_dir: Path,
    split_manifest_path: Path,
) -> tuple[UpstreamState, Path]:
    upstream = _load_verified_upstreams(
        root=root,
        model_config_path=model_config_path,
        ranking_config_path=ranking_config_path,
        protocol_manifest_path=protocol_manifest_path,
        interaction_manifest_path=interaction_manifest_path,
        interaction_output_dir=interaction_output_dir,
        split_manifest_path=split_manifest_path,
    )
    catalogue_path = root / upstream.ranking_config["inputs"]["features_path"]
    configured_hash = upstream.ranking_config["inputs"]["features_sha256"]
    if file_sha256(catalogue_path) != configured_hash:
        raise ValueError("game_features differs from the frozen input hash")
    return upstream, catalogue_path


def generate_feature_artifacts(
    *,
    project_root: str | Path = PROJECT_ROOT,
    model_config_path: str | Path = DEFAULT_MODEL_CONFIG,
    ranking_config_path: str | Path = DEFAULT_RANKING_CONFIG,
    protocol_manifest_path: str | Path = DEFAULT_PROTOCOL_MANIFEST,
    interaction_manifest_path: str | Path = DEFAULT_INTERACTION_MANIFEST,
    interaction_output_dir: str | Path = DEFAULT_INTERACTION_OUTPUT_DIR,
    split_manifest_path: str | Path = DEFAULT_SPLIT_MANIFEST,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    coverage_path: str | Path = DEFAULT_COVERAGE_TABLE,
) -> dict[str, Any]:
    """Stage and publish one immutable, cycle-bound S1.3 feature set."""

    root = Path(project_root).resolve()
    model_path = Path(model_config_path).resolve()
    ranking_path = Path(ranking_config_path).resolve()
    protocol_path = Path(protocol_manifest_path).resolve()
    interaction_manifest = Path(interaction_manifest_path).resolve()
    interaction_output = Path(interaction_output_dir).resolve()
    split_manifest = Path(split_manifest_path).resolve()
    output = Path(output_dir).resolve()
    destination = Path(manifest_path).resolve()
    coverage = Path(coverage_path).resolve()

    exists, complete = _publication_exists(
        output_dir=output,
        manifest_path=destination,
        coverage_path=coverage,
    )
    if exists:
        if not complete:
            raise FileExistsError(
                "refusing partial existing S1.3 feature publication"
            )
        return verify_feature_artifacts(
            project_root=root,
            model_config_path=model_path,
            ranking_config_path=ranking_path,
            protocol_manifest_path=protocol_path,
            interaction_manifest_path=interaction_manifest,
            interaction_output_dir=interaction_output,
            split_manifest_path=split_manifest,
            output_dir=output,
            manifest_path=destination,
            coverage_path=coverage,
        )

    upstream, catalogue_path = _generation_inputs(
        root=root,
        model_config_path=model_path,
        ranking_config_path=ranking_path,
        protocol_manifest_path=protocol_path,
        interaction_manifest_path=interaction_manifest,
        interaction_output_dir=interaction_output,
        split_manifest_path=split_manifest,
    )
    state = build_feature_freeze_state(
        catalogue_path=catalogue_path,
        canonical_item_ids=upstream.canonical_item_ids,
        warm_item_ids=upstream.warm_item_ids,
    )
    inputs = _input_entries(
        root=root,
        model_config_path=model_path,
        ranking_config_path=ranking_path,
        protocol_manifest_path=protocol_path,
        interaction_manifest_path=interaction_manifest,
        split_manifest_path=split_manifest,
        catalogue_path=catalogue_path,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(prefix=".stage1-features-", dir=output.parent)
    )
    staged_output = staging_root / "stage1_features"
    staged_coverage = staging_root / "stage1_feature_coverage.csv"
    staged_manifest = staging_root / "item_feature_manifest.json"
    try:
        save_feature_artifacts(
            state.artifacts,
            staged_output,
            input_files={
                label: root / entry["path"]
                for label, entry in inputs.items()
            },
            path_root=root,
        )
        staged_coverage.write_bytes(state.coverage_bytes)
        artifact_entries = _artifact_entries(
            source_dir=staged_output,
            published_dir=output,
            root=root,
        )
        manifest = _build_public_manifest(
            state=state,
            upstream=upstream,
            inputs=inputs,
            artifact_entries=artifact_entries,
            coverage_path=coverage,
            root=root,
        )
        write_manifest(manifest, staged_manifest)
        _publish_staged(
            staged_output=staged_output,
            output_dir=output,
            staged_coverage=staged_coverage,
            coverage_path=coverage,
            staged_manifest=staged_manifest,
            manifest_path=destination,
        )
    finally:
        if staging_root.exists():
            shutil.rmtree(staging_root)
    return manifest


def verify_feature_artifacts(
    *,
    project_root: str | Path = PROJECT_ROOT,
    model_config_path: str | Path = DEFAULT_MODEL_CONFIG,
    ranking_config_path: str | Path = DEFAULT_RANKING_CONFIG,
    protocol_manifest_path: str | Path = DEFAULT_PROTOCOL_MANIFEST,
    interaction_manifest_path: str | Path = DEFAULT_INTERACTION_MANIFEST,
    interaction_output_dir: str | Path = DEFAULT_INTERACTION_OUTPUT_DIR,
    split_manifest_path: str | Path = DEFAULT_SPLIT_MANIFEST,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    coverage_path: str | Path = DEFAULT_COVERAGE_TABLE,
) -> dict[str, Any]:
    """Regenerate S1.3 semantics and verify all saved bytes without writes."""

    root = Path(project_root).resolve()
    model_path = Path(model_config_path).resolve()
    ranking_path = Path(ranking_config_path).resolve()
    protocol_path = Path(protocol_manifest_path).resolve()
    interaction_manifest = Path(interaction_manifest_path).resolve()
    interaction_output = Path(interaction_output_dir).resolve()
    split_manifest = Path(split_manifest_path).resolve()
    output = Path(output_dir).resolve()
    destination = Path(manifest_path).resolve()
    coverage = Path(coverage_path).resolve()

    manifest = load_json(destination)
    _validate_public_manifest_id(manifest)
    if manifest.get("schema_version") != PUBLIC_SCHEMA_VERSION:
        raise ValueError("unsupported item-feature public manifest schema")

    upstream, catalogue_path = _generation_inputs(
        root=root,
        model_config_path=model_path,
        ranking_config_path=ranking_path,
        protocol_manifest_path=protocol_path,
        interaction_manifest_path=interaction_manifest,
        interaction_output_dir=interaction_output,
        split_manifest_path=split_manifest,
    )
    state = build_feature_freeze_state(
        catalogue_path=catalogue_path,
        canonical_item_ids=upstream.canonical_item_ids,
        warm_item_ids=upstream.warm_item_ids,
    )
    inputs = _input_entries(
        root=root,
        model_config_path=model_path,
        ranking_config_path=ranking_path,
        protocol_manifest_path=protocol_path,
        interaction_manifest_path=interaction_manifest,
        split_manifest_path=split_manifest,
        catalogue_path=catalogue_path,
    )
    if not coverage.is_file() or coverage.read_bytes() != state.coverage_bytes:
        raise ValueError("saved feature coverage table changed")

    loaded = load_feature_artifacts(
        output,
        expected_item_ids=state.artifacts.item_ids,
    )
    expected_semantics = build_feature_manifest(state.artifacts)
    actual_semantics = build_feature_manifest(loaded)
    for key in ("contract", "item_map", "blocks"):
        if actual_semantics[key] != expected_semantics[key]:
            raise ValueError(f"saved feature semantics changed: {key}")

    artifact_entries = _artifact_entries(
        source_dir=output,
        published_dir=output,
        root=root,
    )
    expected_manifest = _build_public_manifest(
        state=state,
        upstream=upstream,
        inputs=inputs,
        artifact_entries=artifact_entries,
        coverage_path=coverage,
        root=root,
    )
    expected_manifest["provenance"]["modules"] = manifest["provenance"][
        "modules"
    ]
    unsigned = dict(expected_manifest)
    unsigned.pop("manifest_id", None)
    expected_manifest["manifest_id"] = semantic_sha256(unsigned)
    if manifest != expected_manifest:
        raise ValueError("saved item-feature public manifest changed")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate or verify frozen Stage 1 item features"
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
        "--interaction-output-dir",
        type=Path,
        default=DEFAULT_INTERACTION_OUTPUT_DIR,
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=DEFAULT_SPLIT_MANIFEST,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--coverage-table",
        type=Path,
        default=DEFAULT_COVERAGE_TABLE,
    )
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)

    function = (
        verify_feature_artifacts
        if args.check_only
        else generate_feature_artifacts
    )
    manifest = function(
        project_root=args.project_root,
        model_config_path=args.models,
        ranking_config_path=args.ranking,
        protocol_manifest_path=args.protocol_manifest,
        interaction_manifest_path=args.interaction_manifest,
        interaction_output_dir=args.interaction_output_dir,
        split_manifest_path=args.split_manifest,
        output_dir=args.output_dir,
        manifest_path=args.manifest,
        coverage_path=args.coverage_table,
    )
    print(
        json.dumps(
            {
                "feature_set_id": manifest["feature_set_id"],
                "manifest_id": manifest["manifest_id"],
                "full_catalogue_items": manifest["summary"][
                    "full_catalogue_items"
                ],
                "warm_training_items": manifest["summary"][
                    "warm_training_items"
                ],
                "genre_features": manifest["summary"]["genre_features"],
                "check_only": bool(args.check_only),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
