"""Tests for the cycle-bound S1.3 feature publication."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.features import FeatureAlignmentError
from src.stage1_feature_artifacts import (
    DEFAULT_MANIFEST,
    _publish_staged,
    build_feature_freeze_state,
    verify_feature_artifacts,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEGACY_INTERACTION_TABLE = PROJECT_ROOT / "outputs" / "tables" / "user_items_df.csv"


def _write_catalogue(path: Path, rows: list[str]) -> None:
    path.write_text(
        "item_id,genres\n" + "\n".join(rows) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def test_freeze_state_aligns_rows_and_reports_warm_coverage(tmp_path):
    catalogue = tmp_path / "game_features.csv"
    _write_catalogue(
        catalogue,
        [
            '30,"Action, RPG"',
            "10,Action",
            "20,",
        ],
    )
    state = build_feature_freeze_state(
        catalogue_path=catalogue,
        canonical_item_ids=np.asarray([10, 20, 30], dtype=np.int64),
        warm_item_ids=np.asarray([10, 30], dtype=np.int64),
    )

    assert state.artifacts.item_ids.tolist() == [10, 20, 30]
    assert state.artifacts.genre_feature_names.tolist() == [
        "genre::Action",
        "genre::RPG",
    ]
    np.testing.assert_allclose(
        state.artifacts.genre.toarray(),
        np.asarray(
            [
                [1.0, 0.0],
                [0.0, 0.0],
                [0.5, 0.5],
            ],
            dtype=np.float32,
        ),
    )
    assert state.summary == {
        "full_catalogue_items": 3,
        "warm_training_items": 2,
        "identity_features": 3,
        "genre_features": 2,
        "genre_nonzeros": 3,
        "full_genre_covered_items": 2,
        "full_zero_content_items": 1,
        "warm_genre_covered_items": 2,
        "warm_zero_content_items": 0,
        "genre_features_absent_from_warm": 0,
        "minimum_warm_items_per_genre": 1,
    }
    assert b"\r" not in state.coverage_bytes
    coverage = state.coverage_bytes.decode("utf-8")
    assert "genre::Action,2,2,0,true" in coverage
    assert "genre::RPG,1,1,0,true" in coverage


def test_freeze_state_rejects_duplicate_noncanonical_and_misaligned_items(
    tmp_path,
):
    duplicate = tmp_path / "duplicate.csv"
    _write_catalogue(duplicate, ["10,Action", "10,RPG"])
    with pytest.raises(ValueError, match="duplicate item rows"):
        build_feature_freeze_state(
            catalogue_path=duplicate,
            canonical_item_ids=np.asarray([10], dtype=np.int64),
            warm_item_ids=np.asarray([10], dtype=np.int64),
        )

    padded = tmp_path / "padded.csv"
    _write_catalogue(padded, ["010,Action"])
    with pytest.raises(ValueError, match="canonical nonnegative decimal"):
        build_feature_freeze_state(
            catalogue_path=padded,
            canonical_item_ids=np.asarray([10], dtype=np.int64),
            warm_item_ids=np.asarray([10], dtype=np.int64),
        )

    missing = tmp_path / "missing.csv"
    _write_catalogue(missing, ["10,Action"])
    with pytest.raises(FeatureAlignmentError, match="length mismatch"):
        build_feature_freeze_state(
            catalogue_path=missing,
            canonical_item_ids=np.asarray([10, 20], dtype=np.int64),
            warm_item_ids=np.asarray([10], dtype=np.int64),
        )


def test_staged_publication_is_manifest_last_and_refuses_overwrite(tmp_path):
    staging = tmp_path / "staging"
    staged_output = staging / "payload"
    staged_output.mkdir(parents=True)
    (staged_output / "artifact.bin").write_bytes(b"artifact")
    staged_coverage = staging / "coverage.csv"
    staged_coverage.write_bytes(b"coverage\n")
    staged_manifest = staging / "manifest.json"
    staged_manifest.write_bytes(b"{}\n")

    output = tmp_path / "published" / "features"
    coverage = tmp_path / "public" / "coverage.csv"
    manifest = tmp_path / "public" / "manifest.json"
    _publish_staged(
        staged_output=staged_output,
        output_dir=output,
        staged_coverage=staged_coverage,
        coverage_path=coverage,
        staged_manifest=staged_manifest,
        manifest_path=manifest,
    )
    assert (output / "artifact.bin").read_bytes() == b"artifact"
    assert coverage.read_bytes() == b"coverage\n"
    assert manifest.read_bytes() == b"{}\n"

    second_staging = tmp_path / "second"
    second_output = second_staging / "payload"
    second_output.mkdir(parents=True)
    (second_output / "artifact.bin").write_bytes(b"different")
    second_coverage = second_staging / "coverage.csv"
    second_coverage.write_bytes(b"different\n")
    second_manifest = second_staging / "manifest.json"
    second_manifest.write_bytes(b'{"different":true}\n')
    with pytest.raises(FileExistsError):
        _publish_staged(
            staged_output=second_output,
            output_dir=output,
            staged_coverage=second_coverage,
            coverage_path=coverage,
            staged_manifest=second_manifest,
            manifest_path=manifest,
        )
    assert (output / "artifact.bin").read_bytes() == b"artifact"
    assert coverage.read_bytes() == b"coverage\n"
    assert manifest.read_bytes() == b"{}\n"


def test_manifest_link_failure_removes_only_new_partial_publication(tmp_path):
    staging = tmp_path / "staging"
    staged_output = staging / "payload"
    staged_output.mkdir(parents=True)
    (staged_output / "artifact.bin").write_bytes(b"artifact")
    staged_coverage = staging / "coverage.csv"
    staged_coverage.write_bytes(b"coverage\n")
    staged_manifest = staging / "manifest.json"
    staged_manifest.write_bytes(b"new\n")

    output = tmp_path / "published" / "features"
    coverage = tmp_path / "public" / "coverage.csv"
    manifest = tmp_path / "public" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_bytes(b"existing\n")
    with pytest.raises(FileExistsError):
        _publish_staged(
            staged_output=staged_output,
            output_dir=output,
            staged_coverage=staged_coverage,
            coverage_path=coverage,
            staged_manifest=staged_manifest,
            manifest_path=manifest,
        )
    assert not output.exists()
    assert not coverage.exists()
    assert manifest.read_bytes() == b"existing\n"


@pytest.mark.skipif(
    not DEFAULT_MANIFEST.exists() or not LEGACY_INTERACTION_TABLE.exists(),
    reason="frozen S1.3 regeneration requires the ignored legacy interaction table",
)
def test_frozen_publication_exactly_regenerates_without_protected_outcomes():
    manifest = verify_feature_artifacts()
    assert manifest["summary"]["full_catalogue_items"] == 10_978
    assert manifest["summary"]["warm_training_items"] == 6_721
    assert manifest["summary"]["genre_features"] == 21
    assert manifest["summary"]["genre_features_absent_from_warm"] == 0
    assert manifest["access_boundary"]["pseudo_cold_items"] == (
        "reserved_for_s1_8_not_accessed"
    )
    assert manifest["access_boundary"]["design_test_targets"] == (
        "sealed_not_accessed"
    )
    assert manifest["access_boundary"]["assessment_ids_or_histories"] == (
        "sealed_not_accessed"
    )
