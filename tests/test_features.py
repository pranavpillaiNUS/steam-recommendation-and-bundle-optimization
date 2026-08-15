"""Tests for the frozen sparse item-feature and alignment contract."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from src.features import (
    FeatureAlignmentError,
    aggregate_genre_records,
    assert_exact_item_alignment,
    build_feature_manifest,
    build_item_features,
    canonical_item_ids,
    csr_semantic_sha256,
    file_sha256,
    load_feature_artifacts,
    model_feature_view,
    normalize_feature_token,
    save_feature_artifacts,
    semantic_sha256,
    validate_feature_artifacts,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GAME_FEATURES = PROJECT_ROOT / "outputs" / "tables" / "game_features.csv"


def test_numeric_item_contract_preserves_caller_order_and_rejects_bad_ids():
    result = canonical_item_ids([2, "10", np.int64(100)])
    assert result.tolist() == [2, 10, 100]
    assert result.dtype == np.int64

    for bad in (
        ["10", "2"],
        ["2", "2"],
        ["02"],
        [-1],
        ["2.0"],
        [2.5],
        [True],
        [2**63],
    ):
        with pytest.raises(ValueError):
            canonical_item_ids(bad)
    np.testing.assert_array_equal(canonical_item_ids([0, 2.0]), [0, 2])


def test_alignment_guard_rejects_a_pure_permutation():
    assert_exact_item_alignment([2, 10, 100], ["2", "10", "100"])
    with pytest.raises(FeatureAlignmentError, match="order mismatch at row 0"):
        assert_exact_item_alignment([2, 10, 100], [10, 2, 100])
    with pytest.raises(FeatureAlignmentError, match="length mismatch"):
        assert_exact_item_alignment([2, 10], [2])


def test_token_normalization_is_conservative_and_duplicate_union_is_order_free():
    assert (
        normalize_feature_token("  Design &amp;\tIllustration  ")
        == "Design & Illustration"
    )
    assert normalize_feature_token("Ａction") == "Action"
    assert normalize_feature_token("RPG") != normalize_feature_token("rpg")

    records = [
        ("10", ["RPG", " Action "]),
        (10, ["Action", "Design &amp; Illustration"]),
        ("20", None),
    ]
    forward = aggregate_genre_records(records)
    backward = aggregate_genre_records(reversed(records))
    assert forward == backward
    assert forward[10] == ("Action", "Design & Illustration", "RPG")
    assert forward[20] == ()


def test_sparse_blocks_have_exact_identity_and_normalized_genre_rows():
    artifacts = build_item_features(
        [10, 20, 100],
        {
            "10": ["RPG", "Action", "Action"],
            "20": None,
            "100": ["Design &amp; Illustration"],
        },
    )
    validate_feature_artifacts(artifacts, expected_item_ids=[10, 20, 100])

    assert sp.isspmatrix_csr(artifacts.identity)
    assert sp.isspmatrix_csr(artifacts.genre)
    assert artifacts.identity.dtype == np.float32
    assert artifacts.genre.dtype == np.float32
    assert artifacts.identity.shape == (3, 3)
    assert artifacts.identity.nnz == 3
    np.testing.assert_array_equal(artifacts.identity.toarray(), np.eye(3))

    assert artifacts.genre_feature_names.tolist() == [
        "genre::Action",
        "genre::Design & Illustration",
        "genre::RPG",
    ]
    np.testing.assert_allclose(
        artifacts.genre.toarray(),
        np.asarray(
            [
                [0.5, 0.0, 0.5],
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )


def test_builder_rejects_unknown_items_unsorted_or_pruned_vocabularies():
    with pytest.raises(FeatureAlignmentError, match="unknown item IDs"):
        build_item_features([10, 20], {10: ["Action"], 30: ["RPG"]})
    with pytest.raises(ValueError, match="lexical order"):
        build_item_features(
            [10, 20], {10: ["Action"]}, genre_vocabulary=["RPG", "Action"]
        )
    with pytest.raises(ValueError, match="absent from the frozen vocabulary"):
        build_item_features(
            [10, 20], {10: ["Action"]}, genre_vocabulary=["RPG"]
        )


def test_validator_rejects_non_int64_ids_non_csr_and_unequal_genre_weights():
    artifacts = build_item_features(
        [10, 20],
        {10: ["Action", "RPG"], 20: ["RPG"]},
    )
    with pytest.raises(ValueError, match="one-dimensional int64"):
        validate_feature_artifacts(
            replace(artifacts, item_ids=artifacts.item_ids.astype("<U2"))
        )
    with pytest.raises(ValueError, match="must be a CSR"):
        validate_feature_artifacts(
            replace(artifacts, genre=artifacts.genre.tocsc())
        )

    unequal = artifacts.genre.copy()
    unequal.data[:2] = np.asarray([0.9, 0.1], dtype=np.float32)
    with pytest.raises(ValueError, match="exactly equally"):
        validate_feature_artifacts(replace(artifacts, genre=unequal))


def test_model_feature_view_projects_rows_and_changes_only_genre_toggle():
    artifacts = build_item_features(
        [10, 20, 100],
        {10: ["Action", "RPG"], 20: None, 100: ["RPG"]},
    )
    identity_only = model_feature_view(
        artifacts,
        include_genre=False,
        requested_item_ids=[10, 100],
    )
    with_genre = model_feature_view(
        artifacts,
        include_genre=True,
        requested_item_ids=[10, 100],
    )

    assert identity_only.item_ids.tolist() == [10, 100]
    assert identity_only.matrix.shape == (2, 3)
    assert with_genre.matrix.shape == (2, 5)
    np.testing.assert_array_equal(
        with_genre.matrix[:, :3].toarray(),
        identity_only.matrix.toarray(),
    )
    np.testing.assert_array_equal(
        identity_only.matrix.toarray(),
        artifacts.identity[[0, 2], :].toarray(),
    )
    np.testing.assert_array_equal(
        with_genre.matrix[:, 3:].toarray(),
        artifacts.genre[[0, 2], :].toarray(),
    )
    assert with_genre.feature_names[:3].tolist() == (
        artifacts.identity_feature_names.tolist()
    )
    assert with_genre.feature_names[3:].tolist() == (
        artifacts.genre_feature_names.tolist()
    )

    with pytest.raises(ValueError, match="ascending"):
        model_feature_view(
            artifacts,
            include_genre=False,
            requested_item_ids=[100, 10],
        )
    with pytest.raises(FeatureAlignmentError, match="absent"):
        model_feature_view(
            artifacts,
            include_genre=False,
            requested_item_ids=[10, 30],
        )
    with pytest.raises(ValueError, match="boolean"):
        model_feature_view(artifacts, include_genre=1)


def test_record_order_does_not_change_sparse_semantics_or_manifest_id():
    records = [
        (10, ["Action"]),
        (20, ["RPG", "Indie"]),
        (20, ["RPG"]),
        (100, None),
    ]
    left = build_item_features([10, 20, 100], records)
    right = build_item_features([10, 20, 100], reversed(records))
    assert csr_semantic_sha256(left.identity) == csr_semantic_sha256(right.identity)
    assert csr_semantic_sha256(left.genre) == csr_semantic_sha256(right.genre)
    assert build_feature_manifest(left)["feature_set_id"] == build_feature_manifest(
        right
    )["feature_set_id"]


def test_save_load_verifies_hashes_unicode_arrays_and_expected_alignment(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("item_id,genres\n10,Action\n", encoding="utf-8")
    artifacts = build_item_features(
        [10, 20, 100], {10: ["Action"], 20: None, 100: ["RPG"]}
    )
    manifest = save_feature_artifacts(
        artifacts,
        tmp_path / "features",
        input_files={"game_features": source},
        path_root=tmp_path,
    )
    loaded = load_feature_artifacts(
        tmp_path / "features", expected_item_ids=[10, 20, 100]
    )
    assert loaded.item_ids.dtype == np.int64
    assert loaded.identity_feature_names.dtype.kind == "U"
    assert loaded.genre_feature_names.dtype.kind == "U"
    assert manifest["blocks"]["genre"]["zero_content_items"] == 1
    assert manifest["inputs"]["game_features"]["path"] == "source.csv"

    manifest_on_disk = json.loads(
        (tmp_path / "features" / "item_feature_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest_on_disk["manifest_id"] == manifest["manifest_id"]
    with pytest.raises(FeatureAlignmentError):
        load_feature_artifacts(
            tmp_path / "features", expected_item_ids=[20, 10, 100]
        )

    genre_path = tmp_path / "features" / "item_features_genre.npz"
    genre_path.write_bytes(genre_path.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="artifact size mismatch"):
        load_feature_artifacts(tmp_path / "features")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda manifest: manifest.update(feature_set_id="0" * 64),
            "set semantic hash",
        ),
        (
            lambda manifest: manifest["artifacts"].update(
                unexpected=dict(
                    path="unexpected.bin",
                    size_bytes=0,
                    sha256="0" * 64,
                )
            ),
            "inventory changed",
        ),
    ],
)
def test_loader_rejects_resigned_set_id_and_inventory_changes(
    tmp_path,
    mutation,
    message,
):
    output = tmp_path / "features"
    save_feature_artifacts(
        build_item_features([10], {10: ["Action"]}),
        output,
    )
    manifest_path = output / "item_feature_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutation(manifest)
    manifest.pop("manifest_id")
    manifest["manifest_id"] = semantic_sha256(manifest)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(ValueError, match=message):
        load_feature_artifacts(output)


def test_manifest_rejects_inputs_outside_declared_root(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("item_id,genres\n10,Action\n", encoding="utf-8")
    with pytest.raises(ValueError, match="outside path_root"):
        build_feature_manifest(
            build_item_features([10], {10: ["Action"]}),
            input_files={"game_features": source},
            path_root=tmp_path / "different-root",
        )


def test_loader_rejects_non_csr_storage_even_with_matching_physical_manifest(tmp_path):
    output = tmp_path / "features"
    artifacts = build_item_features(
        [10, 20],
        {10: ["Action"], 20: ["RPG"]},
    )
    save_feature_artifacts(artifacts, output)

    genre_path = output / "item_features_genre.npz"
    sp.save_npz(genre_path, artifacts.genre.tocsc(), compressed=True)
    manifest_path = output / "item_feature_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["genre"]["size_bytes"] = genre_path.stat().st_size
    manifest["artifacts"]["genre"]["sha256"] = file_sha256(genre_path)
    manifest.pop("manifest_id")
    manifest["manifest_id"] = semantic_sha256(manifest)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(ValueError, match="genre must be a CSR"):
        load_feature_artifacts(output)


@pytest.mark.skipif(not GAME_FEATURES.exists(), reason="derived feature table unavailable")
def test_full_snapshot_genre_contract_and_reconciliation():
    table = pd.read_csv(GAME_FEATURES, dtype={"item_id": "string"})
    item_ids = sorted(table["item_id"].tolist(), key=int)
    genres = {
        str(row.item_id): (
            [] if pd.isna(row.genres) else str(row.genres).split(", ")
        )
        for row in table[["item_id", "genres"]].itertuples(index=False)
    }
    artifacts = build_item_features(item_ids, genres)

    assert artifacts.item_ids.size == 10_978
    assert len(set(artifacts.item_ids.tolist())) == 10_978
    assert artifacts.identity.shape == (10_978, 10_978)
    assert artifacts.identity.nnz == 10_978
    assert artifacts.genre.shape == (10_978, 21)
    assert artifacts.genre.nnz == 21_559
    row_mass = np.asarray(artifacts.genre.sum(axis=1)).ravel()
    assert np.count_nonzero(row_mass) == 8_658
    assert np.count_nonzero(row_mass == 0.0) == 2_320
    np.testing.assert_allclose(row_mass[row_mass > 0.0], 1.0, atol=2e-7)
    assert "genre::Design & Illustration" in artifacts.genre_feature_names
    assert "genre::Design &amp; Illustration" not in artifacts.genre_feature_names
    manifest = build_feature_manifest(artifacts)
    assert manifest["item_map"] == {
        "count": 10_978,
        "array_sha256": (
            "663f35b594a6a989ea085a1a89760c415ffdc7d2004763700723e164837d5d49"
        ),
        "semantic_sha256": (
            "b7dfcbec1d505243bfc843cf4133d6c65f374f731ab513252960e33ff64cb426"
        ),
    }

    sparse_bytes = sum(
        matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
        for matrix in (artifacts.identity, artifacts.genre)
    )
    assert sparse_bytes < 1024 * 1024

    # The old notebook used first-occurrence order; construction must reindex
    # to the frozen numeric order rather than accepting that incidental order.
    assert [int(value) for value in table["item_id"]] != artifacts.item_ids.tolist()
