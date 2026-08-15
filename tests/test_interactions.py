"""Tests for the sparse Stage 1 interaction and confidence contract."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from src.interactions import (
    CanonicalEdges,
    SparseInteractionData,
    assert_aligned_item_rows,
    assert_exact_id_alignment,
    build_sparse_interactions,
    canonicalize_edges,
    csr_semantic_sha256,
    edge_sha256,
    id_map_sha256,
    load_interaction_csv_audited,
    load_sparse_interactions,
    observed_confidence,
    remove_observed_pairs,
    save_sparse_interactions,
    sparse_storage_bytes,
)
from src.stage1_interaction_artifacts import (
    EXPECTED_PROTOCOL_ID,
    generate_interaction_artifacts,
    verify_interaction_artifacts,
)
from src.stage1_protocol import file_sha256


def _fixture(order=None):
    rows = np.array(
        [
            [20, 200, 0, 0],
            [10, 300, 8, 1],
            [10, 100, 0, 0],
            [20, 200, 5, 2],
            [10, 300, 3, 0],
        ],
        dtype=np.int64,
    )
    if order is not None:
        rows = rows[order]
    return canonicalize_edges(rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3])


def test_duplicate_collapse_is_max_playtime_and_permutation_invariant():
    first = _fixture()
    second = _fixture([4, 2, 0, 3, 1])
    assert first.input_row_count == 5
    assert first.duplicate_excess_rows == 2
    assert first.n_edges == 3
    assert first.user_id.tolist() == [10, 10, 20]
    assert first.item_id.tolist() == [100, 300, 200]
    assert first.playtime_forever.tolist() == [0, 8, 5]
    assert first.playtime_2weeks.tolist() == [0, 1, 2]
    assert edge_sha256(first) == edge_sha256(second)


def test_signed_zero_is_canonical_and_permutation_invariant():
    first = canonicalize_edges([1, 1], [2, 2], [0.0, -0.0], [-0.0, 0.0])
    second = canonicalize_edges([1, 1], [2, 2], [-0.0, 0.0], [0.0, -0.0])
    assert not np.signbit(first.playtime_forever[0])
    assert not np.signbit(first.playtime_2weeks[0])
    assert edge_sha256(first) == edge_sha256(second)


def test_direct_edge_contract_normalizes_storage_and_supports_subsets():
    edges = CanonicalEdges([1], [2], [0], [-0.0], 1, 0)
    assert edges.user_id.dtype == np.int64
    assert edges.item_id.dtype == np.int64
    assert edges.playtime_forever.dtype == np.float64
    assert edges.playtime_2weeks.dtype == np.float64
    assert not np.signbit(edges.playtime_2weeks[0])
    assert edges.subset([True]).n_edges == 1
    assert edge_sha256(edges) == edge_sha256(
        CanonicalEdges(
            np.asarray([1], dtype=np.int32),
            np.asarray([2], dtype=np.int32),
            np.asarray([0], dtype=np.float32),
            np.asarray([0], dtype=np.float32),
            1,
            0,
        )
    )


def test_sparse_matrices_share_exact_id_order_and_pattern():
    edges = _fixture()
    data = build_sparse_interactions(edges)
    assert data.user_ids.tolist() == [10, 20]
    assert data.item_ids.tolist() == [100, 200, 300]
    assert sp.isspmatrix_csr(data.ownership)
    assert data.ownership.shape == (2, 3)
    assert data.ownership.nnz == 3
    assert np.array_equal(data.ownership.indices, data.playtime_forever.indices)
    assert np.array_equal(data.ownership.indptr, data.playtime_forever.indptr)


def test_explicit_permuted_id_contract_fails_loudly():
    edges = _fixture()
    with pytest.raises(ValueError, match="ascending"):
        build_sparse_interactions(edges, item_ids=[300, 200, 100])
    with pytest.raises(ValueError, match="absent"):
        build_sparse_interactions(edges, item_ids=[100, 200])
    with pytest.raises(ValueError, match="order mismatch"):
        assert_exact_id_alignment([100, 200, 300], [200, 100, 300], label="item")
    permuted_features = sp.csr_matrix(np.eye(3, dtype=np.float32))[[1, 0, 2]]
    with pytest.raises(ValueError, match="order mismatch"):
        assert_aligned_item_rows(
            [100, 200, 300],
            [200, 100, 300],
            permuted_features,
        )


def test_owned_unplayed_confidence_exceeds_unobserved_and_cap_is_exact():
    data = build_sparse_interactions(_fixture())
    confidence = observed_confidence(
        data.ownership,
        data.playtime_forever,
        alpha_o=20,
        alpha_p=2,
        tau=1.0,
    )
    assert confidence.shape == data.ownership.shape
    assert confidence.nnz == data.ownership.nnz
    assert float(confidence[0, 0]) == 21.0
    assert np.isclose(float(confidence[0, 2]), 23.0)
    assert float(confidence[1, 0]) == 0.0  # unobserved baseline one is implicit
    assert np.all(confidence.data > 1.0)


def test_invalid_playtime_or_confidence_contract_is_rejected():
    with pytest.raises(ValueError, match="negative"):
        canonicalize_edges([1], [2], [-1], [0])
    matrix = sp.csr_matrix([[1.0, 0.0]], dtype=np.float32)
    other = sp.csr_matrix([[0.0, 1.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="identical sparsity"):
        observed_confidence(matrix, other, alpha_o=20, alpha_p=0, tau=0)
    with pytest.raises(ValueError, match="finite"):
        observed_confidence(matrix, matrix, alpha_o=np.nan, alpha_p=0, tau=0)
    with pytest.raises(ValueError, match="nonnegative"):
        observed_confidence(matrix, -matrix, alpha_o=20, alpha_p=1, tau=1)
    with pytest.raises(ValueError, match="finite in float32"):
        observed_confidence(matrix, matrix, alpha_o=1e300, alpha_p=0, tau=0)


@pytest.mark.parametrize(
    "user_id",
    [
        [1.5],
        [-1],
        [True],
        ["01"],
        ["1.0"],
        [str(np.iinfo(np.int64).max + 1)],
    ],
)
def test_noncanonical_ids_fail_without_numeric_truncation(user_id):
    with pytest.raises(ValueError):
        canonicalize_edges(user_id, [2], [0], [0])


def test_audited_csv_exclusions_reconcile_and_are_permutation_invariant(tmp_path):
    frame = pd.DataFrame(
        [
            ["10", "100", 1.0, 0.0],
            ["10", "100", 5.0, 2.0],
            ["20", "200", 0.0, 0.0],
            [None, "300", 1.0, 0.0],
            ["user-x", "300", 1.0, 0.0],
            ["30", None, 1.0, 0.0],
            ["40", "04", 1.0, 0.0],
            ["bad", None, 1.0, 0.0],
        ],
        columns=[
            "user_id",
            "item_id",
            "playtime_forever",
            "playtime_2weeks",
        ],
    )
    first_path = tmp_path / "first.csv"
    second_path = tmp_path / "second.csv"
    frame.to_csv(first_path, index=False)
    frame.sample(frac=1.0, random_state=7).to_csv(second_path, index=False)

    first = load_interaction_csv_audited(first_path, chunksize=2)
    second = load_interaction_csv_audited(second_path, chunksize=3)

    assert first.audit.source_rows == 8
    assert first.audit.eligible_rows_before_duplicate_collapse == 3
    assert first.audit.excluded_rows == 5
    assert first.audit.excluded_by_primary_reason == {
        "missing_user_id": 1,
        "invalid_user_id": 2,
        "missing_item_id": 1,
        "invalid_item_id": 1,
    }
    assert first.audit.raw_invalid_flags == {
        "missing_user_id": 1,
        "invalid_user_id": 2,
        "missing_item_id": 2,
        "invalid_item_id": 1,
    }
    assert first.edges.duplicate_excess_rows == 1
    assert (
        first.audit.source_rows
        == first.audit.eligible_rows_before_duplicate_collapse
        + first.audit.excluded_rows
    )
    assert (
        first.audit.eligible_rows_before_duplicate_collapse
        == first.edges.n_edges + first.edges.duplicate_excess_rows
    )
    assert first.edges.user_id.tolist() == [10, 20]
    assert first.edges.item_id.tolist() == [100, 200]
    assert first.edges.playtime_forever.tolist() == [5.0, 0.0]
    assert first.edges.playtime_2weeks.tolist() == [2.0, 0.0]
    assert edge_sha256(first.edges) == edge_sha256(second.edges)
    assert (
        first.audit.excluded_by_primary_reason
        == second.audit.excluded_by_primary_reason
    )
    assert first.audit.raw_invalid_flags == second.audit.raw_invalid_flags


def test_invalid_playtime_fails_even_when_the_row_id_is_ineligible(tmp_path):
    path = tmp_path / "invalid.csv"
    pd.DataFrame(
        [["not-an-id", "2", -1.0, 0.0]],
        columns=[
            "user_id",
            "item_id",
            "playtime_forever",
            "playtime_2weeks",
        ],
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match="negative"):
        load_interaction_csv_audited(path, chunksize=1)


@pytest.mark.parametrize("playtime", [np.inf, -np.inf])
def test_nonfinite_csv_playtime_fails(tmp_path, playtime):
    path = tmp_path / "nonfinite.csv"
    pd.DataFrame(
        [["1", "2", playtime, 0.0]],
        columns=[
            "user_id",
            "item_id",
            "playtime_forever",
            "playtime_2weeks",
        ],
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match="finite"):
        load_interaction_csv_audited(path, chunksize=1)


def test_heldout_removal_updates_preference_and_all_confidence_inputs():
    data = build_sparse_interactions(_fixture())
    original_hashes = [
        csr_semantic_sha256(matrix)
        for matrix in (
            data.ownership,
            data.playtime_forever,
            data.playtime_2weeks,
        )
    ]

    retained = remove_observed_pairs(data, [10, 20], [100, 200])

    assert retained.ownership.nnz == 1
    assert retained.playtime_forever.nnz == 1
    assert retained.playtime_2weeks.nnz == 1
    assert np.array_equal(retained.user_ids, data.user_ids)
    assert np.array_equal(retained.item_ids, data.item_ids)
    for matrix in (
        retained.ownership,
        retained.playtime_forever,
        retained.playtime_2weeks,
    ):
        assert float(matrix[0, 0]) == 0.0
        assert float(matrix[1, 1]) == 0.0
    assert float(retained.ownership[0, 2]) == 1.0
    assert float(retained.playtime_forever[0, 2]) == 8.0
    assert float(retained.playtime_2weeks[0, 2]) == 1.0
    assert [csr_semantic_sha256(matrix) for matrix in (
        data.ownership,
        data.playtime_forever,
        data.playtime_2weeks,
    )] == original_hashes

    confidence = observed_confidence(
        retained.ownership,
        retained.playtime_forever,
        alpha_o=20,
        alpha_p=2,
        tau=1,
    )
    assert confidence.nnz == retained.ownership.nnz
    with pytest.raises(ValueError, match="unique"):
        remove_observed_pairs(data, [10, 10], [100, 100])
    with pytest.raises(ValueError, match="not an observed"):
        remove_observed_pairs(data, [20], [100])


def test_sparse_contract_rejects_permuted_ids_and_nonbinary_ownership():
    data = build_sparse_interactions(_fixture())
    with pytest.raises(ValueError, match="ascending"):
        SparseInteractionData(
            ownership=data.ownership,
            playtime_forever=data.playtime_forever,
            playtime_2weeks=data.playtime_2weeks,
            user_ids=data.user_ids,
            item_ids=data.item_ids[::-1].copy(),
        )

    invalid_ownership = data.ownership.copy()
    invalid_ownership.data[0] = 2.0
    with pytest.raises(ValueError, match="binary"):
        SparseInteractionData(
            ownership=invalid_ownership,
            playtime_forever=data.playtime_forever,
            playtime_2weeks=data.playtime_2weeks,
            user_ids=data.user_ids,
            item_ids=data.item_ids,
        )


def test_sparse_contract_copies_and_freezes_caller_id_maps():
    edges = _fixture()
    users = np.asarray([10, 20], dtype=np.int64)
    items = np.asarray([100, 200, 300], dtype=np.int64)
    data = build_sparse_interactions(edges, user_ids=users, item_ids=items)
    users[0] = 0
    items[0] = 0
    assert data.user_ids.tolist() == [10, 20]
    assert data.item_ids.tolist() == [100, 200, 300]
    assert not data.user_ids.flags.writeable
    assert not data.item_ids.flags.writeable


def test_explicit_zero_support_catalogue_columns_are_preserved():
    edges = canonicalize_edges([10], [100], [0], [0])
    data = build_sparse_interactions(
        edges,
        user_ids=[10],
        item_ids=[100, 200],
    )
    assert data.item_ids.tolist() == [100, 200]
    assert data.ownership.shape == (1, 2)
    assert data.ownership.getnnz(axis=0).tolist() == [1, 0]
    assert id_map_sha256(data.item_ids, label="item") == id_map_sha256(
        [100, 200], label="item"
    )


def test_sparse_storage_is_smaller_than_dense_equivalent():
    edges = canonicalize_edges(
        [0, 50, 99],
        [0, 100, 199],
        [0.0, 1.0, 2.0],
        [0.0, 0.0, 1.0],
    )
    data = build_sparse_interactions(
        edges,
        user_ids=np.arange(100, dtype=np.int64),
        item_ids=np.arange(200, dtype=np.int64),
    )
    dense_matrix_bytes = 3 * 100 * 200 * np.dtype(np.float32).itemsize
    dense_contract_bytes = (
        dense_matrix_bytes + data.user_ids.nbytes + data.item_ids.nbytes
    )
    assert sparse_storage_bytes(data) < dense_contract_bytes


def test_sparse_round_trip_verifies_physical_and_semantic_hashes(tmp_path):
    data = build_sparse_interactions(_fixture())
    hashes = save_sparse_interactions(data, tmp_path, prefix="fixture")
    loaded = load_sparse_interactions(
        tmp_path,
        prefix="fixture",
        expected_file_hashes=hashes,
    )

    for original, restored in (
        (data.ownership, loaded.ownership),
        (data.playtime_forever, loaded.playtime_forever),
        (data.playtime_2weeks, loaded.playtime_2weeks),
    ):
        assert csr_semantic_sha256(original) == csr_semantic_sha256(restored)
    assert np.array_equal(data.user_ids, loaded.user_ids)
    assert np.array_equal(data.item_ids, loaded.item_ids)

    wrong_hashes = dict(hashes)
    wrong_hashes["ownership"] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        load_sparse_interactions(
            tmp_path,
            prefix="fixture",
            expected_file_hashes=wrong_hashes,
        )


def test_csr_semantic_hash_normalizes_signed_zero():
    positive = sp.csr_matrix(
        (
            np.asarray([0.0], dtype=np.float32),
            np.asarray([0], dtype=np.int32),
            np.asarray([0, 1], dtype=np.int32),
        ),
        shape=(1, 1),
    )
    negative = positive.copy()
    negative.data[0] = -0.0
    assert csr_semantic_sha256(positive) == csr_semantic_sha256(negative)


def _write_builder_inputs(
    root,
    interactions,
    feature_item_ids,
    *,
    suffix,
):
    interaction_path = root / f"interactions_{suffix}.csv"
    feature_path = root / f"features_{suffix}.csv"
    ranking_path = root / f"ranking_{suffix}.json"
    protocol_path = root / "protocol.json"
    interactions.to_csv(interaction_path, index=False)
    pd.DataFrame({"item_id": feature_item_ids}).to_csv(feature_path, index=False)
    ranking = {
        "schema_version": 1,
        "cycle_id": "s1-v1-20260718",
        "inputs": {
            "interactions_path": interaction_path.name,
            "interactions_sha256": file_sha256(interaction_path),
            "features_path": feature_path.name,
            "features_sha256": file_sha256(feature_path),
        },
        "id_contract": {
            "user_order": "ascending_numeric_user_id",
            "item_order": "ascending_numeric_item_id",
            "duplicate_rule": (
                "one ownership edge; maximum nonnegative playtime_forever "
                "and playtime_2weeks"
            ),
            "missing_id_policy": "exclude_and_count",
            "negative_or_nonfinite_playtime_policy": "fail",
            "hash_algorithm": "sha256",
            "canonical_id_encoding": "utf8_decimal_without_padding",
        },
    }
    ranking_path.write_text(
        json.dumps(ranking, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not protocol_path.exists():
        protocol_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "cycle_id": "s1-v1-20260718",
                    "protocol_id": EXPECTED_PROTOCOL_ID,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return ranking_path, protocol_path


def test_artifact_builder_is_verifiable_and_row_permutation_invariant(tmp_path):
    columns = [
        "user_id",
        "item_id",
        "playtime_forever",
        "playtime_2weeks",
    ]
    interactions = pd.DataFrame(
        [
            ["10", "100", 1.0, 0.0],
            ["10", "100", 5.0, 2.0],
            ["20", "200", 0.0, 0.0],
            ["bad", "100", 1.0, 0.0],
        ],
        columns=columns,
    )
    feature_item_ids = ["300", "100", "200"]
    first_ranking, protocol = _write_builder_inputs(
        tmp_path,
        interactions,
        feature_item_ids,
        suffix="first",
    )
    first_output = tmp_path / "artifacts_first"
    first_manifest_path = tmp_path / "manifest_first.json"
    first = generate_interaction_artifacts(
        project_root=tmp_path,
        ranking_config_path=first_ranking,
        protocol_manifest_path=protocol,
        output_dir=first_output,
        manifest_path=first_manifest_path,
    )

    assert first["matrices"]["ownership"]["shape"] == [2, 3]
    assert first["matrices"]["ownership"]["nnz"] == 2
    assert first["load_audit"]["excluded_rows"] == 1
    assert first["load_audit"]["duplicate_excess_rows"] == 1
    assert first["item_reconciliation"]["metadata_only_item_count"] == 1
    assert not first["item_reconciliation"]["feature_source_order_is_canonical"]
    original_manifest_bytes = first_manifest_path.read_bytes()
    verified = verify_interaction_artifacts(
        project_root=tmp_path,
        ranking_config_path=first_ranking,
        protocol_manifest_path=protocol,
        output_dir=first_output,
        manifest_path=first_manifest_path,
    )
    assert verified["interaction_set_id"] == first["interaction_set_id"]
    assert first_manifest_path.read_bytes() == original_manifest_bytes

    second_ranking, _ = _write_builder_inputs(
        tmp_path,
        interactions.iloc[[3, 1, 0, 2]],
        feature_item_ids,
        suffix="second",
    )
    second = generate_interaction_artifacts(
        project_root=tmp_path,
        ranking_config_path=second_ranking,
        protocol_manifest_path=protocol,
        output_dir=tmp_path / "artifacts_second",
        manifest_path=tmp_path / "manifest_second.json",
    )
    assert second["interaction_set_id"] == first["interaction_set_id"]


def test_artifact_builder_rejects_interaction_items_without_metadata(tmp_path):
    interactions = pd.DataFrame(
        [["10", "200", 1.0, 0.0]],
        columns=[
            "user_id",
            "item_id",
            "playtime_forever",
            "playtime_2weeks",
        ],
    )
    ranking, protocol = _write_builder_inputs(
        tmp_path,
        interactions,
        ["100"],
        suffix="missing",
    )
    with pytest.raises(ValueError, match="absent from the feature"):
        generate_interaction_artifacts(
            project_root=tmp_path,
            ranking_config_path=ranking,
            protocol_manifest_path=protocol,
            output_dir=tmp_path / "artifacts",
            manifest_path=tmp_path / "manifest.json",
        )
