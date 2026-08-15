import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.stage1_split_artifacts as split_artifacts
from src.interactions import (
    SparseInteractionData,
    build_sparse_interactions,
    canonicalize_edges,
    observed_confidence,
    remove_observed_pairs,
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
from src.stage1_split_artifacts import build_split_state
from src.stage1_protocol import stable_hash_uint64, write_manifest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def ranking_config():
    return json.loads(
        (ROOT / "configs" / "ranking_evaluation.json").read_text(encoding="utf-8")
    )


def test_outer_split_is_exact_disjoint_and_permutation_invariant(ranking_config):
    band_sizes = [10, 11, 9, 5, 6, 10]
    band_activity = [5, 10, 25, 50, 100, 200]
    rows = []
    user_id = 1000
    for size, activity in zip(band_sizes, band_activity):
        for _ in range(size):
            rows.append((str(user_id), activity))
            user_id += 1
    for _ in range(4):
        rows.append((str(user_id), 4))
        user_id += 1
    users = pd.DataFrame(rows, columns=["user_id", "raw_ownership_count"])
    spec = ranking_config["outer_user_split"]

    first = activity_band_outer_split(users, spec)
    second = activity_band_outer_split(
        users.sample(frac=1.0, random_state=917).reset_index(drop=True), spec
    )
    pd.testing.assert_frame_equal(first, second)

    excluded = first.loc[first["split"].eq(EXCLUDED_LOW_ACTIVITY)]
    assert len(excluded) == 4
    assert (excluded["raw_ownership_count"] < 5).all()
    eligible = first.loc[first["split"].ne(EXCLUDED_LOW_ACTIVITY)]
    design_ids = set(eligible.loc[eligible["split"].eq("design"), "user_id"])
    assessment_ids = set(
        eligible.loc[eligible["split"].eq("assessment"), "user_id"]
    )
    assert design_ids.isdisjoint(assessment_ids)
    assert design_ids | assessment_ids == set(eligible["user_id"])

    for band, size in enumerate(band_sizes):
        band_rows = eligible.loc[eligible["activity_band"].eq(band)].copy()
        assert len(band_rows) == size
        assert band_rows["split"].eq("assessment").sum() == size // 5
        expected = sorted(
            band_rows["user_id"],
            key=lambda uid: (
                stable_hash_uint64(spec["namespace"], band, str(uid)),
                uid,
            ),
        )[: size // 5]
        actual = band_rows.loc[
            band_rows["split"].eq("assessment"), "user_id"
        ].tolist()
        assert set(actual) == set(expected)


def test_outer_split_rejects_nonunique_users(ranking_config):
    users = pd.DataFrame(
        {"user_id": ["1", "1"], "raw_ownership_count": [5, 5]}
    )
    with pytest.raises(ValueError, match="one row per user"):
        activity_band_outer_split(users, ranking_config["outer_user_split"])


def test_outer_split_ignores_duplicate_caller_index_labels(ranking_config):
    users = pd.DataFrame(
        {
            "user_id": np.arange(1, 13, dtype=np.int64),
            "raw_ownership_count": np.full(12, 5, dtype=np.int64),
        },
        index=[0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
    )
    expected = activity_band_outer_split(
        users.reset_index(drop=True),
        ranking_config["outer_user_split"],
    )
    actual = activity_band_outer_split(
        users,
        ranking_config["outer_user_split"],
    )
    pd.testing.assert_frame_equal(actual, expected)


def test_edge_split_is_transactional_capacity_aware_and_permutation_invariant(
    ranking_config,
):
    # Four fully supported items give enough total capacity for all eight users.
    edges = pd.DataFrame(
        [
            (str(user_id), str(item_id))
            for user_id in range(1, 9)
            for item_id in (10, 20, 30, 40)
        ]
        + [(str(user_id), "99") for user_id in range(1, 5)],
        columns=["user_id", "item_id"],
    )
    edges["outer_split"] = "design"
    first = capacity_aware_edge_split(
        edges,
        ranking_config["warm_catalogue"],
        ranking_config["nested_interaction_split"],
    )
    second = capacity_aware_edge_split(
        edges.sample(frac=1.0, random_state=2718).reset_index(drop=True),
        ranking_config["warm_catalogue"],
        ranking_config["nested_interaction_split"],
    )
    pd.testing.assert_frame_equal(first, second)

    assert first.loc[first["item_id"].eq(99), "role"].eq(
        EXCLUDED_NONWARM_ITEM
    ).all()
    warm = first.loc[first["is_warm_item"]]
    assert set(warm["role"]) == {"training", "validation", "test"}
    assert warm.loc[warm["role"].eq("training")].groupby("item_id").size().min() >= 3

    evaluable = first.loc[first["evaluable_user"], "user_id"].unique()
    assert len(evaluable) == 8
    for user_id in evaluable:
        user_rows = first.loc[first["user_id"].eq(user_id)]
        validation = set(user_rows.loc[user_rows["role"].eq("validation"), "item_id"])
        test = set(user_rows.loc[user_rows["role"].eq("test"), "item_id"])
        training = set(user_rows.loc[user_rows["role"].eq("training"), "item_id"])
        assert len(validation) == len(test) == 1
        assert validation.isdisjoint(test)
        assert validation.isdisjoint(training)
        assert test.isdisjoint(training)

    # A bottleneck has capacity for only three complete two-edge holdouts.
    bottleneck = pd.DataFrame(
        [
            (str(user_id), str(item_id))
            for user_id in range(1, 6)
            for item_id in (10, 20, 30)
        ],
        columns=["user_id", "item_id"],
    )
    bottleneck["outer_split"] = "design"
    constrained = capacity_aware_edge_split(
        bottleneck,
        ranking_config["warm_catalogue"],
        ranking_config["nested_interaction_split"],
    )
    assert constrained.loc[constrained["evaluable_user"], "user_id"].nunique() == 3
    assert constrained.loc[
        constrained["role"].eq("training")
    ].groupby("item_id").size().min() == 3
    for _, user_rows in constrained.groupby("user_id"):
        n_validation = user_rows["role"].eq("validation").sum()
        n_test = user_rows["role"].eq("test").sum()
        assert (n_validation, n_test) in {(0, 0), (1, 1)}


def test_edge_split_rejects_duplicate_edges(ranking_config):
    edges = pd.DataFrame(
        {
            "user_id": ["1", "1"],
            "item_id": ["10", "10"],
            "outer_split": ["design", "design"],
        }
    )
    with pytest.raises(ValueError, match="unique ownership edges"):
        capacity_aware_edge_split(
            edges,
            ranking_config["warm_catalogue"],
            ranking_config["nested_interaction_split"],
        )


def test_edge_split_accepts_nonempty_zero_evaluable_cohort(ranking_config):
    edges = pd.DataFrame(
        {
            "user_id": [1, 2, 3],
            "item_id": [10, 10, 10],
            "outer_split": ["design", "design", "design"],
        }
    )
    result = capacity_aware_edge_split(
        edges,
        ranking_config["warm_catalogue"],
        ranking_config["nested_interaction_split"],
    )
    assert not result["evaluable_user"].any()
    assert result["role"].eq(EXCLUDED_NONWARM_ITEM).all()
    assert result["user_split_status"].eq(
        "insufficient_warm_history"
    ).all()


def test_evaluation_sample_is_exact_proportional_and_permutation_invariant(
    ranking_config,
):
    users = pd.DataFrame(
        {
            "user_id": [str(value) for value in range(1, 10001)],
            "activity_band": [0] * 6000 + [1] * 3000 + [2] * 1000,
            "split": ["design"] * 10000,
            "evaluable_user": [True] * 10000,
        }
    )
    spec = ranking_config["evaluation_users"]
    first = proportional_evaluation_user_sample(users, spec)
    second = proportional_evaluation_user_sample(
        users.sample(frac=1.0, random_state=1618).reset_index(drop=True), spec
    )
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 5000
    assert first["user_id"].is_unique
    assert first.groupby("activity_band").size().to_dict() == {
        0: 3000,
        1: 1500,
        2: 500,
    }


def test_evaluation_sample_reconciles_fractional_quotas_deterministically():
    users = pd.DataFrame(
        {
            "user_id": [str(value) for value in range(1, 11)],
            "activity_band": [0] * 3 + [1] * 3 + [2] * 4,
            "split": ["design"] * 10,
            "evaluable_user": [True] * 10,
        }
    )
    spec = {"namespace": "unit:evaluation", "sample_size": 5}
    result = proportional_evaluation_user_sample(users, spec)
    assert result.groupby("activity_band").size().to_dict() == {0: 2, 1: 1, 2: 2}


def test_pseudo_cold_selection_is_exact_stratified_and_permutation_invariant(
    ranking_config,
):
    rows = []
    item_id = 10000
    genre_counts = {"Action": 60, "Indie": 36, "Strategy": 24}
    bands = ranking_config["pseudo_cold"]["support_bands"]
    for band_index, (lower, upper) in enumerate(bands):
        width = upper - lower
        for genre, count in genre_counts.items():
            for offset in range(count):
                rows.append(
                    (
                        str(item_id),
                        lower + (offset % width),
                        genre,
                    )
                )
                item_id += 1
        rows.append((str(item_id), lower, None))
        item_id += 1
    items = pd.DataFrame(
        rows,
        columns=["item_id", "design_training_support", "primary_genre"],
    )
    spec = ranking_config["pseudo_cold"]

    first = select_pseudo_cold_items(items, spec)
    second = select_pseudo_cold_items(
        items.sample(frac=1.0, random_state=5772).reset_index(drop=True), spec
    )
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 300
    assert first["item_id"].is_unique
    assert first["primary_genre"].notna().all()
    for band_index, (lower, upper) in enumerate(bands):
        band_rows = first.loc[first["support_band_index"].eq(band_index)]
        assert len(band_rows) == 100
        assert band_rows["design_training_support"].between(
            lower, upper, inclusive="left"
        ).all()
        assert band_rows.groupby("primary_genre").size().to_dict() == {
            "Action": 50,
            "Indie": 30,
            "Strategy": 20,
        }


def test_pseudo_cold_selection_fails_on_preregistered_band_shortfall(
    ranking_config,
):
    items = pd.DataFrame(
        {
            "item_id": [str(value) for value in range(1, 100)],
            "design_training_support": [5] * 99,
            "primary_genre": ["Action"] * 99,
        }
    )
    with pytest.raises(ValueError, match="fewer than the frozen 100"):
        select_pseudo_cold_items(items, ranking_config["pseudo_cold"])


def test_primary_genre_rule_is_normalized_order_independent_and_missing_safe():
    assert (
        primary_genre_from_csv(" RPG, Action, Design &amp; Illustration ")
        == "Action"
    )
    assert primary_genre_from_csv("Action, RPG, Action") == "Action"
    assert primary_genre_from_csv("RPG, Action") == primary_genre_from_csv(
        "Action, RPG"
    )
    assert primary_genre_from_csv(None) is None
    assert primary_genre_from_csv(" , ") is None


def test_split_helpers_reject_assessment_leakage(ranking_config):
    edges = pd.DataFrame(
        {
            "user_id": [1, 2, 3],
            "item_id": [10, 10, 10],
            "outer_split": ["design", "assessment", "design"],
        }
    )
    with pytest.raises(ValueError, match="design users only"):
        capacity_aware_edge_split(
            edges,
            ranking_config["warm_catalogue"],
            ranking_config["nested_interaction_split"],
        )

    users = pd.DataFrame(
        {
            "user_id": np.arange(1, 5001, dtype=np.int64),
            "activity_band": np.zeros(5000, dtype=np.int64),
            "split": ["assessment"] * 5000,
            "evaluable_user": [True] * 5000,
        }
    )
    with pytest.raises(ValueError, match="restricted to design"):
        proportional_evaluation_user_sample(
            users,
            ranking_config["evaluation_users"],
        )
    users["split"] = "design"
    users["evaluable_user"] = False
    with pytest.raises(ValueError, match="restricted to evaluable"):
        proportional_evaluation_user_sample(
            users,
            ranking_config["evaluation_users"],
        )


def test_split_helpers_reject_nullable_access_labels(ranking_config):
    edges = pd.DataFrame(
        {
            "user_id": [1, 2],
            "item_id": [10, 20],
            "outer_split": pd.Series(["design", pd.NA], dtype="string"),
        }
    )
    with pytest.raises(ValueError, match="design users only"):
        capacity_aware_edge_split(
            edges,
            ranking_config["warm_catalogue"],
            ranking_config["nested_interaction_split"],
        )

    users = pd.DataFrame(
        {
            "user_id": np.arange(1, 5001, dtype=np.int64),
            "activity_band": np.zeros(5000, dtype=np.int64),
            "split": pd.Series(["design"] * 4999 + [pd.NA], dtype="string"),
            "evaluable_user": pd.Series(
                [True] * 5000,
                dtype="boolean",
            ),
        }
    )
    with pytest.raises(ValueError, match="restricted to design"):
        proportional_evaluation_user_sample(
            users,
            ranking_config["evaluation_users"],
        )
    users["split"] = pd.Series(["design"] * 5000, dtype="string")
    users["evaluable_user"] = pd.Series(
        [True] * 4999 + [pd.NA],
        dtype="boolean",
    )
    with pytest.raises(ValueError, match="restricted to evaluable"):
        proportional_evaluation_user_sample(
            users,
            ranking_config["evaluation_users"],
        )


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("user_id", str(2**63)),
        ("raw_ownership_count", str(2**63)),
        ("raw_ownership_count", True),
    ],
)
def test_outer_split_rejects_out_of_contract_ids_and_counts(
    ranking_config,
    column,
    value,
):
    users = pd.DataFrame(
        {
            "user_id": pd.Series(["1"], dtype=object),
            "raw_ownership_count": pd.Series([5], dtype=object),
        }
    )
    users.loc[0, column] = value
    with pytest.raises(ValueError):
        activity_band_outer_split(users, ranking_config["outer_user_split"])


def test_empty_design_edge_input_has_declared_schema(ranking_config):
    result = capacity_aware_edge_split(
        pd.DataFrame(columns=["user_id", "item_id", "outer_split"]),
        ranking_config["warm_catalogue"],
        ranking_config["nested_interaction_split"],
    )
    assert result.empty
    assert result.columns.tolist() == [
        "user_id",
        "item_id",
        "role",
        "is_warm_item",
        "evaluable_user",
        "user_split_status",
        "item_support_before",
        "item_training_support_after",
    ]


def test_integrated_design_split_removes_all_preference_and_confidence_leakage(
    ranking_config,
):
    design_users = np.arange(1, 7, dtype=np.int64)
    assessment_users = np.arange(101, 107, dtype=np.int64)
    design_rows = [
        (user_id, item_id, "design")
        for user_id in design_users
        for item_id in (10, 20, 30)
    ]
    assessment_rows = [
        (user_id, 90, "assessment") for user_id in assessment_users
    ]
    all_rows = design_rows + assessment_rows
    edges = pd.DataFrame(
        all_rows,
        columns=["user_id", "item_id", "outer_split"],
    )
    nested = capacity_aware_edge_split(
        edges.loc[edges["outer_split"].eq("design")],
        ranking_config["warm_catalogue"],
        ranking_config["nested_interaction_split"],
    )

    assert 90 not in set(nested["item_id"])
    assert set(nested["role"]) == {"training", "validation", "test"}
    assert (
        len(nested)
        == nested["role"].eq("training").sum()
        + nested["role"].eq("validation").sum()
        + nested["role"].eq("test").sum()
    )
    status_by_user = nested.groupby("user_id")["user_split_status"].first()
    assert set(status_by_user) <= {"evaluable", "capacity_exhausted"}

    canonical = canonicalize_edges(
        [row[0] for row in design_rows],
        [row[1] for row in design_rows],
        np.arange(1, len(design_rows) + 1, dtype=np.float64),
        np.zeros(len(design_rows), dtype=np.float64),
    )
    design_data = build_sparse_interactions(
        canonical,
        user_ids=design_users,
        item_ids=[10, 20, 30],
    )
    heldout = nested.loc[nested["role"].isin(["validation", "test"])]
    training = remove_observed_pairs(
        design_data,
        heldout["user_id"].to_numpy(),
        heldout["item_id"].to_numpy(),
    )
    assert isinstance(training, SparseInteractionData)
    assert design_data.ownership.nnz == len(design_rows)
    assert training.ownership.nnz == nested["role"].eq("training").sum()
    for user_id, item_id in heldout[["user_id", "item_id"]].itertuples(
        index=False,
        name=None,
    ):
        row = int(np.searchsorted(training.user_ids, user_id))
        col = int(np.searchsorted(training.item_ids, item_id))
        assert float(training.ownership[row, col]) == 0.0
        assert float(training.playtime_forever[row, col]) == 0.0
        assert float(training.playtime_2weeks[row, col]) == 0.0
    confidence = observed_confidence(
        training.ownership,
        training.playtime_forever,
        alpha_o=20,
        alpha_p=2,
        tau=1,
    )
    assert confidence.nnz == training.ownership.nnz


def _build_synthetic_split_state(
    ranking_config,
    tmp_path,
):
    ranking = json.loads(json.dumps(ranking_config))
    ranking["outer_user_split"].update(
        {
            "minimum_raw_ownership_count": 3,
            "activity_bands": [3],
        }
    )
    ranking["warm_catalogue"].update(
        {
            "minimum_design_user_support_before_holdout": 2,
            "minimum_design_training_support_after_holdout": 1,
        }
    )
    ranking["evaluation_users"]["sample_size"] = 4
    ranking["pseudo_cold"].update(
        {
            "support_bands": [[1, 10]],
            "items_per_band": 1,
        }
    )

    users = np.arange(1, 11, dtype=np.int64)
    items = np.asarray([10, 20, 30], dtype=np.int64)
    edge_users = np.repeat(users, items.size)
    edge_items = np.tile(items, users.size)
    canonical = canonicalize_edges(
        edge_users,
        edge_items,
        np.arange(1, edge_users.size + 1, dtype=np.float64),
        np.zeros(edge_users.size, dtype=np.float64),
    )
    data = build_sparse_interactions(
        canonical,
        user_ids=users,
        item_ids=items,
    )
    feature_path = tmp_path / "features.csv"
    pd.DataFrame(
        {
            "item_id": [30, 10, 20],
            "genres": ["RPG", "Action, RPG", "Strategy"],
        }
    ).to_csv(feature_path, index=False)

    state = build_split_state(
        ranking,
        data,
        feature_path=feature_path,
    )
    return ranking, state


def test_split_state_builder_keeps_assessment_and_test_targets_sealed(
    ranking_config,
    tmp_path,
):
    _, state = _build_synthetic_split_state(ranking_config, tmp_path)

    outer = state.arrays["outer_user_split"]
    assessment_ids = set(outer["user_ids"][outer["split_code"] == 2])
    design_ids = set(outer["user_ids"][outer["split_code"] == 1])
    nested = state.arrays["nested_interaction_split"]
    assert assessment_ids
    assert design_ids
    assert set(nested["user_ids"]).isdisjoint(assessment_ids)
    assert set(state.training.user_ids) == design_ids
    assert state.summary["nested_interactions"]["training_edges"] == (
        state.training.ownership.nnz
    )
    assert state.arrays["validation_targets"]["user_ids"].size == (
        state.summary["nested_interactions"]["evaluable_users"]
    )
    assert set(state.arrays["validation_targets"]) == {
        "user_ids",
        "item_ids",
    }
    assert set(state.arrays["validation_target_diagnostics"]) == {
        "playtime_forever",
        "playtime_2weeks",
    }
    assert state.arrays["design_test_targets"]["user_ids"].size == (
        state.summary["nested_interactions"]["evaluable_users"]
    )
    np.testing.assert_array_equal(
        state.arrays["validation_other_holdout_mask"]["user_ids"],
        state.arrays["design_test_targets"]["user_ids"],
    )
    np.testing.assert_array_equal(
        state.arrays["validation_other_holdout_mask"]["item_ids"],
        state.arrays["design_test_targets"]["item_ids"],
    )
    assert state.arrays["evaluation_user_sample"]["user_ids"].size == 4
    assert state.arrays["pseudo_cold_items"]["item_ids"].size == 1


def _save_synthetic_split_publication(
    ranking_config,
    tmp_path,
):
    ranking, state = _build_synthetic_split_state(
        ranking_config,
        tmp_path,
    )
    output_dir = (
        tmp_path
        / "outputs"
        / "modeling"
        / "protected"
        / split_artifacts.EXPECTED_CYCLE_ID
        / "stage1_splits"
    )
    artifacts = split_artifacts._save_artifacts(
        state,
        root=tmp_path,
        output_dir=output_dir,
    )
    manifest = split_artifacts._add_manifest_ids(
        {
            "schema_version": 1,
            "artifact": "frozen_stage1_outer_and_nested_splits",
            "cycle_id": split_artifacts.EXPECTED_CYCLE_ID,
            "protocol_id": split_artifacts.EXPECTED_PROTOCOL_ID,
            "interaction_set_id": "synthetic-interaction-set",
            "contract": split_artifacts._contract(ranking),
            "inputs": {},
            "summary": dict(state.summary),
            "artifact_semantics": split_artifacts._artifact_identity(
                state
            ),
            "training_semantics": split_artifacts._training_semantics(
                state.training
            ),
            "artifacts": artifacts,
            "access_boundary": split_artifacts._access_boundary(),
            "provenance": {"fixture": "synthetic"},
        }
    )
    manifest_path = (
        tmp_path / "outputs" / "modeling" / "stage1_split_manifest.json"
    )
    write_manifest(manifest, manifest_path)
    return state, output_dir, manifest_path, manifest


def test_scoped_loaders_verify_artifacts_and_apply_validation_mask(
    ranking_config,
    tmp_path,
    monkeypatch,
):
    state, output_dir, manifest_path, manifest = (
        _save_synthetic_split_publication(ranking_config, tmp_path)
    )
    split_artifacts._validate_public_manifest_redaction(manifest)
    leaky_manifest = json.loads(json.dumps(manifest))
    leaky_manifest["summary"]["user_ids"] = [1, 2]
    with pytest.raises(ValueError, match="identifier fields"):
        split_artifacts._validate_public_manifest_redaction(
            leaky_manifest
        )
    assert manifest["contract"]["artifact_schema"]["outer_user_split"][
        "split_code_label_to_value"
    ] == {
        EXCLUDED_LOW_ACTIVITY: 0,
        "design": 1,
        "assessment": 2,
    }
    assert manifest["contract"]["artifact_schema"][
        "nested_interaction_split"
    ]["nonwarm_training_support_int32_sentinel"] == -1
    assert (
        manifest["access_boundary"]["assessment_item_histories_saved"]
        is False
    )
    assert (
        manifest["access_boundary"][
            "assessment_activity_count_saved_in_audit_only_split"
        ]
        is True
    )

    before = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    opened_npz: list[Path] = []
    original_read_npz = split_artifacts._read_npz

    def recording_read_npz(path):
        opened_npz.append(Path(path))
        return original_read_npz(path)

    monkeypatch.setattr(
        split_artifacts,
        "_read_npz",
        recording_read_npz,
    )
    targets = split_artifacts.load_validation_targets(
        project_root=tmp_path,
        manifest_path=manifest_path,
    )
    assert set(targets) == {"user_ids", "item_ids"}
    assert opened_npz == [
        output_dir
        / split_artifacts.NPZ_RELATIVE_PATHS["validation_targets"]
    ]

    training = split_artifacts.load_design_training_artifacts(
        project_root=tmp_path,
        manifest_path=manifest_path,
    )
    assert split_artifacts._training_semantics(training) == (
        manifest["training_semantics"]
    )

    users = targets["user_ids"]
    items = training.item_ids
    candidate_mask = np.ones((users.size, items.size), dtype=bool)
    for row, user_id in enumerate(users):
        training_row = int(np.searchsorted(training.user_ids, user_id))
        candidate_mask[row, training.ownership[training_row].indices] = False
    for user_id, item_id in zip(
        targets["user_ids"],
        targets["item_ids"],
    ):
        row = int(np.flatnonzero(users == user_id)[0])
        column = int(np.searchsorted(items, item_id))
        assert candidate_mask[row, column]

    opened_npz.clear()
    masked = split_artifacts.mask_validation_other_holdouts(
        candidate_mask,
        users,
        items,
        project_root=tmp_path,
        manifest_path=manifest_path,
    )
    assert opened_npz == [
        output_dir
        / split_artifacts.NPZ_RELATIVE_PATHS[
            "validation_other_holdout_mask"
        ]
    ]
    for user_id, item_id in zip(
        targets["user_ids"],
        targets["item_ids"],
    ):
        row = int(np.flatnonzero(users == user_id)[0])
        column = int(np.searchsorted(items, item_id))
        assert masked[row, column]
    sealed_test = state.arrays["design_test_targets"]
    for user_id, item_id in zip(
        sealed_test["user_ids"],
        sealed_test["item_ids"],
    ):
        row = int(np.flatnonzero(users == user_id)[0])
        column = int(np.searchsorted(items, item_id))
        assert not masked[row, column]

    opened_npz.clear()
    sample = split_artifacts.load_evaluation_user_sample(
        project_root=tmp_path,
        manifest_path=manifest_path,
    )
    assert sample["user_ids"].size == 4
    assert opened_npz == [
        output_dir
        / split_artifacts.NPZ_RELATIVE_PATHS[
            "evaluation_user_sample"
        ]
    ]
    after = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_validation_mask_rejects_unknown_users_and_incomplete_catalogue(
    ranking_config,
    tmp_path,
):
    _, _, manifest_path, _ = _save_synthetic_split_publication(
        ranking_config,
        tmp_path,
    )
    targets = split_artifacts.load_validation_targets(
        project_root=tmp_path,
        manifest_path=manifest_path,
    )
    training = split_artifacts.load_design_training_artifacts(
        project_root=tmp_path,
        manifest_path=manifest_path,
    )
    items = training.item_ids
    with pytest.raises(ValueError, match="every validation batch user"):
        split_artifacts.mask_validation_other_holdouts(
            np.ones((1, items.size), dtype=bool),
            [np.iinfo(np.int64).max],
            items,
            project_root=tmp_path,
            manifest_path=manifest_path,
        )

    users = targets["user_ids"][:1]
    incomplete_items = items[:-1]
    with pytest.raises(ValueError, match="frozen warm item map"):
        split_artifacts.mask_validation_other_holdouts(
            np.ones((1, incomplete_items.size), dtype=bool),
            users,
            incomplete_items,
            project_root=tmp_path,
            manifest_path=manifest_path,
        )


def test_scoped_loaders_reject_npz_and_sparse_corruption(
    ranking_config,
    tmp_path,
):
    _, _, manifest_path, manifest = _save_synthetic_split_publication(
        ranking_config,
        tmp_path,
    )
    validation_path = (
        tmp_path / manifest["artifacts"]["validation_targets"]["path"]
    )
    validation_bytes = validation_path.read_bytes()
    validation_path.write_bytes(validation_bytes + b"x")
    with pytest.raises(ValueError, match="size changed"):
        split_artifacts.load_validation_targets(
            project_root=tmp_path,
            manifest_path=manifest_path,
        )
    validation_path.write_bytes(validation_bytes)

    ownership_path = (
        tmp_path
        / manifest["artifacts"]["design_training_ownership"]["path"]
    )
    ownership_bytes = ownership_path.read_bytes()
    ownership_path.write_bytes(ownership_bytes + b"x")
    with pytest.raises(ValueError, match="size changed"):
        split_artifacts.load_design_training_artifacts(
            project_root=tmp_path,
            manifest_path=manifest_path,
        )


def test_generation_refuses_partial_or_nonidentical_publication(
    tmp_path,
):
    common = {
        "ranking_config_path": ROOT
        / "configs"
        / "ranking_evaluation.json",
        "protocol_manifest_path": ROOT
        / "outputs"
        / "modeling"
        / "stage1_protocol_manifest.json",
        "interaction_manifest_path": ROOT
        / "outputs"
        / "modeling"
        / "stage1_interaction_manifest.json",
    }

    partial_root = tmp_path / "partial"
    partial_output = (
        partial_root
        / "outputs"
        / "modeling"
        / "protected"
        / split_artifacts.EXPECTED_CYCLE_ID
        / "stage1_splits"
    )
    partial_output.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="partial"):
        split_artifacts.generate_split_artifacts(
            project_root=partial_root,
            output_dir=partial_output,
            manifest_path=partial_root
            / "outputs"
            / "modeling"
            / "stage1_split_manifest.json",
            **common,
        )

    frozen_root = tmp_path / "frozen"
    frozen_output = (
        frozen_root
        / "outputs"
        / "modeling"
        / "protected"
        / split_artifacts.EXPECTED_CYCLE_ID
        / "stage1_splits"
    )
    frozen_output.mkdir(parents=True)
    frozen_manifest = (
        frozen_root
        / "outputs"
        / "modeling"
        / "stage1_split_manifest.json"
    )
    frozen_manifest.parent.mkdir(parents=True, exist_ok=True)
    frozen_manifest.write_text("{}", encoding="utf-8")
    before = frozen_manifest.read_bytes()
    with pytest.raises(FileExistsError, match="nonidentical"):
        split_artifacts.generate_split_artifacts(
            project_root=frozen_root,
            output_dir=frozen_output,
            manifest_path=frozen_manifest,
            **common,
        )
    assert frozen_manifest.read_bytes() == before


def test_generation_path_must_be_directly_under_protected_cycle(
    tmp_path,
):
    with pytest.raises(ValueError, match="protected cycle"):
        split_artifacts._validate_generation_paths(
            root=tmp_path,
            output_dir=tmp_path
            / "outputs"
            / "modeling"
            / "protected"
            / "wrong"
            / split_artifacts.EXPECTED_CYCLE_ID
            / "stage1_splits",
            manifest_path=tmp_path
            / "outputs"
            / "modeling"
            / "stage1_split_manifest.json",
            cycle_id=split_artifacts.EXPECTED_CYCLE_ID,
        )


def test_staged_publication_is_no_clobber_and_manifest_last(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "complete.bin").write_bytes(b"complete")
    output = tmp_path / "published"
    temporary_manifest = tmp_path / "manifest.tmp"
    temporary_manifest.write_bytes(b"manifest")
    destination = tmp_path / "manifest.json"

    split_artifacts._publish_staged_publication(
        staging_dir=staging,
        output_dir=output,
        temporary_manifest=temporary_manifest,
        manifest_path=destination,
    )
    assert not staging.exists()
    assert (output / "complete.bin").read_bytes() == b"complete"
    assert destination.read_bytes() == b"manifest"
    assert not temporary_manifest.exists()

    next_staging = tmp_path / "next-staging"
    next_staging.mkdir()
    next_temporary_manifest = tmp_path / "next-manifest.tmp"
    next_temporary_manifest.write_bytes(b"next")
    with pytest.raises(FileExistsError, match="overwrite"):
        split_artifacts._publish_staged_publication(
            staging_dir=next_staging,
            output_dir=output,
            temporary_manifest=next_temporary_manifest,
            manifest_path=destination,
        )
    assert next_staging.exists()
    assert next_temporary_manifest.exists()
    assert destination.read_bytes() == b"manifest"


def test_staged_publication_cleans_directory_if_manifest_publish_fails(
    tmp_path,
    monkeypatch,
):
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "complete.bin").write_bytes(b"complete")
    output = tmp_path / "published"
    temporary_manifest = tmp_path / "manifest.tmp"
    temporary_manifest.write_bytes(b"manifest")
    destination = tmp_path / "manifest.json"

    def fail_link(_source, _destination):
        raise OSError("injected no-clobber failure")

    monkeypatch.setattr(split_artifacts.os, "link", fail_link)
    with pytest.raises(OSError, match="injected"):
        split_artifacts._publish_staged_publication(
            staging_dir=staging,
            output_dir=output,
            temporary_manifest=temporary_manifest,
            manifest_path=destination,
        )
    assert not staging.exists()
    assert not output.exists()
    assert not temporary_manifest.exists()
    assert not destination.exists()
