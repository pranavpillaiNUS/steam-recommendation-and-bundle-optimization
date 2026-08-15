"""Tests for the preregistered Stage 1 protocol and stable hash rules."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from src import stage1_protocol as protocol


EXPECTED_PROTOCOL_ID = (
    "00b18d784ee34196e34a90c354fb03f45fa025082039eb7a90cc662b23a22f6f"
)
LEGACY_INTERACTION_TABLE = (
    Path(__file__).resolve().parents[1] / "outputs" / "tables" / "user_items_df.csv"
)


def _configs():
    models = protocol.load_json(protocol.DEFAULT_MODEL_CONFIG)
    ranking = protocol.load_json(protocol.DEFAULT_RANKING_CONFIG)
    return models, ranking


def test_repository_configs_validate_and_have_frozen_grid_sizes():
    models, ranking = _configs()
    protocol.validate_protocol_configs(models, ranking)

    als = protocol.enumerate_als_configurations(models)
    bpr = protocol.enumerate_bpr_configurations(models)
    assert len(als) == 24
    assert len(bpr) == 4
    assert len({row["configuration_id"] for row in als}) == 24
    assert len({row["configuration_id"] for row in bpr}) == 4
    assert models["training_seeds"] == [104729, 130363, 155921]


def test_semantic_hash_ignores_json_key_order_and_whitespace():
    value = {"z": [3, 2, 1], "a": {"y": False, "x": 1.25}}
    reordered = json.loads(json.dumps(value, indent=4, sort_keys=False))
    reordered = {"a": reordered["a"], "z": reordered["z"]}
    assert protocol.semantic_sha256(value) == protocol.semantic_sha256(reordered)


def test_stable_hash_partition_is_permutation_invariant_and_exact_by_stratum():
    ids = [101, 102, 103, 104, 105, 201, 202, 203, 204, 205]
    strata = [0] * 5 + [1] * 5
    expected = protocol.stratified_hash_partition(
        ids,
        strata,
        assessment_fraction=0.2,
        namespace="test:outer",
    )
    reversed_result = protocol.stratified_hash_partition(
        list(reversed(ids)),
        list(reversed(strata)),
        assessment_fraction=0.2,
        namespace="test:outer",
    )
    assert expected == reversed_result
    assert sum(value == "assessment" for value in expected.values()) == 2
    assert sum(expected[identifier] == "assessment" for identifier in ids[:5]) == 1
    assert sum(expected[identifier] == "assessment" for identifier in ids[5:]) == 1


def test_hash_fields_are_unambiguous_and_namespace_separated():
    assert protocol.stable_hash_uint64("x", 1, 23) != protocol.stable_hash_uint64(
        "x", 12, 3
    )
    assert protocol.stable_hash_uint64("x", 1) != protocol.stable_hash_uint64(
        "y", 1
    )
    assert protocol.stable_hash_uint64("x", 1) == protocol.stable_hash_uint64(
        "x", 1
    )


def test_validation_rejects_protected_outcome_and_ladder_changes():
    models, ranking = _configs()
    bad_ranking = copy.deepcopy(ranking)
    bad_ranking["protected_outcomes"]["stage2_objectives_available"] = True
    with pytest.raises(ValueError, match="Stage 2 objectives"):
        protocol.validate_protocol_configs(models, bad_ranking)

    bad_models = copy.deepcopy(models)
    bad_models["model_ladder"].append("tags")
    with pytest.raises(ValueError, match="four-rung"):
        protocol.validate_protocol_configs(bad_models, ranking)


def test_manifest_is_deterministic_without_reading_input_files():
    models, ranking = _configs()
    first = protocol.build_protocol_manifest(models, ranking, verify_inputs=False)
    second = protocol.build_protocol_manifest(
        json.loads(json.dumps(models, sort_keys=True)),
        json.loads(json.dumps(ranking, sort_keys=True)),
        verify_inputs=False,
    )
    assert first == second
    assert first["protected_outcomes"]["design_test_status"].startswith("sealed")
    assert first["protected_outcomes"]["assessment_user_status"].startswith(
        "sealed"
    )


@pytest.mark.skipif(
    not LEGACY_INTERACTION_TABLE.exists(),
    reason="frozen v1 input verification requires the ignored legacy interaction table",
)
def test_repository_manifest_reproduces_frozen_identity_and_inputs():
    models, ranking = _configs()
    rebuilt = protocol.build_protocol_manifest(models, ranking, verify_inputs=True)
    saved = protocol.load_json(protocol.DEFAULT_MANIFEST)

    assert rebuilt["protocol_id"] == EXPECTED_PROTOCOL_ID
    assert saved["protocol_id"] == EXPECTED_PROTOCOL_ID
    for field in (
        "configs",
        "grid",
        "inputs",
        "protected_outcomes",
        "first_permitted_next_action",
    ):
        assert saved[field] == rebuilt[field]
