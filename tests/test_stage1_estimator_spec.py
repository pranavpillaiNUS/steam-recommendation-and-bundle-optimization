"""Tests for the public S1.4 estimator-specification freeze."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.preference_model import (
    DEFAULT_JITTER_SEQUENCE,
    PLAYTIME_CONFIDENCE_SOURCE,
    estimator_specification,
)
from src.stage1_estimator_spec import (
    DEFAULT_OUTPUT,
    EXPECTED_CYCLE_ID,
    build_estimator_spec_manifest,
    generate_estimator_spec_manifest,
    verify_estimator_spec_manifest,
)
from src.stage1_interaction_artifacts import EXPECTED_PROTOCOL_ID


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_reference_contract_matches_frozen_numerical_and_feature_conventions():
    specification = estimator_specification()
    assert PLAYTIME_CONFIDENCE_SOURCE == "playtime_forever"
    assert specification["confidence"]["playtime_source"] == (
        "playtime_forever"
    )
    assert specification["wrmf"]["jitter_sequence"] == list(
        DEFAULT_JITTER_SEQUENCE
    )
    assert specification["wrmf"]["accumulator_dtype"] == "float64"
    assert specification["wrmf"]["stored_factor_dtype"] == "float32"
    assert specification["feature_sum_bpr"][
        "inactive_content_excluded_from_penalty"
    ] is True
    assert specification["serialization"][
        "dense_user_item_score_matrix"
    ] is False

    source = (
        PROJECT_ROOT / "src" / "preference_model.py"
    ).read_text(encoding="utf-8")
    assert "linalg.inv" not in source


def test_manifest_binds_every_completed_upstream_and_defers_backends():
    manifest = build_estimator_spec_manifest()
    assert manifest["cycle_id"] == EXPECTED_CYCLE_ID
    assert manifest["protocol_id"] == EXPECTED_PROTOCOL_ID
    assert len(manifest["interaction_set_id"]) == 64
    assert len(manifest["split_set_id"]) == 64
    assert len(manifest["feature_set_id"]) == 64
    assert len(manifest["specification_id"]) == 64
    assert manifest["conventions"]["implicit_als"]["playtime_field"] == (
        "playtime_forever"
    )
    sampler = manifest["conventions"]["pairwise_feature_sum"]["sampler"]
    assert sampler["positive"] == (
        "uniform_training_edges_with_replacement"
    )
    assert sampler["full_catalogue_user"] == "fail_before_sampling"
    assert sampler["epoch_policy"] == (
        "one_continuing_stream_not_reset_per_epoch"
    )
    assert manifest["backend_status"]["implicit"]["equivalence"] == (
        "not_tested_until_s1_5"
    )
    assert manifest["backend_status"]["lightfm"]["equivalence"] == (
        "not_tested_until_s1_5"
    )
    assert set(manifest["parameter_schemas"]["popularity"]) == {
        "item_counts"
    }
    assert set(
        manifest["parameter_schemas"]["feature_sum_bpr_identity_genre"]
    ) == {
        "user_factors",
        "identity_factors",
        "feature_factors",
        "item_bias",
    }
    assert manifest["access_boundary"]["real_model_fit"] is False
    assert manifest["access_boundary"]["design_test_targets"] == (
        "sealed_not_accessed"
    )
    assert manifest["access_boundary"]["pseudo_cold_items"] == (
        "reserved_for_s1_8_not_accessed"
    )


def test_generation_is_exact_checkable_and_refuses_nonidentical_overwrite(
    tmp_path,
):
    output = tmp_path / "stage1_estimator_spec_manifest.json"
    generated = generate_estimator_spec_manifest(output_path=output)
    assert verify_estimator_spec_manifest(output_path=output) == generated
    assert generate_estimator_spec_manifest(output_path=output) == generated

    changed = json.loads(output.read_text(encoding="utf-8"))
    changed["backend_status"]["implicit"]["equivalence"] = "changed"
    output.write_text(
        json.dumps(changed, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(FileExistsError, match="nonidentical"):
        generate_estimator_spec_manifest(output_path=output)


@pytest.mark.skipif(
    not DEFAULT_OUTPUT.exists(),
    reason="frozen S1.4 estimator specification is not generated yet",
)
def test_frozen_estimator_specification_exactly_regenerates():
    manifest = verify_estimator_spec_manifest()
    assert manifest["access_boundary"]["real_model_fit"] is False
