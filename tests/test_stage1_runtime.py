"""Successor runtime regressions against independent dense ranking equations."""

import hashlib
import json

import numpy as np
import pytest

from src.pseudo_utility import fit_global_parameters, transform
from src.ranking import evaluate_score_block, topk_inclusion_probabilities
from src.stage1_runtime import (
    FactorScoreSource,
    evaluate_factor_ranking,
    fit_global_parameters_disk,
    iter_pseudo_utility_rows,
    load_bound_parameter_archive,
    map_ids_sha256,
    save_bound_parameter_archive,
    validated_fold_in_triples,
)


def _problem(**kwargs):
    users = np.asarray([[1, 0], [0, 1], [1, 1]], dtype=np.float32)
    items = np.asarray([[2, 0], [1, 1], [1, 1], [0, 2], [-1, -1], [0, 0], [3, -2]], dtype=np.float32)
    bias = np.asarray([0, 1, 1, 0, 0, -1, 0], dtype=np.float64)
    return FactorScoreSource(users, items, bias, **kwargs)


@pytest.mark.parametrize("user_block,item_block,byte_limit", [(1, 1, 9), (2, 2, 36), (7, 20, 1000), (3, 20, 54)])
def test_two_axis_ranking_matches_dense_exact_ties_and_coverage(user_block, item_block, byte_limit):
    source = _problem(user_block_size=user_block, item_block_size=item_block, maximum_score_block_bytes=byte_limit)
    users = np.asarray([2, 0, 1])
    targets = np.asarray([1, 3, 5])
    exclusions = [[4, 6], [0, 5], [0, 1, 2, 3, 4, 6]]
    scores = source.user_factors[users].astype(np.float64) @ source.item_factors.astype(np.float64).T + source.item_bias
    mask = np.ones(scores.shape, dtype=bool)
    for row, indices in enumerate(exclusions):
        mask[row, indices] = False
    ks = (1, 3, 20)
    expected = evaluate_score_block(scores, targets, mask, ks=ks)
    probabilities = np.vstack([topk_inclusion_probabilities(row, 20, m) for row, m in zip(scores, mask)])
    top_mask = np.asarray([True, False, False, True, False, False, False])
    result = evaluate_factor_ranking(source, users, targets, exclusions, ks=ks, top_item_mask=top_mask)
    for name, values in expected.items():
        np.testing.assert_array_equal(result.metrics[name], values)
    assert result.aggregate["expected_catalogue_coverage_at_20"] == pytest.approx(np.mean(1 - np.prod(1 - probabilities, axis=0)))
    assert result.aggregate["expected_top_item_concentration_at_20"] == pytest.approx(probabilities[:, top_mask].sum() / probabilities.sum())
    assert result.aggregate["evaluation_users"] == 3
    assert result.resource_contract["score_passes"] == 2
    for _, rows in source.user_batches(users):
        for _, _, values in source.item_blocks(rows):
            assert values.size * 9 <= byte_limit


def test_fractional_boundary_coverage_includes_ties_across_item_blocks():
    source = FactorScoreSource(np.ones((2, 1)), np.ones((7, 1)), user_block_size=1, item_block_size=2)
    result = evaluate_factor_ranking(source, [0, 1], [0, 6], [[], []], ks=(3,))
    np.testing.assert_array_equal(result.metrics["strictly_above"], [0, 0])
    np.testing.assert_array_equal(result.metrics["tied_block_size"], [7, 7])
    np.testing.assert_allclose(result.metrics["recall_at_3"], [3 / 7, 3 / 7])
    assert result.aggregate["expected_catalogue_coverage_at_3"] == pytest.approx(1 - (1 - 3 / 7) ** 2)


def test_factor_scores_and_exact_ties_are_independent_of_block_shape():
    rng = np.random.default_rng(391)
    users = rng.normal(size=(7, 64)) * np.geomspace(1e-10, 1e10, 64)
    items = rng.normal(size=(19, 64))
    items[17] = items[1]  # Preserve exact ties, even with cancellation.
    references = []
    for user_block, item_block in [(1, 1), (3, 4), (7, 19)]:
        source = FactorScoreSource(users, items, user_block_size=user_block, item_block_size=item_block)
        dense = np.empty((7, 19))
        for offset, rows in source.user_batches(np.arange(7)):
            for start, stop, scores in source.item_blocks(rows):
                dense[offset : offset + len(rows), start:stop] = scores
        references.append(dense)
    np.testing.assert_array_equal(references[0], references[1])
    np.testing.assert_array_equal(references[0], references[2])
    np.testing.assert_array_equal(references[0][:, 1], references[0][:, 17])


@pytest.mark.parametrize("kwargs", [dict(maximum_score_block_bytes=8), dict(user_block_size=True), dict(item_block_size=1.5)])
def test_source_rejects_invalid_resource_settings(kwargs):
    with pytest.raises(ValueError):
        _problem(**kwargs)


def test_score_refuses_oversized_request_before_allocating(monkeypatch):
    source = _problem(maximum_score_block_bytes=36, user_block_size=3, item_block_size=7)
    def forbidden(*args, **kwargs):
        raise AssertionError("score multiplication must not be reached")
    monkeypatch.setattr(np, "einsum", forbidden)
    with pytest.raises(MemoryError, match="byte budget"):
        source.score([0, 1], 0, 3)


@pytest.mark.parametrize("users,targets,exclusions,kwargs", [
    ([0], [0], [[0]], {}),
    ([0, 0], [1, 2], [[], []], {}),
    ([0], [1], [[7]], {}),
    ([0], [1], [[False]], {}),
    ([0], [1], [[]], {"ks": (1, 1)}),
    ([0], [1], [[]], {"top_item_mask": np.ones(7)}),
])
def test_ranking_rejects_invalid_candidates(users, targets, exclusions, kwargs):
    with pytest.raises((ValueError, IndexError)):
        evaluate_factor_ranking(_problem(), users, targets, exclusions, **kwargs)


def test_factor_validation_rejects_misalignment_and_nonfinite_inputs():
    with pytest.raises(ValueError, match="same factor count"):
        FactorScoreSource(np.zeros((2, 2)), np.zeros((3, 3)))
    with pytest.raises(ValueError, match="finite"):
        FactorScoreSource(np.zeros((2, 2)), np.asarray([[1.0, np.nan]]))
    with pytest.raises(ValueError, match="item_bias"):
        FactorScoreSource(np.zeros((2, 2)), np.zeros((3, 2)), np.zeros(2))
    source = FactorScoreSource(np.asarray([[1e308]]), np.asarray([[1e308]]))
    with pytest.raises(FloatingPointError):
        source.score([0], 0, 1)


def test_disk_quantiles_and_prepartition_hash_match_frozen_dense_reference(tmp_path):
    source = _problem(user_block_size=2, item_block_size=2, maximum_score_block_bytes=36)
    users = np.asarray([2, 0])
    dense = source.user_factors[users].astype(np.float64) @ source.item_factors.astype(np.float64).T + source.item_bias
    expected = fit_global_parameters(dense)
    fitted = fit_global_parameters_disk(source, users, scratch_directory=tmp_path, maximum_disk_bytes=dense.nbytes)
    assert fitted["parameters"] == expected
    assert fitted["score_sample_sha256"] == hashlib.sha256(np.ascontiguousarray(dense, dtype="<f8").tobytes()).hexdigest()
    assert fitted["resource_contract"]["temporary_disk_bytes"] == dense.nbytes
    assert fitted["score_sample_shape"] == [2, 7]
    assert list(tmp_path.iterdir()) == []


def test_quantiles_refuse_disk_cap_before_creating_scratch(tmp_path):
    scratch = tmp_path / "scratch"
    with pytest.raises(MemoryError, match="maximum_disk_bytes"):
        fit_global_parameters_disk(_problem(), [0, 1], scratch_directory=scratch, maximum_disk_bytes=1)
    assert not scratch.exists()


def test_quantile_failure_removes_temporary_mapped_file_on_windows(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("injected partition failure")
    monkeypatch.setattr(np, "quantile", fail)
    with pytest.raises(RuntimeError, match="injected"):
        fit_global_parameters_disk(_problem(), [0], scratch_directory=tmp_path, maximum_disk_bytes=1000)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("scenario", ["global_shift_q90_scale", "global_robust_softplus", "within_user_midrank_percentile", "positive_part_user_standardization"])
def test_pseudo_utilities_use_complete_catalogue_and_match_frozen_transform(scenario):
    source = _problem(user_block_size=2, item_block_size=2)
    dense = source.user_factors.astype(np.float64) @ source.item_factors.astype(np.float64).T + source.item_bias
    parameters = fit_global_parameters(dense)
    actual = list(iter_pseudo_utility_rows(source, [2, 0], scenario, parameters=parameters, maximum_row_workspace_bytes=7 * 128))
    assert [user for user, _ in actual] == [2, 0]
    np.testing.assert_array_equal(np.vstack([row for _, row in actual]), transform(scenario, dense[[2, 0]], parameters))
    assert all(row.size == source.n_items for _, row in actual)
    if scenario == "within_user_midrank_percentile":
        assert not np.array_equal(actual[0][1][:2], transform(scenario, dense[2, :2]))


def test_pseudo_utility_row_cap_is_checked_before_scoring(monkeypatch):
    source = _problem()
    def forbidden(*args, **kwargs):
        raise AssertionError("scoring must not be reached")
    monkeypatch.setattr(source, "score", forbidden)
    with pytest.raises(MemoryError, match="row workspace"):
        list(iter_pseudo_utility_rows(source, [0], "within_user_midrank_percentile", maximum_row_workspace_bytes=7 * 128 - 1))


def _archive(tmp_path):
    source = _problem()
    arrays = {"user_factors": source.user_factors, "item_factors": source.item_factors}
    users, items = np.asarray([70, 12, 42]), np.asarray([8, 2, 13, 6, 52, 70, 99])
    path = tmp_path / "model.npz"
    metadata = save_bound_parameter_archive(path, arrays, family="implicit_als", user_ids=users, item_ids=items)
    arguments = dict(expected_sha256=metadata["sha256"], expected_family="implicit_als", expected_user_ids=users, expected_item_ids=items)
    return path, arrays, arguments


def test_bound_archive_roundtrip_and_ordered_map_binding(tmp_path):
    path, original, kwargs = _archive(tmp_path)
    loaded = load_bound_parameter_archive(path, **kwargs)
    for name, values in original.items():
        np.testing.assert_array_equal(loaded[name], values)
    assert map_ids_sha256(kwargs["expected_item_ids"]) != map_ids_sha256(kwargs["expected_item_ids"][::-1])
    for key in ("expected_user_ids", "expected_item_ids"):
        mismatched = {**kwargs, key: kwargs[key][::-1]}
        with pytest.raises(ValueError, match="binding mismatch"):
            load_bound_parameter_archive(path, **mismatched)


def test_bound_archive_rejects_wrong_hash_family_and_size_before_loading(tmp_path):
    path, _, kwargs = _archive(tmp_path)
    with pytest.raises(ValueError, match="hash mismatch"):
        load_bound_parameter_archive(path, **{**kwargs, "expected_sha256": "0" * 64})
    with pytest.raises(ValueError, match="family"):
        load_bound_parameter_archive(path, **{**kwargs, "expected_family": "popularity"})
    with pytest.raises(MemoryError, match="uncompressed"):
        load_bound_parameter_archive(path, **kwargs, maximum_uncompressed_bytes=1)


@pytest.mark.parametrize("change", ["extra", "shape", "nan", "integer_factors", "bool_schema", "embedded_map"])
def test_bound_archive_rejects_malformed_hash_valid_payloads(tmp_path, change):
    path, _, kwargs = _archive(tmp_path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    if change == "extra":
        arrays["unknown"] = np.zeros(1)
    elif change == "shape":
        arrays["item_factors"] = arrays["item_factors"][:-1]
    elif change == "nan":
        arrays["item_factors"][0, 0] = np.nan
    elif change == "integer_factors":
        arrays["user_factors"] = arrays["user_factors"].astype(np.int64)
    elif change == "bool_schema":
        meta = json.loads(arrays["metadata"].item())
        meta["schema_version"] = True
        arrays["metadata"] = np.asarray(json.dumps(meta))
    elif change == "embedded_map":
        arrays["item_ids"] = arrays["item_ids"][::-1]
    np.savez_compressed(path, **arrays)
    kwargs["expected_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        load_bound_parameter_archive(path, **kwargs)


@pytest.mark.parametrize("family", ["popularity", "feature_sum_bpr_identity", "feature_sum_bpr_identity_genre"])
def test_bound_archive_validates_supported_non_als_schemas(tmp_path, family):
    arrays = {"item_counts": np.asarray([0, 1, 2])}
    if family != "popularity":
        arrays = {
            "user_factors": np.zeros((2, 4), dtype=np.float32),
            "identity_factors": np.zeros((3, 4), dtype=np.float32),
            "feature_factors": np.zeros((0 if family.endswith("identity") else 2, 4), dtype=np.float32),
            "item_bias": np.zeros(3, dtype=np.float32),
        }
    path = tmp_path / "other.npz"
    meta = save_bound_parameter_archive(path, arrays, family=family, user_ids=[1, 2], item_ids=[3, 4, 5])
    loaded = load_bound_parameter_archive(path, expected_sha256=meta["sha256"], expected_family=family, expected_user_ids=[1, 2], expected_item_ids=[3, 4, 5])
    assert set(loaded) == set(arrays)


@pytest.mark.parametrize("ids", [[1, 1], [True, False], [-1], np.asarray([2**63], dtype=np.uint64)])
def test_id_map_requires_unique_nonnegative_int64_ids(ids):
    with pytest.raises(ValueError):
        map_ids_sha256(ids)


@pytest.mark.parametrize("positives", [[-1, 0, 1, 2], [0, 1, 2, 4], [0, 1, 2, 3, 4]])
def test_fold_in_rejects_bad_bounds_before_all_owned_early_return(positives):
    with pytest.raises((ValueError, IndexError)):
        validated_fold_in_triples(positives, n_items=4, cycle_id="successor", user_id=8)


def test_fold_in_retains_frozen_deterministic_sampling_for_valid_histories():
    from src.stage1_backend import construct_fold_in_triples
    kwargs = dict(n_items=7, cycle_id="successor", user_id=8)
    np.testing.assert_array_equal(validated_fold_in_triples([1, 3], **kwargs), construct_fold_in_triples([1, 3], **kwargs))
    assert validated_fold_in_triples([], **kwargs).shape == (0, 2)
    assert validated_fold_in_triples(np.arange(7), **kwargs).shape == (0, 2)
