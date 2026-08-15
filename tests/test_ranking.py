import json
from pathlib import Path

import numpy as np
import pytest

from src import ranking


def test_defaults_match_frozen_ranking_configuration():
    config_path = Path(__file__).parents[1] / "configs" / "ranking_evaluation.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    assert ranking.DEFAULT_KS == tuple(config["ranking"]["ks"])
    assert (
        ranking.DEFAULT_MAX_SCORE_BLOCK_BYTES
        == config["ranking"]["maximum_score_block_bytes"]
    )
    assert (
        ranking.DEFAULT_BOOTSTRAP_REPLICATES
        == config["uncertainty"]["paired_user_bootstrap_replicates"]
    )
    assert ranking.DEFAULT_BOOTSTRAP_SEED == config["uncertainty"]["bootstrap_seed"]
    assert ranking.DEFAULT_CONFIDENCE_LEVEL == config["uncertainty"]["confidence_level"]


def test_expected_tie_metrics_cross_topk_boundary():
    result = ranking.expected_tie_metrics(8, 4, ks=(10, 20))

    assert result.recall[10] == pytest.approx(0.5)
    expected_at_10 = (1 / np.log2(10) + 1 / np.log2(11)) / 4
    assert result.ndcg[10] == pytest.approx(expected_at_10)
    assert result.recall[20] == pytest.approx(1.0)
    expected_at_20 = sum(1 / np.log2(r + 1) for r in range(9, 13)) / 4
    assert result.ndcg[20] == pytest.approx(expected_at_20)
    assert result.expected_rank == pytest.approx(10.5)
    assert result.as_flat_dict()["ndcg_at_20"] == pytest.approx(expected_at_20)


def test_expected_tie_metrics_all_tied_and_below_cutoff():
    all_tied = ranking.expected_tie_metrics(0, 100, ks=(10,))
    assert all_tied.recall[10] == pytest.approx(0.1)
    assert all_tied.ndcg[10] == pytest.approx(
        sum(1 / np.log2(r + 1) for r in range(1, 11)) / 100
    )

    below = ranking.expected_tie_metrics(20, 3, ks=(10,))
    assert below.recall[10] == 0.0
    assert below.ndcg[10] == 0.0


@pytest.mark.parametrize(
    "g,e",
    [(-1, 1), (0, 0), (1.5, 2), (1, 2.5)],
)
def test_expected_tie_metrics_reject_invalid_counts(g, e):
    with pytest.raises(ValueError):
        ranking.expected_tie_metrics(g, e)


def test_target_tie_counts_use_exact_equality_and_candidate_mask():
    just_above_two = np.nextafter(2.0, np.inf)
    scores = np.array([3.0, just_above_two, 2.0, 2.0, 1.0])

    assert ranking.target_tie_counts(scores, 2) == (2, 2)
    mask = np.array([False, True, True, True, True])
    assert ranking.target_tie_counts(scores, 2, mask) == (1, 2)

    mask[2] = False
    with pytest.raises(ValueError, match="target"):
        ranking.target_tie_counts(scores, 2, mask)


def test_streamed_tie_counter_matches_direct_evaluation():
    scores = np.array([4.0, 3.0, 3.0, 2.0, 1.0, 3.0])
    direct = ranking.evaluate_target_scores(scores, target_index=1, ks=(2, 4))

    counter = ranking.TargetTieCounter(target_score=3.0)
    counter.update(scores[:2], target_offset=1)
    counter.update(scores[2:5], candidate_mask=[True, True, False])
    counter.update(scores[5:])
    streamed = counter.metrics(ks=(2, 4))

    # Direct evaluation must use the same exclusion as the streamed version.
    mask = np.array([True, True, True, True, False, True])
    direct = ranking.evaluate_target_scores(scores, 1, mask, ks=(2, 4))
    assert streamed == direct


def test_streamed_tie_counter_requires_the_actual_target_coordinate():
    counter = ranking.TargetTieCounter(target_score=2.0)
    counter.update([3.0, 2.0])
    with pytest.raises(ValueError, match="not identified"):
        counter.metrics()

    counter = ranking.TargetTieCounter(target_score=2.0)
    with pytest.raises(ValueError, match="differs"):
        counter.update([3.0, 1.0], target_offset=1)


def test_ranking_indices_and_masks_reject_silent_numeric_coercion():
    with pytest.raises(ValueError, match="integers"):
        ranking.candidate_masks_from_exclusions(4, [[1.5]])
    with pytest.raises(ValueError, match="integers"):
        ranking.evaluate_score_block(np.ones((1, 4)), [1.5])
    with pytest.raises(ValueError, match="booleans"):
        ranking.evaluate_target_scores([3.0, 2.0], 0, [1, 1])


def test_candidate_masks_enforce_target_and_masked_copy_is_nonmutating():
    exclusions = [np.array([0, 2]), np.array([1])]
    masks = ranking.candidate_masks_from_exclusions(
        5, exclusions, required_targets=np.array([4, 3])
    )
    assert masks.tolist() == [
        [False, True, False, True, True],
        [True, False, True, True, True],
    ]

    scores = np.arange(10.0).reshape(2, 5)
    masked = ranking.masked_score_copy(scores, masks)
    assert np.isneginf(masked[0, 0])
    assert scores[0, 0] == 0.0

    with pytest.raises(ValueError, match="target"):
        ranking.candidate_masks_from_exclusions(
            5, exclusions, required_targets=np.array([2, 3])
        )


def test_topk_fractional_inclusion_at_tied_boundary():
    scores = np.array([5.0, 4.0, 4.0, 4.0, 1.0])
    probabilities = ranking.topk_inclusion_probabilities(scores, k=3)

    assert probabilities == pytest.approx([1.0, 2 / 3, 2 / 3, 2 / 3, 0.0])
    assert probabilities.sum() == pytest.approx(3.0)

    all_tied = ranking.topk_inclusion_probabilities(np.ones(4), k=2)
    assert all_tied == pytest.approx(np.full(4, 0.5))


def test_streamed_topk_boundary_and_exclusions_match_direct_result():
    scores = np.array([5.0, 4.0, 4.0, 4.0, 1.0, 6.0])
    mask = np.array([True, True, False, True, True, True])

    accumulator = ranking.TopKBoundaryAccumulator(k=3)
    accumulator.update(scores[:2], mask[:2])
    accumulator.update(scores[2:4], mask[2:4])
    accumulator.update(scores[4:], mask[4:])
    boundary = accumulator.boundary()

    assert boundary.threshold == 4.0
    assert boundary.strictly_above == 2
    assert boundary.tied_block_size == 2
    assert boundary.boundary_inclusion_probability == pytest.approx(0.5)
    streamed = ranking.inclusion_probabilities_at_boundary(scores, boundary, mask)
    direct = ranking.topk_inclusion_probabilities(scores, 3, mask)
    assert streamed == pytest.approx(direct)
    assert direct.sum() == pytest.approx(3.0)


def test_topk_when_k_exceeds_candidate_count_includes_every_candidate():
    scores = np.array([3.0, 2.0, 1.0, 0.0])
    mask = np.array([True, False, True, False])
    probabilities = ranking.topk_inclusion_probabilities(scores, k=10, candidate_mask=mask)
    assert probabilities.tolist() == [1.0, 0.0, 1.0, 0.0]


def test_bounded_score_block_evaluation_matches_row_oracles():
    scores = np.array(
        [
            [5.0, 4.0, 4.0, 4.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ]
    )
    targets = np.array([1, 2])
    masks = ranking.candidate_masks_from_exclusions(
        5, [np.array([4]), np.array([0])], required_targets=targets
    )
    result = ranking.evaluate_score_block(scores, targets, masks, ks=(2, 4))

    for row in range(scores.shape[0]):
        oracle = ranking.evaluate_target_scores(
            scores[row], targets[row], masks[row], ks=(2, 4)
        )
        assert result["strictly_above"][row] == oracle.strictly_above
        assert result["tied_block_size"][row] == oracle.tied_block_size
        assert result["expected_rank"][row] == pytest.approx(oracle.expected_rank)
        assert result["recall_at_2"][row] == pytest.approx(oracle.recall[2])
        assert result["ndcg_at_4"][row] == pytest.approx(oracle.ndcg[4])

    with pytest.raises(MemoryError):
        ranking.evaluate_score_block(
            scores,
            targets,
            masks,
            maximum_score_block_bytes=scores.nbytes + masks.nbytes - 1,
        )


def test_nonfinite_candidate_scores_fail_but_excluded_values_are_ignored():
    scores = np.array([2.0, np.nan, 1.0])
    with pytest.raises(ValueError, match="finite"):
        ranking.evaluate_target_scores(scores, 0)

    mask = np.array([True, False, True])
    result = ranking.evaluate_target_scores(scores, 0, mask, ks=(1,))
    assert result.recall[1] == 1.0


def test_expected_coverage_and_top_item_concentration():
    first = np.array([1.0, 0.5, 0.5, 0.0])
    second = np.array([0.0, 1.0, 0.0, 1.0])
    coverage = ranking.ExpectedCoverageAccumulator(n_items=4)
    coverage.update(first)
    coverage.update(second)

    assert coverage.n_users == 2
    assert coverage.expected_item_count == pytest.approx(3.5)
    assert coverage.expected_fraction == pytest.approx(0.875)

    block = np.vstack([first, second])
    top_items = np.array([True, True, False, False])
    assert ranking.expected_top_item_concentration(block, top_items) == pytest.approx(
        2.5 / 4.0
    )


def test_coverage_sparse_updates_reject_duplicate_item_indices():
    coverage = ranking.ExpectedCoverageAccumulator(n_items=5)
    coverage.update([1.0, 0.5], item_indices=[0, 3])
    assert coverage.expected_item_count == pytest.approx(1.5)

    with pytest.raises(ValueError, match="duplicates"):
        coverage.update([0.5, 0.5], item_indices=[1, 1])


def test_paired_bootstrap_constant_difference_and_determinism():
    reference = np.array([0.1, 0.4, 0.2, 0.8, 0.3])
    candidate = reference + 0.05
    constant = ranking.paired_bootstrap_mean_difference(
        candidate, reference, replicates=101, seed=314159
    )
    assert constant.mean_difference == pytest.approx(0.05)
    assert constant.lower == pytest.approx(0.05)
    assert constant.upper == pytest.approx(0.05)
    assert constant.bootstrap_standard_error == pytest.approx(0.0, abs=1e-16)

    variable_candidate = np.array([0.2, 0.1, 0.6, 0.4, 0.9])
    first = ranking.paired_bootstrap_mean_difference(
        variable_candidate,
        reference,
        replicates=257,
        seed=11,
        maximum_index_bytes=5 * 8,
    )
    second = ranking.paired_bootstrap_mean_difference(
        variable_candidate,
        reference,
        replicates=257,
        seed=11,
        maximum_index_bytes=5 * 8 * 100,
    )
    assert first == second


@pytest.mark.parametrize(
    "candidate,reference",
    [([], []), ([1.0], [1.0, 2.0]), ([np.nan], [0.0])],
)
def test_paired_bootstrap_rejects_invalid_pairs(candidate, reference):
    with pytest.raises(ValueError):
        ranking.paired_bootstrap_mean_difference(candidate, reference, replicates=10)
