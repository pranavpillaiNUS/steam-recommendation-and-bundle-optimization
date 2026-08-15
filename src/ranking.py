"""Exact, bounded ranking metrics for Stage 1 latent preference scores.

The primary evaluation ranks one held-out ownership edge against the complete
eligible catalogue.  Scores are latent ranking scores, not utilities, purchase
probabilities, willingness to pay, or monetary quantities.

The frozen tie convention is uniform random ordering within an *exactly*
score-tied block.  No numerical tolerance is used to merge distinct scores.
This module never constructs or saves a full user-by-catalogue score matrix;
its largest primitive is a caller-bounded user score block.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


DEFAULT_KS = (10, 20)
DEFAULT_MAX_SCORE_BLOCK_BYTES = 67_108_864
DEFAULT_BOOTSTRAP_REPLICATES = 2_000
DEFAULT_BOOTSTRAP_SEED = 314_159
DEFAULT_CONFIDENCE_LEVEL = 0.95


def _strict_integer(value: object, *, name: str, positive: bool = False) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if (positive and result <= 0) or (not positive and result < 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be a {qualifier} integer")
    return result


def _strict_integer_array(values: object, *, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.size == 0:
        return np.asarray(raw, dtype=np.int64)
    if raw.dtype.kind not in "iu" or raw.dtype.kind == "b":
        raise ValueError(f"{name} must contain integers")
    return np.asarray(raw, dtype=np.int64)


def _validated_ks(ks: Sequence[int]) -> tuple[int, ...]:
    raw = tuple(ks)
    result = tuple(_strict_integer(k, name="k", positive=True) for k in raw)
    if not result:
        raise ValueError("ks must contain at least one positive integer")
    if len(set(result)) != len(result):
        raise ValueError("ks must not contain duplicates")
    return result


def _validated_candidate_mask(
    scores: np.ndarray, candidate_mask: np.ndarray | None
) -> np.ndarray:
    if candidate_mask is None:
        mask = np.ones(scores.shape, dtype=bool)
    else:
        mask = np.asarray(candidate_mask)
        if mask.dtype.kind != "b":
            raise ValueError("candidate_mask must contain booleans")
        if mask.shape != scores.shape:
            raise ValueError("candidate_mask must have the same shape as scores")
    if not np.all(np.isfinite(scores[mask])):
        raise ValueError("every candidate score must be finite")
    return mask


@dataclass(frozen=True)
class ExpectedTieMetrics:
    """Expected metrics for one target under uniform ordering in its tie block."""

    strictly_above: int
    tied_block_size: int
    expected_rank: float
    recall: dict[int, float]
    ndcg: dict[int, float]

    def as_flat_dict(self) -> dict[str, float | int]:
        result: dict[str, float | int] = {
            "strictly_above": self.strictly_above,
            "tied_block_size": self.tied_block_size,
            "expected_rank": self.expected_rank,
        }
        for k, value in self.recall.items():
            result[f"recall_at_{k}"] = value
        for k, value in self.ndcg.items():
            result[f"ndcg_at_{k}"] = value
        return result


def expected_tie_metrics(
    strictly_above: int,
    tied_block_size: int,
    ks: Sequence[int] = DEFAULT_KS,
) -> ExpectedTieMetrics:
    """Return exact expected Recall/NDCG values for a single held-out target.

    ``strictly_above`` is the number of eligible candidates with a score
    strictly greater than the target score. ``tied_block_size`` includes the
    target itself. Ranks are one-based in the NDCG discount.
    """

    g = int(strictly_above)
    e = int(tied_block_size)
    if g != strictly_above or g < 0:
        raise ValueError("strictly_above must be a nonnegative integer")
    if e != tied_block_size or e < 1:
        raise ValueError("tied_block_size must be a positive integer")
    checked_ks = _validated_ks(ks)

    recall: dict[int, float] = {}
    ndcg: dict[int, float] = {}
    for k in checked_ks:
        positions_in_k = min(max(k - g, 0), e)
        recall[k] = positions_in_k / e
        if positions_in_k == 0:
            ndcg[k] = 0.0
        else:
            ranks = np.arange(g + 1, g + positions_in_k + 1, dtype=float)
            ndcg[k] = float(np.sum(1.0 / np.log2(ranks + 1.0)) / e)

    return ExpectedTieMetrics(
        strictly_above=g,
        tied_block_size=e,
        expected_rank=float(g + (e + 1) / 2.0),
        recall=recall,
        ndcg=ndcg,
    )


def target_tie_counts(
    scores: Sequence[float] | np.ndarray,
    target_index: int,
    candidate_mask: Sequence[bool] | np.ndarray | None = None,
) -> tuple[int, int]:
    """Count candidates strictly above and exactly tied with one target."""

    values = np.asarray(scores, dtype=float)
    if values.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    target = _strict_integer(target_index, name="target_index")
    if target >= values.size:
        raise IndexError("target_index is outside scores")
    mask = _validated_candidate_mask(
        values, None if candidate_mask is None else np.asarray(candidate_mask)
    )
    if not mask[target]:
        raise ValueError("the held-out target must remain an eligible candidate")
    target_score = values[target]
    if not np.isfinite(target_score):
        raise ValueError("the held-out target score must be finite")

    candidates = values[mask]
    return int(np.count_nonzero(candidates > target_score)), int(
        np.count_nonzero(candidates == target_score)
    )


def evaluate_target_scores(
    scores: Sequence[float] | np.ndarray,
    target_index: int,
    candidate_mask: Sequence[bool] | np.ndarray | None = None,
    ks: Sequence[int] = DEFAULT_KS,
) -> ExpectedTieMetrics:
    """Evaluate one target against every candidate in a score vector."""

    g, e = target_tie_counts(scores, target_index, candidate_mask)
    return expected_tie_metrics(g, e, ks=ks)


class TargetTieCounter:
    """Bounded item-block counter for a known finite target score.

    Every eligible catalogue item, including the target, must be passed exactly
    once across calls to :meth:`update`. The class stores only two counts.
    """

    def __init__(self, target_score: float) -> None:
        self.target_score = float(target_score)
        if not np.isfinite(self.target_score):
            raise ValueError("target_score must be finite")
        self.strictly_above = 0
        self.tied_block_size = 0
        self.candidate_count = 0
        self._target_seen = False

    def update(
        self,
        score_block: Sequence[float] | np.ndarray,
        candidate_mask: Sequence[bool] | np.ndarray | None = None,
        *,
        target_offset: int | None = None,
    ) -> None:
        values = np.asarray(score_block, dtype=float)
        if values.ndim != 1:
            raise ValueError("score_block must be one-dimensional")
        mask = _validated_candidate_mask(
            values, None if candidate_mask is None else np.asarray(candidate_mask)
        )
        candidates = values[mask]
        if target_offset is not None:
            offset = _strict_integer(target_offset, name="target_offset")
            if offset >= values.size:
                raise IndexError("target_offset is outside score_block")
            if self._target_seen:
                raise ValueError("the held-out target was streamed more than once")
            if not mask[offset]:
                raise ValueError("the held-out target must be an eligible candidate")
            if values[offset] != self.target_score:
                raise ValueError("the streamed target score differs from target_score")
            self._target_seen = True
        self.strictly_above += int(np.count_nonzero(candidates > self.target_score))
        self.tied_block_size += int(np.count_nonzero(candidates == self.target_score))
        self.candidate_count += int(candidates.size)

    def metrics(self, ks: Sequence[int] = DEFAULT_KS) -> ExpectedTieMetrics:
        if self.candidate_count == 0:
            raise ValueError("no candidates were supplied")
        if not self._target_seen:
            raise ValueError("the held-out target was not identified in the stream")
        if self.tied_block_size == 0:
            raise ValueError(
                "the streamed catalogue did not contain the held-out target score"
            )
        return expected_tie_metrics(
            self.strictly_above, self.tied_block_size, ks=ks
        )


def candidate_masks_from_exclusions(
    n_items: int,
    excluded_by_user: Sequence[Sequence[int] | np.ndarray],
    required_targets: Sequence[int] | np.ndarray | None = None,
) -> np.ndarray:
    """Build per-user candidate masks without modifying model scores.

    Training positives and the non-target holdout belong in
    ``excluded_by_user``. If provided, each required target is checked rather
    than silently restored after an erroneous exclusion.
    """

    item_count = _strict_integer(n_items, name="n_items", positive=True)
    n_users = len(excluded_by_user)
    masks = np.ones((n_users, item_count), dtype=bool)
    for row, excluded in enumerate(excluded_by_user):
        indices = _strict_integer_array(excluded, name="exclusions")
        if indices.ndim != 1:
            raise ValueError("each exclusion list must be one-dimensional")
        if np.any((indices < 0) | (indices >= item_count)):
            raise IndexError("an excluded item index is outside the catalogue")
        masks[row, indices] = False

    if required_targets is not None:
        targets = _strict_integer_array(required_targets, name="required_targets")
        if targets.shape != (n_users,):
            raise ValueError("required_targets must contain one target per user")
        if np.any((targets < 0) | (targets >= item_count)):
            raise IndexError("a required target index is outside the catalogue")
        if not np.all(masks[np.arange(n_users), targets]):
            raise ValueError("a held-out target was included in the exclusions")
    return masks


def masked_score_copy(
    scores: Sequence[float] | np.ndarray,
    candidate_mask: Sequence[bool] | np.ndarray,
    fill_value: float = -np.inf,
) -> np.ndarray:
    """Return a display/selection copy with excluded entries replaced.

    Metric functions should still receive the candidate mask explicitly; they
    reject nonfinite values among eligible candidates.
    """

    values = np.asarray(scores, dtype=float)
    mask = np.asarray(candidate_mask, dtype=bool)
    if values.shape != mask.shape:
        raise ValueError("candidate_mask must have the same shape as scores")
    result = values.copy()
    result[~mask] = fill_value
    return result


@dataclass(frozen=True)
class TopKBoundary:
    """Exact score boundary and tied-block inclusion probability for top-K."""

    requested_k: int
    selected_count: int
    candidate_count: int
    threshold: float
    strictly_above: int
    tied_block_size: int
    boundary_inclusion_probability: float


class TopKBoundaryAccumulator:
    """Find a top-K boundary while retaining only O(K) levels between blocks."""

    def __init__(self, k: int) -> None:
        self.k = _strict_integer(k, name="k", positive=True)
        self.candidate_count = 0
        self._top_level_counts: dict[float, int] = {}

    def update(
        self,
        score_block: Sequence[float] | np.ndarray,
        candidate_mask: Sequence[bool] | np.ndarray | None = None,
    ) -> None:
        values = np.asarray(score_block, dtype=float)
        if values.ndim != 1:
            raise ValueError("score_block must be one-dimensional")
        mask = _validated_candidate_mask(
            values, None if candidate_mask is None else np.asarray(candidate_mask)
        )
        candidates = values[mask]
        if candidates.size == 0:
            return
        levels, counts = np.unique(candidates, return_counts=True)
        for level, count in zip(levels, counts):
            key = float(level)
            self._top_level_counts[key] = self._top_level_counts.get(key, 0) + int(
                count
            )
        self.candidate_count += int(candidates.size)

        # The Kth score can only move upward as more candidates arrive. Once a
        # score level is below the current boundary, it can never be needed.
        cumulative = 0
        retained: dict[float, int] = {}
        for level in sorted(self._top_level_counts, reverse=True):
            count = self._top_level_counts[level]
            retained[level] = count
            cumulative += count
            if cumulative >= self.k:
                break
        self._top_level_counts = retained

    def boundary(self) -> TopKBoundary:
        if self.candidate_count == 0:
            raise ValueError("no candidates were supplied")
        selected = min(self.k, self.candidate_count)
        cumulative = 0
        for level in sorted(self._top_level_counts, reverse=True):
            count = self._top_level_counts[level]
            if cumulative + count >= selected:
                probability = (selected - cumulative) / count
                return TopKBoundary(
                    requested_k=self.k,
                    selected_count=selected,
                    candidate_count=self.candidate_count,
                    threshold=float(level),
                    strictly_above=cumulative,
                    tied_block_size=count,
                    boundary_inclusion_probability=float(probability),
                )
            cumulative += count
        raise RuntimeError("top-K boundary bookkeeping is inconsistent")


def inclusion_probabilities_at_boundary(
    scores: Sequence[float] | np.ndarray,
    boundary: TopKBoundary,
    candidate_mask: Sequence[bool] | np.ndarray | None = None,
) -> np.ndarray:
    """Return exact top-K inclusion probabilities for one score block."""

    values = np.asarray(scores, dtype=float)
    if values.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    mask = _validated_candidate_mask(
        values, None if candidate_mask is None else np.asarray(candidate_mask)
    )
    probabilities = np.zeros(values.shape, dtype=float)
    probabilities[mask & (values > boundary.threshold)] = 1.0
    probabilities[mask & (values == boundary.threshold)] = (
        boundary.boundary_inclusion_probability
    )
    return probabilities


def topk_inclusion_probabilities(
    scores: Sequence[float] | np.ndarray,
    k: int,
    candidate_mask: Sequence[bool] | np.ndarray | None = None,
) -> np.ndarray:
    """Return per-item expected inclusion under exact random tie ordering."""

    values = np.asarray(scores, dtype=float)
    if values.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    accumulator = TopKBoundaryAccumulator(k)
    accumulator.update(values, candidate_mask)
    return inclusion_probabilities_at_boundary(
        values, accumulator.boundary(), candidate_mask
    )


def evaluate_score_block(
    score_block: np.ndarray,
    target_indices: Sequence[int] | np.ndarray,
    candidate_masks: np.ndarray | None = None,
    ks: Sequence[int] = DEFAULT_KS,
    maximum_score_block_bytes: int = DEFAULT_MAX_SCORE_BLOCK_BYTES,
) -> dict[str, np.ndarray]:
    """Evaluate a bounded user-by-catalogue block without retaining scores.

    The returned arrays contain one row per user. The caller owns score
    generation and must pass the complete frozen catalogue in its established
    item order.
    """

    scores = np.asarray(score_block)
    if scores.ndim != 2:
        raise ValueError("score_block must be two-dimensional")
    byte_limit = int(maximum_score_block_bytes)
    if byte_limit != maximum_score_block_bytes or byte_limit <= 0:
        raise ValueError("maximum_score_block_bytes must be a positive integer")
    auxiliary_bytes = 0 if candidate_masks is None else np.asarray(candidate_masks).nbytes
    if scores.nbytes + auxiliary_bytes > byte_limit:
        raise MemoryError("score block and candidate mask exceed the frozen byte budget")

    targets = _strict_integer_array(target_indices, name="target_indices")
    if targets.shape != (scores.shape[0],):
        raise ValueError("target_indices must contain one target per score row")
    if np.any((targets < 0) | (targets >= scores.shape[1])):
        raise IndexError("a target index is outside the score block")
    if candidate_masks is None:
        masks = None
    else:
        masks = np.asarray(candidate_masks, dtype=bool)
        if masks.shape != scores.shape:
            raise ValueError("candidate_masks must match score_block")

    checked_ks = _validated_ks(ks)
    rows = [
        evaluate_target_scores(
            scores[row],
            int(targets[row]),
            None if masks is None else masks[row],
            checked_ks,
        )
        for row in range(scores.shape[0])
    ]
    result: dict[str, np.ndarray] = {
        "strictly_above": np.asarray([x.strictly_above for x in rows], dtype=np.int64),
        "tied_block_size": np.asarray(
            [x.tied_block_size for x in rows], dtype=np.int64
        ),
        "expected_rank": np.asarray([x.expected_rank for x in rows], dtype=float),
    }
    for k in checked_ks:
        result[f"recall_at_{k}"] = np.asarray([x.recall[k] for x in rows])
        result[f"ndcg_at_{k}"] = np.asarray([x.ndcg[k] for x in rows])
    return result


class ExpectedCoverageAccumulator:
    """Accumulate expected distinct top-K catalogue coverage across users."""

    def __init__(self, n_items: int) -> None:
        self.n_items = _strict_integer(n_items, name="n_items", positive=True)
        self._not_included_probability = np.ones(self.n_items, dtype=float)
        self.n_users = 0

    def update(
        self,
        inclusion_probabilities: Sequence[float] | np.ndarray,
        item_indices: Sequence[int] | np.ndarray | None = None,
    ) -> None:
        probabilities = np.asarray(inclusion_probabilities, dtype=float)
        if probabilities.ndim != 1:
            raise ValueError("inclusion_probabilities must be one-dimensional")
        if np.any(~np.isfinite(probabilities)) or np.any(
            (probabilities < 0.0) | (probabilities > 1.0)
        ):
            raise ValueError("inclusion probabilities must be finite and in [0, 1]")
        if item_indices is None:
            if probabilities.size != self.n_items:
                raise ValueError("a full probability row must have n_items entries")
            indices = np.arange(self.n_items)
        else:
            indices = _strict_integer_array(item_indices, name="item_indices")
            if indices.shape != probabilities.shape:
                raise ValueError("item_indices and probabilities must have equal shape")
            if np.any((indices < 0) | (indices >= self.n_items)):
                raise IndexError("an item index is outside the catalogue")
            if np.unique(indices).size != indices.size:
                raise ValueError("item_indices must not contain duplicates within one user")
        self._not_included_probability[indices] *= 1.0 - probabilities
        self.n_users += 1

    @property
    def expected_item_count(self) -> float:
        return float(np.sum(1.0 - self._not_included_probability))

    @property
    def expected_fraction(self) -> float:
        return self.expected_item_count / self.n_items


def expected_top_item_concentration(
    inclusion_probabilities: Sequence[float] | np.ndarray,
    top_item_mask: Sequence[bool] | np.ndarray,
) -> float:
    """Expected share of recommendation exposure assigned to a fixed item set."""

    probabilities = np.asarray(inclusion_probabilities, dtype=float)
    if probabilities.ndim not in (1, 2):
        raise ValueError("probabilities must be a vector or user-by-item block")
    top_mask = np.asarray(top_item_mask, dtype=bool)
    if probabilities.ndim == 1:
        if top_mask.shape != probabilities.shape:
            raise ValueError("top_item_mask must have one entry per item")
        selected_exposure = float(np.sum(probabilities[top_mask]))
    else:
        if top_mask.shape == (probabilities.shape[1],):
            selected_exposure = float(np.sum(probabilities[:, top_mask]))
        elif top_mask.shape == probabilities.shape:
            selected_exposure = float(np.sum(probabilities[top_mask]))
        else:
            raise ValueError("top_item_mask must be item-level or match probabilities")
    if np.any(~np.isfinite(probabilities)) or np.any(
        (probabilities < 0.0) | (probabilities > 1.0)
    ):
        raise ValueError("inclusion probabilities must be finite and in [0, 1]")
    exposure = float(np.sum(probabilities))
    if exposure <= 0.0:
        raise ValueError("total expected exposure must be positive")
    return selected_exposure / exposure


@dataclass(frozen=True)
class PairedBootstrapResult:
    """Percentile interval for a paired user-level mean difference."""

    mean_difference: float
    lower: float
    upper: float
    confidence_level: float
    replicates: int
    seed: int
    bootstrap_standard_error: float


def paired_bootstrap_mean_difference(
    candidate_metric: Sequence[float] | np.ndarray,
    reference_metric: Sequence[float] | np.ndarray,
    *,
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    maximum_index_bytes: int = DEFAULT_MAX_SCORE_BLOCK_BYTES,
) -> PairedBootstrapResult:
    """Compute a deterministic percentile interval from paired user rows.

    Resampling is over paired users. Training-seed variation is deliberately
    outside this interval and must be reported separately.
    """

    candidate = np.asarray(candidate_metric, dtype=float)
    reference = np.asarray(reference_metric, dtype=float)
    if candidate.ndim != 1 or reference.ndim != 1 or candidate.shape != reference.shape:
        raise ValueError("candidate and reference metrics must be equal-length vectors")
    if candidate.size == 0:
        raise ValueError("paired metrics must not be empty")
    if not np.all(np.isfinite(candidate)) or not np.all(np.isfinite(reference)):
        raise ValueError("paired metrics must be finite")
    n_replicates = int(replicates)
    if n_replicates != replicates or n_replicates <= 0:
        raise ValueError("replicates must be a positive integer")
    rng_seed = int(seed)
    if rng_seed != seed or rng_seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    level = float(confidence_level)
    if not 0.0 < level < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")
    byte_budget = int(maximum_index_bytes)
    if byte_budget != maximum_index_bytes or byte_budget < np.dtype(np.int64).itemsize:
        raise ValueError("maximum_index_bytes is too small")

    differences = candidate - reference
    rng = np.random.default_rng(rng_seed)
    bootstrap_means = np.empty(n_replicates, dtype=float)
    bytes_per_resample = differences.size * np.dtype(np.int64).itemsize
    if bytes_per_resample > byte_budget:
        raise ValueError("maximum_index_bytes cannot hold one paired resample")
    rows_per_chunk = max(1, byte_budget // max(bytes_per_resample, 1))
    for start in range(0, n_replicates, rows_per_chunk):
        stop = min(start + rows_per_chunk, n_replicates)
        indices = rng.integers(
            0, differences.size, size=(stop - start, differences.size), dtype=np.int64
        )
        bootstrap_means[start:stop] = differences[indices].mean(axis=1)

    alpha = (1.0 - level) / 2.0
    lower, upper = np.quantile(bootstrap_means, [alpha, 1.0 - alpha])
    standard_error = (
        float(np.std(bootstrap_means, ddof=1)) if n_replicates > 1 else 0.0
    )
    return PairedBootstrapResult(
        mean_difference=float(np.mean(differences)),
        lower=float(lower),
        upper=float(upper),
        confidence_level=level,
        replicates=n_replicates,
        seed=rng_seed,
        bootstrap_standard_error=standard_error,
    )
