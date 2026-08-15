"""Deterministic nonnegative score transformations for the Stage 1/2 boundary."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def _scores(values: Any) -> np.ndarray:
    scores = np.asarray(values, dtype=np.float64)
    if scores.ndim == 0 or not np.all(np.isfinite(scores)):
        raise ValueError("scores must be a finite vector or matrix")
    return scores


def fit_global_parameters(values: Any) -> dict[str, float]:
    scores = _scores(values).reshape(-1)
    quantiles = np.quantile(scores, [0.05, 0.25, 0.5, 0.75, 0.95], method="linear")
    return {
        "global_min": float(scores.min()),
        "global_q05": float(quantiles[0]),
        "global_q25": float(quantiles[1]),
        "global_median": float(quantiles[2]),
        "global_q75": float(quantiles[3]),
        "global_q95": float(quantiles[4]),
        "sample_count": int(scores.size),
    }


def global_shift_q90_scale(values: Any, parameters: Mapping[str, Any]) -> np.ndarray:
    scores = _scores(values)
    minimum = float(parameters["global_min"])
    span = max(float(parameters["global_q95"]) - float(parameters["global_q05"]), 1e-6)
    if not np.isfinite(minimum) or not np.isfinite(span):
        raise ValueError("global shift parameters must be finite")
    result = np.clip((scores - minimum) / span, 0.0, 1_000_000.0)
    return _validated_output(result)


def global_robust_softplus(values: Any, parameters: Mapping[str, Any]) -> np.ndarray:
    scores = _scores(values)
    median = float(parameters["global_median"])
    scale = max(float(parameters["global_q75"]) - float(parameters["global_q25"]), 1e-6)
    if not np.isfinite(median) or not np.isfinite(scale):
        raise ValueError("global robust parameters must be finite")
    result = np.log1p(np.exp(np.clip((scores - median) / scale, -40.0, 40.0)))
    return _validated_output(result)


def within_user_midrank_percentile(values: Any) -> np.ndarray:
    scores = _scores(values)
    was_vector = scores.ndim == 1
    rows = scores[None, :] if was_vector else scores
    if rows.ndim != 2 or rows.shape[1] == 0:
        raise ValueError("within-user percentiles require nonempty score rows")
    result = np.empty_like(rows)
    for row, vector in enumerate(rows):
        levels, inverse, counts = np.unique(vector, return_inverse=True, return_counts=True)
        lower = np.concatenate(([0], np.cumsum(counts[:-1])))
        result[row] = (lower[inverse] + 0.5 * counts[inverse]) / vector.size
    result = _validated_output(result)
    return result[0] if was_vector else result


def positive_part_user_standardization(values: Any) -> np.ndarray:
    scores = _scores(values)
    was_vector = scores.ndim == 1
    rows = scores[None, :] if was_vector else scores
    if rows.ndim != 2 or rows.shape[1] == 0:
        raise ValueError("user standardization requires nonempty score rows")
    mean = rows.mean(axis=1, keepdims=True, dtype=np.float64)
    scale = np.maximum(rows.std(axis=1, keepdims=True, dtype=np.float64), 1e-6)
    result = np.clip(np.maximum(0.0, (rows - mean) / scale), 0.0, 100.0)
    result = _validated_output(result)
    return result[0] if was_vector else result


def transform(
    scenario_id: str, values: Any, parameters: Mapping[str, Any] | None = None
) -> np.ndarray:
    supplied = {} if parameters is None else parameters
    if scenario_id == "global_shift_q90_scale":
        return global_shift_q90_scale(values, supplied)
    if scenario_id == "global_robust_softplus":
        return global_robust_softplus(values, supplied)
    if scenario_id == "within_user_midrank_percentile":
        return within_user_midrank_percentile(values)
    if scenario_id == "positive_part_user_standardization":
        return positive_part_user_standardization(values)
    raise ValueError(f"unknown pseudo-utility scenario: {scenario_id}")


def _validated_output(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(result)) or np.any(result < 0.0):
        raise FloatingPointError("pseudo-utilities must be finite and nonnegative")
    return result
