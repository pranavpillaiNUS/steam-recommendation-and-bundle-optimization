"""Prospective minibatch BPR; this module does not alter the frozen v2 fit.

The NEW objective is the expected sampled-triple logistic loss plus sampled
parameter regularization. For a batch of N triples (u, i, j), it is

    mean[softplus(-margin) + lambda * (||x_u||^2 + ||eta_i||^2
         + ||eta_j||^2 + b_i^2 + b_j^2)] + lambda * ||G||_F^2.

The G term applies only when content is active. Repeated rows receive their
sampling multiplicity. This is deliberately different from the frozen summed
epoch objective with a single global penalty: its regularization must be tuned
afresh. Only touched user/item rows are updated, so batch work does not grow
with all user parameters. No validation, test, or assessment data is accepted.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import time
from typing import Any

import numpy as np
import scipy.sparse as sp
from scipy.special import expit

from src.preference_model import FeatureSumParameters
from src.stage1_backend import (
    BPRFitResult,
    CycleBPRTripleSampler,
    initialize_feature_sum_parameters,
)


OBJECTIVE = "mean_logistic_plus_mean_sampled_parameter_l2_plus_global_genre_l2"
_FIELDS = ("user_factors", "identity_factors", "feature_factors", "item_bias")


def _integer(value: Any, name: str, *, minimum: int = 1) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _scalar(value: Any, name: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or np.ndim(value) != 0:
        raise ValueError(f"{name} must be a finite scalar")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite scalar") from exc
    if not np.isfinite(result) or (result < 0 if allow_zero else result <= 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")
    return result


def _training_inputs(
    ownership: sp.spmatrix, genre_features: sp.spmatrix
) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    owned = sp.csr_matrix(ownership, dtype=np.float64, copy=True)
    if not all(owned.shape) or not np.all(np.isfinite(owned.data)):
        raise ValueError("ownership must be finite with nonempty dimensions")
    if np.any((owned.data != 0) & (owned.data != 1)):
        raise ValueError("ownership must be binary")
    owned.sum_duplicates()
    owned.eliminate_zeros()
    owned.sort_indices()
    if owned.nnz == 0 or np.any(owned.data != 1):
        raise ValueError("ownership must contain unique binary positive edges")
    if np.any(np.diff(owned.indptr) == owned.shape[1]):
        raise ValueError("every sampled user must have an unowned training item")
    genres = sp.csr_matrix(genre_features, dtype=np.float64, copy=True)
    genres.sum_duplicates()
    genres.eliminate_zeros()
    genres.sort_indices()
    if genres.shape[0] != owned.shape[1]:
        raise ValueError("genre rows must match ownership item columns")
    if not np.all(np.isfinite(genres.data)) or np.any(genres.data < 0):
        raise ValueError("genre features must be finite and nonnegative")
    return owned, genres


@dataclass(frozen=True)
class BPRBatchGradient:
    """Sparse by parameter row, dense within each selected factor row."""

    user_rows: np.ndarray
    item_rows: np.ndarray
    user_factors: np.ndarray
    identity_factors: np.ndarray
    feature_factors: np.ndarray
    item_bias: np.ndarray


def _batch_gradient(
    parameters: FeatureSumParameters,
    genres: sp.csr_matrix,
    triples: np.ndarray,
    regularization: float,
) -> tuple[float, BPRBatchGradient]:
    u, i, j = triples.T
    user_rows, user_inverse, user_counts = np.unique(
        u, return_inverse=True, return_counts=True
    )
    item_rows, item_inverse, item_counts = np.unique(
        np.concatenate((i, j)), return_inverse=True, return_counts=True
    )
    n = len(triples)
    positive_inverse, negative_inverse = item_inverse[:n], item_inverse[n:]
    users = np.asarray(parameters.user_factors[user_rows], dtype=np.float64)
    identity = np.asarray(parameters.identity_factors[item_rows], dtype=np.float64)
    bias = np.asarray(parameters.item_bias[item_rows], dtype=np.float64)
    content = np.asarray(parameters.feature_factors, dtype=np.float64)
    active = content.shape[0] > 0
    vectors = identity + genres[item_rows] @ content if active else identity
    selected_users = users[user_inverse]
    differences = vectors[positive_inverse] - vectors[negative_inverse]
    margin = (
        bias[positive_inverse] - bias[negative_inverse]
        + np.einsum("ij,ij->i", selected_users, differences)
    )
    coefficient = -expit(-margin) / n
    user_gradient = np.zeros_like(users)
    item_gradient = np.zeros_like(identity)
    bias_gradient = np.zeros_like(bias)
    np.add.at(user_gradient, user_inverse, coefficient[:, None] * differences)
    weighted_users = coefficient[:, None] * selected_users
    np.add.at(item_gradient, positive_inverse, weighted_users)
    np.add.at(item_gradient, negative_inverse, -weighted_users)
    np.add.at(bias_gradient, positive_inverse, coefficient)
    np.add.at(bias_gradient, negative_inverse, -coefficient)
    content_gradient = (
        np.asarray((genres[i] - genres[j]).T @ weighted_users)
        if active else np.zeros_like(content)
    )

    user_weights = user_counts / n
    item_weights = item_counts / n
    penalty = regularization * (
        float(np.sum(user_weights[:, None] * users * users))
        + float(np.sum(item_weights[:, None] * identity * identity))
        + float(np.sum(item_weights * bias * bias))
        + float(np.sum(content * content))
    )
    user_gradient += 2 * regularization * user_weights[:, None] * users
    item_gradient += 2 * regularization * item_weights[:, None] * identity
    bias_gradient += 2 * regularization * item_weights * bias
    content_gradient += 2 * regularization * content
    loss = float(np.mean(np.logaddexp(0.0, -margin))) + penalty
    gradients = BPRBatchGradient(
        user_rows, item_rows, user_gradient, item_gradient,
        content_gradient, bias_gradient,
    )
    if not np.isfinite(loss) or any(
        not np.all(np.isfinite(getattr(gradients, name))) for name in _FIELDS
    ):
        raise FloatingPointError("minibatch BPR loss or gradient became nonfinite")
    return loss, gradients


def minibatch_loss_and_gradients(
    parameters: FeatureSumParameters,
    genre_features: sp.spmatrix,
    triples: np.ndarray,
    *,
    regularization: float,
) -> tuple[float, BPRBatchGradient]:
    """Validated equation entry point for numerical checks and small probes.

    Training itself supplies checked sampler output to the same equation. The
    triple argument must be nonempty, integer, in bounds, with i != j. This
    equation alone cannot decide which items are owned; the trainer's sampler
    rejects every training positive when drawing negatives.
    """
    penalty = _scalar(regularization, "regularization", allow_zero=True)
    arrays = {name: np.asarray(getattr(parameters, name)) for name in _FIELDS}
    users, items = arrays["user_factors"], arrays["identity_factors"]
    content, bias = arrays["feature_factors"], arrays["item_bias"]
    if (
        users.ndim != 2 or items.ndim != 2 or content.ndim != 2
        or not all(users.shape) or not all(items.shape)
        or users.shape[1] != items.shape[1] or content.shape[1] != items.shape[1]
        or bias.shape != (items.shape[0],)
    ):
        raise ValueError("invalid BPR parameter shapes")
    if any(not np.all(np.isfinite(value)) for value in arrays.values()):
        raise ValueError("BPR parameters must be finite")
    genres = sp.csr_matrix(genre_features, dtype=np.float64)
    if genres.shape[0] != len(items) or (
        len(content) > 0 and genres.shape[1] != len(content)
    ):
        raise ValueError("genre and parameter shapes differ")
    if not np.all(np.isfinite(genres.data)) or np.any(genres.data < 0):
        raise ValueError("genre features must be finite and nonnegative")
    sample = np.asarray(triples)
    if sample.ndim != 2 or sample.shape[1] != 3 or len(sample) == 0 or sample.dtype.kind not in "iu":
        raise ValueError("triples must be a nonempty integer N by 3 array")
    if np.any(sample < 0) or np.any(sample[:, 0] >= len(users)) or np.any(sample[:, 1:] >= len(items)):
        raise ValueError("triple index is outside parameter bounds")
    if np.any(sample[:, 1] == sample[:, 2]):
        raise ValueError("positive and negative items must differ")
    return _batch_gradient(parameters, genres, sample.astype(np.int64), penalty)


def fit_minibatch_bpr(
    ownership: sp.spmatrix,
    genre_features: sp.spmatrix,
    *,
    cycle_id: str,
    training_seed: int,
    factors: int,
    regularization: float,
    learning_rate: float,
    epochs: int,
    samples_per_epoch: int,
    batch_size: int = 4096,
    include_genre: bool = False,
    diagnostic_sample_size: int = 4096,
    adagrad_epsilon: float = 1e-8,
) -> BPRFitResult:
    """Fit with one AdaGrad update PER BATCH and train-only fixed probes.

    Sampling continues across epochs; the independent fixed probe stream does
    not consume optimization triples. Diagnostics do not choose a checkpoint.
    Future selection must use separately protected validation orchestration.
    """
    if not isinstance(cycle_id, str) or not cycle_id.strip():
        raise ValueError("cycle_id must be a nonempty string")
    if not isinstance(include_genre, (bool, np.bool_)):
        raise ValueError("include_genre must be boolean")
    seed = _integer(training_seed, "training_seed", minimum=0)
    k = _integer(factors, "factors")
    epoch_count = _integer(epochs, "epochs")
    sample_count = _integer(samples_per_epoch, "samples_per_epoch")
    size = _integer(batch_size, "batch_size")
    probe_count = _integer(diagnostic_sample_size, "diagnostic_sample_size")
    penalty = _scalar(regularization, "regularization", allow_zero=True)
    rate = _scalar(learning_rate, "learning_rate")
    epsilon = _scalar(adagrad_epsilon, "adagrad_epsilon")
    owned, genres = _training_inputs(ownership, genre_features)
    content_active = bool(include_genre and genres.nnz)
    parameters = initialize_feature_sum_parameters(
        n_users=owned.shape[0], n_items=owned.shape[1],
        n_features=genres.shape[1] if content_active else 0,
        factors=k, cycle_id=cycle_id, training_seed=seed,
    )
    accumulators = {
        name: np.zeros_like(getattr(parameters, name), dtype=np.float64)
        for name in _FIELDS
    }
    sampler = CycleBPRTripleSampler(owned, cycle_id=cycle_id, training_seed=seed)
    probe = CycleBPRTripleSampler(
        owned, cycle_id=f"{cycle_id}:fixed-training-probe", training_seed=seed
    ).sample(probe_count)
    initial_loss = _batch_gradient(parameters, genres, probe, penalty)[0]
    trace = []
    update_count = 0
    started = time.perf_counter()
    for epoch in range(epoch_count):
        epoch_started = time.perf_counter()
        rejected_before = sampler.rejected_draws
        stream_digest = hashlib.sha256()
        weighted_loss = 0.0
        epoch_updates = 0
        for start in range(0, sample_count, size):
            triples = sampler.sample(min(size, sample_count - start))
            stream_digest.update(np.ascontiguousarray(triples, dtype="<i8").tobytes())
            loss, gradients = _batch_gradient(parameters, genres, triples, penalty)
            weighted_loss += loss * len(triples)
            for name in _FIELDS:
                rows = (
                    gradients.user_rows if name == "user_factors"
                    else slice(None) if name == "feature_factors"
                    else gradients.item_rows
                )
                gradient = getattr(gradients, name)
                current = getattr(parameters, name)
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    accumulated = accumulators[name][rows] + gradient * gradient
                    updated = np.asarray(current[rows], dtype=np.float64) - rate * gradient / (
                        np.sqrt(accumulated) + epsilon
                    )
                    stored = updated.astype(np.float32)
                if not np.all(np.isfinite(stored)):
                    raise FloatingPointError("minibatch BPR parameters became nonfinite")
                accumulators[name][rows] = accumulated
                current[rows] = stored
            update_count += 1
            epoch_updates += 1
        trace.append({
            "epoch": epoch + 1,
            "updates": epoch_updates,
            "cumulative_updates": update_count,
            "accepted_triples": sample_count,
            "rejected_negative_draws": sampler.rejected_draws - rejected_before,
            "triple_stream_sha256": stream_digest.hexdigest(),
            "mean_online_batch_objective": weighted_loss / sample_count,
            "fixed_training_probe_objective": _batch_gradient(parameters, genres, probe, penalty)[0],
            "runtime_seconds": time.perf_counter() - epoch_started,
        })
    return BPRFitResult(parameters=parameters, diagnostics={
        "backend": "prospective_numpy_minibatch_bpr",
        "optimizer": "touched_row_minibatch_adagrad",
        "objective": OBJECTIVE,
        "regularization": penalty,
        "regularization_compatible_with_frozen_v2": False,
        "learning_rate": rate,
        "batch_size": size,
        "samples_per_epoch": sample_count,
        "update_count": update_count,
        "adagrad_epsilon": epsilon,
        "content_active": content_active,
        "training_seed": seed,
        "cycle_id": cycle_id,
        "initial_fixed_training_probe_objective": initial_loss,
        "training_probe_count": probe_count,
        "training_probe_sha256": hashlib.sha256(np.ascontiguousarray(probe, dtype="<i8").tobytes()).hexdigest(),
        "probe_role": "training_diagnostic_only_not_a_validation_metric",
        "epochs": trace,
        "runtime_seconds": time.perf_counter() - started,
    })
