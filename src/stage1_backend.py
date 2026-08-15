"""Production training, scoring, serialization, and fold-in for Stage 1.

This module is cycle-neutral.  Every random stream receives an explicit cycle
identifier, and every routine operates on already-authorized sparse inputs.
It never discovers or opens protected coordinates on its own.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import scipy.optimize as opt
import scipy.sparse as sp
from threadpoolctl import threadpool_limits

from src.preference_model import (
    FeatureSumParameters,
    feature_sum_item_factors,
    observed_confidence,
    triple_stream_sha256,
    update_user_factors,
)
from src.stage1_protocol import canonical_json_bytes, file_sha256


ADAGRAD_EPSILON = 1e-8
INITIALIZATION_SCALE = 0.01


def namespaced_seed(cycle_id: str, training_seed: int, purpose: str) -> int:
    if not isinstance(cycle_id, str) or not cycle_id:
        raise ValueError("cycle_id must be a nonempty string")
    if isinstance(training_seed, (bool, np.bool_)) or not isinstance(
        training_seed, (int, np.integer)
    ):
        raise ValueError("training_seed must be an integer")
    if int(training_seed) < 0 or not isinstance(purpose, str) or not purpose:
        raise ValueError("invalid namespaced seed fields")
    payload = f"{cycle_id}:bpr:{int(training_seed)}:{purpose}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


class CycleBPRTripleSampler:
    """Frozen scalar-order PCG64 stream with an explicit cycle namespace."""

    def __init__(
        self, ownership: sp.spmatrix, *, cycle_id: str, training_seed: int
    ) -> None:
        owned = sp.csr_matrix(ownership, dtype=np.float32)
        owned.sum_duplicates()
        owned.sort_indices()
        if owned.ndim != 2 or owned.nnz == 0:
            raise ValueError("ownership must be a nonempty two-dimensional matrix")
        if np.any(owned.data != 1.0):
            raise ValueError("ownership must be binary")
        row_counts = np.diff(owned.indptr)
        if np.any(row_counts[row_counts > 0] >= owned.shape[1]):
            raise ValueError("a sampled user owns the complete catalogue")
        self.ownership = owned
        self.edge_rows = np.repeat(
            np.arange(owned.shape[0], dtype=np.int64), row_counts
        )
        self.rng = np.random.Generator(
            np.random.PCG64(
                namespaced_seed(cycle_id, training_seed, "triple-sampler")
            )
        )
        self.rejected_draws = 0

    def sample(self, count: int) -> np.ndarray:
        if isinstance(count, (bool, np.bool_)) or not isinstance(
            count, (int, np.integer)
        ) or int(count) <= 0:
            raise ValueError("count must be a positive integer")
        triples = np.empty((int(count), 3), dtype=np.int64)
        owned = self.ownership
        for row in range(int(count)):
            edge = int(self.rng.integers(0, owned.nnz))
            user = int(self.edge_rows[edge])
            positive = int(owned.indices[edge])
            start, stop = int(owned.indptr[user]), int(owned.indptr[user + 1])
            known = owned.indices[start:stop]
            while True:
                negative = int(self.rng.integers(0, owned.shape[1]))
                location = int(np.searchsorted(known, negative))
                if location >= known.size or known[location] != negative:
                    break
                self.rejected_draws += 1
            triples[row] = user, positive, negative
        return triples


def initialize_feature_sum_parameters(
    *,
    n_users: int,
    n_items: int,
    n_features: int,
    factors: int,
    cycle_id: str,
    training_seed: int,
) -> FeatureSumParameters:
    shape_values = (n_users, n_items, n_features, factors)
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) < (0 if index == 2 else 1)
        for index, value in enumerate(shape_values)
    ):
        raise ValueError("invalid feature-sum parameter dimensions")
    shared = np.random.Generator(
        np.random.PCG64(
            namespaced_seed(cycle_id, training_seed, "shared-parameter-initialization")
        )
    )
    users = shared.normal(0.0, INITIALIZATION_SCALE, (n_users, factors))
    identity = shared.normal(0.0, INITIALIZATION_SCALE, (n_items, factors))
    bias = shared.normal(0.0, INITIALIZATION_SCALE, n_items)
    if n_features:
        genre_rng = np.random.Generator(
            np.random.PCG64(
                namespaced_seed(cycle_id, training_seed, "genre-parameter-initialization")
            )
        )
        features = genre_rng.normal(
            0.0, INITIALIZATION_SCALE, (n_features, factors)
        )
    else:
        features = np.empty((0, factors), dtype=np.float64)
    return FeatureSumParameters(
        user_factors=users.astype(np.float32),
        identity_factors=identity.astype(np.float32),
        feature_factors=features.astype(np.float32),
        item_bias=bias.astype(np.float32),
    )


@dataclass(frozen=True)
class BPRFitResult:
    parameters: FeatureSumParameters
    diagnostics: Mapping[str, Any]


def _bpr_epoch_gradient(
    parameters: FeatureSumParameters,
    genre_features: sp.csr_matrix,
    triples: np.ndarray,
    *,
    regularization: float,
    include_genre: bool,
    chunk_size: int = 100_000,
) -> tuple[float, dict[str, np.ndarray]]:
    users = np.asarray(parameters.user_factors, dtype=np.float64)
    identity = np.asarray(parameters.identity_factors, dtype=np.float64)
    feature_factors = np.asarray(parameters.feature_factors, dtype=np.float64)
    bias = np.asarray(parameters.item_bias, dtype=np.float64)
    gradients = {
        "user_factors": np.zeros_like(users),
        "identity_factors": np.zeros_like(identity),
        "feature_factors": np.zeros_like(feature_factors),
        "item_bias": np.zeros_like(bias),
    }
    data_loss = 0.0
    for start in range(0, triples.shape[0], chunk_size):
        block = triples[start : start + chunk_size]
        u, i, j = block[:, 0], block[:, 1], block[:, 2]
        x = users[u]
        qi = identity[i].copy()
        qj = identity[j].copy()
        if include_genre:
            qi += genre_features[i] @ feature_factors
            qj += genre_features[j] @ feature_factors
        difference = qi - qj
        margin = bias[i] - bias[j] + np.einsum("ij,ij->i", x, difference)
        data_loss += float(np.logaddexp(0.0, -margin).sum(dtype=np.float64))
        positive = margin >= 0.0
        h = np.empty_like(margin)
        h[positive] = np.exp(-margin[positive]) / (1.0 + np.exp(-margin[positive]))
        h[~positive] = 1.0 / (1.0 + np.exp(margin[~positive]))
        coefficient = -h
        np.add.at(gradients["user_factors"], u, coefficient[:, None] * difference)
        weighted_users = coefficient[:, None] * x
        np.add.at(gradients["identity_factors"], i, weighted_users)
        np.add.at(gradients["identity_factors"], j, -weighted_users)
        np.add.at(gradients["item_bias"], i, coefficient)
        np.add.at(gradients["item_bias"], j, -coefficient)
        if include_genre:
            feature_difference = genre_features[i] - genre_features[j]
            gradients["feature_factors"] += feature_difference.T @ weighted_users

    active = ("user_factors", "identity_factors", "item_bias")
    penalty = float(regularization)
    regularization_term = 0.0
    for name in active:
        values = {
            "user_factors": users,
            "identity_factors": identity,
            "item_bias": bias,
        }[name]
        regularization_term += penalty * float(np.sum(values * values))
        gradients[name] += 2.0 * penalty * values
    if include_genre:
        regularization_term += penalty * float(
            np.sum(feature_factors * feature_factors)
        )
        gradients["feature_factors"] += 2.0 * penalty * feature_factors
    return data_loss + regularization_term, gradients


def fit_feature_sum_bpr(
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
    include_genre: bool,
) -> BPRFitResult:
    """Fit the prospective full-epoch-gradient NumPy AdaGrad fallback."""

    owned = sp.csr_matrix(ownership, dtype=np.float32)
    owned.sum_duplicates()
    owned.sort_indices()
    genres = sp.csr_matrix(genre_features, dtype=np.float64)
    if genres.shape[0] != owned.shape[1]:
        raise ValueError("genre rows and model items differ")
    if regularization <= 0 or learning_rate <= 0 or epochs <= 0 or samples_per_epoch <= 0:
        raise ValueError("BPR optimization parameters must be positive")
    parameters = initialize_feature_sum_parameters(
        n_users=owned.shape[0],
        n_items=owned.shape[1],
        n_features=genres.shape[1] if include_genre else 0,
        factors=factors,
        cycle_id=cycle_id,
        training_seed=training_seed,
    )
    accumulators = {
        name: np.zeros_like(np.asarray(getattr(parameters, name)), dtype=np.float64)
        for name in ("user_factors", "identity_factors", "feature_factors", "item_bias")
    }
    sampler = CycleBPRTripleSampler(
        owned, cycle_id=cycle_id, training_seed=training_seed
    )
    traces: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(int(epochs)):
        epoch_started = time.perf_counter()
        rejected_before = sampler.rejected_draws
        triples = sampler.sample(int(samples_per_epoch))
        loss, gradients = _bpr_epoch_gradient(
            parameters,
            genres,
            triples,
            regularization=float(regularization),
            include_genre=include_genre,
        )
        arrays: dict[str, np.ndarray] = {}
        for name, gradient in gradients.items():
            accumulators[name] += gradient * gradient
            current = np.asarray(getattr(parameters, name), dtype=np.float64)
            updated = current - float(learning_rate) * gradient / (
                np.sqrt(accumulators[name]) + ADAGRAD_EPSILON
            )
            arrays[name] = updated.astype(np.float32)
        parameters = FeatureSumParameters(**arrays)
        finite = all(
            np.all(np.isfinite(np.asarray(getattr(parameters, name))))
            for name in arrays
        )
        traces.append(
            {
                "epoch": epoch,
                "sampled_loss_before_update": loss,
                "triple_stream_sha256": triple_stream_sha256(triples),
                "accepted_triples": int(triples.shape[0]),
                "rejected_negative_draws": sampler.rejected_draws - rejected_before,
                "gradient_l2": {
                    name: float(np.linalg.norm(value))
                    for name, value in gradients.items()
                },
                "runtime_seconds": time.perf_counter() - epoch_started,
                "finite_after_update": bool(finite),
            }
        )
        if not finite:
            raise FloatingPointError("BPR parameters became nonfinite")
    return BPRFitResult(
        parameters=parameters,
        diagnostics={
            "backend": "tested_numpy_feature_sum_bpr",
            "optimizer": "full_sampled_epoch_gradient_adagrad",
            "adagrad_epsilon": ADAGRAD_EPSILON,
            "initialization_scale": INITIALIZATION_SCALE,
            "epochs": traces,
            "runtime_seconds": time.perf_counter() - started,
        },
    )


@dataclass(frozen=True)
class ALSFitResult:
    user_factors: np.ndarray
    item_factors: np.ndarray
    diagnostics: Mapping[str, Any]


def fit_implicit_als(
    ownership: sp.spmatrix,
    playtime: sp.spmatrix,
    *,
    factors: int,
    regularization: float,
    alpha_o: float,
    alpha_p: float,
    tau: float,
    iterations: int,
    training_seed: int,
    num_threads: int,
) -> ALSFitResult:
    """Fit the frozen WRMF equation through implicit 0.7.2's exact solver."""

    from implicit.als import AlternatingLeastSquares

    confidence = observed_confidence(
        ownership,
        playtime,
        alpha_o=alpha_o,
        alpha_p=alpha_p,
        tau=tau,
    )
    trace: list[dict[str, Any]] = []
    started = time.perf_counter()
    with threadpool_limits(limits=1, user_api="blas"):
        model = AlternatingLeastSquares(
            factors=int(factors),
            regularization=float(regularization),
            alpha=1.0,
            dtype=np.float32,
            use_native=True,
            use_cg=False,
            use_gpu=False,
            iterations=int(iterations),
            calculate_training_loss=True,
            num_threads=int(num_threads),
            random_state=int(training_seed),
        )
        model.fit(
            confidence,
            show_progress=False,
            callback=lambda iteration, elapsed, loss: trace.append(
                {
                    "iteration": int(iteration),
                    "runtime_seconds": float(elapsed),
                    "backend_loss": None if loss is None else float(loss),
                }
            ),
        )
    users = np.asarray(model.user_factors, dtype=np.float32)
    items = np.asarray(model.item_factors, dtype=np.float32)
    if not np.all(np.isfinite(users)) or not np.all(np.isfinite(items)):
        raise FloatingPointError("ALS parameters became nonfinite")
    return ALSFitResult(
        user_factors=users,
        item_factors=items,
        diagnostics={
            "backend": "implicit",
            "backend_version": "0.7.2",
            "solver": "native_exact_least_squares",
            "iterations": trace,
            "runtime_seconds": time.perf_counter() - started,
        },
    )


def fold_in_als(
    ownership: sp.spmatrix,
    playtime: sp.spmatrix,
    item_factors: np.ndarray,
    *,
    regularization: float,
    alpha_o: float,
    alpha_p: float,
    tau: float,
) -> np.ndarray:
    confidence = observed_confidence(
        ownership, playtime, alpha_o=alpha_o, alpha_p=alpha_p, tau=tau
    )
    folded, _ = update_user_factors(
        ownership,
        confidence,
        item_factors,
        regularization=regularization,
    )
    return folded.astype(np.float32)


def construct_fold_in_triples(
    positive_items: Sequence[int] | np.ndarray,
    *,
    n_items: int,
    cycle_id: str,
    user_id: int,
) -> np.ndarray:
    positives = np.asarray(positive_items)
    if positives.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    if positives.dtype.kind not in "iu" or positives.dtype.kind == "b" or positives.ndim != 1:
        raise ValueError("positive_items must be a one-dimensional integer array")
    positives = np.unique(positives.astype(np.int64, copy=False))
    if positives.size == 0 or positives.size >= int(n_items):
        return np.empty((0, 2), dtype=np.int64)
    if np.any((positives < 0) | (positives >= int(n_items))):
        raise IndexError("fold-in positive is outside the catalogue")
    payload = f"{cycle_id}:fold-in:{int(user_id)}".encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    rng = np.random.Generator(np.random.PCG64(seed))
    pairs = np.empty((positives.size, 2), dtype=np.int64)
    for row, positive in enumerate(positives):
        while True:
            negative = int(rng.integers(0, int(n_items)))
            location = int(np.searchsorted(positives, negative))
            if location >= positives.size or positives[location] != negative:
                break
        pairs[row] = int(positive), negative
    return pairs


def fold_in_bpr_user(
    positive_items: Sequence[int] | np.ndarray,
    *,
    user_id: int,
    cycle_id: str,
    identity_factors: np.ndarray,
    item_bias: np.ndarray,
    genre_features: sp.spmatrix,
    feature_factors: np.ndarray,
    regularization: float,
    tolerance: float = 1e-8,
    max_iterations: int = 250,
) -> tuple[np.ndarray, Mapping[str, Any]]:
    identity = np.asarray(identity_factors, dtype=np.float64)
    bias = np.asarray(item_bias, dtype=np.float64)
    genres = sp.csr_matrix(genre_features, dtype=np.float64)
    content = np.asarray(feature_factors, dtype=np.float64)
    item_factors = identity + (genres @ content if content.shape[0] else 0.0)
    pairs = construct_fold_in_triples(
        positive_items,
        n_items=identity.shape[0],
        cycle_id=cycle_id,
        user_id=user_id,
    )
    if pairs.size == 0:
        return np.zeros(identity.shape[1], dtype=np.float32), {
            "status": "insufficient_history",
            "triple_count": 0,
        }
    differences = item_factors[pairs[:, 0]] - item_factors[pairs[:, 1]]
    bias_differences = bias[pairs[:, 0]] - bias[pairs[:, 1]]

    def objective(vector: np.ndarray) -> tuple[float, np.ndarray]:
        margin = bias_differences + differences @ vector
        loss = float(np.logaddexp(0.0, -margin).sum()) + float(
            regularization
        ) * float(vector @ vector)
        positive = margin >= 0.0
        h = np.empty_like(margin)
        h[positive] = np.exp(-margin[positive]) / (1.0 + np.exp(-margin[positive]))
        h[~positive] = 1.0 / (1.0 + np.exp(margin[~positive]))
        gradient = -(h @ differences) + 2.0 * float(regularization) * vector
        return loss, gradient

    result = opt.minimize(
        objective,
        np.zeros(identity.shape[1], dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        options={"ftol": float(tolerance), "gtol": float(tolerance), "maxiter": int(max_iterations)},
    )
    vector = np.asarray(result.x, dtype=np.float32)
    if not np.all(np.isfinite(vector)):
        raise FloatingPointError("pairwise fold-in produced nonfinite factors")
    return vector, {
        "status": "complete" if result.success else "solver_stopped",
        "triple_count": int(pairs.shape[0]),
        "triple_sha256": hashlib.sha256(
            np.ascontiguousarray(pairs, dtype="<i8").tobytes()
        ).hexdigest(),
        "objective": float(result.fun),
        "iterations": int(result.nit),
        "gradient_linf": float(np.max(np.abs(result.jac))),
        "message": str(result.message),
    }


def save_parameter_archive(path: str | Path, **arrays: np.ndarray) -> Mapping[str, Any]:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canonical: dict[str, np.ndarray] = {}
    for name, values in arrays.items():
        array = np.asarray(values)
        if not np.all(np.isfinite(array)):
            raise ValueError(f"nonfinite parameter array: {name}")
        canonical[name] = array
    np.savez_compressed(destination, **canonical)
    return {
        "path": destination.as_posix(),
        "size_bytes": destination.stat().st_size,
        "sha256": file_sha256(destination),
        "arrays": {
            name: {
                "shape": list(values.shape),
                "dtype": str(values.dtype),
                "sha256": hashlib.sha256(
                    np.ascontiguousarray(values).tobytes()
                ).hexdigest(),
            }
            for name, values in sorted(canonical.items())
        },
    }


def load_parameter_archive(
    path: str | Path, *, expected_sha256: str | None = None
) -> dict[str, np.ndarray]:
    source = Path(path)
    if expected_sha256 is not None and file_sha256(source) != expected_sha256:
        raise ValueError("parameter archive hash mismatch")
    with np.load(source, allow_pickle=False) as payload:
        result = {name: np.asarray(payload[name]) for name in payload.files}
    if any(not np.all(np.isfinite(value)) for value in result.values()):
        raise ValueError("parameter archive contains nonfinite values")
    return result
