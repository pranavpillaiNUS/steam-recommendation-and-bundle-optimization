"""Reference mathematics for the frozen Stage 1 preference estimators.

This module defines backend-neutral numerical oracles. It does not load
repository artifacts, inspect outcomes, or fit production models. The
confidence source for the Stage 1 ALS specification is frozen as
``playtime_forever``. Playtime changes confidence only; binary ownership is
the preference target.

All dense linear algebra accumulates in float64. Reference ALS fits expose
float32 stored factors, matching the frozen storage policy. No function
materializes a full user-by-item confidence or score matrix unless the caller
explicitly requests and byte-bounds a score block.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np
import scipy.sparse as sp


SCHEMA_VERSION = 1
PLAYTIME_CONFIDENCE_SOURCE = "playtime_forever"
CONFIDENCE_EQUATION = (
    "1 + alpha_o * owned + alpha_p * min(log1p(playtime_forever), tau)"
)
WRMF_SCORE_EQUATION = "s_ui = x_u dot q_i"
FEATURE_SUM_SCORE_EQUATION = (
    "s_ui = b_i + x_u dot (eta_i + rho * sum_a f_ia * g_a)"
)
BPR_LOSS_EQUATION = (
    "sum_(u,i,j) log(1 + exp(-(s_ui-s_uj))) + lambda * ||theta||_2^2"
)
DEFAULT_JITTER_SEQUENCE = (0.0, 1e-8, 1e-6)
DEFAULT_MAXIMUM_SCORE_BLOCK_BYTES = 67_108_864
TRIPLE_SAMPLER = (
    "sha256_namespaced_numpy_pcg64_scalar_uniform_training_edge_with_"
    "replacement_then_uniform_warm_item_reject_training_positives_only"
)


@dataclass(frozen=True)
class LinearSolveDiagnostic:
    """Diagnostics for one Cholesky solve."""

    jitter: float
    residual_norm: float
    relative_residual_norm: float
    original_residual_norm: float
    original_relative_residual_norm: float


@dataclass(frozen=True)
class ALSIterationDiagnostic:
    """Objective and solve diagnostics for one complete ALS iteration."""

    iteration: int
    objective_before: float
    objective_after_users: float
    objective_after_items: float
    user_solve_diagnostics: tuple[LinearSolveDiagnostic, ...]
    item_solve_diagnostics: tuple[LinearSolveDiagnostic, ...]
    runtime_seconds: float


@dataclass(frozen=True)
class ALSFitDiagnostic:
    """Complete diagnostic trace for the reference fixed-iteration ALS fit."""

    initial_objective: float
    final_objective: float
    iterations: tuple[ALSIterationDiagnostic, ...]
    runtime_seconds: float
    numerical_failures: int


@dataclass(frozen=True)
class ALSReferenceModel:
    """Reference ALS factors and their diagnostic trace."""

    user_factors: np.ndarray
    item_factors: np.ndarray
    diagnostics: ALSFitDiagnostic
    seed: int
    regularization: float


@dataclass(frozen=True)
class FeatureSumParameters:
    """Parameters for the pairwise identity-plus-feature score equation."""

    user_factors: np.ndarray
    identity_factors: np.ndarray
    feature_factors: np.ndarray
    item_bias: np.ndarray


def _finite_scalar(value: Any, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite scalar") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite scalar")
    return result


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if result != value or result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _nonnegative_integer(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a nonnegative integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a nonnegative integer") from exc
    if result != value or result < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return result


def _validate_csr(
    matrix: sp.spmatrix,
    *,
    name: str,
    binary: bool = False,
    nonnegative: bool = False,
) -> sp.csr_matrix:
    if not sp.isspmatrix_csr(matrix):
        raise ValueError(f"{name} must be a CSR matrix")
    if matrix.dtype != np.dtype(np.float32):
        raise ValueError(f"{name} must use float32 data")
    if not matrix.has_canonical_format or not matrix.has_sorted_indices:
        raise ValueError(f"{name} must use canonical sorted CSR storage")
    if not np.all(np.isfinite(matrix.data)):
        raise ValueError(f"{name} contains nonfinite values")
    if binary and np.any(matrix.data != 1.0):
        raise ValueError(f"{name} must contain binary observed values")
    if nonnegative and np.any(matrix.data < 0.0):
        raise ValueError(f"{name} contains negative values")
    return matrix


def _validate_ownership(ownership: sp.spmatrix) -> sp.csr_matrix:
    return _validate_csr(ownership, name="ownership", binary=True)


def _validate_observed_pair(
    ownership: sp.spmatrix,
    observed_values: sp.spmatrix,
    *,
    value_name: str,
    confidence: bool = False,
) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    owned = _validate_ownership(ownership)
    values = _validate_csr(
        observed_values,
        name=value_name,
        nonnegative=True,
    )
    if owned.shape != values.shape:
        raise ValueError(f"ownership and {value_name} shapes differ")
    if not (
        np.array_equal(owned.indptr, values.indptr)
        and np.array_equal(owned.indices, values.indices)
    ):
        raise ValueError(
            f"ownership and {value_name} must have identical observed entries"
        )
    if confidence and np.any(values.data <= 1.0):
        raise ValueError("observed confidence must exceed the unobserved baseline")
    return owned, values


def _factor_matrix(
    values: Any,
    *,
    name: str,
    rows: int | None = None,
    columns: int | None = None,
) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 2 or array.dtype.kind not in "fc":
        raise ValueError(f"{name} must be a two-dimensional floating array")
    if np.iscomplexobj(array) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite and real")
    if rows is not None and array.shape[0] != rows:
        raise ValueError(f"{name} row count is misaligned")
    if columns is not None and array.shape[1] != columns:
        raise ValueError(f"{name} factor dimension is misaligned")
    return np.asarray(array, dtype=np.float64)


def _index_vector(
    values: Any,
    *,
    name: str,
    upper_bound: int,
) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.dtype.kind not in "iu" or array.dtype.kind == "b":
        raise ValueError(f"{name} must be a one-dimensional integer array")
    result = array.astype(np.int64, copy=False)
    if np.any((result < 0) | (result >= upper_bound)):
        raise ValueError(f"{name} contains an out-of-range index")
    return result


def popularity_scores(ownership: sp.spmatrix) -> np.ndarray:
    """Return exact training-only ownership counts in item order."""

    owned = _validate_ownership(ownership)
    return np.bincount(
        owned.indices,
        minlength=owned.shape[1],
    ).astype(np.int64, copy=False)


def observed_confidence(
    ownership: sp.spmatrix,
    playtime: sp.spmatrix,
    *,
    alpha_o: float,
    alpha_p: float,
    tau: float,
) -> sp.csr_matrix:
    """Return observed full confidence values for ``playtime_forever``.

    Unobserved confidence is one and remains implicit. The playtime matrix may
    omit explicit zero entries; values at owned positions are then zero.
    """

    owned = _validate_ownership(ownership)
    played = _validate_csr(
        playtime,
        name=PLAYTIME_CONFIDENCE_SOURCE,
        nonnegative=True,
    )
    if played.shape != owned.shape:
        raise ValueError("ownership and playtime shapes differ")
    for row in range(played.shape[0]):
        played_start = int(played.indptr[row])
        played_stop = int(played.indptr[row + 1])
        if played_start == played_stop:
            continue
        owned_start = int(owned.indptr[row])
        owned_stop = int(owned.indptr[row + 1])
        owned_columns = owned.indices[owned_start:owned_stop]
        played_columns = played.indices[played_start:played_stop]
        locations = np.searchsorted(owned_columns, played_columns)
        clipped = np.minimum(locations, max(owned_columns.size - 1, 0))
        if owned_columns.size == 0 or np.any(
            (locations >= owned_columns.size)
            | (owned_columns[clipped] != played_columns)
        ):
            raise ValueError("playtime contains an unowned interaction")
    ao = _finite_scalar(alpha_o, name="alpha_o")
    ap = _finite_scalar(alpha_p, name="alpha_p")
    cap = _finite_scalar(tau, name="tau")
    if ao <= 0.0 or ap < 0.0 or cap < 0.0:
        raise ValueError("confidence parameters violate the frozen contract")

    rows = np.repeat(
        np.arange(owned.shape[0], dtype=np.int64),
        np.diff(owned.indptr),
    )
    columns = owned.indices
    playtime_at_owned = np.asarray(
        played[rows, columns],
        dtype=np.float64,
    ).reshape(-1)
    with np.errstate(over="ignore", invalid="ignore"):
        confidence = (
            1.0
            + ao
            + ap * np.minimum(np.log1p(playtime_at_owned), cap)
        )
    if (
        not np.all(np.isfinite(confidence))
        or np.any(confidence <= 1.0)
        or np.any(confidence > np.finfo(np.float32).max)
    ):
        raise ValueError("confidence values are not finite float32 values")
    return sp.csr_matrix(
        (
            confidence.astype(np.float32),
            owned.indices.copy(),
            owned.indptr.copy(),
        ),
        shape=owned.shape,
    )


def _regularization(value: Any) -> float:
    result = _finite_scalar(value, name="regularization")
    if result < 0.0:
        raise ValueError("regularization must be nonnegative")
    return result


def wrmf_objective(
    ownership: sp.spmatrix,
    confidence: sp.spmatrix,
    user_factors: Any,
    item_factors: Any,
    *,
    regularization: float,
) -> float:
    """Evaluate the exact WRMF objective without a dense score matrix."""

    owned, weights = _validate_observed_pair(
        ownership,
        confidence,
        value_name="confidence",
        confidence=True,
    )
    users = _factor_matrix(
        user_factors,
        name="user_factors",
        rows=owned.shape[0],
    )
    items = _factor_matrix(
        item_factors,
        name="item_factors",
        rows=owned.shape[1],
        columns=users.shape[1],
    )
    penalty = _regularization(regularization)

    user_gram = users.T @ users
    item_gram = items.T @ items
    objective = float(np.sum(user_gram * item_gram))
    for row in range(owned.shape[0]):
        start = int(owned.indptr[row])
        stop = int(owned.indptr[row + 1])
        if start == stop:
            continue
        columns = owned.indices[start:stop]
        scores = items[columns] @ users[row]
        observed_weights = weights.data[start:stop].astype(
            np.float64,
            copy=False,
        )
        objective += float(
            np.sum(
                observed_weights * np.square(1.0 - scores)
                - np.square(scores)
            )
        )
    objective += penalty * float(
        np.sum(np.square(users)) + np.sum(np.square(items))
    )
    if not np.isfinite(objective):
        raise ValueError("WRMF objective is nonfinite")
    return objective


def _normal_equation_from_row(
    fixed_factors: np.ndarray,
    observed_indices: np.ndarray,
    observed_confidence_values: np.ndarray,
    *,
    regularization: float,
    gram: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    fixed = _factor_matrix(fixed_factors, name="fixed_factors")
    indices = _index_vector(
        observed_indices,
        name="observed_indices",
        upper_bound=fixed.shape[0],
    )
    confidence_values = np.asarray(
        observed_confidence_values,
        dtype=np.float64,
    )
    if confidence_values.shape != indices.shape:
        raise ValueError("observed confidence and index lengths differ")
    if (
        not np.all(np.isfinite(confidence_values))
        or np.any(confidence_values <= 1.0)
    ):
        raise ValueError("observed confidence must be finite and above one")
    penalty = _regularization(regularization)
    base = fixed.T @ fixed if gram is None else np.asarray(gram, dtype=np.float64)
    if base.shape != (fixed.shape[1], fixed.shape[1]):
        raise ValueError("fixed-factor Gram matrix is misaligned")
    matrix = base.copy()
    target = np.zeros(fixed.shape[1], dtype=np.float64)
    if indices.size:
        observed = fixed[indices]
        matrix += observed.T @ (
            (confidence_values - 1.0)[:, None] * observed
        )
        target = observed.T @ confidence_values
    matrix.flat[:: fixed.shape[1] + 1] += penalty
    return matrix, target


def user_normal_equation(
    ownership: sp.spmatrix,
    confidence: sp.spmatrix,
    item_factors: Any,
    user_index: int,
    *,
    regularization: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the exact ridge system for one user block row."""

    owned, weights = _validate_observed_pair(
        ownership,
        confidence,
        value_name="confidence",
        confidence=True,
    )
    items = _factor_matrix(
        item_factors,
        name="item_factors",
        rows=owned.shape[1],
    )
    user = _nonnegative_integer(user_index, name="user_index")
    if user >= owned.shape[0]:
        raise ValueError("user_index is out of range")
    start = int(owned.indptr[user])
    stop = int(owned.indptr[user + 1])
    return _normal_equation_from_row(
        items,
        owned.indices[start:stop],
        weights.data[start:stop],
        regularization=regularization,
    )


def item_normal_equation(
    ownership: sp.spmatrix,
    confidence: sp.spmatrix,
    user_factors: Any,
    item_index: int,
    *,
    regularization: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the exact ridge system for one item block row."""

    owned, weights = _validate_observed_pair(
        ownership,
        confidence,
        value_name="confidence",
        confidence=True,
    )
    users = _factor_matrix(
        user_factors,
        name="user_factors",
        rows=owned.shape[0],
    )
    item = _nonnegative_integer(item_index, name="item_index")
    if item >= owned.shape[1]:
        raise ValueError("item_index is out of range")
    owned_t = owned.transpose().tocsr()
    weights_t = weights.transpose().tocsr()
    start = int(owned_t.indptr[item])
    stop = int(owned_t.indptr[item + 1])
    return _normal_equation_from_row(
        users,
        owned_t.indices[start:stop],
        weights_t.data[start:stop],
        regularization=regularization,
    )


def _jitter_values(values: Sequence[float]) -> tuple[float, ...]:
    result = tuple(_finite_scalar(value, name="jitter") for value in values)
    if not result or any(value < 0.0 for value in result):
        raise ValueError("jitter_sequence must contain nonnegative values")
    if any(right <= left for left, right in zip(result, result[1:])):
        raise ValueError("jitter_sequence must be strictly increasing")
    return result


def solve_spd(
    matrix: Any,
    target: Any,
    *,
    jitter_sequence: Sequence[float] = DEFAULT_JITTER_SEQUENCE,
) -> tuple[np.ndarray, LinearSolveDiagnostic]:
    """Solve a symmetric positive definite system by Cholesky factorization."""

    system = np.asarray(matrix, dtype=np.float64)
    vector = np.asarray(target, dtype=np.float64)
    if (
        system.ndim != 2
        or system.shape[0] != system.shape[1]
        or vector.shape != (system.shape[0],)
        or system.shape[0] == 0
    ):
        raise ValueError("matrix and target shapes do not define a square system")
    if not np.all(np.isfinite(system)) or not np.all(np.isfinite(vector)):
        raise ValueError("linear system must be finite")
    if not np.allclose(system, system.T, rtol=1e-12, atol=1e-12):
        raise ValueError("linear system must be symmetric")
    jitters = _jitter_values(jitter_sequence)
    identity = np.eye(system.shape[0], dtype=np.float64)
    last_error: Exception | None = None
    for jitter in jitters:
        adjusted = system if jitter == 0.0 else system + jitter * identity
        try:
            cholesky = np.linalg.cholesky(adjusted)
            intermediate = np.linalg.solve(cholesky, vector)
            solution = np.linalg.solve(cholesky.T, intermediate)
        except np.linalg.LinAlgError as exc:
            last_error = exc
            continue
        if not np.all(np.isfinite(solution)):
            last_error = np.linalg.LinAlgError("solution is nonfinite")
            continue
        adjusted_residual = adjusted @ solution - vector
        original_residual = system @ solution - vector
        residual_norm = float(np.linalg.norm(adjusted_residual))
        original_norm = float(np.linalg.norm(original_residual))
        denominator = max(1.0, float(np.linalg.norm(vector)))
        return solution, LinearSolveDiagnostic(
            jitter=float(jitter),
            residual_norm=residual_norm,
            relative_residual_norm=residual_norm / denominator,
            original_residual_norm=original_norm,
            original_relative_residual_norm=original_norm / denominator,
        )
    raise np.linalg.LinAlgError(
        "Cholesky solve failed for every declared jitter"
    ) from last_error


def _update_factor_rows(
    ownership: sp.csr_matrix,
    confidence: sp.csr_matrix,
    fixed_factors: np.ndarray,
    *,
    regularization: float,
    jitter_sequence: Sequence[float],
) -> tuple[np.ndarray, tuple[LinearSolveDiagnostic, ...]]:
    fixed = _factor_matrix(
        fixed_factors,
        name="fixed_factors",
        rows=ownership.shape[1],
    )
    gram = fixed.T @ fixed
    updated = np.empty(
        (ownership.shape[0], fixed.shape[1]),
        dtype=np.float64,
    )
    diagnostics: list[LinearSolveDiagnostic] = []
    for row in range(ownership.shape[0]):
        start = int(ownership.indptr[row])
        stop = int(ownership.indptr[row + 1])
        matrix, target = _normal_equation_from_row(
            fixed,
            ownership.indices[start:stop],
            confidence.data[start:stop],
            regularization=regularization,
            gram=gram,
        )
        solution, diagnostic = solve_spd(
            matrix,
            target,
            jitter_sequence=jitter_sequence,
        )
        updated[row] = solution
        diagnostics.append(diagnostic)
    return updated, tuple(diagnostics)


def update_user_factors(
    ownership: sp.spmatrix,
    confidence: sp.spmatrix,
    item_factors: Any,
    *,
    regularization: float,
    jitter_sequence: Sequence[float] = DEFAULT_JITTER_SEQUENCE,
) -> tuple[np.ndarray, tuple[LinearSolveDiagnostic, ...]]:
    """Solve every user ridge problem with item factors fixed."""

    owned, weights = _validate_observed_pair(
        ownership,
        confidence,
        value_name="confidence",
        confidence=True,
    )
    return _update_factor_rows(
        owned,
        weights,
        _factor_matrix(
            item_factors,
            name="item_factors",
            rows=owned.shape[1],
        ),
        regularization=regularization,
        jitter_sequence=jitter_sequence,
    )


def update_item_factors(
    ownership: sp.spmatrix,
    confidence: sp.spmatrix,
    user_factors: Any,
    *,
    regularization: float,
    jitter_sequence: Sequence[float] = DEFAULT_JITTER_SEQUENCE,
) -> tuple[np.ndarray, tuple[LinearSolveDiagnostic, ...]]:
    """Solve every item ridge problem with user factors fixed."""

    owned, weights = _validate_observed_pair(
        ownership,
        confidence,
        value_name="confidence",
        confidence=True,
    )
    users = _factor_matrix(
        user_factors,
        name="user_factors",
        rows=owned.shape[0],
    )
    return _update_factor_rows(
        owned.transpose().tocsr(),
        weights.transpose().tocsr(),
        users,
        regularization=regularization,
        jitter_sequence=jitter_sequence,
    )


def fit_als_reference(
    ownership: sp.spmatrix,
    confidence: sp.spmatrix,
    *,
    factors: int,
    regularization: float,
    iterations: int,
    seed: int,
    jitter_sequence: Sequence[float] = DEFAULT_JITTER_SEQUENCE,
    initialization_scale: float = 0.01,
) -> ALSReferenceModel:
    """Run deterministic exact fixed-iteration ALS for small reference cases."""

    owned, weights = _validate_observed_pair(
        ownership,
        confidence,
        value_name="confidence",
        confidence=True,
    )
    rank = _positive_integer(factors, name="factors")
    steps = _positive_integer(iterations, name="iterations")
    rng_seed = _nonnegative_integer(seed, name="seed")
    penalty = _regularization(regularization)
    if penalty <= 0.0:
        raise ValueError("reference ALS requires positive regularization")
    scale = _finite_scalar(initialization_scale, name="initialization_scale")
    if scale <= 0.0:
        raise ValueError("initialization_scale must be positive")
    jitters = _jitter_values(jitter_sequence)

    started = time.perf_counter()
    rng = np.random.Generator(np.random.PCG64(rng_seed))
    users = rng.normal(
        0.0,
        scale,
        size=(owned.shape[0], rank),
    ).astype(np.float32)
    items = rng.normal(
        0.0,
        scale,
        size=(owned.shape[1], rank),
    ).astype(np.float32)
    initial = wrmf_objective(
        owned,
        weights,
        users,
        items,
        regularization=penalty,
    )
    trace: list[ALSIterationDiagnostic] = []
    objective_before = initial
    for iteration in range(steps):
        iteration_started = time.perf_counter()
        users64, user_diagnostics = update_user_factors(
            owned,
            weights,
            items,
            regularization=penalty,
            jitter_sequence=jitters,
        )
        users = users64.astype(np.float32)
        after_users = wrmf_objective(
            owned,
            weights,
            users,
            items,
            regularization=penalty,
        )
        items64, item_diagnostics = update_item_factors(
            owned,
            weights,
            users,
            regularization=penalty,
            jitter_sequence=jitters,
        )
        items = items64.astype(np.float32)
        after_items = wrmf_objective(
            owned,
            weights,
            users,
            items,
            regularization=penalty,
        )
        trace.append(
            ALSIterationDiagnostic(
                iteration=iteration,
                objective_before=objective_before,
                objective_after_users=after_users,
                objective_after_items=after_items,
                user_solve_diagnostics=user_diagnostics,
                item_solve_diagnostics=item_diagnostics,
                runtime_seconds=time.perf_counter() - iteration_started,
            )
        )
        objective_before = after_items
    total_runtime = time.perf_counter() - started
    return ALSReferenceModel(
        user_factors=users,
        item_factors=items,
        diagnostics=ALSFitDiagnostic(
            initial_objective=initial,
            final_objective=objective_before,
            iterations=tuple(trace),
            runtime_seconds=total_runtime,
            numerical_failures=0,
        ),
        seed=rng_seed,
        regularization=penalty,
    )


def score_factor_pairs(
    user_factors: Any,
    item_factors: Any,
    user_indices: Any,
    item_indices: Any,
) -> np.ndarray:
    """Score explicitly requested user-item pairs in float64."""

    users = _factor_matrix(user_factors, name="user_factors")
    items = _factor_matrix(
        item_factors,
        name="item_factors",
        columns=users.shape[1],
    )
    user_rows = _index_vector(
        user_indices,
        name="user_indices",
        upper_bound=users.shape[0],
    )
    item_rows = _index_vector(
        item_indices,
        name="item_indices",
        upper_bound=items.shape[0],
    )
    if user_rows.shape != item_rows.shape:
        raise ValueError("user_indices and item_indices must have equal shape")
    return np.einsum(
        "ij,ij->i",
        users[user_rows],
        items[item_rows],
        dtype=np.float64,
    )


def score_factor_block(
    user_factors: Any,
    item_factors: Any,
    user_indices: Any,
    item_indices: Any,
    *,
    maximum_score_block_bytes: int = DEFAULT_MAXIMUM_SCORE_BLOCK_BYTES,
) -> np.ndarray:
    """Score one explicitly byte-bounded user-by-item block."""

    users = _factor_matrix(user_factors, name="user_factors")
    items = _factor_matrix(
        item_factors,
        name="item_factors",
        columns=users.shape[1],
    )
    user_rows = _index_vector(
        user_indices,
        name="user_indices",
        upper_bound=users.shape[0],
    )
    item_rows = _index_vector(
        item_indices,
        name="item_indices",
        upper_bound=items.shape[0],
    )
    byte_limit = _positive_integer(
        maximum_score_block_bytes,
        name="maximum_score_block_bytes",
    )
    required = (
        int(user_rows.size)
        * int(item_rows.size)
        * np.dtype(np.float64).itemsize
    )
    if required > byte_limit:
        raise ValueError("requested score block exceeds the byte limit")
    return users[user_rows] @ items[item_rows].T


def _feature_matrix(
    item_features: Any,
    *,
    n_items: int,
    n_features: int,
) -> np.ndarray | sp.csr_matrix:
    if sp.issparse(item_features):
        if not sp.isspmatrix_csr(item_features):
            raise ValueError("item_features must be dense or canonical CSR")
        features = item_features
        if not features.has_canonical_format or not features.has_sorted_indices:
            raise ValueError("item_features must use canonical sorted CSR storage")
        if features.dtype.kind not in "fc":
            raise ValueError("item_features must use floating data")
        if np.iscomplexobj(features.data) or not np.all(np.isfinite(features.data)):
            raise ValueError("item_features must be finite and real")
        if features.shape != (n_items, n_features):
            raise ValueError("item feature shape is misaligned")
        return features.astype(np.float64, copy=False)
    features = np.asarray(item_features)
    if (
        features.ndim != 2
        or features.dtype.kind not in "fc"
        or np.iscomplexobj(features)
        or not np.all(np.isfinite(features))
    ):
        raise ValueError("item_features must be a finite floating matrix")
    if features.shape != (n_items, n_features):
        raise ValueError("item feature shape is misaligned")
    return np.asarray(features, dtype=np.float64)


def _validated_feature_sum_parameters(
    parameters: FeatureSumParameters,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(parameters, FeatureSumParameters):
        raise ValueError("parameters must be FeatureSumParameters")
    users = _factor_matrix(parameters.user_factors, name="user_factors")
    identity = _factor_matrix(
        parameters.identity_factors,
        name="identity_factors",
        columns=users.shape[1],
    )
    features = _factor_matrix(
        parameters.feature_factors,
        name="feature_factors",
        columns=users.shape[1],
    )
    bias = np.asarray(parameters.item_bias)
    if (
        bias.ndim != 1
        or bias.shape[0] != identity.shape[0]
        or bias.dtype.kind not in "fc"
        or np.iscomplexobj(bias)
        or not np.all(np.isfinite(bias))
    ):
        raise ValueError("item_bias must be a finite aligned floating vector")
    return users, identity, features, np.asarray(bias, dtype=np.float64)


def feature_sum_item_factors(
    parameters: FeatureSumParameters,
    item_features: Any,
    *,
    rho: float,
) -> np.ndarray:
    """Return ``eta_i + rho * F_i G`` in float64."""

    _, identity, feature_factors, _ = _validated_feature_sum_parameters(
        parameters
    )
    weight = _finite_scalar(rho, name="rho")
    features = _feature_matrix(
        item_features,
        n_items=identity.shape[0],
        n_features=feature_factors.shape[0],
    )
    if weight == 0.0 or feature_factors.shape[0] == 0:
        return identity.copy()
    return identity + weight * (features @ feature_factors)


def feature_sum_scores(
    parameters: FeatureSumParameters,
    item_features: Any,
    *,
    rho: float,
    user_indices: Any | None = None,
    item_indices: Any | None = None,
    maximum_score_block_bytes: int = DEFAULT_MAXIMUM_SCORE_BLOCK_BYTES,
) -> np.ndarray:
    """Score an explicitly bounded feature-sum block."""

    users, identity, _, bias = _validated_feature_sum_parameters(parameters)
    item_vectors = feature_sum_item_factors(
        parameters,
        item_features,
        rho=rho,
    )
    user_rows = (
        np.arange(users.shape[0], dtype=np.int64)
        if user_indices is None
        else _index_vector(
            user_indices,
            name="user_indices",
            upper_bound=users.shape[0],
        )
    )
    item_rows = (
        np.arange(identity.shape[0], dtype=np.int64)
        if item_indices is None
        else _index_vector(
            item_indices,
            name="item_indices",
            upper_bound=identity.shape[0],
        )
    )
    byte_limit = _positive_integer(
        maximum_score_block_bytes,
        name="maximum_score_block_bytes",
    )
    required = user_rows.size * item_rows.size * np.dtype(np.float64).itemsize
    if required > byte_limit:
        raise ValueError("requested score block exceeds the byte limit")
    return users[user_rows] @ item_vectors[item_rows].T + bias[item_rows][None, :]


def _triples(
    values: Any,
    *,
    n_users: int,
    n_items: int,
    allow_empty: bool = False,
) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 2 or array.shape[1:] != (3,):
        raise ValueError("triples must have shape (n, 3)")
    if array.dtype.kind not in "iu" or array.dtype.kind == "b":
        raise ValueError("triples must use integer indices")
    triples = array.astype(np.int64, copy=False)
    if not allow_empty and triples.shape[0] == 0:
        raise ValueError("triples must not be empty")
    if (
        np.any((triples[:, 0] < 0) | (triples[:, 0] >= n_users))
        or np.any((triples[:, 1:] < 0) | (triples[:, 1:] >= n_items))
    ):
        raise ValueError("a triple index is out of range")
    if np.any(triples[:, 1] == triples[:, 2]):
        raise ValueError("positive and negative items must differ")
    return triples


def _sigmoid_negative_margin(delta: np.ndarray) -> np.ndarray:
    result = np.empty(delta.shape, dtype=np.float64)
    nonnegative = delta >= 0.0
    exp_negative = np.exp(-delta[nonnegative])
    result[nonnegative] = exp_negative / (1.0 + exp_negative)
    exp_positive = np.exp(delta[~nonnegative])
    result[~nonnegative] = 1.0 / (1.0 + exp_positive)
    return result


def _features_are_zero(features: np.ndarray | sp.csr_matrix) -> bool:
    if sp.issparse(features):
        return features.nnz == 0 or np.all(features.data == 0.0)
    return not np.any(features != 0.0)


def bpr_loss_and_gradients(
    parameters: FeatureSumParameters,
    item_features: Any,
    triples: Any,
    *,
    rho: float,
    regularization: float,
) -> tuple[float, FeatureSumParameters]:
    """Return summed stable BPR loss and exact accumulated gradients.

    When ``rho`` is zero or the feature block is identically zero, the feature
    parameters are inactive and excluded from both the penalty and gradients.
    This makes both paths exactly equal to the identity-only specification.
    """

    users, identity, feature_factors, bias = (
        _validated_feature_sum_parameters(parameters)
    )
    weight = _finite_scalar(rho, name="rho")
    penalty = _regularization(regularization)
    features = _feature_matrix(
        item_features,
        n_items=identity.shape[0],
        n_features=feature_factors.shape[0],
    )
    sample = _triples(
        triples,
        n_users=users.shape[0],
        n_items=identity.shape[0],
    )
    content_active = (
        weight != 0.0
        and feature_factors.shape[0] > 0
        and not _features_are_zero(features)
    )
    item_vectors = (
        identity + weight * (features @ feature_factors)
        if content_active
        else identity
    )
    user_rows = sample[:, 0]
    positive_rows = sample[:, 1]
    negative_rows = sample[:, 2]
    selected_users = users[user_rows]
    delta = (
        bias[positive_rows]
        - bias[negative_rows]
        + np.einsum(
            "ij,ij->i",
            selected_users,
            item_vectors[positive_rows] - item_vectors[negative_rows],
        )
    )
    data_loss = float(np.sum(np.logaddexp(0.0, -delta)))
    regularized_norm = (
        float(np.sum(users * users))
        + float(np.sum(identity * identity))
        + float(np.sum(bias * bias))
    )
    if content_active:
        regularized_norm += float(np.sum(feature_factors * feature_factors))
    loss = data_loss + penalty * regularized_norm
    if not np.isfinite(loss):
        raise ValueError("BPR loss is nonfinite")

    coefficient = -_sigmoid_negative_margin(delta)
    user_gradient = np.zeros_like(users)
    identity_gradient = np.zeros_like(identity)
    feature_gradient = np.zeros_like(feature_factors)
    bias_gradient = np.zeros_like(bias)
    np.add.at(
        user_gradient,
        user_rows,
        coefficient[:, None]
        * (item_vectors[positive_rows] - item_vectors[negative_rows]),
    )
    np.add.at(
        identity_gradient,
        positive_rows,
        coefficient[:, None] * selected_users,
    )
    np.add.at(
        identity_gradient,
        negative_rows,
        -coefficient[:, None] * selected_users,
    )
    np.add.at(bias_gradient, positive_rows, coefficient)
    np.add.at(bias_gradient, negative_rows, -coefficient)

    if content_active:
        for row, positive, negative in zip(
            range(sample.shape[0]),
            positive_rows,
            negative_rows,
        ):
            scale = coefficient[row] * weight
            user_vector = selected_users[row]
            if sp.issparse(features):
                for item, sign in ((positive, 1.0), (negative, -1.0)):
                    start = int(features.indptr[item])
                    stop = int(features.indptr[item + 1])
                    feature_indices = features.indices[start:stop]
                    feature_values = features.data[start:stop]
                    np.add.at(
                        feature_gradient,
                        feature_indices,
                        (
                            scale
                            * sign
                            * feature_values[:, None]
                            * user_vector[None, :]
                        ),
                    )
            else:
                difference = features[positive] - features[negative]
                feature_gradient += (
                    scale * difference[:, None] * user_vector[None, :]
                )

    user_gradient += 2.0 * penalty * users
    identity_gradient += 2.0 * penalty * identity
    bias_gradient += 2.0 * penalty * bias
    if content_active:
        feature_gradient += 2.0 * penalty * feature_factors
    gradients = FeatureSumParameters(
        user_factors=user_gradient,
        identity_factors=identity_gradient,
        feature_factors=feature_gradient,
        item_bias=bias_gradient,
    )
    return float(loss), gradients


class BPRTripleSampler:
    """Stateful frozen triple stream that can continue across epochs."""

    def __init__(self, ownership: sp.spmatrix, *, seed: int) -> None:
        owned = _validate_ownership(ownership)
        training_seed = _nonnegative_integer(seed, name="seed")
        if owned.nnz == 0:
            raise ValueError(
                "cannot sample positives from an empty interaction matrix"
            )
        row_counts = np.diff(owned.indptr)
        positive_users = row_counts > 0
        if np.any(row_counts[positive_users] >= owned.shape[1]):
            raise ValueError(
                "a positive user owns the complete catalogue and has no negative"
            )
        namespace = (
            f"s1-v1-20260718:bpr:{training_seed}:triple-sampler"
        ).encode("utf-8")
        derived_seed = int.from_bytes(
            hashlib.sha256(namespace).digest()[:8],
            "big",
            signed=False,
        )
        self._ownership = owned.copy()
        self._edge_rows = np.repeat(
            np.arange(owned.shape[0], dtype=np.int64),
            row_counts,
        )
        self._rng = np.random.Generator(np.random.PCG64(derived_seed))

    def sample(self, n_samples: int) -> np.ndarray:
        """Consume the next ordered triples from the continuing stream."""

        count = _positive_integer(n_samples, name="n_samples")
        triples = np.empty((count, 3), dtype=np.int64)
        owned = self._ownership
        for row in range(count):
            edge_position = int(self._rng.integers(0, owned.nnz))
            user = int(self._edge_rows[edge_position])
            positive = int(owned.indices[edge_position])
            start = int(owned.indptr[user])
            stop = int(owned.indptr[user + 1])
            known = owned.indices[start:stop]
            while True:
                negative = int(self._rng.integers(0, owned.shape[1]))
                position = int(np.searchsorted(known, negative))
                if position >= known.size or known[position] != negative:
                    break
            triples[row] = (user, positive, negative)
        return triples


def sample_bpr_triples(
    ownership: sp.spmatrix,
    *,
    n_samples: int,
    seed: int,
) -> np.ndarray:
    """Return the first triples from the frozen namespaced stream."""

    return BPRTripleSampler(ownership, seed=seed).sample(n_samples)


def triple_stream_sha256(triples: Any) -> str:
    """Hash ordered triple rows as canonical little-endian int64 bytes."""

    array = np.asarray(triples)
    if array.ndim != 2 or array.shape[1:] != (3,):
        raise ValueError("triples must have shape (n, 3)")
    if array.dtype.kind not in "iu" or array.dtype.kind == "b":
        raise ValueError("triples must use integer indices")
    canonical = np.ascontiguousarray(array, dtype="<i8")
    digest = hashlib.sha256()
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def save_feature_sum_parameters(
    parameters: FeatureSumParameters,
    path: str | Path,
) -> None:
    """Save validated feature-sum parameters without pickle."""

    _validated_feature_sum_parameters(parameters)
    arrays = {
        "user_factors": np.asarray(parameters.user_factors),
        "identity_factors": np.asarray(parameters.identity_factors),
        "feature_factors": np.asarray(parameters.feature_factors),
        "item_bias": np.asarray(parameters.item_bias),
    }
    if any(array.dtype != np.dtype(np.float32) for array in arrays.values()):
        raise ValueError("stored feature-sum parameters must use float32")
    np.savez_compressed(
        Path(path),
        schema_version=np.asarray([SCHEMA_VERSION], dtype=np.int16),
        **arrays,
    )


def load_feature_sum_parameters(
    path: str | Path,
) -> FeatureSumParameters:
    """Load and validate feature-sum parameters without pickle."""

    with np.load(Path(path), allow_pickle=False) as payload:
        expected = {
            "schema_version",
            "user_factors",
            "identity_factors",
            "feature_factors",
            "item_bias",
        }
        if set(payload.files) != expected:
            raise ValueError("feature-sum parameter fields changed")
        schema = np.asarray(payload["schema_version"])
        if schema.shape != (1,) or int(schema[0]) != SCHEMA_VERSION:
            raise ValueError("unsupported feature-sum parameter schema")
        result = FeatureSumParameters(
            user_factors=np.asarray(payload["user_factors"]),
            identity_factors=np.asarray(payload["identity_factors"]),
            feature_factors=np.asarray(payload["feature_factors"]),
            item_bias=np.asarray(payload["item_bias"]),
        )
        if any(
            np.asarray(getattr(result, name)).dtype != np.dtype(np.float32)
            for name in (
                "user_factors",
                "identity_factors",
                "feature_factors",
                "item_bias",
            )
        ):
            raise ValueError("stored feature-sum parameters must use float32")
    _validated_feature_sum_parameters(result)
    return result


def save_model(
    parameters: FeatureSumParameters,
    path: str | Path,
) -> None:
    """Save a feature-sum parameter artifact under the public model API."""

    save_feature_sum_parameters(parameters, path)


def load_model(path: str | Path) -> FeatureSumParameters:
    """Load a feature-sum parameter artifact under the public model API."""

    return load_feature_sum_parameters(path)


def estimator_specification() -> dict[str, Any]:
    """Return the outcome-free S1.4 mathematical and serialization contract."""

    return {
        "schema_version": SCHEMA_VERSION,
        "confidence": {
            "playtime_source": PLAYTIME_CONFIDENCE_SOURCE,
            "equation": CONFIDENCE_EQUATION,
            "unobserved_confidence": 1.0,
        },
        "wrmf": {
            "score_equation": WRMF_SCORE_EQUATION,
            "accumulator_dtype": "float64",
            "stored_factor_dtype": "float32",
            "solve": "cholesky_without_explicit_inverse",
            "jitter_sequence": list(DEFAULT_JITTER_SEQUENCE),
        },
        "feature_sum_bpr": {
            "score_equation": FEATURE_SUM_SCORE_EQUATION,
            "loss_equation": BPR_LOSS_EQUATION,
            "sampler": TRIPLE_SAMPLER,
            "inactive_content_excluded_from_penalty": True,
        },
        "serialization": {
            "schema_version": SCHEMA_VERSION,
            "dense_user_item_score_matrix": False,
            "allow_pickle": False,
        },
    }
