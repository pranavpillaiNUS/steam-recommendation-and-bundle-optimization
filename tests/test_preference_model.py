"""Synthetic mathematical-oracle tests for the Stage 1 estimators."""

from __future__ import annotations

from dataclasses import fields, replace
import hashlib
import inspect
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import scipy.sparse as sp

from src import preference_model as pm


def _tiny_implicit_problem():
    ownership = sp.csr_matrix(
        np.asarray(
            [
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
    )
    playtime = sp.csr_matrix(
        np.asarray(
            [
                [0.0, 0.0, 9.0],
                [0.0, 4.0, 0.0],
            ],
            dtype=np.float32,
        )
    )
    user_factors = np.asarray(
        [[0.2, -0.3], [0.7, 0.1]],
        dtype=np.float64,
    )
    item_factors = np.asarray(
        [[0.4, 0.5], [-0.2, 0.6], [0.3, -0.7]],
        dtype=np.float64,
    )
    confidence = pm.observed_confidence(
        ownership,
        playtime,
        alpha_o=2.0,
        alpha_p=3.0,
        tau=1.0,
    )
    return ownership, playtime, confidence, user_factors, item_factors


def _dense_confidence(
    ownership: sp.csr_matrix,
    observed_confidence: sp.csr_matrix,
) -> np.ndarray:
    result = np.ones(ownership.shape, dtype=np.float64)
    rows, columns = ownership.nonzero()
    result[rows, columns] = np.asarray(
        observed_confidence[rows, columns]
    ).ravel()
    return result


def _diagnostic_value(diagnostic: Any, name: str) -> Any:
    if isinstance(diagnostic, dict):
        return diagnostic[name]
    return getattr(diagnostic, name)


def _score_block(
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    user_indices: np.ndarray,
    item_indices: np.ndarray,
    *,
    maximum_bytes: int,
) -> np.ndarray:
    parameters = inspect.signature(pm.score_factor_block).parameters
    keyword = (
        "maximum_score_block_bytes"
        if "maximum_score_block_bytes" in parameters
        else "max_block_bytes"
    )
    return pm.score_factor_block(
        user_factors,
        item_factors,
        user_indices,
        item_indices,
        **{keyword: maximum_bytes},
    )


def _parameters(
    *,
    user_factors: np.ndarray | None = None,
    identity_factors: np.ndarray | None = None,
    feature_factors: np.ndarray | None = None,
    item_bias: np.ndarray | None = None,
):
    return pm.FeatureSumParameters(
        user_factors=np.asarray(
            (
                [[0.3, -0.4], [0.2, 0.6]]
                if user_factors is None
                else user_factors
            ),
            dtype=np.float64,
        ),
        identity_factors=np.asarray(
            (
                [[0.5, 0.1], [-0.3, 0.7], [0.2, -0.6]]
                if identity_factors is None
                else identity_factors
            ),
            dtype=np.float64,
        ),
        feature_factors=np.asarray(
            (
                [[0.4, -0.2], [-0.1, 0.3]]
                if feature_factors is None
                else feature_factors
            ),
            dtype=np.float64,
        ),
        item_bias=np.asarray(
            [0.05, -0.1, 0.2] if item_bias is None else item_bias,
            dtype=np.float64,
        ),
    )


def _parameter_array(parameters: Any, name: str) -> np.ndarray:
    return np.asarray(getattr(parameters, name))


def _replace_parameter(parameters: Any, name: str, value: np.ndarray):
    return replace(parameters, **{name: value})


def _loss_and_gradients(
    parameters: Any,
    item_features: np.ndarray | sp.csr_matrix,
    triples: np.ndarray,
    *,
    rho: float,
    regularization: float,
):
    result = pm.bpr_loss_and_gradients(
        parameters,
        item_features,
        triples,
        rho=rho,
        regularization=regularization,
    )
    if isinstance(result, tuple):
        return float(result[0]), result[1]
    return float(result.loss), result.gradients


def _finite_difference(
    parameters: Any,
    item_features: np.ndarray | sp.csr_matrix,
    triples: np.ndarray,
    *,
    block: str,
    index: tuple[int, ...],
    rho: float,
    regularization: float,
    step: float = 1e-6,
) -> float:
    values = _parameter_array(parameters, block)
    left = values.copy()
    right = values.copy()
    left[index] -= step
    right[index] += step
    loss_left, _ = _loss_and_gradients(
        _replace_parameter(parameters, block, left),
        item_features,
        triples,
        rho=rho,
        regularization=regularization,
    )
    loss_right, _ = _loss_and_gradients(
        _replace_parameter(parameters, block, right),
        item_features,
        triples,
        rho=rho,
        regularization=regularization,
    )
    return (loss_right - loss_left) / (2.0 * step)


def _reference_triple_stream(
    ownership: sp.csr_matrix,
    *,
    n_samples: int,
    training_seed: int,
) -> np.ndarray:
    canonical = ownership.copy()
    canonical.sum_duplicates()
    canonical.sort_indices()
    edge_rows = np.repeat(
        np.arange(canonical.shape[0], dtype=np.int64),
        np.diff(canonical.indptr),
    )
    namespace = (
        f"s1-v1-20260718:bpr:{training_seed}:triple-sampler"
    ).encode("utf-8")
    derived_seed = int.from_bytes(
        hashlib.sha256(namespace).digest()[:8],
        "big",
        signed=False,
    )
    rng = np.random.Generator(np.random.PCG64(derived_seed))
    triples = np.empty((n_samples, 3), dtype=np.int64)
    for row in range(n_samples):
        edge_position = int(rng.integers(0, canonical.nnz))
        user = int(edge_rows[edge_position])
        positive = int(canonical.indices[edge_position])
        start = int(canonical.indptr[user])
        stop = int(canonical.indptr[user + 1])
        known = canonical.indices[start:stop]
        while True:
            negative = int(rng.integers(0, canonical.shape[1]))
            position = int(np.searchsorted(known, negative))
            if position >= known.size or known[position] != negative:
                break
        triples[row] = (user, positive, negative)
    return triples


def test_observed_confidence_has_exact_edge_and_cap_semantics():
    ownership, _, confidence, _, _ = _tiny_implicit_problem()
    np.testing.assert_array_equal(confidence.indptr, ownership.indptr)
    np.testing.assert_array_equal(confidence.indices, ownership.indices)
    assert confidence.dtype == np.float32
    assert float(confidence[0, 0]) == 3.0
    assert float(confidence[0, 2]) == 6.0
    assert float(confidence[1, 1]) == 6.0
    assert float(confidence[1, 0]) == 0.0

    ownership_only = pm.observed_confidence(
        ownership,
        sp.csr_matrix(ownership.shape, dtype=np.float32),
        alpha_o=20.0,
        alpha_p=0.0,
        tau=0.0,
    )
    np.testing.assert_array_equal(
        ownership_only.data,
        np.full(ownership.nnz, 21.0, dtype=np.float32),
    )


@pytest.mark.parametrize(
    "alpha_o,alpha_p,tau",
    [
        (0.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (1.0, -1.0, 0.0),
        (1.0, 1.0, -1.0),
        (np.nan, 0.0, 0.0),
        (1.0, np.inf, 0.0),
    ],
)
def test_observed_confidence_rejects_invalid_parameters(
    alpha_o,
    alpha_p,
    tau,
):
    ownership = sp.csr_matrix([[1.0]], dtype=np.float32)
    playtime = sp.csr_matrix([[0.0]], dtype=np.float32)
    with pytest.raises(ValueError):
        pm.observed_confidence(
            ownership,
            playtime,
            alpha_o=alpha_o,
            alpha_p=alpha_p,
            tau=tau,
        )


def test_wrmf_objective_matches_a_tiny_dense_oracle():
    ownership, _, confidence, users, items = _tiny_implicit_problem()
    regularization = 0.17
    preference = ownership.toarray().astype(np.float64)
    dense_confidence = _dense_confidence(ownership, confidence)
    residual = preference - users @ items.T
    expected = float(
        np.sum(dense_confidence * residual * residual)
        + regularization
        * (np.sum(users * users) + np.sum(items * items))
    )
    actual = pm.wrmf_objective(
        ownership,
        confidence,
        users,
        items,
        regularization=regularization,
    )
    assert actual == pytest.approx(expected, rel=2e-13, abs=2e-13)


def test_user_and_item_normal_equations_match_dense_oracles():
    ownership, _, confidence, users, items = _tiny_implicit_problem()
    regularization = 0.23
    preference = ownership.toarray().astype(np.float64)
    dense_confidence = _dense_confidence(ownership, confidence)

    for user in range(ownership.shape[0]):
        expected_a = (
            items.T @ (dense_confidence[user, :, None] * items)
            + regularization * np.eye(items.shape[1])
        )
        expected_b = items.T @ (
            dense_confidence[user] * preference[user]
        )
        actual_a, actual_b = pm.user_normal_equation(
            ownership,
            confidence,
            items,
            user,
            regularization=regularization,
        )
        np.testing.assert_allclose(actual_a, expected_a, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(actual_b, expected_b, rtol=1e-13, atol=1e-13)

    for item in range(ownership.shape[1]):
        expected_a = (
            users.T @ (dense_confidence[:, item, None] * users)
            + regularization * np.eye(users.shape[1])
        )
        expected_b = users.T @ (
            dense_confidence[:, item] * preference[:, item]
        )
        actual_a, actual_b = pm.item_normal_equation(
            ownership,
            confidence,
            users,
            item,
            regularization=regularization,
        )
        np.testing.assert_allclose(actual_a, expected_a, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(actual_b, expected_b, rtol=1e-13, atol=1e-13)


def test_stable_spd_solve_reports_residual_and_controlled_jitter():
    matrix = np.asarray([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64)
    target = np.asarray([1.0, 2.0], dtype=np.float64)
    solution, diagnostic = pm.solve_spd(
        matrix,
        target,
        jitter_sequence=(0.0, 1e-8),
    )
    np.testing.assert_allclose(solution, np.linalg.solve(matrix, target))
    assert _diagnostic_value(diagnostic, "jitter") == 0.0
    assert _diagnostic_value(diagnostic, "residual_norm") < 1e-12

    singular = np.asarray([[1.0, 1.0], [1.0, 1.0]], dtype=np.float64)
    solution, diagnostic = pm.solve_spd(
        singular,
        np.asarray([1.0, 1.0]),
        jitter_sequence=(0.0, 1e-8),
    )
    assert np.isfinite(solution).all()
    assert _diagnostic_value(diagnostic, "jitter") == pytest.approx(1e-8)
    assert _diagnostic_value(diagnostic, "residual_norm") < 1e-6

    _, small_target_diagnostic = pm.solve_spd(
        singular,
        np.asarray([0.1, 0.1]),
        jitter_sequence=(0.0, 1e-8),
    )
    assert small_target_diagnostic.original_relative_residual_norm == (
        pytest.approx(
            small_target_diagnostic.original_residual_norm
            / max(1.0, np.linalg.norm([0.1, 0.1]))
        )
    )


def test_block_updates_match_oracles_and_do_not_increase_objective():
    ownership, _, confidence, users, items = _tiny_implicit_problem()
    regularization = 0.2
    updated_users, user_diagnostics = pm.update_user_factors(
        ownership,
        confidence,
        items,
        regularization=regularization,
    )
    for user in range(ownership.shape[0]):
        matrix, target = pm.user_normal_equation(
            ownership,
            confidence,
            items,
            user,
            regularization=regularization,
        )
        np.testing.assert_allclose(
            updated_users[user],
            np.linalg.solve(matrix, target),
            rtol=1e-11,
            atol=1e-11,
        )
    assert all(
        _diagnostic_value(row, "residual_norm") < 1e-9
        for row in user_diagnostics
    )
    before = pm.wrmf_objective(
        ownership,
        confidence,
        users,
        items,
        regularization=regularization,
    )
    after_users = pm.wrmf_objective(
        ownership,
        confidence,
        updated_users,
        items,
        regularization=regularization,
    )
    assert after_users <= before + 1e-12

    updated_items, item_diagnostics = pm.update_item_factors(
        ownership,
        confidence,
        updated_users,
        regularization=regularization,
    )
    for item in range(ownership.shape[1]):
        matrix, target = pm.item_normal_equation(
            ownership,
            confidence,
            updated_users,
            item,
            regularization=regularization,
        )
        np.testing.assert_allclose(
            updated_items[item],
            np.linalg.solve(matrix, target),
            rtol=1e-11,
            atol=1e-11,
        )
    assert all(
        _diagnostic_value(row, "residual_norm") < 1e-9
        for row in item_diagnostics
    )
    after_items = pm.wrmf_objective(
        ownership,
        confidence,
        updated_users,
        updated_items,
        regularization=regularization,
    )
    assert after_items <= after_users + 1e-12


def test_reference_als_is_deterministic_and_records_block_descent_diagnostics():
    ownership, _, confidence, _, _ = _tiny_implicit_problem()
    first = pm.fit_als_reference(
        ownership,
        confidence,
        factors=2,
        regularization=0.2,
        iterations=3,
        seed=104729,
    )
    second = pm.fit_als_reference(
        ownership,
        confidence,
        factors=2,
        regularization=0.2,
        iterations=3,
        seed=104729,
    )
    np.testing.assert_array_equal(first.user_factors, second.user_factors)
    np.testing.assert_array_equal(first.item_factors, second.item_factors)
    assert first.user_factors.dtype == np.float32
    assert first.item_factors.dtype == np.float32
    assert first.diagnostics.initial_objective == (
        second.diagnostics.initial_objective
    )
    assert first.diagnostics.final_objective == (
        second.diagnostics.final_objective
    )
    assert first.diagnostics.numerical_failures == 0
    assert len(first.diagnostics.iterations) == 3

    previous = first.diagnostics.initial_objective
    for index, row in enumerate(first.diagnostics.iterations):
        assert row.iteration == index
        assert np.isfinite(row.objective_before)
        assert np.isfinite(row.objective_after_users)
        assert np.isfinite(row.objective_after_items)
        assert row.objective_before == pytest.approx(previous)
        assert row.objective_after_users <= row.objective_before + 1e-5
        assert row.objective_after_items <= row.objective_after_users + 1e-5
        assert len(row.user_solve_diagnostics) == ownership.shape[0]
        assert len(row.item_solve_diagnostics) == ownership.shape[1]
        assert max(
            diagnostic.relative_residual_norm
            for diagnostic in (
                row.user_solve_diagnostics
                + row.item_solve_diagnostics
            )
        ) < 1e-9
        assert row.runtime_seconds >= 0.0
        previous = row.objective_after_items
    assert first.diagnostics.final_objective == pytest.approx(previous)
    assert first.diagnostics.runtime_seconds >= 0.0


def test_factor_pair_and_bounded_block_scores_match_direct_products():
    users = np.asarray(
        [[0.2, 0.3], [-0.4, 0.5], [0.7, -0.1]],
        dtype=np.float64,
    )
    items = np.asarray(
        [[0.1, 0.8], [0.6, -0.2], [-0.3, 0.4], [0.2, 0.9]],
        dtype=np.float64,
    )
    user_indices = np.asarray([2, 0, 2, 1], dtype=np.int64)
    item_indices = np.asarray([3, 1, 0, 2], dtype=np.int64)
    pairs = pm.score_factor_pairs(
        users,
        items,
        user_indices,
        item_indices,
    )
    expected_pairs = np.sum(
        users[user_indices] * items[item_indices],
        axis=1,
    )
    np.testing.assert_allclose(pairs, expected_pairs)

    block_users = np.asarray([2, 0], dtype=np.int64)
    block_items = np.asarray([3, 1, 0], dtype=np.int64)
    expected_block = users[block_users] @ items[block_items].T
    actual_block = _score_block(
        users,
        items,
        block_users,
        block_items,
        maximum_bytes=expected_block.nbytes,
    )
    np.testing.assert_allclose(actual_block, expected_block)
    with pytest.raises(ValueError):
        _score_block(
            users,
            items,
            block_users,
            block_items,
            maximum_bytes=expected_block.nbytes - 1,
        )


def test_popularity_is_an_exact_integer_training_count():
    ownership = sp.csr_matrix(
        np.asarray(
            [[1, 0, 1], [1, 1, 0], [0, 1, 0]],
            dtype=np.float32,
        )
    )
    scores = pm.popularity_scores(ownership)
    assert scores.dtype == np.int64
    np.testing.assert_array_equal(scores, np.asarray([2, 2, 1]))


def test_feature_sum_item_vectors_and_scores_match_direct_equations():
    parameters = _parameters()
    features = np.asarray(
        [[1.0, 0.0], [0.25, 0.75], [0.0, 1.0]],
        dtype=np.float64,
    )
    rho = 0.6
    expected_items = (
        parameters.identity_factors
        + rho * features @ parameters.feature_factors
    )
    actual_items = pm.feature_sum_item_factors(
        parameters,
        features,
        rho=rho,
    )
    np.testing.assert_allclose(actual_items, expected_items)

    expected_scores = (
        parameters.user_factors @ expected_items.T
        + parameters.item_bias[None, :]
    )
    actual_scores = pm.feature_sum_scores(
        parameters,
        features,
        rho=rho,
    )
    np.testing.assert_allclose(actual_scores, expected_scores)


@pytest.mark.parametrize(
    "block,index",
    [
        ("user_factors", (0, 1)),
        ("user_factors", (1, 0)),
        ("identity_factors", (0, 0)),
        ("identity_factors", (2, 1)),
        ("feature_factors", (0, 1)),
        ("feature_factors", (1, 0)),
        ("item_bias", (0,)),
        ("item_bias", (2,)),
    ],
)
def test_bpr_gradients_match_finite_differences_for_every_parameter_block(
    block,
    index,
):
    parameters = _parameters()
    features = np.asarray(
        [[1.0, 0.0], [0.25, 0.75], [0.0, 1.0]],
        dtype=np.float64,
    )
    triples = np.asarray(
        [[0, 0, 1], [1, 2, 0], [0, 0, 2]],
        dtype=np.int64,
    )
    rho = 0.7
    regularization = 0.031
    _, gradients = _loss_and_gradients(
        parameters,
        features,
        triples,
        rho=rho,
        regularization=regularization,
    )
    analytic = _parameter_array(gradients, block)[index]
    numerical = _finite_difference(
        parameters,
        features,
        triples,
        block=block,
        index=index,
        rho=rho,
        regularization=regularization,
    )
    assert analytic == pytest.approx(numerical, rel=2e-5, abs=2e-6)


def test_repeated_bpr_indices_accumulate_instead_of_overwriting():
    parameters = _parameters()
    features = np.asarray(
        [[1.0, 0.0], [0.25, 0.75], [0.0, 1.0]],
        dtype=np.float64,
    )
    triples = np.asarray(
        [[0, 0, 1], [0, 0, 2], [0, 0, 1]],
        dtype=np.int64,
    )
    _, joint = _loss_and_gradients(
        parameters,
        features,
        triples,
        rho=0.8,
        regularization=0.0,
    )
    separate = [
        _loss_and_gradients(
            parameters,
            features,
            triples[index : index + 1],
            rho=0.8,
            regularization=0.0,
        )[1]
        for index in range(triples.shape[0])
    ]
    for field in fields(parameters):
        name = field.name
        expected = sum(
            (_parameter_array(gradient, name) for gradient in separate),
            np.zeros_like(_parameter_array(parameters, name)),
        )
        np.testing.assert_allclose(
            _parameter_array(joint, name),
            expected,
            rtol=1e-13,
            atol=1e-13,
        )


def test_bpr_loss_and_gradients_are_finite_at_extreme_margins():
    parameters = _parameters(
        user_factors=[[1e4], [1e4]],
        identity_factors=[[1e4], [-1e4], [0.0]],
        feature_factors=[[1e4]],
        item_bias=[1e4, -1e4, 0.0],
    )
    features = np.asarray([[1.0], [0.0], [-1.0]], dtype=np.float64)
    triples = np.asarray([[0, 0, 1], [1, 1, 0]], dtype=np.int64)
    loss, gradients = _loss_and_gradients(
        parameters,
        features,
        triples,
        rho=1.0,
        regularization=1e-8,
    )
    assert np.isfinite(loss)
    for field in fields(parameters):
        assert np.isfinite(
            _parameter_array(gradients, field.name)
        ).all()


def test_dense_and_csr_feature_paths_are_identical():
    parameters = _parameters()
    dense = np.asarray(
        [[1.0, 0.0], [0.25, 0.75], [0.0, 1.0]],
        dtype=np.float64,
    )
    sparse = sp.csr_matrix(dense)
    triples = np.asarray([[0, 0, 1], [1, 2, 0]], dtype=np.int64)
    np.testing.assert_allclose(
        pm.feature_sum_item_factors(parameters, dense, rho=0.8),
        pm.feature_sum_item_factors(parameters, sparse, rho=0.8),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        pm.feature_sum_scores(parameters, dense, rho=0.8),
        pm.feature_sum_scores(parameters, sparse, rho=0.8),
        rtol=0.0,
        atol=0.0,
    )
    dense_loss, dense_gradient = _loss_and_gradients(
        parameters,
        dense,
        triples,
        rho=0.8,
        regularization=0.04,
    )
    sparse_loss, sparse_gradient = _loss_and_gradients(
        parameters,
        sparse,
        triples,
        rho=0.8,
        regularization=0.04,
    )
    assert sparse_loss == dense_loss
    for field in fields(parameters):
        np.testing.assert_allclose(
            _parameter_array(sparse_gradient, field.name),
            _parameter_array(dense_gradient, field.name),
            rtol=0.0,
            atol=0.0,
        )


def test_rho_zero_and_zero_column_features_equal_identity_with_regularization():
    parameters = _parameters()
    features = np.asarray(
        [[1.0, 0.0], [0.25, 0.75], [0.0, 1.0]],
        dtype=np.float64,
    )
    triples = np.asarray([[0, 0, 1], [1, 2, 0]], dtype=np.int64)
    identity_parameters = _parameters(
        feature_factors=np.empty(
            (0, parameters.user_factors.shape[1]),
            dtype=np.float64,
        )
    )
    no_features = np.empty(
        (parameters.identity_factors.shape[0], 0),
        dtype=np.float64,
    )
    regularization = 0.09
    identity_loss, identity_gradient = _loss_and_gradients(
        identity_parameters,
        no_features,
        triples,
        rho=0.0,
        regularization=regularization,
    )
    rho_zero_loss, rho_zero_gradient = _loss_and_gradients(
        parameters,
        features,
        triples,
        rho=0.0,
        regularization=regularization,
    )
    zero_feature_loss, zero_feature_gradient = _loss_and_gradients(
        identity_parameters,
        no_features,
        triples,
        rho=1.0,
        regularization=regularization,
    )

    assert rho_zero_loss == pytest.approx(identity_loss, abs=1e-14)
    assert zero_feature_loss == pytest.approx(identity_loss, abs=1e-14)
    for block in ("user_factors", "identity_factors", "item_bias"):
        expected = _parameter_array(identity_gradient, block)
        np.testing.assert_allclose(
            _parameter_array(rho_zero_gradient, block),
            expected,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            _parameter_array(zero_feature_gradient, block),
            expected,
            atol=1e-14,
        )
    np.testing.assert_array_equal(
        rho_zero_gradient.feature_factors,
        np.zeros_like(parameters.feature_factors),
    )
    assert zero_feature_gradient.feature_factors.shape == (0, 2)


def test_feature_sum_score_matrix_obeys_the_centered_rank_bound():
    parameters = _parameters()
    features = np.asarray(
        [[1.0, 0.0], [0.25, 0.75], [0.0, 1.0]],
        dtype=np.float64,
    )
    scores = pm.feature_sum_scores(parameters, features, rho=0.7)
    centered = scores - parameters.item_bias[None, :]
    assert np.linalg.matrix_rank(centered, tol=1e-10) <= (
        parameters.user_factors.shape[1]
    )


def test_triple_sampling_is_deterministic_hashable_and_rejects_known_positives():
    ownership = sp.csr_matrix(
        np.asarray(
            [
                [1, 0, 1, 0],
                [0, 1, 0, 0],
                [1, 0, 0, 1],
            ],
            dtype=np.float32,
        )
    )
    first = pm.sample_bpr_triples(
        ownership,
        n_samples=100,
        seed=104729,
    )
    second = pm.sample_bpr_triples(
        ownership,
        n_samples=100,
        seed=104729,
    )
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(
        first,
        _reference_triple_stream(
            ownership,
            n_samples=100,
            training_seed=104729,
        ),
    )
    continuing = pm.BPRTripleSampler(ownership, seed=104729)
    np.testing.assert_array_equal(
        np.vstack((continuing.sample(40), continuing.sample(60))),
        first,
    )
    assert pm.triple_stream_sha256(first) == pm.triple_stream_sha256(second)
    expected_hash = hashlib.sha256(
        np.ascontiguousarray(first, dtype="<i8").tobytes(order="C")
    ).hexdigest()
    assert pm.triple_stream_sha256(first) == expected_hash
    assert first.shape == (100, 3)
    assert first.dtype == np.int64
    assert np.all(first[:, 1] != first[:, 2])
    for user, positive, negative in first:
        assert ownership[int(user), int(positive)] == 1.0
        assert ownership[int(user), int(negative)] == 0.0

    different = pm.sample_bpr_triples(
        ownership,
        n_samples=100,
        seed=130363,
    )
    assert pm.triple_stream_sha256(different) != (
        pm.triple_stream_sha256(first)
    )


def test_triple_sampling_fails_when_the_full_catalogue_is_owned():
    ownership = sp.csr_matrix(
        np.ones((1, 4), dtype=np.float32)
    )
    with pytest.raises(ValueError, match="negative|catalogue|owns"):
        pm.sample_bpr_triples(
            ownership,
            n_samples=1,
            seed=104729,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "nonfinite_user",
        "wrong_feature_rows",
        "bad_triple_shape",
        "out_of_range_triple",
        "equal_positive_negative",
        "negative_regularization",
        "nonfinite_rho",
    ],
)
def test_bpr_validation_failures_are_explicit(mutation):
    parameters = _parameters()
    features = np.asarray(
        [[1.0, 0.0], [0.25, 0.75], [0.0, 1.0]],
        dtype=np.float64,
    )
    triples = np.asarray([[0, 0, 1]], dtype=np.int64)
    rho = 0.7
    regularization = 0.03

    if mutation == "nonfinite_user":
        values = parameters.user_factors.copy()
        values[0, 0] = np.nan
        parameters = _replace_parameter(
            parameters,
            "user_factors",
            values,
        )
    elif mutation == "wrong_feature_rows":
        features = features[:-1]
    elif mutation == "bad_triple_shape":
        triples = np.asarray([[0, 1]], dtype=np.int64)
    elif mutation == "out_of_range_triple":
        triples = np.asarray([[9, 0, 1]], dtype=np.int64)
    elif mutation == "equal_positive_negative":
        triples = np.asarray([[0, 1, 1]], dtype=np.int64)
    elif mutation == "negative_regularization":
        regularization = -0.1
    elif mutation == "nonfinite_rho":
        rho = np.inf

    with pytest.raises(ValueError):
        _loss_and_gradients(
            parameters,
            features,
            triples,
            rho=rho,
            regularization=regularization,
        )


def test_wrmf_validation_rejects_shape_binary_and_finiteness_failures():
    ownership, _, confidence, users, items = _tiny_implicit_problem()
    with pytest.raises(ValueError):
        pm.wrmf_objective(
            ownership[:, :-1],
            confidence,
            users,
            items,
            regularization=0.1,
        )
    nonbinary = ownership.copy()
    nonbinary.data[0] = 2.0
    with pytest.raises(ValueError):
        pm.wrmf_objective(
            nonbinary,
            confidence,
            users,
            items,
            regularization=0.1,
        )
    bad_users = users.copy()
    bad_users[0, 0] = np.nan
    with pytest.raises(ValueError):
        pm.wrmf_objective(
            ownership,
            confidence,
            bad_users,
            items,
            regularization=0.1,
        )
    with pytest.raises(ValueError):
        pm.wrmf_objective(
            ownership,
            confidence,
            users,
            items,
            regularization=-0.1,
        )


def test_feature_sum_serialization_reproduces_scores_when_supported(
    tmp_path: Path,
):
    save = getattr(pm, "save_model", None)
    load = getattr(pm, "load_model", None)
    if save is None or load is None:
        pytest.skip("feature-sum parameter serialization is not implemented at S1.4")

    baseline = _parameters()
    parameters = pm.FeatureSumParameters(
        user_factors=baseline.user_factors.astype(np.float32),
        identity_factors=baseline.identity_factors.astype(np.float32),
        feature_factors=baseline.feature_factors.astype(np.float32),
        item_bias=baseline.item_bias.astype(np.float32),
    )
    features = sp.csr_matrix(
        np.asarray(
            [[1.0, 0.0], [0.25, 0.75], [0.0, 1.0]],
            dtype=np.float64,
        )
    )
    expected = pm.feature_sum_scores(parameters, features, rho=0.7)
    path = tmp_path / "feature_sum_parameters.npz"
    save(parameters, path)
    with np.load(path, allow_pickle=False) as payload:
        for name in (
            "user_factors",
            "identity_factors",
            "feature_factors",
            "item_bias",
        ):
            assert payload[name].dtype == np.float32
    restored = load(path)
    actual = pm.feature_sum_scores(restored, features, rho=0.7)
    np.testing.assert_array_equal(actual, expected)
    for field in fields(parameters):
        np.testing.assert_array_equal(
            _parameter_array(restored, field.name),
            _parameter_array(parameters, field.name),
        )
