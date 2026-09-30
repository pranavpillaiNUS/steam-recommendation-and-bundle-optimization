"""Independent numerical and behavioral checks for the prospective trainer."""

import math

import numpy as np
import pytest
import scipy.sparse as sp

from src.preference_model import FeatureSumParameters, bpr_loss_and_gradients
from src.stage1_backend import initialize_feature_sum_parameters
from src.stage1_training import fit_minibatch_bpr, minibatch_loss_and_gradients


FIELDS = ("user_factors", "identity_factors", "feature_factors", "item_bias")


def tiny_problem():
    ownership = sp.csr_matrix([[1, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 1]])
    genres = sp.csr_matrix([[1, 0], [1, 0], [0, 1], [0.5, 0.5]])
    return ownership, genres


def kwargs(**overrides):
    return {
        "cycle_id": "prospective-unit-test", "training_seed": 17,
        "factors": 3, "regularization": 0.002, "learning_rate": 0.05,
        "epochs": 4, "samples_per_epoch": 101, "batch_size": 16,
        "include_genre": True, "diagnostic_sample_size": 127, **overrides,
    }


def expanded(gradient, parameters):
    arrays = {name: np.zeros_like(getattr(parameters, name), dtype=float) for name in FIELDS}
    arrays["user_factors"][gradient.user_rows] = gradient.user_factors
    arrays["identity_factors"][gradient.item_rows] = gradient.identity_factors
    arrays["item_bias"][gradient.item_rows] = gradient.item_bias
    arrays["feature_factors"][:] = gradient.feature_factors
    return arrays


@pytest.mark.parametrize("include_genre", [False, True])
def test_batch_data_gradient_matches_independent_frozen_equation_oracle(include_genre):
    _, genres = tiny_problem()
    parameters = initialize_feature_sum_parameters(
        n_users=3, n_items=4, n_features=2 if include_genre else 0,
        factors=2, cycle_id="oracle", training_seed=4,
    )
    # Repeated users and item rows verify accumulated rather than overwritten gradients.
    triples = np.asarray([[0, 0, 1], [0, 2, 1], [1, 1, 2], [2, 3, 1], [0, 0, 1]])
    loss, gradient = minibatch_loss_and_gradients(parameters, genres, triples, regularization=0)
    oracle_features = genres if include_genre else sp.csr_matrix((4, 0))
    oracle_loss, oracle = bpr_loss_and_gradients(
        parameters, oracle_features, triples, rho=float(include_genre), regularization=0,
    )
    assert loss == pytest.approx(oracle_loss / len(triples), abs=1e-14)
    for name, values in expanded(gradient, parameters).items():
        np.testing.assert_allclose(values, getattr(oracle, name) / len(triples), rtol=0, atol=1e-14)


def test_sampled_regularization_and_all_gradients_match_finite_differences():
    _, genres = tiny_problem()
    initial = initialize_feature_sum_parameters(
        n_users=3, n_items=4, n_features=2, factors=2, cycle_id="finite", training_seed=9,
    )
    parameters = FeatureSumParameters(**{
        name: np.asarray(getattr(initial, name), dtype=float) for name in FIELDS
    })
    triples = np.asarray([[0, 0, 1], [0, 0, 1], [1, 1, 2]])
    penalty = 0.03
    loss, gradient = minibatch_loss_and_gradients(parameters, genres, triples, regularization=penalty)
    data_loss = bpr_loss_and_gradients(parameters, genres, triples, rho=1, regularization=0)[0]
    u, i, j = triples.T
    sampled_norm = np.mean(
        np.sum(parameters.user_factors[u] ** 2, axis=1)
        + np.sum(parameters.identity_factors[i] ** 2, axis=1)
        + np.sum(parameters.identity_factors[j] ** 2, axis=1)
        + parameters.item_bias[i] ** 2 + parameters.item_bias[j] ** 2
    )
    expected_loss = data_loss / len(triples) + penalty * (sampled_norm + np.sum(parameters.feature_factors ** 2))
    assert loss == pytest.approx(expected_loss, abs=1e-14)
    step = 1e-6
    for name, analytic in expanded(gradient, parameters).items():
        array = getattr(parameters, name)
        numeric = np.zeros_like(array)
        for index in np.ndindex(array.shape):
            saved = array[index]
            array[index] = saved + step
            plus = minibatch_loss_and_gradients(parameters, genres, triples, regularization=penalty)[0]
            array[index] = saved - step
            minus = minibatch_loss_and_gradients(parameters, genres, triples, regularization=penalty)[0]
            array[index] = saved
            numeric[index] = (plus - minus) / (2 * step)
        np.testing.assert_allclose(analytic, numeric, rtol=1e-6, atol=1e-9)


def test_fit_repeats_counts_updates_and_improves_fixed_training_probe():
    ownership, genres = tiny_problem()
    first = fit_minibatch_bpr(ownership, genres, **kwargs())
    again = fit_minibatch_bpr(ownership, genres, **kwargs())
    for name in FIELDS:
        np.testing.assert_array_equal(getattr(first.parameters, name), getattr(again.parameters, name))
        assert getattr(first.parameters, name).dtype == np.float32
    assert first.diagnostics["update_count"] == 4 * math.ceil(101 / 16)
    assert first.diagnostics["epochs"][-1]["fixed_training_probe_objective"] < first.diagnostics["initial_fixed_training_probe_objective"]
    assert first.diagnostics["regularization_compatible_with_frozen_v2"] is False
    assert [x["triple_stream_sha256"] for x in first.diagnostics["epochs"]] == [
        x["triple_stream_sha256"] for x in again.diagnostics["epochs"]
    ]


def test_batch_and_genre_changes_preserve_the_optimization_triple_stream():
    ownership, genres = tiny_problem()
    identity = fit_minibatch_bpr(ownership, genres, **kwargs(include_genre=False))
    genre = fit_minibatch_bpr(ownership, genres, **kwargs(batch_size=20))
    assert [x["triple_stream_sha256"] for x in identity.diagnostics["epochs"]] == [
        x["triple_stream_sha256"] for x in genre.diagnostics["epochs"]
    ]
    assert identity.parameters.feature_factors.shape == (0, 3)


def test_sampler_uses_training_negatives_and_empty_users_stay_unmodified(monkeypatch):
    import src.stage1_training as module

    ownership, genres = tiny_problem()
    ownership = sp.vstack((ownership, sp.csr_matrix((1, 4))), format="csr")
    original = module._batch_gradient
    inspected = []

    def audit(parameters, features, triples, regularization):
        u, i, j = triples.T
        assert np.all(np.asarray(ownership[u, i]).ravel() == 1)
        assert np.all(np.asarray(ownership[u, j]).ravel() == 0)
        inspected.append(len(triples))
        return original(parameters, features, triples, regularization)

    monkeypatch.setattr(module, "_batch_gradient", audit)
    fitted = fit_minibatch_bpr(ownership, genres, **kwargs())
    initial = initialize_feature_sum_parameters(
        n_users=4, n_items=4, n_features=2, factors=3,
        cycle_id=kwargs()["cycle_id"], training_seed=17,
    )
    np.testing.assert_array_equal(fitted.parameters.user_factors[-1], initial.user_factors[-1])
    assert inspected


@pytest.mark.parametrize("option,value", [
    ("batch_size", 0), ("batch_size", True), ("epochs", 1.5),
    ("learning_rate", float("nan")), ("regularization", float("inf")),
    ("regularization", -0.1), ("training_seed", -1), ("include_genre", 1),
    ("diagnostic_sample_size", 0), ("adagrad_epsilon", 0),
])
def test_invalid_optimization_controls_fail_before_training(option, value):
    ownership, genres = tiny_problem()
    with pytest.raises(ValueError):
        fit_minibatch_bpr(ownership, genres, **kwargs(**{option: value}))


@pytest.mark.parametrize("values", [
    [[0, 0]], [[1, 1]], [[float("nan"), 0]], [[-1, 0]], [[2, 0]],
])
def test_invalid_training_edges_fail(values):
    with pytest.raises(ValueError):
        fit_minibatch_bpr(sp.csr_matrix(values), sp.csr_matrix((2, 0)), **kwargs())


def test_empty_content_block_matches_identity_path():
    ownership, _ = tiny_problem()
    empty_genres = sp.csr_matrix((4, 2))
    a = fit_minibatch_bpr(ownership, empty_genres, **kwargs(include_genre=True))
    b = fit_minibatch_bpr(ownership, empty_genres, **kwargs(include_genre=False))
    for name in FIELDS:
        np.testing.assert_array_equal(getattr(a.parameters, name), getattr(b.parameters, name))
    assert a.diagnostics["content_active"] is False


def test_nonfinite_update_is_rejected():
    ownership, genres = tiny_problem()
    with pytest.raises(FloatingPointError):
        fit_minibatch_bpr(ownership, genres, **kwargs(learning_rate=1e300))
