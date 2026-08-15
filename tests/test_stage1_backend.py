import numpy as np
import pytest
import scipy.sparse as sp

from src.preference_model import (
    bpr_loss_and_gradients,
    feature_sum_item_factors,
    score_factor_block,
)
from src.stage1_backend import (
    _bpr_epoch_gradient,
    CycleBPRTripleSampler,
    construct_fold_in_triples,
    fit_feature_sum_bpr,
    fit_implicit_als,
    fold_in_als,
    fold_in_bpr_user,
    load_parameter_archive,
    save_parameter_archive,
    initialize_feature_sum_parameters,
)


def _tiny_problem():
    ownership = sp.csr_matrix(
        np.asarray(
            [[1, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 1]],
            dtype=np.float32,
        )
    )
    playtime = sp.csr_matrix(
        np.asarray(
            [[2, 0, 0, 0], [0, 5, 0, 0], [1, 0, 0, 3]],
            dtype=np.float32,
        )
    )
    genres = sp.csr_matrix(
        np.asarray(
            [[1, 0], [1, 0], [0, 1], [0.5, 0.5]],
            dtype=np.float32,
        )
    )
    return ownership, playtime, genres


def test_cycle_sampler_is_continuing_deterministic_and_cycle_separated():
    ownership, _, _ = _tiny_problem()
    first = CycleBPRTripleSampler(
        ownership, cycle_id="cycle-a", training_seed=11
    )
    combined = np.vstack((first.sample(17), first.sample(23)))
    repeated = CycleBPRTripleSampler(
        ownership, cycle_id="cycle-a", training_seed=11
    ).sample(40)
    different = CycleBPRTripleSampler(
        ownership, cycle_id="cycle-b", training_seed=11
    ).sample(40)
    np.testing.assert_array_equal(combined, repeated)
    assert not np.array_equal(combined, different)
    for user, _, negative in combined:
        assert ownership[int(user), int(negative)] == 0


def test_bpr_fit_reproduces_and_genre_toggle_preserves_shared_initial_stream():
    ownership, _, genres = _tiny_problem()
    kwargs = dict(
        cycle_id="test-cycle",
        training_seed=17,
        factors=3,
        regularization=0.001,
        learning_rate=0.05,
        epochs=2,
        samples_per_epoch=100,
    )
    identity = fit_feature_sum_bpr(
        ownership, genres, include_genre=False, **kwargs
    )
    repeated = fit_feature_sum_bpr(
        ownership, genres, include_genre=False, **kwargs
    )
    for name in ("user_factors", "identity_factors", "item_bias"):
        np.testing.assert_array_equal(
            getattr(identity.parameters, name), getattr(repeated.parameters, name)
        )
    genre = fit_feature_sum_bpr(
        ownership, genres, include_genre=True, **kwargs
    )
    assert [x["triple_stream_sha256"] for x in identity.diagnostics["epochs"]] == [
        x["triple_stream_sha256"] for x in genre.diagnostics["epochs"]
    ]
    assert genre.parameters.feature_factors.shape == (2, 3)


def test_production_epoch_gradient_matches_the_independent_equation_oracle():
    _, _, genres = _tiny_problem()
    parameters = initialize_feature_sum_parameters(
        n_users=3,
        n_items=4,
        n_features=2,
        factors=2,
        cycle_id="oracle-cycle",
        training_seed=5,
    )
    triples = np.asarray([[0, 0, 1], [0, 2, 3], [1, 1, 2], [2, 3, 1]])
    loss, gradients = _bpr_epoch_gradient(
        parameters,
        genres,
        triples,
        regularization=0.003,
        include_genre=True,
        chunk_size=2,
    )
    oracle_loss, oracle = bpr_loss_and_gradients(
        parameters,
        genres,
        triples,
        rho=1.0,
        regularization=0.003,
    )
    assert loss == pytest.approx(oracle_loss, rel=0, abs=1e-12)
    for name in (
        "user_factors",
        "identity_factors",
        "feature_factors",
        "item_bias",
    ):
        np.testing.assert_allclose(
            gradients[name], getattr(oracle, name), rtol=0, atol=1e-12
        )


def test_fold_in_is_deterministic_and_does_not_mutate_shared_parameters():
    ownership, _, genres = _tiny_problem()
    fitted = fit_feature_sum_bpr(
        ownership,
        genres,
        cycle_id="test-cycle",
        training_seed=19,
        factors=2,
        regularization=0.01,
        learning_rate=0.03,
        epochs=1,
        samples_per_epoch=50,
        include_genre=True,
    ).parameters
    identity_before = fitted.identity_factors.copy()
    genre_before = fitted.feature_factors.copy()
    bias_before = fitted.item_bias.copy()
    first, first_diagnostic = fold_in_bpr_user(
        [0, 2],
        user_id=123,
        cycle_id="test-cycle",
        identity_factors=fitted.identity_factors,
        item_bias=fitted.item_bias,
        genre_features=genres,
        feature_factors=fitted.feature_factors,
        regularization=0.01,
    )
    second, second_diagnostic = fold_in_bpr_user(
        [0, 2],
        user_id=123,
        cycle_id="test-cycle",
        identity_factors=fitted.identity_factors,
        item_bias=fitted.item_bias,
        genre_features=genres,
        feature_factors=fitted.feature_factors,
        regularization=0.01,
    )
    np.testing.assert_array_equal(first, second)
    assert first_diagnostic == second_diagnostic
    np.testing.assert_array_equal(fitted.identity_factors, identity_before)
    np.testing.assert_array_equal(fitted.feature_factors, genre_before)
    np.testing.assert_array_equal(fitted.item_bias, bias_before)
    assert construct_fold_in_triples([], n_items=4, cycle_id="x", user_id=1).shape == (0, 2)


def test_implicit_fit_fold_in_scoring_and_serialization(tmp_path):
    ownership, playtime, _ = _tiny_problem()
    fitted = fit_implicit_als(
        ownership,
        playtime,
        factors=2,
        regularization=0.1,
        alpha_o=20.0,
        alpha_p=2.0,
        tau=5.0,
        iterations=2,
        training_seed=23,
        num_threads=1,
    )
    assert fitted.user_factors.shape == (3, 2)
    assert fitted.item_factors.shape == (4, 2)
    direct = np.asarray(fitted.user_factors, dtype=np.float64) @ np.asarray(
        fitted.item_factors, dtype=np.float64
    ).T
    batched = score_factor_block(
        fitted.user_factors,
        fitted.item_factors,
        np.arange(3),
        np.arange(4),
    )
    np.testing.assert_allclose(direct, batched, rtol=0, atol=1e-7)

    folded = fold_in_als(
        ownership[:1],
        playtime[:1],
        fitted.item_factors,
        regularization=0.1,
        alpha_o=20.0,
        alpha_p=2.0,
        tau=5.0,
    )
    assert folded.shape == (1, 2)
    shared_before = fitted.item_factors.copy()
    archive = tmp_path / "model.npz"
    metadata = save_parameter_archive(
        archive,
        user_factors=fitted.user_factors,
        item_factors=fitted.item_factors,
    )
    loaded = load_parameter_archive(archive, expected_sha256=metadata["sha256"])
    np.testing.assert_array_equal(loaded["item_factors"], fitted.item_factors)
    np.testing.assert_array_equal(fitted.item_factors, shared_before)
