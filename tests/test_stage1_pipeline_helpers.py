import numpy as np
import scipy.sparse as sp

from src.interactions import SparseInteractionData
from src.stage1_gate1 import _restore_full_design, _score_function
from src.stage1_production import _production_design


def _training():
    ownership = sp.csr_matrix(
        np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    )
    forever = sp.csr_matrix(
        (np.asarray([2.0, 3.0], dtype=np.float32), ownership.indices.copy(), ownership.indptr.copy()),
        shape=ownership.shape,
    )
    recent = sp.csr_matrix(
        (np.asarray([0.0, 0.0], dtype=np.float32), ownership.indices.copy(), ownership.indptr.copy()),
        shape=ownership.shape,
    )
    return SparseInteractionData(
        ownership=ownership,
        playtime_forever=forever,
        playtime_2weeks=recent,
        user_ids=np.asarray([10, 20], dtype=np.int64),
        item_ids=np.asarray([100, 200, 300], dtype=np.int64),
    )


def test_full_design_and_playtime_restoration_are_aligned():
    training = _training()
    validation = {
        "user_ids": np.asarray([10, 20]),
        "item_ids": np.asarray([200, 300]),
    }
    test = {
        "user_ids": np.asarray([10, 20]),
        "item_ids": np.asarray([300, 100]),
        "playtime_forever": np.asarray([4.0, 5.0]),
        "playtime_2weeks": np.asarray([1.0, 0.0]),
    }
    diagnostics = {
        "playtime_forever": np.asarray([6.0, 7.0]),
        "playtime_2weeks": np.asarray([0.0, 2.0]),
    }
    full_ownership = _restore_full_design(training, validation, test)
    np.testing.assert_array_equal(full_ownership.toarray(), np.ones((2, 3)))

    production = _production_design(
        training, validation, diagnostics, test
    )
    np.testing.assert_array_equal(production.ownership.toarray(), np.ones((2, 3)))
    np.testing.assert_array_equal(
        production.playtime_forever.toarray(),
        np.asarray([[2, 6, 4], [5, 3, 7]], dtype=np.float32),
    )


def test_selected_score_functions_match_direct_equations():
    genres = sp.csr_matrix(
        np.asarray([[1.0], [0.0], [0.5]], dtype=np.float32)
    )
    users = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    identity = np.asarray(
        [[0.5, 0.0], [0.0, 0.5], [1.0, 1.0]], dtype=np.float32
    )
    feature = np.asarray([[0.2, 0.4]], dtype=np.float32)
    bias = np.asarray([0.0, 0.1, -0.1], dtype=np.float32)
    arrays = {
        "user_factors": users,
        "identity_factors": identity,
        "feature_factors": feature,
        "item_bias": bias,
    }
    identity_score = _score_function(
        {"family": "feature_sum_bpr_identity"}, arrays, genres
    )(np.asarray([1]))
    np.testing.assert_allclose(identity_score, users[[1]] @ identity.T + bias)
    genre_score = _score_function(
        {"family": "feature_sum_bpr_identity_genre"}, arrays, genres
    )(np.asarray([1]))
    expected_items = identity + genres @ feature
    np.testing.assert_allclose(genre_score, users[[1]] @ expected_items.T + bias)
