"""Archive regression tests for the rejected monetary anchor.

These are exact-recovery and invariant checks on synthetic data where the true
(a, b, sigma) are known: the aggregate fit recovers them from exact rates, the
nonlinear refinement recovers them when within-game dispersion breaks the
aggregate's Jensen approximation, free disposal floors valuations at zero, the
score-to-valuation map is monotone, and the moment bridge produces a well-
conditioned Sigma. No Steam data is involved.
"""

import os
import sys

import numpy as np
import pytest
from scipy import stats

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import calibration as cal  # noqa: E402


# ---------------------------------------------------------------------------
# Aggregate fit: exact recovery from exact rates
# ---------------------------------------------------------------------------

def test_aggregate_recovers_known_params_from_exact_rates():
    rng = np.random.default_rng(0)
    G = 200
    s_bar = rng.normal(0.0, 1.0, G)
    price = rng.uniform(2.0, 12.0, G)
    a_true, b_true, sigma_true = 4.0, 3.0, 8.0

    z = (a_true + b_true * s_bar - price) / sigma_true
    rates = stats.norm.cdf(z)  # exact, no clipping at these params

    calib = cal.fit_calibration_aggregate(s_bar, price, rates,
                                          n_users=70000, link="probit")
    assert calib.valid
    assert calib.diagnostics["price_coef_negative"]
    assert calib.a == pytest.approx(a_true, abs=1e-4)
    assert calib.b == pytest.approx(b_true, abs=1e-4)
    assert calib.sigma == pytest.approx(sigma_true, abs=1e-4)
    assert calib.diagnostics["r2"] == pytest.approx(1.0, abs=1e-6)


def test_aggregate_invalid_when_price_slope_wrong_sign():
    # Ownership that rises with price (holding score fixed) cannot identify a
    # positive dollar scale; the fit must flag itself invalid rather than return a
    # negative b or sigma.
    rng = np.random.default_rng(1)
    G = 100
    s_bar = rng.normal(0.0, 1.0, G)
    price = rng.uniform(2.0, 12.0, G)
    z = (0.5 * s_bar + 0.3 * price)  # positive price slope
    rates = stats.norm.cdf(z - z.mean())

    calib = cal.fit_calibration_aggregate(s_bar, price, rates,
                                          n_users=70000, link="probit")
    assert not calib.valid
    assert np.isnan(calib.b) and np.isnan(calib.sigma) and np.isnan(calib.a)


def test_aggregate_logit_link_runs_and_is_valid():
    rng = np.random.default_rng(2)
    G = 150
    s_bar = rng.normal(0.0, 1.0, G)
    price = rng.uniform(2.0, 12.0, G)
    a_true, b_true, sigma_true = 3.0, 2.5, 6.0
    z = (a_true + b_true * s_bar - price) / sigma_true
    rates = 1.0 / (1.0 + np.exp(-z))
    calib = cal.fit_calibration_aggregate(s_bar, price, rates,
                                          n_users=70000, link="logit")
    assert calib.valid
    assert calib.b == pytest.approx(b_true, abs=1e-4)
    assert calib.sigma == pytest.approx(sigma_true, abs=1e-4)


# ---------------------------------------------------------------------------
# Quantile anchor (the cleanest censoring test) and normalization
# ---------------------------------------------------------------------------

def test_censoring_quantiles_pick_the_right_quantile():
    # Column of 0..99; rate 0.10 -> the 0.90 quantile is ~89-90.
    S = np.tile(np.arange(100.0)[:, None], (1, 3))
    rates = np.array([0.10, 0.50, 0.90])
    q = cal.censoring_quantiles(S, rates)
    assert q[0] == pytest.approx(np.quantile(np.arange(100.0), 0.90))
    assert q[1] == pytest.approx(np.quantile(np.arange(100.0), 0.50))
    assert q[2] == pytest.approx(np.quantile(np.arange(100.0), 0.10))


def test_quantile_anchor_recovers_scale_when_valid():
    rng = np.random.default_rng(6)
    G = 120
    q_g = rng.uniform(0.0, 1.0, G)
    a_true, b_true = 5.0, 30.0
    prices = a_true + b_true * q_g  # exact linear, b > 0
    calib = cal.fit_calibration_quantile(q_g, prices)
    assert calib.valid
    assert calib.a == pytest.approx(a_true, abs=1e-6)
    assert calib.b == pytest.approx(b_true, abs=1e-6)
    assert calib.diagnostics["spearman_price_quantile"] == pytest.approx(1.0, abs=1e-9)


def test_quantile_anchor_invalid_when_slope_negative():
    rng = np.random.default_rng(7)
    G = 120
    q_g = rng.uniform(0.0, 1.0, G)
    prices = 40.0 - 25.0 * q_g  # price falls with the threshold quantile
    calib = cal.fit_calibration_quantile(q_g, prices)
    assert not calib.valid
    assert np.isnan(calib.a) and np.isnan(calib.b)
    assert calib.diagnostics["spearman_price_quantile"] < 0


def test_normalize_calibration_matches_moments_and_is_flagged():
    rng = np.random.default_rng(8)
    s_bar = rng.normal(0.2, 0.1, 500)
    prices = rng.uniform(2.0, 40.0, 500)
    calib = cal.normalize_calibration(s_bar, prices)
    assert calib.valid
    assert calib.diagnostics["method"] == "normalization"
    # Median of implied valuations at mean scores matches the price median.
    implied = calib.a + calib.b * s_bar
    assert np.median(implied) == pytest.approx(np.median(prices), rel=1e-6)
    # a_shift moves the location by exactly that many dollars.
    shifted = cal.normalize_calibration(s_bar, prices, a_shift=7.0)
    assert shifted.a == pytest.approx(calib.a + 7.0, rel=1e-9)


# ---------------------------------------------------------------------------
# Nonlinear refinement corrects the within-game Jensen bias
# ---------------------------------------------------------------------------

def test_refine_recovers_params_under_within_game_dispersion():
    rng = np.random.default_rng(3)
    U, G = 4000, 150
    s_bar = rng.normal(0.0, 1.0, G)
    price = rng.uniform(2.0, 12.0, G)
    S = s_bar[None, :] + rng.normal(0.0, 0.6, (U, G))
    a_true, b_true, sigma_true = 4.0, 3.0, 7.0

    true_rates = stats.norm.cdf((a_true + b_true * S - price[None, :]) / sigma_true
                                ).mean(axis=0)

    calib0 = cal.fit_calibration_aggregate(S.mean(axis=0), price, true_rates,
                                           n_users=U, link="probit")
    refined = cal.refine_calibration_rates(S, price, true_rates, calib0,
                                           link="probit")

    assert refined.a == pytest.approx(a_true, rel=0.10)
    assert refined.b == pytest.approx(b_true, rel=0.10)
    assert refined.sigma == pytest.approx(sigma_true, rel=0.10)

    # The refinement should fit the rates at least as well as the aggregate seed.
    agg_pred = cal.predicted_rates(S, price, calib0.a, calib0.b, calib0.sigma,
                                   link="probit")
    ref_pred = cal.predicted_rates(S, price, refined.a, refined.b, refined.sigma,
                                   link="probit")
    assert np.mean((ref_pred - true_rates) ** 2) <= np.mean((agg_pred - true_rates) ** 2)


# ---------------------------------------------------------------------------
# Valuation map: free disposal, monotonicity, scale
# ---------------------------------------------------------------------------

def test_valuation_free_disposal_floor_and_monotonicity():
    calib = cal.CalibrationResult(a=-1.0, b=2.0, sigma=5.0, link="probit",
                                  valid=True, diagnostics={})
    scores = np.array([-3.0, -0.5, 0.0, 1.0, 4.0])

    v_floor = calib.valuation(scores, free_disposal=True)
    v_raw = calib.valuation(scores, free_disposal=False)

    assert np.all(v_floor >= 0.0)
    assert np.allclose(v_raw, -1.0 + 2.0 * scores)
    # Floored where a + b s < 0 (scores -3.0 and -0.5 give -7 and -2), raw elsewhere.
    assert v_floor[0] == 0.0 and v_floor[1] == 0.0
    assert v_floor[-1] == pytest.approx(7.0)
    # Monotone non-decreasing in score under free disposal.
    assert np.all(np.diff(v_floor) >= -1e-12)


# ---------------------------------------------------------------------------
# Moment bridge wiring
# ---------------------------------------------------------------------------

def test_valuation_moments_shapes_and_conditioning():
    rng = np.random.default_rng(4)
    n_users, n_items, k = 2000, 12, 6
    P = rng.normal(0.0, 1.0, (n_users, k))
    Q = rng.normal(0.0, 1.0, (n_items, k))
    calib = cal.CalibrationResult(a=20.0, b=5.0, sigma=8.0, link="probit",
                                  valid=True, diagnostics={})

    item_indices = np.arange(n_items)
    sizes = [1, 2, 3, 5, 8]

    omega, Sigma, V, info = cal.valuation_moments(
        P, Q, item_indices, sizes, calib, cov_method="ledoit_wolf")

    assert omega.shape == (len(sizes),)
    assert Sigma.shape == (len(sizes), len(sizes))
    assert V.shape == (n_users, n_items)
    assert np.all(V >= 0.0)  # free disposal floor
    assert np.allclose(Sigma, Sigma.T)
    # omega is increasing in size (more items in the bundle, larger top-s sum).
    assert np.all(np.diff(omega) > 0)
    # Ledoit-Wolf shrinkage keeps Sigma strictly positive definite.
    assert np.linalg.eigvalsh(Sigma)[0] > 0
    assert info["cov"]["method"] == "ledoit_wolf"


def test_score_helpers_consistent():
    rng = np.random.default_rng(5)
    n_users, n_items, k = 500, 20, 4
    P = rng.normal(size=(n_users, k))
    Q = rng.normal(size=(n_items, k))
    # mean_scores equals the column mean of the full score matrix.
    full = cal.score_matrix(P, Q, np.arange(n_items))
    assert np.allclose(cal.mean_scores(P, Q), full.mean(axis=0))


def test_unknown_link_raises():
    with pytest.raises(ValueError):
        cal._link_ppf(np.array([0.3]), "cloglog")
