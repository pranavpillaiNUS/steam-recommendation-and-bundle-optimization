"""Unit tests for the Stage 0 BSP/CMM optimizer (src/bundle_pricing.py).

These are the *exact* checks: hand-computed values, finite-difference gradient
checks, internal cross-checks between the simplex and SDP demand, and agreement
of the convex optimizer with a CMM-specific grid oracle. The approximation
comparison between the CMM optimum and the empirical synthetic optimum is a
research finding and lives in notebook 06, not here.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import bundle_pricing as bp  # noqa: E402


# ---------------------------------------------------------------------------
# Hand-computed values
# ---------------------------------------------------------------------------

def test_evaluate_schedule_hand_computed():
    # 3 users, sizes {1, 2}, zero cost, prices [4, 7].
    # A=[10,1] -> w=(10,11); B=[3,2] -> w=(3,5); C=[1,0.5] -> w=(1,1.5).
    # A picks size 1 (surplus 6) -> profit 4; B and C take the outside option.
    valuations = np.array([[10.0, 1.0], [3.0, 2.0], [1.0, 0.5]])
    profit = bp.evaluate_schedule(valuations, [4.0, 7.0], [1, 2], 0.0)
    assert profit == pytest.approx((4.0 + 0.0 + 0.0) / 3.0)


def test_evaluate_schedule_tiebreak_prefers_lower_price():
    # w1 = 6, w2 = 8, prices [2, 4]: both surpluses equal 4. The tie-break
    # takes the lower-priced (size 1) option, so realised profit is its margin 2.
    valuations = np.array([[6.0, 2.0]])
    profit = bp.evaluate_schedule(valuations, [2.0, 4.0], [1, 2], 0.0)
    assert profit == pytest.approx(2.0)


def test_moments_from_valuations_hand_computed():
    V = np.array([[3.0, 1.0, 2.0], [5.0, 0.0, 1.0]])
    omega, sigma, W = bp.moments_from_valuations(V, [1, 2, 3])
    # Row sorts: [3,2,1] -> cumsum [3,5,6]; [5,1,0] -> cumsum [5,6,6].
    np.testing.assert_allclose(W, np.array([[3.0, 5.0, 6.0], [5.0, 6.0, 6.0]]))
    np.testing.assert_allclose(omega, np.array([4.0, 5.5, 6.0]))
    # ddof=1 sample covariance of W.
    np.testing.assert_allclose(sigma, np.cov(W, rowvar=False, ddof=1))


def test_normalize_costs_scalar_and_vector():
    # Scalar is a per-game unit cost: c_s = s * unit_cost.
    np.testing.assert_allclose(bp.normalize_costs(0.5, [1, 2, 4]), [0.5, 1.0, 2.0])
    # Array is total cost per size, passed through.
    np.testing.assert_allclose(bp.normalize_costs([1.0, 3.0], [2, 5]), [1.0, 3.0])
    with pytest.raises(ValueError):
        bp.normalize_costs([1.0, 2.0, 3.0], [1, 2])  # wrong length


# ---------------------------------------------------------------------------
# Gradient and single-size correctness
# ---------------------------------------------------------------------------

def test_cmm_grad_f_matches_finite_difference():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(3, 3))
    Sigma = A @ A.T + np.eye(3)  # SPD
    q = np.array([0.3, 0.25, 0.2])  # interior, sum < 1
    grad = bp.cmm_grad_f(q, Sigma)
    num = np.zeros(3)
    h = 1e-6
    for i in range(3):
        e = np.zeros(3)
        e[i] = h
        num[i] = (bp.cmm_f(q + e, Sigma) - bp.cmm_f(q - e, Sigma)) / (2 * h)
    np.testing.assert_allclose(grad, num, rtol=1e-4, atol=1e-6)


def test_single_size_price_demand_round_trip():
    omega_i, sigma_i = 5.0, 2.0
    for x in [0.1, 0.4, 0.65, 0.9]:
        p = bp.single_size_price(omega_i, sigma_i, x)
        x_back = bp.single_size_demand(omega_i, sigma_i, p)
        assert x_back == pytest.approx(x, abs=1e-9)


def test_single_size_root_matches_direct_optimization():
    from scipy.optimize import minimize_scalar

    omega_i, sigma_i, c_i = 5.0, 2.0, 0.0
    res = bp.optimize_single_size(np.array([omega_i]), np.array([[sigma_i**2]]),
                                  0.0, [1])
    x_root = res["best"]["demand"]
    direct = minimize_scalar(
        lambda x: -bp._single_size_profit(x, omega_i, sigma_i, c_i),
        bounds=(1e-7, 1 - 1e-7), method="bounded",
    )
    assert x_root == pytest.approx(direct.x, abs=1e-4)
    assert res["best"]["solution_type"] == "interior_root"
    # The Corollary 1 root equation is satisfied at the interior root.
    assert bp._single_size_root_lhs(x_root, omega_i, sigma_i, c_i) == pytest.approx(0.0, abs=1e-6)


def test_single_size_high_cost_is_boundary():
    # A cost above the mean valuation pushes the optimum to ~zero demand.
    res = bp.optimize_single_size(np.array([2.0]), np.array([[1.0]]), 100.0, [1])
    assert res["best"]["solution_type"] in ("lower_boundary", "interior_root")
    assert res["best"]["demand"] < 0.5


# ---------------------------------------------------------------------------
# Demand: simplex vs SDP
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 2, 5])
def test_demand_simplex_matches_sdp(seed):
    rng = np.random.default_rng(seed)
    m = 2
    A = rng.normal(size=(m, m))
    Sigma = A @ A.T + np.eye(m)
    omega = rng.uniform(3, 8, size=m)
    p = omega - rng.uniform(0.5, 2.0, size=m)
    ds = bp.cmm_demand_simplex(p, omega, Sigma)
    dd = bp.cmm_demand_sdp(p, omega, Sigma)
    # Same choice probabilities and objective via the shared DemandResult.
    np.testing.assert_allclose(ds.q, dd.q, atol=5e-3)
    assert ds.objective == pytest.approx(dd.objective, abs=5e-3)
    # SDP self-computed feasibility (not a solver-reported residual key).
    assert dd.diagnostics["min_block_eigenvalue"] >= -1e-5
    assert dd.diagnostics["objective_gap"] < 1e-6
    assert dd.q0_outside == pytest.approx(1.0 - dd.q.sum(), abs=1e-9)


# ---------------------------------------------------------------------------
# Multi-size optimizer vs the CMM grid oracle, and the softmax cross-check
# ---------------------------------------------------------------------------

def test_optimize_convex_matches_grid_oracle():
    omega = np.array([5.0, 9.0])
    Sigma = np.array([[4.0, 1.0], [1.0, 9.0]])
    oc = bp.optimize_convex(omega, Sigma, 0.0, [1, 2])
    bf = bp.brute_force_cmm_q(omega, Sigma, 0.0, [1, 2], grid=120)
    assert oc["profit"] == pytest.approx(bf["profit"], abs=1e-3)
    np.testing.assert_allclose(oc["q"], bf["q"], atol=1e-2)
    # Recovered prices reproduce q* (round trip) and are finite/nonnegative.
    assert oc["roundtrip_max_abs_err"] < 1e-3
    assert oc["prices_finite"] and oc["prices_nonnegative"]


def test_softmax_cross_check_agrees_with_convex():
    omega = np.array([5.0, 9.0])
    Sigma = np.array([[4.0, 1.0], [1.0, 9.0]])
    oc = bp.optimize_convex(omega, Sigma, 0.0, [1, 2])
    sm = bp.optimize_convex_softmax(omega, Sigma, 0.0, [1, 2], n_restarts=6)
    assert sm["profit"] == pytest.approx(oc["profit"], abs=2e-3)
    # Diagnostics are reported.
    assert "max_abs_logit" in sm and "min_probability" in sm


# ---------------------------------------------------------------------------
# Numerical robustness
# ---------------------------------------------------------------------------

def test_optimize_convex_stable_across_delta_sweep():
    omega = np.array([4.0, 7.0, 9.0])
    rng = np.random.default_rng(3)
    A = rng.normal(size=(3, 3))
    Sigma = A @ A.T + 2 * np.eye(3)
    profits = []
    for delta in (1e-6, 1e-8, 1e-10):
        oc = bp.optimize_convex(omega, Sigma, 0.0, [1, 2, 3], delta=delta)
        profits.append(oc["profit"])
        assert oc["min_q"] > 0.0  # not pinned to zero
    assert max(profits) - min(profits) < 1e-3


def test_eigenvalue_policy_clips_float_error_but_raises_on_material():
    # Float-error negativity is clipped silently-with-warning, not fatal.
    w, V, diag = bp._eigh_clip(np.diag([1.0, -1e-13]), eig_tol=1e-9)
    assert diag["n_clipped"] == 1
    assert np.all(w >= 0)
    # Materially negative eigenvalue is fatal.
    with pytest.raises(bp.NumericalError):
        bp._eigh_clip(np.diag([1.0, -1.0]), eig_tol=1e-9)


def test_moments_input_validation():
    with pytest.raises(ValueError):
        bp.moments_from_valuations(np.array([1.0, 2.0, 3.0]), [1])  # 1-D
    with pytest.raises(ValueError):
        bp.moments_from_valuations(np.array([[1.0, 2.0]]), [1])  # <2 users
    with pytest.raises(ValueError):
        bp.moments_from_valuations(np.ones((3, 2)), [1, 3])  # size > n_items
    with pytest.raises(ValueError):
        bp.moments_from_valuations(np.ones((3, 3)), [1, 1])  # duplicate sizes


# ---------------------------------------------------------------------------
# Vectorized empirical path: choice rule and precomputed-W agreement
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_vectorized_choice_matches_reference_loop(seed):
    # Generic continuous data: no ties, the two implementations must agree
    # user by user, including who takes the outside option.
    rng = np.random.default_rng(seed)
    n_users, m = 500, 4
    W = np.sort(rng.gamma(2.0, 3.0, size=(n_users, m)), axis=1).cumsum(axis=1)
    sizes = np.array([1, 2, 3, 4])
    prices = rng.uniform(2.0, 25.0, size=m)
    costs = np.zeros(m)
    fast = bp._user_choice_indices(W, prices, sizes, costs)
    ref = bp._user_choice_indices_reference(W, prices, sizes, costs)
    np.testing.assert_array_equal(fast, ref)


def test_vectorized_choice_exact_ties_and_outside():
    # Exact tie in surplus: lower price wins; price tie: smaller size wins;
    # non-positive best surplus: outside option (-1).
    W = np.array([[6.0, 8.0],    # surpluses (4, 4): tie -> lower price (size 1)
                  [5.0, 7.0],    # surpluses (3, 3): tie -> lower price
                  [1.0, 2.0]])   # surpluses (-1, -2): outside
    prices = np.array([2.0, 4.0])
    sizes = np.array([1, 2])
    choices = bp._user_choice_indices(W, prices, sizes, np.zeros(2))
    np.testing.assert_array_equal(choices, [0, 0, -1])
    # Price tie between sizes: the smaller size wins.
    W2 = np.array([[6.0, 6.0]])
    choices2 = bp._user_choice_indices(W2, np.array([3.0, 3.0]), sizes, np.zeros(2))
    np.testing.assert_array_equal(choices2, [0])


def test_evaluate_schedule_precomputed_W_matches():
    rng = np.random.default_rng(4)
    V = rng.gamma(2.0, 2.0, size=(300, 5))
    sizes = [1, 3, 5]
    prices = [3.0, 7.0, 9.0]
    W = bp._top_s_sums(V, np.asarray(sizes), free_disposal=True)
    direct = bp.evaluate_schedule(V, prices, sizes, 0.0)
    cached = bp.evaluate_schedule(None, prices, sizes, 0.0, W=W)
    assert direct == pytest.approx(cached)


def test_optimize_empirical_schedule_beats_grid_oracle():
    rng = np.random.default_rng(5)
    V = rng.gamma(2.0, 2.0, size=(400, 3))
    sizes = [1, 3]
    de = bp.optimize_empirical_schedule(V, sizes, 0.0, seed=0, maxiter=60)
    bf = bp.brute_force_empirical_prices(V, sizes, 0.0, grid=30)
    assert de["profit"] >= bf["profit"] - 1e-9


def test_optimize_empirical_schedule_x0_weakly_improves():
    rng = np.random.default_rng(6)
    V = rng.gamma(2.0, 2.0, size=(400, 3))
    sizes = [1, 2, 3]
    x0 = np.array([3.0, 5.0, 6.0])
    base = bp.evaluate_schedule(V, x0, sizes, 0.0)
    de = bp.optimize_empirical_schedule(V, sizes, 0.0, seed=0, maxiter=40, x0=x0)
    assert de["profit"] >= base - 1e-9


# ---------------------------------------------------------------------------
# Benchmark policies: separate selling and pure bundling
# ---------------------------------------------------------------------------

def test_separate_selling_profit_hand_computed():
    # Item 0 valuations [4, 2]: price 4 -> 4*(1/2)=2; price 2 -> 2*(2/2)=2 -> 2.
    # Item 1 valuations [6, 1]: price 6 -> 3; price 1 -> 1 -> 3. Total 5.
    V = np.array([[4.0, 6.0], [2.0, 1.0]])
    assert bp.separate_selling_profit(V) == pytest.approx(5.0)


def test_pure_bundling_profit_hand_computed():
    # Bundle valuations: [10, 3]. Price 10 -> 5; price 3 -> 3 -> 5.
    V = np.array([[4.0, 6.0], [2.0, 1.0]])
    assert bp.pure_bundling_profit(V) == pytest.approx(5.0)
    # A positive per-unit story with cost: cost 4 -> price 10 gives (10-4)/2 = 3.
    assert bp.pure_bundling_profit(V, cost=4.0) == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# No strictly dominated *materially active* size (point 9)
# ---------------------------------------------------------------------------

def test_no_dominated_materially_active_size():
    # Universal "prices increase with size" is NOT a property of the unconstrained
    # CMM optimum: with independent valuations and low cost, pure bundling is
    # near-optimal (Abdallah et al. 2017; the paper's Section 5.2.3 degeneracy),
    # so only the grand bundle is materially active and the near-inactive smaller
    # sizes carry residual smooth-approximation mass with non-monotone prices.
    # The robust invariant is therefore: among *materially active* sizes
    # (q > 0.05) no larger size is priced below a smaller one (no dominance).
    rng = np.random.default_rng(7)
    V = rng.gamma(2.0, 2.0, size=(4000, 4))
    sizes = np.array([1, 2, 3, 4])
    omega, sigma, W = bp.moments_from_valuations(V, sizes)
    Sigma, _ = bp.estimate_covariance(W, method="ridge")
    oc = bp.optimize_convex(omega, Sigma, 0.0, sizes)
    active = oc["q"] > 0.05
    prices_active = oc["prices"][active]
    assert np.all(np.diff(prices_active) >= -1e-6)
    # Documented invariants that always hold: finite prices and a valid round trip.
    assert oc["prices_finite"]
    assert oc["roundtrip_max_abs_err"] < 1e-3
