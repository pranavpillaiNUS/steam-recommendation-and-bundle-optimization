"""Archived bundle-size pricing (BSP) under the cross-moment model (CMM).

This preserves the pre-pivot optimizer for the paper:
    "Convex Optimization for the Bundle Size Pricing Problem".

CMM lets a customer construct any set of a priced size. It is not the live
fixed-curated-bundle mechanism and supplies no objective, price, or guarantee to
the CP-anchored SBA pipeline.

It is pure numpy/scipy/cvxpy. Section and equation pointers below refer to
the paper and its appendix A.

Two objectives are kept strictly separate:

- the CMM model objective ``pi(q)``, which uses only the moments
  ``(omega, Sigma)`` of the size-wise maximum valuations, and
- the empirical sample-average objective ``evaluate_schedule``, which uses the
  supplied per-user assumed input panel.

They approximate each other but are not the same optimization problem, so they
get separate optimizers and separate brute-force oracles. Neither is model-free,
and a differential-evolution result is the best solution found unless certified.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import warnings
import numpy as np
from scipy import optimize, stats


"""
Shortcuts:
1. _eigh_clip
2. _matrix_sqrt
3. _trace_sqrt
4. _trace_sqrt_psd
5. _inv_sqrt
6. _S(x)
7. moments_from_valuations
8. estimate_covariance
9. normalize_costs
10. cmm_f
11. cmm_grad_f
12. cmm_profit_q
13. numerical_profit_gradient
14. cmm_demand_simplex
15. cmm_demand_sdp
16. cmm_demand
17. single_size_price
18. single_size_demand
19. _single_size_root_lhs
20. _single_size_profit
21. optimize_single_size
22. optimize_convex
23. optimize_convex_softmax
24. _user_choice_indices
25. _user_choice_indices_reference
26. evaluate_schedule
27. optimize_empirical_schedule
28. separate_selling_profit
29. pure_bundling_profit
30. brute_force_cmm_q
31. brute_force_empirical_prices

"""


class NumericalError(RuntimeError):
    """Raised when a covariance/Gram matrix has a materially negative eigenvalue.

    Tiny negative eigenvalues from floating point are clipped to zero with a
    warning; only economically material negativity (below ``-eig_tol``) is fatal.
    """


# Symmetric matrix functions with an explicit eigenvalue policy
def _eigh_clip(A, eig_tol=1e-9):
    """Symmetric eigendecomposition with the project's clip-or-raise policy.

    Returns ``(eigvals_clipped, eigvecs, diagnostics)``. Raises ``NumericalError``
    if the smallest eigenvalue is below ``-eig_tol`` (silent or material clipping
    is exactly what we want to avoid); otherwise small negatives are clipped to 0.
    """
    A = np.asarray(A, dtype=float)
    A = 0.5 * (A + A.T)  # symmetrize against round-off
    w, V = np.linalg.eigh(A)
    lam_min = float(w[0])
    if lam_min < -eig_tol:
        raise NumericalError(
            f"matrix has eigenvalue {lam_min:.3e} < -eig_tol ({-eig_tol:.1e}); "
            "this is material, not float error"
        )
    n_clipped = int(np.sum(w < 0))
    largest_clip = float(-w[0]) if lam_min < 0 else 0.0
    w_clipped = np.clip(w, 0.0, None)
    diag = {
        "min_raw_eigenvalue": lam_min,
        "eig_tol": eig_tol,
        "n_clipped": n_clipped,
        "largest_clipped_magnitude": largest_clip,
    }
    return w_clipped, V, diag


def _matrix_sqrt(A, eig_tol=1e-9):
    """Unique PSD square root of a symmetric PSD matrix."""
    w, V, _ = _eigh_clip(A, eig_tol)
    return (V * np.sqrt(w)) @ V.T


def _trace_sqrt(A, eig_tol=1e-9):
    """tr(A^{1/2}) = sum of sqrt of eigenvalues, without forming A^{1/2}."""
    w, _, _ = _eigh_clip(A, eig_tol)
    return float(np.sum(np.sqrt(w)))


def _trace_sqrt_psd(A):
    """tr(A^{1/2}) for a matrix that is PSD up to round-off (silent clip).

    Used for T(q) = Sigma^{1/2} S(q) Sigma^{1/2}, a function of the decision
    variable q. The optimizer can probe q values that transiently violate
    ``sum(q) <= 1`` and make S(q) indefinite; there we clip negative eigenvalues
    to zero rather than raise, since the true optimum is interior and is found
    once the constraints bind. (The raise policy in ``_eigh_clip`` is reserved
    for the Sigma modeling input.)
    """
    A = 0.5 * (np.asarray(A, dtype=float) + np.asarray(A, dtype=float).T)
    w = np.linalg.eigvalsh(A)
    return float(np.sum(np.sqrt(np.clip(w, 0.0, None))))


def _inv_sqrt(A, eig_floor=1e-12):
    """Moore-Penrose inverse symmetric square root A^{-1/2} of a PSD matrix.

    Matches the pseudoinverse used in the paper's closed-form solution (App. A,
    eq. 20). On the simplex interior with Sigma > 0 this is the true inverse
    square root; near the boundary (or at transiently infeasible q) the near-zero
    and clipped eigenvalues contribute zero, which keeps f and its gradient
    finite and defined everywhere for the optimizer.
    """
    A = 0.5 * (np.asarray(A, dtype=float) + np.asarray(A, dtype=float).T)
    w, V = np.linalg.eigh(A)
    inv = np.where(w > eig_floor, 1.0 / np.sqrt(np.clip(w, eig_floor, None)), 0.0)
    return (V * inv) @ V.T


def _S(x):
    """S(x) = Diag(x) - x x^T."""
    x = np.asarray(x, dtype=float)
    return np.diag(x) - np.outer(x, x)


# The empirical moment bridge and covariance estimation
def _top_s_sums(valuations, sizes, free_disposal=False):
    """Per-user top-s partial sums w_s of the valuation rows.

    For each user, sort their item valuations in decreasing order and take the
    cumulative sum at each requested size. With ``free_disposal`` the customer
    drops negatively valued items (floor at zero); with non-negative valuations
    the two coincide, which is the Stage 0 regime.
    """
    V = np.asarray(valuations, dtype=float)
    sizes = np.asarray(sizes, dtype=int)
    sorted_desc = -np.sort(-V, axis=1)
    if free_disposal:
        sorted_desc = np.maximum(sorted_desc, 0.0)
    cumsum = np.cumsum(sorted_desc, axis=1)
    # cumsum[:, s-1] is the size-s partial sum.
    return cumsum[:, sizes - 1]


def moments_from_valuations(valuations, sizes):
    """Empirical CMM inputs from per-user valuation vectors.

    Parameters
    ----------
    valuations : (n_users, n_items) array
    sizes      : iterable of bundle sizes S (each in 1..n_items)

    Returns
    -------
    omega        : (m,) sample mean of the size-wise maximum valuations
    sample_sigma : (m, m) sample covariance (ddof=1)
    W            : (n_users, m) the per-user size-wise maximum valuations themselves, so
                covariance estimators can use the observations, not a finished matrix.

    The user-supplied size order is preserved in all three outputs.
    """
    V = np.asarray(valuations, dtype=float)
    if V.ndim != 2:
        raise ValueError("valuations must be a 2-D (n_users, n_items) array")
    if not np.all(np.isfinite(V)):
        raise ValueError("valuations must be finite")
    n_users, n_items = V.shape
    if n_users < 2:
        raise ValueError("need at least two users for an unbiased sample covariance")

    sizes = np.asarray(list(sizes), dtype=int)
    if np.any(sizes < 1) or np.any(sizes > n_items):
        raise ValueError(f"every size must satisfy 1 <= s <= n_items ({n_items})")
    if len(set(sizes.tolist())) != len(sizes):
        raise ValueError("duplicate sizes are not allowed; pass each size once")

    W = _top_s_sums(V, sizes, free_disposal=False)
    omega = W.mean(axis=0)
    sample_sigma = np.cov(W, rowvar=False, ddof=1)
    sample_sigma = np.atleast_2d(sample_sigma)
    return omega, sample_sigma, W


def estimate_covariance(W, method="sample", eps_rel=1e-8):
    """Estimate the partial-sum covariance Sigma from the observations W.

    Successive partial sums are mechanically correlated (w_{s+1} = w_s + the next
    order statistic), so the raw sample covariance can be near singular while the
    CMM needs Sigma > 0. Methods:

    - ``"sample"``: plain sample covariance (ddof=1).
    - ``"ridge"``: sample covariance plus a *relative* ridge
      ``eps_rel * (tr(Sigma)/m) * I`` (absolute epsilon is wrong because the
      covariance units depend on the valuation scale).
    - ``"ledoit_wolf"``: Ledoit-Wolf shrinkage, whose intensity is estimated
      from the observations W (not from a finished matrix), via sklearn.

    Returns ``(Sigma, info)`` where ``info`` records the condition number before
    and after regularization.
    """
    W = np.asarray(W, dtype=float)
    if W.ndim != 2:
        raise ValueError("W must be a 2-D (n_users, m) array")
    m = W.shape[1]

    sample = np.atleast_2d(np.cov(W, rowvar=False, ddof=1))
    cond_before = float(np.linalg.cond(sample)) if m > 1 else 1.0

    if method == "sample":
        Sigma = sample
    elif method == "ridge":
        ridge = eps_rel * (np.trace(sample) / m)
        Sigma = sample + ridge * np.eye(m)
    elif method == "ledoit_wolf":
        from sklearn.covariance import LedoitWolf

        lw = LedoitWolf().fit(W)
        Sigma = np.atleast_2d(lw.covariance_)
    else:
        raise ValueError(f"unknown method {method!r}")

    cond_after = float(np.linalg.cond(Sigma)) if m > 1 else 1.0
    info = {
        "method": method,
        "cond_before": cond_before,
        "cond_after": cond_after,
        "m": m,
    }
    return Sigma, info



# Cost convention
def normalize_costs(cost, sizes):
    """Return a size-aligned cost vector costs_by_size = (c_{s_1}, ..., c_{s_m}).

    Scalar ``cost`` is a per-game unit cost, so ``c_s = s * unit_cost``. Array
    ``cost`` is the total cost for each corresponding bundle size and must match
    ``len(sizes)``. A scalar is never silently treated as the same total cost
    for every size.
    """
    sizes = np.asarray(list(sizes), dtype=int)
    if np.isscalar(cost) or (np.ndim(cost) == 0):
        return float(cost) * sizes.astype(float)
    cost = np.asarray(cost, dtype=float)
    if cost.shape != sizes.shape:
        raise ValueError(
            "array cost must give a total cost per size; "
            f"got {cost.shape}, expected {sizes.shape}"
        )
    return cost



# CMM core: f, grad f, profit, numerical profit gradient
def cmm_f(x, Sigma, eig_tol=1e-9):
    """f(x) = tr( (Sigma^{1/2} S(x) Sigma^{1/2})^{1/2} ).

    Equivalently the sum of the square roots of the eigenvalues of
    Sigma^{1/2} S(x) Sigma^{1/2}. Valid on the closed simplex.
    """
    x = np.asarray(x, dtype=float)
    Sigma = np.atleast_2d(np.asarray(Sigma, dtype=float))
    Sigma_half = _matrix_sqrt(Sigma, eig_tol)  # Sigma input: clip-or-raise policy
    M = Sigma_half @ _S(x) @ Sigma_half
    return _trace_sqrt_psd(M)  # M depends on q: silent clip, no raise


def cmm_grad_f(q, Sigma, eig_tol=1e-9):
    """Gradient of f at an interior point q (note's price-demand bijection).

    grad f(q) = 1/2 diag( Sigma^{1/2} T(q)^{-1/2} Sigma^{1/2} )
                - Sigma^{1/2} T(q)^{-1/2} Sigma^{1/2} q,
    with T(q) = Sigma^{1/2} S(q) Sigma^{1/2}. Interior use only.
    """
    q = np.asarray(q, dtype=float)
    Sigma = np.atleast_2d(np.asarray(Sigma, dtype=float))
    Sigma_half = _matrix_sqrt(Sigma, eig_tol)
    T = Sigma_half @ _S(q) @ Sigma_half
    T_inv_half = _inv_sqrt(T)
    G = Sigma_half @ T_inv_half @ Sigma_half  # symmetric
    return 0.5 * np.diag(G) - G @ q


def cmm_profit_q(q, omega, Sigma, costs_by_size, eig_tol=1e-9):
    """CMM expected profit in choice probabilities (note's eq. (6)).

    pi(q) = q^T (omega - costs_by_size) + q^T grad_f(q), using the reparametrized
    price p = omega + grad_f(q). Concave on the interior (Theorem 2).
    """
    q = np.asarray(q, dtype=float)
    omega = np.asarray(omega, dtype=float)
    costs_by_size = np.asarray(costs_by_size, dtype=float)
    grad = cmm_grad_f(q, Sigma, eig_tol)
    return float(q @ (omega - costs_by_size) + q @ grad)


def numerical_profit_gradient(q, omega, Sigma, costs_by_size, step=None, eig_tol=1e-9):
    """Central finite-difference gradient of cmm_profit_q in q-coordinates.

    Stage 0 deliberately uses only this numerical outer gradient: the analytic
    gradient of pi involves the Hessian action of the matrix square root, which
    is not derived here. The optimizer is validated against the grid oracle, not
    against an analytic gradient.
    """
    q = np.asarray(q, dtype=float)
    m = q.size
    if step is None:
        step = 1e-6
    g = np.zeros(m)
    for i in range(m):
        e = np.zeros(m)
        e[i] = step
        f_plus = cmm_profit_q(q + e, omega, Sigma, costs_by_size, eig_tol)
        f_minus = cmm_profit_q(q - e, omega, Sigma, costs_by_size, eig_tol)
        g[i] = (f_plus - f_minus) / (2 * step)
    return g



# CMM demand q(p)
@dataclass
class DemandResult:
    """Shared return type for both demand implementations."""

    q: np.ndarray
    q0_outside: float
    objective: float
    success: bool
    status: str
    method: str
    diagnostics: dict = field(default_factory=dict)


def _maximize_concave_on_simplex(neg_obj, neg_grad, m, delta, method="trust-constr"):
    """Maximize a concave objective over {q_i >= delta, sum q <= 1 - delta}.

    Implemented as a minimization of ``neg_obj`` with linear constraints; the
    feasible set is convex and the objective concave, so the local optimum is
    global up to the interior margin delta.
    """
    x0 = np.full(m, (1.0 - delta) / (m + 1.0))  # interior start, mass on outside too
    bounds = optimize.Bounds(np.full(m, delta), np.full(m, 1.0 - delta))
    lin = optimize.LinearConstraint(np.ones((1, m)), -np.inf, 1.0 - delta)
    res = optimize.minimize(
        neg_obj,
        x0,
        jac=neg_grad,
        method=method,
        bounds=bounds,
        constraints=[lin],
        options={"gtol": 1e-10, "xtol": 1e-12, "maxiter": 2000},
    )
    return res


def cmm_demand_simplex(p, omega, Sigma, delta=1e-9, eig_tol=1e-12):
    """CMM choice probabilities q(p) via the concave simplex program (Theorem 1).

    Maximizes (omega - p)^T x + f(x) over the simplex. The objective gradient is
    analytic, (omega - p) + grad_f(x); only the demand (not the pricing) problem
    needs it, and no Hessian is involved.
    """
    p = np.asarray(p, dtype=float)
    omega = np.asarray(omega, dtype=float)
    Sigma = np.atleast_2d(np.asarray(Sigma, dtype=float))
    m = omega.size
    alpha = omega - p

    def neg_obj(x):
        return -(alpha @ x + cmm_f(x, Sigma, eig_tol))

    def neg_grad(x):
        return -(alpha + cmm_grad_f(x, Sigma, eig_tol))

    res = _maximize_concave_on_simplex(neg_obj, neg_grad, m, delta)
    q = np.clip(res.x, 0.0, None)
    return DemandResult(
        q=q,
        q0_outside=float(1.0 - q.sum()),
        objective=float(-res.fun),
        success=bool(res.success),
        status=str(res.message),
        method="simplex",
        diagnostics={"min_q": float(q.min()), "nit": int(res.nit)},
    )


def cmm_demand_sdp(p, omega, Sigma, solver=None):
    """CMM choice probabilities q(p) via the reduced SDP (paper eq. 12 / CMM2).

    Builds the exact block matrix with alpha = omega - p, Pi = Sigma + alpha alpha^T,
    maximizes tr(Y). Returns the shared DemandResult, with solver-independent
    feasibility diagnostics computed by us (min q_i, q0, the minimum eigenvalue of
    the constructed block at the returned (x, Y), and reported-vs-recomputed tr(Y)),
    rather than relying on a solver-specific residual key.
    """
    import cvxpy as cp

    p = np.asarray(p, dtype=float)
    omega = np.asarray(omega, dtype=float)
    Sigma = np.atleast_2d(np.asarray(Sigma, dtype=float))
    m = omega.size
    alpha = (omega - p).reshape(m, 1)
    Pi = Sigma + alpha @ alpha.T

    x = cp.Variable(m, nonneg=True)
    Y = cp.Variable((m, m))
    block = cp.bmat([
        [Pi, Y.T, alpha],
        [Y, cp.diag(x), cp.reshape(x, (m, 1), order="C")],
        [alpha.T, cp.reshape(x, (1, m), order="C"), np.array([[1.0]])],
    ])
    constraints = [cp.sum(x) <= 1, block >> 0]
    prob = cp.Problem(cp.Maximize(cp.trace(Y)), constraints)
    prob.solve(solver=solver)

    q = np.asarray(x.value, dtype=float).ravel()
    q = np.clip(q, 0.0, None)
    Yv = np.asarray(Y.value, dtype=float)

    # Solver-independent feasibility: rebuild the block at the returned (x, Y).
    block_val = np.block([
        [Pi, Yv.T, alpha],
        [Yv, np.diag(q), q.reshape(m, 1)],
        [alpha.T, q.reshape(1, m), np.array([[1.0]])],
    ])
    block_val = 0.5 * (block_val + block_val.T)
    min_block_eig = float(np.linalg.eigvalsh(block_val)[0])
    reported = float(prob.value)
    recomputed = float(np.trace(Yv))

    diagnostics = {
        "min_q": float(q.min()) if q.size else float("nan"),
        "q0_outside": float(1.0 - q.sum()),
        "min_block_eigenvalue": min_block_eig,
        "objective_reported": reported,
        "objective_recomputed_trace_Y": recomputed,
        "objective_gap": abs(reported - recomputed),
        "solver_name": prob.solver_stats.solver_name if prob.solver_stats else None,
    }
    return DemandResult(
        q=q,
        q0_outside=float(1.0 - q.sum()),
        objective=reported,
        success=prob.status in ("optimal", "optimal_inaccurate"),
        status=str(prob.status),
        method="sdp",
        diagnostics=diagnostics,
    )


def cmm_demand(p, omega, Sigma, **kwargs):
    """Default CMM demand: the simplex form."""
    return cmm_demand_simplex(p, omega, Sigma, **kwargs)



# Single-size pricing (Corollary 1) plus the closed-form oracle from App. B.3
def single_size_price(omega_i, sigma_i, x):
    """Price-demand map for one size: p = omega + sigma (1 - 2x) / (2 sqrt(x - x^2))."""
    return float(omega_i + sigma_i * (1.0 - 2.0 * x) / (2.0 * np.sqrt(x - x * x)))


def single_size_demand(omega_i, sigma_i, p):
    """Closed-form single-size CMM demand (paper App. B.3, eq. 23 solution).

    D(p) = 1/2 - (p - omega) / (2 sqrt((p - omega)^2 + sigma^2)); the CDF of a
    t-distribution with two degrees of freedom. Used as an exact demand oracle.
    """
    d = p - omega_i
    return float(0.5 - d / (2.0 * np.sqrt(d * d + sigma_i * sigma_i)))


def _single_size_root_lhs(x, omega_i, sigma_i, c_i):
    """LHS of Corollary 1: omega - c + sigma (4x^2 - 6x + 1) / (4(1-x) sqrt(x-x^2))."""
    return omega_i - c_i + sigma_i * (4 * x * x - 6 * x + 1) / (
        4.0 * (1.0 - x) * np.sqrt(x - x * x)
    )


def _single_size_profit(x, omega_i, sigma_i, c_i):
    p = single_size_price(omega_i, sigma_i, x)
    return (p - c_i) * x


def optimize_single_size(omega, Sigma, cost, sizes, tol=1e-9):
    """Best single-size policy under CMM (Corollary 1), with boundary handling.

    For each size i the optimal demand solves the Corollary 1 root equation on
    (0, 1). brentq is not assumed to bracket an interior root for every
    (omega_i, sigma_i, c_i): we look for an interior root and also compare the
    feasible boundary candidates by profit, returning a ``solution_type`` flag.
    sigma_i is the standard deviation sqrt(Sigma_ii).
    """
    omega = np.asarray(omega, dtype=float)
    Sigma = np.atleast_2d(np.asarray(Sigma, dtype=float))
    costs_by_size = normalize_costs(cost, sizes)
    sigmas = np.sqrt(np.diag(Sigma))

    per_size = []
    for idx in range(len(omega)):
        oi, si, ci = omega[idx], sigmas[idx], costs_by_size[idx]
        lo, hi = tol, 1.0 - tol

        candidates = []  # (x, solution_type)
        f_lo = _single_size_root_lhs(lo, oi, si, ci)
        f_hi = _single_size_root_lhs(hi, oi, si, ci)
        if np.isfinite(f_lo) and np.isfinite(f_hi) and f_lo * f_hi < 0:
            x_root = optimize.brentq(_single_size_root_lhs, lo, hi, args=(oi, si, ci))
            candidates.append((x_root, "interior_root"))
        # Boundary candidates: a near-zero-demand and a near-one-demand price.
        candidates.append((lo, "lower_boundary"))
        candidates.append((hi, "upper_boundary"))

        best = max(candidates, key=lambda cx: _single_size_profit(cx[0], oi, si, ci))
        x_star, sol_type = best
        p_star = single_size_price(oi, si, x_star)
        per_size.append({
            "size": int(np.asarray(list(sizes), dtype=int)[idx]),
            "demand": float(x_star),
            "price": float(p_star),
            "profit": float(_single_size_profit(x_star, oi, si, ci)),
            "solution_type": sol_type,
        })

    best_overall = max(per_size, key=lambda d: d["profit"])
    return {"best": best_overall, "per_size": per_size}



# Multi-size CMM optimizer (the concave program in native q variables)
def optimize_convex(omega, Sigma, cost, sizes, delta=1e-8, method="trust-constr"):
    """Multi-size BSP optimum under CMM, solved in the native q variables.

    Maximizes the concave pi(q) (Theorem 2) over {q_i >= delta, sum q <= 1 - delta}.
    The interior margin keeps iterates off the singular boundary while preserving
    concavity; the constraints are linear so only the objective is finite-
    differenced. Recovers p* = omega + grad_f(q*) and checks the round trip.
    Returns the optimum plus an ``activity`` report (the q_s > 1e-6 threshold) and
    the recovered prices.
    """
    omega = np.asarray(omega, dtype=float)
    Sigma = np.atleast_2d(np.asarray(Sigma, dtype=float))
    sizes = np.asarray(list(sizes), dtype=int)
    costs_by_size = normalize_costs(cost, sizes)
    m = omega.size

    def neg_obj(q):
        return -cmm_profit_q(q, omega, Sigma, costs_by_size)

    def neg_grad(q):
        return -numerical_profit_gradient(q, omega, Sigma, costs_by_size)

    res = _maximize_concave_on_simplex(neg_obj, neg_grad, m, delta, method=method)
    q_star = np.clip(res.x, 0.0, None)
    grad = cmm_grad_f(q_star, Sigma)
    p_star = omega + grad
    profit = cmm_profit_q(q_star, omega, Sigma, costs_by_size)

    # Round-trip check: does p_star reproduce q_star as CMM demand?
    rt = cmm_demand_simplex(p_star, omega, Sigma)
    roundtrip_err = float(np.max(np.abs(rt.q - q_star)))

    active = q_star > 1e-6
    return {
        "q": q_star,
        "q0_outside": float(1.0 - q_star.sum()),
        "prices": p_star,
        "profit": float(profit),
        "sizes": sizes,
        "success": bool(res.success),
        "status": str(res.message),
        "delta": delta,
        "min_q": float(q_star.min()),
        "active_sizes": sizes[active].tolist(),
        "prices_finite": bool(np.all(np.isfinite(p_star))),
        "prices_nonnegative": bool(np.all(p_star >= -1e-9)),
        "roundtrip_max_abs_err": roundtrip_err,
    }


def optimize_convex_softmax(omega, Sigma, cost, sizes, n_restarts=5, seed=0):
    """Diagnostic cross-check: optimize pi over softmax logits (m+1 categories).

    NOT the primary path: pi(softmax(z)) is generally non-concave in z, so this
    is only a robustness cross-check and a boundary diagnostic. The objective is
    finite-differenced directly in logit space (scipy default), so the q-space
    numerical_profit_gradient is never passed here. One logit (the outside option)
    is fixed at zero for identifiability. Reports max|logit| and the smallest
    probability, which flag a near-boundary, poorly conditioned solution.
    """
    omega = np.asarray(omega, dtype=float)
    Sigma = np.atleast_2d(np.asarray(Sigma, dtype=float))
    sizes = np.asarray(list(sizes), dtype=int)
    costs_by_size = normalize_costs(cost, sizes)
    m = omega.size
    rng = np.random.default_rng(seed)

    def q_of_z(z_free):
        z = np.concatenate([[0.0], z_free])  # outside logit fixed at 0
        z = z - z.max()
        e = np.exp(z)
        probs = e / e.sum()
        return probs[1:], probs  # size probs, full incl. outside

    def neg_obj(z_free):
        q, _ = q_of_z(z_free)
        return -cmm_profit_q(q, omega, Sigma, costs_by_size)

    best = None
    for _ in range(n_restarts):
        z0 = rng.normal(scale=1.0, size=m)
        res = optimize.minimize(neg_obj, z0, method="Nelder-Mead",
                                options={"xatol": 1e-9, "fatol": 1e-12, "maxiter": 20000})
        if best is None or res.fun < best.fun:
            best = res

    z_full = np.concatenate([[0.0], best.x])
    q, probs = q_of_z(best.x)
    grad = cmm_grad_f(q, Sigma)
    p_star = omega + grad
    return {
        "q": q,
        "q0_outside": float(probs[0]),
        "prices": p_star,
        "profit": float(-best.fun),
        "max_abs_logit": float(np.max(np.abs(z_full - z_full.mean()))),
        "min_probability": float(probs.min()),
    }



# Archived empirical sample-average objective and uncertified search
def _user_choice_indices(W, prices, sizes, costs_by_size):
    """Each user's chosen size index (or -1 for the outside option).

    Deterministic tie-break: the outside option wins when the best surplus is
    non-positive; among equal positive surpluses choose the lower-priced option;
    if price ties too, choose the smaller size. ``W`` is (n_users, m) of size-wise
    maximum valuations.

    Vectorized: columns are scanned in tie-break order (price asc, then size asc)
    and each user takes the first option within the float tolerance of their best
    surplus. Agrees with the per-user reference loop below except on surpluses
    that differ by less than the 1e-12 tolerance without being exactly tied,
    where either answer is a knife-edge float coincidence.
    """
    prices = np.asarray(prices, dtype=float)
    sizes = np.asarray(sizes, dtype=int)
    m = prices.size
    order = np.asarray(sorted(range(m), key=lambda j: (prices[j], sizes[j])),
                       dtype=int)
    surplus_ord = W[:, order] - prices[order]  # (n_users, m), tie-break order
    best = surplus_ord.max(axis=1)
    first_hit = np.argmax(surplus_ord >= (best[:, None] - 1e-12), axis=1)
    return np.where(best > 1e-12, order[first_hit], -1)


def _user_choice_indices_reference(W, prices, sizes, costs_by_size):
    """Per-user loop reference for ``_user_choice_indices`` (tests only)."""
    prices = np.asarray(prices, dtype=float)
    sizes = np.asarray(sizes, dtype=int)
    surplus = W - prices  # (n_users, m)

    n_users, m = W.shape
    choices = np.full(n_users, -1, dtype=int)
    # Order sizes by the tie-break key (price asc, then size asc) once.
    order = sorted(range(m), key=lambda j: (prices[j], sizes[j]))
    for u in range(n_users):
        best_j = -1
        best_surplus = 0.0  # outside option surplus
        for j in order:
            s = surplus[u, j]
            if s > best_surplus + 1e-12:  # strictly better than current best
                best_surplus = s
                best_j = j
            # ties resolved by the pre-sorted order (lower price, then size)
        choices[u] = best_j
    return choices


def evaluate_schedule(valuations, prices, sizes, cost, free_disposal=True,
                      W=None):
    """Realized sample-average objective of a size-price menu on an input panel.

    The result is conditional on the supplied assumed panel and retired CMM
    choice mechanism; it is neither model-free nor actual revenue.

    Each user picks the surplus-maximising offered size (or the outside option)
    under the deterministic tie-break, and contributes ``price - cost`` for the
    chosen size. Piecewise constant in the prices by construction.

    ``W`` optionally passes the precomputed size-wise maximum valuations
    (n_users, m); the top-s sums do not depend on the prices, so callers that
    evaluate many menus on the same sample compute them once.
    """
    sizes = np.asarray(list(sizes), dtype=int)
    costs_by_size = normalize_costs(cost, sizes)
    prices = np.asarray(prices, dtype=float)

    if W is None:
        valuations = np.asarray(valuations, dtype=float)
        W = _top_s_sums(valuations, sizes, free_disposal=free_disposal)
    choices = _user_choice_indices(W, prices, sizes, costs_by_size)
    margins = prices - costs_by_size
    realised = np.where(choices >= 0, margins[choices], 0.0)
    return float(realised.mean())


def optimize_empirical_schedule(valuations, sizes, cost, seed=0, free_disposal=True,
                                maxiter=300, popsize=15, x0=None):
    """Optimize the price menu directly against the per-user sample.

    The independent check on the moment-based CMM optimum. The objective is
    non-smooth (users switch at discrete surplus thresholds), so we use a
    derivative-free optimizer with deterministic price bounds 0 <= p_s <= max_u W_us.
    ``x0`` optionally seeds the population with a known-good menu (e.g. the CMM
    prices), so the search result weakly improves on it.
    """
    valuations = np.asarray(valuations, dtype=float)
    sizes = np.asarray(list(sizes), dtype=int)
    costs_by_size = normalize_costs(cost, sizes)
    W = _top_s_sums(valuations, sizes, free_disposal=free_disposal)
    upper = W.max(axis=0)

    def neg_obj(prices):
        return -evaluate_schedule(None, prices, sizes, costs_by_size,
                                  free_disposal=free_disposal, W=W)

    bounds = [(0.0, float(u)) for u in upper]
    if x0 is not None:
        x0 = np.clip(np.asarray(x0, dtype=float), 0.0, upper)
    res = optimize.differential_evolution(
        neg_obj, bounds, seed=seed, tol=1e-9, polish=True, maxiter=maxiter,
        popsize=popsize, x0=x0,
    )
    return {
        "prices": np.asarray(res.x, dtype=float),
        "profit": float(-res.fun),
        "sizes": sizes,
        "success": bool(res.success),
    }


# Benchmark policies (used by the Stage 3 and 4 notebooks)
def separate_selling_profit(valuations, cost=0.0):
    """Best separate-selling profit per user on a valuation sample.

    Each item gets its own independently optimized uniform price; a user buys
    every item they value above its price, so the total is the sum over items of
    the best single-item profit. The candidate prices for item j are the observed
    valuations v_{., j} (the profit is piecewise constant between them). ``cost``
    is the per-item marginal cost.
    """
    V = np.asarray(valuations, dtype=float)
    n_users = V.shape[0]
    total = 0.0
    for j in range(V.shape[1]):
        v = np.sort(V[:, j])[::-1]
        profit = (v - cost) * (np.arange(1, v.size + 1) / n_users)
        total += max(0.0, float(profit.max()))
    return total


def pure_bundling_profit(valuations, cost=0.0):
    """Best pure-bundling profit per user: one price for the grand bundle.

    The bundle valuation is the row sum (all items; with non-negative valuations
    this equals the size-n top sum), the candidate prices are the observed bundle
    valuations, and ``cost`` is the total cost of the grand bundle.
    """
    V = np.asarray(valuations, dtype=float)
    n_users = V.shape[0]
    w = np.sort(V.sum(axis=1))[::-1]
    profit = (w - cost) * (np.arange(1, w.size + 1) / n_users)
    return max(0.0, float(profit.max()))



# Brute-force oracles (kept separate: CMM vs empirical)
def brute_force_cmm_q(omega, Sigma, cost, sizes, grid=80):
    """Grid search over q of the CMM objective pi(q) (CMM-specific oracle).

    Searches the interior of the simplex on a regular grid. Intended for small m
    (m <= 3) as a test oracle for optimize_convex.
    """
    omega = np.asarray(omega, dtype=float)
    Sigma = np.atleast_2d(np.asarray(Sigma, dtype=float))
    sizes = np.asarray(list(sizes), dtype=int)
    costs_by_size = normalize_costs(cost, sizes)
    m = omega.size
    eps = 1e-4
    axis = np.linspace(eps, 1.0 - eps, grid)

    best_q, best_val = None, -np.inf
    if m == 1:
        for a in axis:
            val = cmm_profit_q(np.array([a]), omega, Sigma, costs_by_size)
            if val > best_val:
                best_val, best_q = val, np.array([a])
    elif m == 2:
        for a in axis:
            for b in axis:
                if a + b < 1.0 - eps:
                    val = cmm_profit_q(np.array([a, b]), omega, Sigma, costs_by_size)
                    if val > best_val:
                        best_val, best_q = val, np.array([a, b])
    elif m == 3:
        for a in axis:
            for b in axis:
                for c in axis:
                    if a + b + c < 1.0 - eps:
                        q = np.array([a, b, c])
                        val = cmm_profit_q(q, omega, Sigma, costs_by_size)
                        if val > best_val:
                            best_val, best_q = val, q
    else:
        raise ValueError("brute_force_cmm_q supports m <= 3")
    return {"q": best_q, "profit": float(best_val)}


def brute_force_empirical_prices(valuations, sizes, cost, grid=40, free_disposal=True):
    """Grid search over the price menu of the empirical objective.

    Same tie-break and price bounds as evaluate_schedule. For small m only.
    """
    valuations = np.asarray(valuations, dtype=float)
    sizes = np.asarray(list(sizes), dtype=int)
    costs_by_size = normalize_costs(cost, sizes)
    W = _top_s_sums(valuations, sizes, free_disposal=free_disposal)
    uppers = W.max(axis=0)
    m = sizes.size
    axes = [np.linspace(0.0, float(u), grid) for u in uppers]

    best_p, best_val = None, -np.inf
    grids = np.meshgrid(*axes, indexing="ij")
    flat = np.stack([g.ravel() for g in grids], axis=1)
    for prices in flat:
        val = evaluate_schedule(None, prices, sizes, costs_by_size,
                                free_disposal=free_disposal, W=W)
        if val > best_val:
            best_val, best_p = val, prices
    return {"prices": np.asarray(best_p, dtype=float), "profit": float(best_val)}
