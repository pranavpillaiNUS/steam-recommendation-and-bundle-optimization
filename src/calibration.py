"""Archived monetary-anchor experiment from the retired CMM pipeline.

This module preserves the estimators and affine transformation used by notebook
09 before the mechanism and identification pivot. All tested price anchors fail
their sign checks on the Steam snapshot. Consequently, the transformation does
not identify utility, willingness to pay, a purchase probability, or a monetary
scale, and none of its outputs is a live input to CP-anchored SBA.

The historical function and field names are retained so the archived notebook
and regression tests remain reproducible. New work must use the separately
specified pseudo-utility interface described in ``planning.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import optimize, stats
from scipy.special import expit, logit

from . import bundle_pricing as bp


# ---------------------------------------------------------------------------
# Link functions
# ---------------------------------------------------------------------------

def _link_cdf(z, link):
    """Symmetric link CDF F, with F((a + b s - price)/sigma) = P(own)."""
    if link == "probit":
        return stats.norm.cdf(z)
    if link == "logit":
        return expit(z)
    raise ValueError(f"unknown link {link!r}; use 'probit' or 'logit'")


def _link_ppf(r, link):
    """Inverse link applied to ownership rates (the transformed response)."""
    if link == "probit":
        return stats.norm.ppf(r)
    if link == "logit":
        return logit(r)
    raise ValueError(f"unknown link {link!r}; use 'probit' or 'logit'")


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class CalibrationResult:
    """Result object for the archived, rejected affine monetary anchor.

    The historical fields ``a``, ``b``, and ``sigma`` are retained for archive API
    compatibility. ``valid=False`` records that the snapshot did not identify the
    intended monetary scale; no fallback result becomes an economic calibration.
    """

    a: float
    b: float
    sigma: float
    link: str
    valid: bool
    diagnostics: dict = field(default_factory=dict)

    def valuation(self, scores, free_disposal=True):
        """Apply the archived affine score transformation ``a + b * s``.

        The historical method name is API compatibility only. Returned numbers are
        not identified valuations or monetary quantities. ``free_disposal`` floors
        the assumed transformed scores at zero for the archived CMM experiment.
        """
        v = self.a + self.b * np.asarray(scores, dtype=float)
        if free_disposal:
            v = np.maximum(v, 0.0)
        return v


# ---------------------------------------------------------------------------
# Aggregate fit (primary): regress transformed rate on mean score and price
# ---------------------------------------------------------------------------

def _clip_rates(obs_rates, n_users):
    """Clip ownership rates off 0 and 1 before the link transform.

    A half-count continuity floor (0.5 / n_users) keeps the transform finite for
    games owned by nobody-but-one or by almost everyone.
    """
    obs_rates = np.asarray(obs_rates, dtype=float)
    floor = 0.5 / float(n_users)
    return np.clip(obs_rates, floor, 1.0 - floor)


def fit_calibration_aggregate(mean_scores, prices, obs_rates, n_users,
                              link="probit"):
    """Fit (a, b, sigma) by a linear regression of the transformed rate.

    For each game g with mean score ``s_bar_g`` and price ``price_g`` and observed
    ownership rate ``r_g``, the censored-link model implies

        F^{-1}(r_g) ~= gamma0 + gamma_s * s_bar_g + gamma_p * price_g,

    absorbing the within-game score dispersion into the link scale. Ordinary least
    squares on the transformed rates recovers the standardized coefficients, and

        sigma = -1 / gamma_p,   b = -gamma_s / gamma_p,   a = -gamma0 / gamma_p,

    so the dollar scale comes from the price coefficient. A positive dollar scale
    needs ``gamma_p < 0`` (higher price, less ownership, holding preference fixed);
    if that fails the fit is returned with ``valid=False``.

    Parameters
    ----------
    mean_scores : (G,) per-game mean preference score s_bar_g = Q_g . mean_u P_u
    prices : (G,) standalone prices (dollars)
    obs_rates : (G,) observed ownership rate ownership_count_g / n_users
    n_users : int, panel size (for the rate clip and standard errors)
    link : 'probit' or 'logit'
    """
    s = np.asarray(mean_scores, dtype=float)
    p = np.asarray(prices, dtype=float)
    G = s.size
    if not (s.shape == p.shape == np.asarray(obs_rates).shape):
        raise ValueError("mean_scores, prices, obs_rates must have equal shape")
    if G < 4:
        raise ValueError("need at least 4 priced games for the aggregate fit")

    z = _link_ppf(_clip_rates(obs_rates, n_users), link)
    X = np.column_stack([np.ones(G), s, p])
    beta, *_ = np.linalg.lstsq(X, z, rcond=None)
    resid = z - X @ beta
    dof = G - X.shape[1]
    sigma2 = float(resid @ resid) / dof if dof > 0 else np.nan
    XtX_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(sigma2 * XtX_inv))
    r2 = 1.0 - float(resid @ resid) / float(((z - z.mean()) ** 2).sum())

    gamma0, gamma_s, gamma_p = beta
    se0, se_s, se_p = se
    valid = bool(gamma_p < 0)

    if valid:
        sigma = -1.0 / gamma_p
        b = -gamma_s / gamma_p
        a = -gamma0 / gamma_p
    else:
        # The cross-section does not give a downward price slope, so a dollar scale
        # cannot be identified. Return finite placeholders and flag invalid.
        sigma = np.nan
        b = np.nan
        a = np.nan

    diagnostics = {
        "method": "aggregate",
        "n_games": int(G),
        "r2": r2,
        "gamma0": float(gamma0), "gamma_s": float(gamma_s), "gamma_p": float(gamma_p),
        "se_gamma0": float(se0), "se_gamma_s": float(se_s), "se_gamma_p": float(se_p),
        "t_gamma_p": float(gamma_p / se_p) if se_p > 0 else np.nan,
        "t_gamma_s": float(gamma_s / se_s) if se_s > 0 else np.nan,
        "price_coef_negative": valid,
    }
    return CalibrationResult(a=float(a), b=float(b), sigma=float(sigma),
                             link=link, valid=valid, diagnostics=diagnostics)


# ---------------------------------------------------------------------------
# Quantile anchor (the cleanest censoring test) and a normalization fallback
# ---------------------------------------------------------------------------

def censoring_quantiles(score_matrix, rates):
    """Per-game score quantile at the ownership threshold.

    For game g owned by a fraction ``r_g``, the marginal owner sits at the
    ``(1 - r_g)`` quantile of the user score distribution, so this returns
    ``q_g = quantile(scores_{.,g}, 1 - r_g)``. ``score_matrix`` is
    (n_users_sample, G); a user subsample is fine for the quantile estimate.
    """
    S = np.asarray(score_matrix, dtype=float)
    r = np.asarray(rates, dtype=float)
    if S.shape[1] != r.size:
        raise ValueError("score_matrix columns must match len(rates)")
    q = np.empty(r.size)
    for g in range(r.size):
        q[g] = np.quantile(S[:, g], 1.0 - r[g])
    return q


def fit_calibration_quantile(quantiles, prices):
    """Identify (a, b) from ``price_g = a + b * q_g`` across games.

    This is the per-game censoring statement done exactly: no cross-game
    monotonicity is assumed, the ownership rate selects the right quantile. A valid
    dollar scale needs ``b > 0`` (the threshold score rises with price). ``sigma``
    is set from the regression residual spread as an implied dollar noise scale; it
    does not enter the affine valuation map.
    """
    q = np.asarray(quantiles, dtype=float)
    p = np.asarray(prices, dtype=float)
    G = q.size
    if q.shape != p.shape:
        raise ValueError("quantiles and prices must have equal shape")
    if G < 4:
        raise ValueError("need at least 4 priced games")

    X = np.column_stack([np.ones(G), q])
    beta, *_ = np.linalg.lstsq(X, p, rcond=None)
    resid = p - X @ beta
    r2 = 1.0 - float(resid @ resid) / float(((p - p.mean()) ** 2).sum())
    a, b = float(beta[0]), float(beta[1])
    rho = float(stats.spearmanr(p, q).statistic)
    valid = bool(b > 0)
    sigma = float(np.std(resid, ddof=2)) if valid else np.nan
    diagnostics = {
        "method": "quantile",
        "n_games": int(G),
        "r2": r2,
        "spearman_price_quantile": rho,
        "b": b, "a": a,
        "price_slope_positive": valid,
    }
    return CalibrationResult(a=a if valid else np.nan,
                             b=b if valid else np.nan,
                             sigma=sigma, link="probit", valid=valid,
                             diagnostics=diagnostics)


def normalize_calibration(mean_scores, prices, b_scale=1.0, a_shift=0.0):
    """Set (a, b) by a transparent normalization, not a censoring identification.

    Used when the price anchors fail to identify a dollar scale. The scale ``b`` is
    chosen so the spread of mean-score-implied valuations matches the spread of
    prices, and the location ``a`` so their medians match, giving a dollar-like but
    explicitly assumed scale. ``b_scale`` and ``a_shift`` are the sensitivity knobs
    swept downstream: ``b -> b_scale * b`` and ``a -> a + a_shift`` (a_shift in
    dollars). Note that *both* knobs move the free-disposal floor in score units
    (``-a/b``) and therefore both can move the revenue ratios: ``a_shift`` shifts
    the floor directly, and ``b_scale`` shifts it too because ``a`` re-anchors to
    the price median at the new spread. The transformation the ratios are exactly
    invariant to is the joint rescaling ``(a, b) -> (lambda*a, lambda*b)``, which
    keeps ``-a/b`` fixed (verified in notebook 09); the floor location is the one
    substantive normalization parameter and the sweeps report it.

    ``valid`` is True (a usable scale is produced), but the diagnostics mark it as a
    normalization so nothing reads it as identified.
    """
    s = np.asarray(mean_scores, dtype=float)
    p = np.asarray(prices, dtype=float)
    s_sd = float(np.std(s))
    if s_sd == 0:
        raise ValueError("mean_scores has zero spread; cannot normalize")
    b = b_scale * float(np.std(p)) / s_sd
    a = (float(np.median(p)) - b * float(np.median(s))) + a_shift
    diagnostics = {
        "method": "normalization",
        "b_scale": float(b_scale),
        "a_shift": float(a_shift),
        "note": "scale assumed, not identified from price; swept in sensitivity",
    }
    return CalibrationResult(a=a, b=b, sigma=float(np.std(p)), link="probit",
                             valid=True, diagnostics=diagnostics)


# ---------------------------------------------------------------------------
# Nonlinear refinement (robustness): match the full per-user averaged rate
# ---------------------------------------------------------------------------

def predicted_rates(score_matrix, prices, a, b, sigma, link="probit"):
    """Model ownership rate per game, averaged over the user score distribution.

    ``predicted_rate_g = mean_u F((a + b * s_{u,g} - price_g) / sigma)`` using the
    full per-user scores rather than the per-game mean. ``score_matrix`` is
    (n_users_sample, G).
    """
    S = np.asarray(score_matrix, dtype=float)
    p = np.asarray(prices, dtype=float)
    z = (a + b * S - p[None, :]) / sigma
    return _link_cdf(z, link).mean(axis=0)


def refine_calibration_rates(score_matrix, prices, obs_rates, calib0,
                             link="probit"):
    """Refine (a, b, sigma) by least squares on the per-user averaged rate.

    Starts from an aggregate-fit ``calib0`` and drops its within-game Jensen
    approximation by averaging the link over the supplied per-user score sample.
    ``sigma`` and ``b`` are optimized in log space to keep them positive.
    """
    S = np.asarray(score_matrix, dtype=float)
    p = np.asarray(prices, dtype=float)
    r = np.asarray(obs_rates, dtype=float)
    if not np.isfinite([calib0.a, calib0.b, calib0.sigma]).all() or calib0.b <= 0:
        raise ValueError("refine needs a valid aggregate calib0 with b > 0")

    def unpack(theta):
        a = theta[0]
        b = np.exp(theta[1])
        sigma = np.exp(theta[2])
        return a, b, sigma

    def resid(theta):
        a, b, sigma = unpack(theta)
        return predicted_rates(S, p, a, b, sigma, link) - r

    theta0 = np.array([calib0.a, np.log(calib0.b), np.log(calib0.sigma)])
    res = optimize.least_squares(resid, theta0, method="lm")
    a, b, sigma = unpack(res.x)
    pred = predicted_rates(S, p, a, b, sigma, link)
    ss_res = float(((pred - r) ** 2).sum())
    ss_tot = float(((r - r.mean()) ** 2).sum())
    diagnostics = {
        "method": "refine_rates",
        "n_games": int(S.shape[1]),
        "n_user_sample": int(S.shape[0]),
        "rate_rmse": float(np.sqrt(ss_res / S.shape[1])),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan,
        "success": bool(res.success),
        "from_aggregate": {"a": calib0.a, "b": calib0.b, "sigma": calib0.sigma},
    }
    return CalibrationResult(a=float(a), b=float(b), sigma=float(sigma),
                             link=link, valid=True, diagnostics=diagnostics)


# ---------------------------------------------------------------------------
# Archived transformed-score construction over an item set
# ---------------------------------------------------------------------------

def mean_scores(P, Q):
    """Per-game mean preference score s_bar_g = Q_g . mean_u P_u for all games."""
    return np.asarray(Q, dtype=float) @ np.asarray(P, dtype=float).mean(axis=0)


def score_matrix(P, Q, item_indices, user_indices=None):
    """Preference-score block s_{u,g} = P_u . Q_g for the given items.

    Returns (n_users_used, len(item_indices)). With ``user_indices=None`` all users
    are used; pass a sample for the nonlinear refinement to bound memory.
    """
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    item_indices = np.asarray(item_indices, dtype=int)
    Psub = P if user_indices is None else P[np.asarray(user_indices, dtype=int)]
    return Psub @ Q[item_indices].T


def calibrated_valuations(P, Q, item_indices, calib, free_disposal=True):
    """Archived per-user affine-transformed score vectors for an item set.

    Returns ``(n_users, len(item_indices))``. This preserves the historical
    computation without granting its output a monetary or utility interpretation.
    """
    S = score_matrix(P, Q, item_indices)
    return calib.valuation(S, free_disposal=free_disposal)


def valuation_moments(P, Q, item_indices, sizes, calib,
                      cov_method="ledoit_wolf", free_disposal=True):
    """Feed archived affine-transformed scores to the retired CMM bridge.

    Wires ``calibrated_valuations`` into the existing moment bridge: forms the
    per-user top-s partial sums, then ``(omega, Sigma)`` with the chosen covariance
    estimator. ``Sigma`` regularization matters because successive partial sums are
    mechanically near-collinear; ``ledoit_wolf`` is the documented default.

    Returns ``(omega, Sigma, V, info)`` where ``V`` is the archived transformed
    score matrix. This function is not part of the live SBA interface.
    """
    V = calibrated_valuations(P, Q, item_indices, calib, free_disposal=free_disposal)
    omega, _sample_sigma, W = bp.moments_from_valuations(V, sizes)
    Sigma, cov_info = bp.estimate_covariance(W, method=cov_method)
    info = {"cov": cov_info,
            "calibration": {"a": calib.a, "b": calib.b, "sigma": calib.sigma,
                            "link": calib.link, "valid": calib.valid}}
    return omega, Sigma, V, info
