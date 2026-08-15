# Stage 1 v2 mathematical appendix

This appendix records the equations actually used by cycle `s1-v2-20260814`. Every fitted output
is a latent score for ownership ranking. It is not money, willingness to pay, a purchase
probability, or an interpersonally comparable utility.

## Weighted implicit ALS

For binary ownership `o_ui`, lifetime playtime `t_ui`, and an observed-edge confidence

`c_ui = 1 + alpha_o + alpha_p min(log(1 + t_ui), tau)`,

the estimator minimizes

`sum_ui c_ui (o_ui - x_u' q_i)^2 + lambda (||X||_F^2 + ||Q||_F^2)`.

Unobserved cells have `o_ui = 0` and confidence one. The score is `s_ui = x_u' q_i`. The
production backend is `implicit==0.7.2` with its native exact least-squares solver, float32 stored
factors, float64 score accumulation, fixed iterations, and the frozen three training seeds. Joint
factorization is nonconvex; the exact statement applies only to each fixed-block ridge solve.

With item factors fixed, a new user is folded in by the unique ridge solution

`x_u = (Q' C_u Q + lambda I)^(-1) Q' C_u o_u`.

The implementation solves the linear system and does not construct an explicit matrix inverse.

## Feature-sum BPR

The pairwise score is

`s_ui = b_i + x_u' (eta_i + rho F_i G)`.

Identity-only sets `rho = 0` and allocates no active genre parameter block. Identity plus genre
sets `rho = 1`; this is the only controlled change. For sampled triples `(u,i,j)`, the summed
objective is

`sum_(u,i,j) log(1 + exp(-(s_ui-s_uj))) + lambda ||theta||_2^2`,

where the norm contains every active user, identity, genre, and item-bias parameter. The
cycle-namespaced PCG64 stream samples a training edge uniformly with replacement and then samples a
warm item uniformly, rejecting that user's training positives. It continues across all 12 epochs.

LightFM 1.17 could not be built in the frozen Windows/Python environment. Before validation access,
S1.5 activated the predeclared NumPy fallback. One million frozen triples define each epoch. The
fallback accumulates that epoch's summed gradient in float64, applies one coordinatewise AdaGrad
step with epsilon `1e-8`, and casts parameters to float32. Initialization is namespaced PCG64 normal
with mean zero and standard deviation `0.01`. The identity and genre fits start from identical
shared parameters and consume identical triples.

With item and genre parameters frozen, pairwise fold-in minimizes only over `x_u`:

`sum_(i,j in T_u) log(1 + exp(-x_u'(q_i-q_j)-(b_i-b_j))) + lambda ||x_u||_2^2`.

`T_u` uses each permitted warm positive once, in ascending item order, and one deterministic
cycle-and-user-namespaced negative per positive. Negatives are drawn from the full warm catalogue
and reject the user's permitted positives. Positive regularization makes this user-only objective
strictly convex. The solver is zero-initialized L-BFGS-B with tolerance `1e-8` and at most 250
iterations. Empty or full-catalogue histories receive a reported zero-vector fallback.

## Low-rank bounded scoring

For a user block `U` and item block `I`, ALS scores are `X_U Q_I'`. Pairwise scores are
`X_U (Eta_I + rho F_I G)' + b_I`. The rank of the centered score block is at most the latent
dimension. All evaluation uses bounded blocks; no full user-by-catalogue score matrix is saved.

One held-out ownership target is ranked against the complete frozen eligible catalogue after the
specified positive masks are applied. If `g` eligible scores are strictly above the target and its
exact score-tied block has size `e`, then

`E[Recall@K] = min(max(K-g,0),e)/e`

and

`E[NDCG@K] = (1/e) sum_(r=g+1)^(min(g+e,K)) 1/log2(r+1)`.

No numerical tolerance merges distinct score levels. Expected coverage and concentration use the
same fractional inclusion probability at a tied top-K boundary.

## Pseudo-utility boundary

Stage 1 freezes four deterministic, finite, nonnegative transformations. Their equations and all
fallbacks are in
`configs/cycles/s1-v2-20260814/pseudo_utility_scenarios.json`. These mappings are scenario
interfaces for Stage 2 robustness. None identifies a true cardinal utility scale, and no bundle
objective is used to choose or fit them.
