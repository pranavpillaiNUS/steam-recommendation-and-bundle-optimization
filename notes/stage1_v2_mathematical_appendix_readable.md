# Stage 1 v2 mathematical appendix: reading version

Presentation updated: 2026-10-02.

I use this version for reading the equations in GitHub and VS Code Markdown preview. It typesets
the same model and evaluation equations as the [frozen appendix](stage1_v2_mathematical_appendix.md).
I keep that original unchanged because its bytes are recorded in the Stage 1 evidence manifest.
This presentation change does not alter the model, fitted parameters, results, or evidence ID.

Every fitted output from cycle `s1-v2-20260814` is a latent score for ownership ranking. It is not
money, willingness to pay, a purchase probability, or an interpersonally comparable utility.

## Weighted implicit ALS

For binary ownership $o_{ui}$, lifetime playtime $t_{ui}$, and observed-edge confidence

```math
c_{ui}=1+\alpha_o+\alpha_p\min\{\log(1+t_{ui}),\tau\},
```

the estimator minimizes

```math
\sum_{u,i}c_{ui}(o_{ui}-x_u^\top q_i)^2
+\lambda\left(\lVert X\rVert_F^2+\lVert Q\rVert_F^2\right).
```

Unobserved cells have $o_{ui}=0$ and confidence one. The score is $s_{ui}=x_u^\top q_i$.
The production backend is `implicit==0.7.2` with its native exact least-squares solver, float32
stored factors, float64 score accumulation, fixed iterations, and the frozen three training
seeds. Joint factorization is nonconvex. Exactness applies only to each fixed-block ridge solve.

With item factors fixed, a new user is folded in by the unique ridge solution

```math
x_u=(Q^\top C_uQ+\lambda I)^{-1}Q^\top C_uo_u.
```

The implementation solves the linear system and does not construct an explicit matrix inverse.

## Feature-sum BPR

The pairwise score is

```math
s_{ui}=b_i+x_u^\top(\eta_i+\rho F_iG).
```

Identity-only sets $\rho=0$ and allocates no active genre parameter block. Identity plus genre
sets $\rho=1$. This is the only controlled change. For sampled triples $(u,i,j)$, the summed
objective is

```math
\sum_{(u,i,j)}\log\left(1+\exp\{-(s_{ui}-s_{uj})\}\right)
+\lambda\lVert\theta\rVert_2^2.
```

The norm contains every active user, identity, genre, and item-bias parameter. The cycle-namespaced
PCG64 stream samples a training edge uniformly with replacement and then a warm item uniformly,
rejecting that user's training positives. It continues across all 12 epochs.

LightFM 1.17 could not be built in the frozen Windows/Python environment. Before validation
access, S1.5 activated the predeclared NumPy fallback. One million frozen triples define each
epoch. The fallback accumulates that epoch's summed gradient in float64, applies one
coordinatewise AdaGrad step with epsilon `1e-8`, and casts parameters to float32. Initialization
is namespaced PCG64 normal with mean zero and standard deviation `0.01`. Identity and genre fits
start from identical shared parameters and consume identical triples.

With item and genre parameters frozen, pairwise fold-in minimizes only over $x_u$:

```math
\sum_{(i,j)\in T_u}
\log\left(1+\exp\{-x_u^\top(q_i-q_j)-(b_i-b_j)\}\right)
+\lambda\lVert x_u\rVert_2^2.
```

$T_u$ uses each permitted warm positive once, in ascending item order, and one deterministic
cycle-and-user-namespaced negative per positive. Negatives are drawn from the full warm catalogue
and reject the user's permitted positives. Positive regularization makes this user-only objective
strictly convex. The solver is zero-initialized L-BFGS-B with tolerance `1e-8` and at most 250
iterations. Empty or full-catalogue histories receive a reported zero-vector fallback.

## Low-rank bounded scoring

For user block $U$ and item block $I$, ALS scores are

```math
S_{U,I}=X_UQ_I^\top.
```

Pairwise scores are

```math
S_{U,I}=X_U(\mathrm{Eta}_I+\rho F_IG)^\top+b_I,
```

with the item-bias row broadcast across users. The centered score block has rank at most the
latent dimension. All evaluation uses bounded blocks. No full user-by-catalogue score matrix is
saved.

One held-out ownership target is ranked against the complete frozen eligible catalogue after the
specified positive masks are applied. If $g$ eligible scores are strictly above the target and its
exact score-tied block has size $e$, then

```math
\mathbb{E}[\mathrm{Recall}\mathord{@}K]=\frac{\min\{\max(K-g,0),e\}}{e}
```

and

```math
\mathbb{E}[\mathrm{NDCG}\mathord{@}K]
=\frac{1}{e}\sum_{r=g+1}^{\min(g+e,K)}\frac{1}{\log_2(r+1)}.
```

An empty sum is zero. No numerical tolerance merges distinct score levels. Expected coverage and
concentration use the same fractional inclusion probability at a tied top $K$ boundary.

## Pseudo-utility boundary

Stage 1 freezes four deterministic, finite, nonnegative transformations. Their equations and
fallbacks are in `configs/cycles/s1-v2-20260814/pseudo_utility_scenarios.json`. These mappings are
scenario interfaces for Stage 2 robustness. None identifies a true cardinal utility scale, and no
bundle objective is used to choose or fit them.
