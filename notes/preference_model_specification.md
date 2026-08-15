# Stage 1 preference-model specification

Last revised: 2026-07-30

Status: governing S1.4 mathematical and artifact contract. It specifies the model ladder and
numerical conventions frozen in `configs/preference_models.json` under cycle
`s1-v1-20260718`. It does not report a fitted model or an empirical result.

## 1. Scope and interpretation

Stage 1 reconstructs held-out ownership within the observed Australian-user Steam snapshot. For
user $u$ and game $i$, let $o_{ui}\in\{0,1\}$ denote ownership and let $t_{ui}\ge 0$ denote
playtime. An unobserved pair is not a negative rating, proof of exposure, or purchase occasion.

Every fitted output is a latent preference score. It is not a utility, valuation, willingness to
pay, purchase probability, consumer surplus, or monetary quantity. The random interaction
holdouts have no acquisition timestamps, so their evaluation is held-out ownership reconstruction,
not future-purchase prediction.

The required model ladder is:

1. training-only popularity;
2. weighted regularized implicit matrix factorization;
3. identity-only feature-sum pairwise matrix factorization; and
4. the same pairwise estimator with the frozen genre block enabled.

Tags and other predictive metadata are outside this contract. Price, bundle membership, ownership
popularity, and playtime are not item features. Playtime appears only in the confidence rule for
the weighted implicit model.

All real-data implementations must bind to the frozen protocol, interaction, split, and feature
manifest IDs. S1.4 uses synthetic oracles to test the equations and artifact interfaces. It does
not access validation outcomes, design-test targets, assessment users, the reserved pseudo-cold
cohort, Stage 2 objectives, candidate-pool outcomes, or bundle outcomes.

## 2. Shared maps and numerical conventions

The row and column maps are explicit inputs, never inferred from incidental data order:

- user IDs are unique ascending nonnegative `int64` values;
- item IDs are unique ascending nonnegative `int64` values;
- the training ownership and playtime matrices share identical maps and sparse structure;
- model item rows are projected explicitly to the frozen design-training item map; and
- feature rows are checked against their declared item IDs before scoring or fitting.

Binary ownership, observed playtime, and item features are canonical CSR matrices. No
$U\times I$ score or confidence matrix is materialized. Requested score blocks must respect the
frozen memory bound.

The frozen numerical policy is:

- stored floating-point parameters use `float32`;
- objectives, gradients, normal equations, and diagnostics accumulate in `float64`;
- BLAS uses one thread where the implementation permits control;
- ALS uses fixed iterations and pairwise training uses fixed epochs;
- nonfinite parameters, objectives, gradients, or scores invalidate that seed;
- an invalid seed is logged and is not silently retried; and
- the only permitted ALS diagonal-jitter sequence is $0$, $10^{-8}$, then $10^{-6}$.

The three frozen training seeds are 104729, 130363, and 155921. Runtime varies with machine load
and is reported, but it is not a model-selection tie-break.

Each fit is limited to 8 GiB peak memory, 2,700 wall-clock seconds, and 256 MiB of saved model
parameters per seed. The complete tuning run is later limited to 43,200 wall-clock seconds. These
budgets are recorded here for interface validation; S1.4 does not start the tuning clock.

## 3. Training-only popularity

For each frozen design-training item, the popularity score is its design-training ownership count:

$$
d_i=\sum_u o_{ui}.
$$

Every user receives the same item score:

$$
s^{\mathrm{pop}}_{ui}=d_i.
$$

The estimator has configuration ID `popularity_train_count` and no trainable parameters. Counts
are accumulated exactly as nonnegative integers and may be cast to the declared scorer dtype only
at the scoring boundary. Validation, test, assessment, and pseudo-cold interactions never enter
$d_i$. Exact count ties remain ties and are handled later by the frozen expected-tie ranking rule.

The saved contract records the training item map, the integer count vector, its semantic hash, and
the training ownership hash.

## 4. Weighted regularized implicit matrix factorization

### 4.1 Preference and confidence

The preference target is binary ownership:

$$
p_{ui}=o_{ui}.
$$

Confidence is

$$
c_{ui}
=1+\alpha_o o_{ui}
+\alpha_p\min\{\log(1+t_{ui}),\tau\},
$$

with $\alpha_o>0$, $\alpha_p\ge 0$, and $\tau\ge 0$. Therefore an owned but unplayed pair has
confidence $1+\alpha_o>1$, while every unobserved pair has the implicit baseline confidence one.
The three frozen confidence schemes are ownership-only, log-playtime, and capped-log-playtime.

| Confidence scheme | $\alpha_p$ | $\tau$ |
| --- | ---: | ---: |
| `ownership_only` | 0 | 0 |
| `log_playtime` | 2 | 14 |
| `capped_log_playtime` | 2 | 5 |

S1.4 prospectively resolves the generic $t_{ui}$ field in the frozen equation as
`playtime_forever`. This field is selected without inspecting any model outcome. It is the
lifetime playtime measure aligned to the ownership edge and is less dependent on an arbitrary
two-week observation window. `playtime_2weeks` remains outside the headline confidence grid. Both
playtime matrices still remove validation and test ownership pairs under the S1.2 leakage
contract.

Only confidence values on observed ownership edges are stored. The unobserved baseline of one is
handled algebraically and is never expanded into a dense matrix.

### 4.2 Objective

For rank $k$, user factors $X\in\mathbb R^{U\times k}$, and item factors
$Q\in\mathbb R^{I\times k}$, the frozen objective is

$$
L_{\mathrm{ALS}}(X,Q)
=
\sum_{u=1}^{U}\sum_{i=1}^{I}
c_{ui}\left(o_{ui}-x_u^\top q_i\right)^2
+\lambda_x\lVert X\rVert_F^2
+\lambda_q\lVert Q\rVert_F^2.
$$

The core grid uses $\lambda_x=\lambda_q=\lambda>0$. Unequal regularizers are not part of the
frozen core. The grid contains $k\in\{32,64\}$, $\lambda\in\{0.05,0.2\}$,
$\alpha_o\in\{20,40\}$, and the three frozen confidence schemes. Every configuration runs for
exactly 12 alternating iterations. The preferred ALS backend requests four model threads while the
underlying BLAS policy remains one thread.

The objective can be evaluated without a dense score matrix. A valid decomposition is

$$
\begin{aligned}
L_{\mathrm{ALS}}
={}&
\sum_{u,i}\left(o_{ui}-x_u^\top q_i\right)^2\\
&+\sum_{(u,i):o_{ui}=1}
(c_{ui}-1)\left(1-x_u^\top q_i\right)^2\\
&+\lambda_x\lVert X\rVert_F^2
+\lambda_q\lVert Q\rVert_F^2,
\end{aligned}
$$

where the first double sum is computed through Gram matrices and sparse observed-edge corrections.

### 4.3 Normal equations

Let $\mathcal I_u=\{i:o_{ui}=1\}$ and let $Q_u$ contain the corresponding item-factor rows. Let
$C_u^{\mathrm{obs}}$ be the diagonal matrix of their observed confidence values. With $Q$ fixed,
the exact user update solves

$$
A_u x_u=r_u,
$$

where

$$
A_u
=Q^\top Q
+Q_u^\top(C_u^{\mathrm{obs}}-I)Q_u
+\lambda_x I
$$

and

$$
r_u=Q_u^\top C_u^{\mathrm{obs}}\mathbf 1.
$$

Similarly, with $\mathcal U_i=\{u:o_{ui}=1\}$ and $X_i$ containing the corresponding user-factor
rows, the item update solves

$$
A_i q_i=r_i,
$$

with

$$
A_i
=X^\top X
+X_i^\top(C_i^{\mathrm{obs}}-I)X_i
+\lambda_q I
$$

and

$$
r_i=X_i^\top C_i^{\mathrm{obs}}\mathbf 1.
$$

The displayed inverse form in explanatory prose defines the mathematical solution only. Code uses
a Cholesky factorization or a stable linear solve and never forms an explicit inverse.

For fixed $Q$, every user block is a strictly convex ridge problem. For fixed $X$, every item
block is a strictly convex ridge problem. The joint factorization is nonconvex. Alternating block
descent, a decreasing objective trace, or convergence of the updates is not a global-optimality
certificate.

### 4.4 Linear solves and diagnostics

Each normal equation is accumulated in `float64`. The solver first attempts the unmodified system.
If factorization or solution fails, it may retry only with the next frozen diagonal jitter. Every
jitter use is counted. Failure after the final jitter invalidates the seed.

For a solved system $Az=r$, record the relative residual

$$
\epsilon_{\mathrm{solve}}
=
\frac{\lVert Az-r\rVert_2}
{\max\{1,\lVert r\rVert_2\}}.
$$

The ALS diagnostic record contains:

- configuration ID, seed, rank, regularization, confidence scheme, and backend;
- requested and completed iteration counts;
- objective values after initialization, each user update, and each item update;
- maximum and mean user- and item-solve residuals by iteration;
- solve counts at each jitter value and failed-solve counts;
- factor norms, maximum absolute factor entries, and nonfinite checks;
- wall time, peak-memory estimate, and serialized-model bytes; and
- a terminal status chosen from `complete`, `nonfinite`, `linear_solve_failure`,
  `resource_limit`, or `exception`.

Objective changes are reported. Small numerical increases are not silently rewritten or described
as theoretical monotonicity.

## 5. Pairwise feature-sum matrix factorization

### 5.1 Feature and score equation

Let $x_u\in\mathbb R^k$ be the user vector, $\eta_i\in\mathbb R^k$ the item-identity vector,
$g_a\in\mathbb R^k$ the vector for genre feature $a$, and $b_i$ the item bias. The item vector and
score are

$$
q_i
=
\eta_i+\rho\sum_a f_{ia}g_a
$$

and

$$
s_{ui}=b_i+x_u^\top q_i.
$$

The genre values $f_{ia}$ are the frozen L1-normalized multi-hot features. A genre-covered item has
$\sum_a f_{ia}=1$; a missing-genre item has a zero content row. Identity has weight one, genre has
block weight one, and the combined row is not renormalized.

The controlled toggle is fixed rather than tuned:

- identity-only uses $\rho=0$ and does not allocate or regularize genre parameters;
- identity plus genre uses $\rho=1$ and the frozen genre block; and
- item bias is enabled while user bias is absent in both models.

No other loss, sampler, rank, regularization, learning rate, epoch count, sample weight, stopping
rule, seed, or tuning budget changes between the controlled pair.

The frozen pairwise grid uses $k\in\{32,64\}$, $\lambda\in\{10^{-4},10^{-3}\}$, learning rate
$0.05$, AdaGrad, 12 fixed epochs, 1,000,000 sampled positives per epoch, one negative per positive,
and one training thread. Every sampled triple occurrence has unit weight.

### 5.2 Deterministic triple sampler

The training edge order is canonical CSR order: ascending user row, then ascending item column.
Let $E$ be the number of design-training edges and let $I$ be the number of frozen warm catalogue
items. Each epoch produces exactly 1,000,000 ordered triples, with one negative per positive.

The ordered multiset is generated as follows:

1. draw one edge index uniformly from $\{0,\ldots,E-1\}$ with replacement;
2. let $(u,i)$ be that positive edge;
3. draw an item index uniformly from $\{0,\ldots,I-1\}$ with replacement;
4. reject and redraw only while the candidate is in user $u$'s design-training positives; and
5. accept the first permitted candidate as $j$.

Repeated positives, negatives, and complete triples are allowed. Rejection is based only on
training positives. The sampler does not load validation or test targets, and it does not reject a
candidate because it is a held-out positive.

Before consuming random numbers, the sampler verifies that every user represented by a positive
edge has at least one warm item outside that user's training positives. A user who owns the entire
warm catalogue makes the sampler invalid; the implementation fails explicitly instead of entering
an unbounded rejection loop.

The reference generator is NumPy `Generator(PCG64)`. Independent streams are derived from the
training seed by interpreting the first eight SHA-256 digest bytes as one unsigned big-endian
integer. The digest input is the UTF-8 canonical string

`s1-v1-20260718:bpr:<training_seed>:<purpose>`

where `<purpose>` is one of `shared-parameter-initialization`,
`genre-parameter-initialization`, or `triple-sampler`. The triple-sampler stream continues across
epochs and is not reset at an epoch boundary. Draws occur in the scalar order above, including
every rejected negative draw.

The shared parameter stream initializes $X$, $\eta$, and $b$ in the same canonical order for the
identity and genre runs. Genre parameters use their separate stream. The triple stream is generated
independently of model parameters and is identical for the controlled pair. Each epoch records a
SHA-256 hash of its ordered triples, encoded row by row as little-endian `int64`
$(u,i,j)$ indices, so stream equality can be verified without committing the triple data.

The parameter initialization distribution and scale remain backend details for the S1.5
feasibility decision. Regardless of that decision, the shared initial parameter arrays must be
identical for the controlled pair, and parameter initialization must not advance the triple stream.

These stream details are deterministic, outcome-free S1.4 implementation clarifications. A backend
that cannot consume or reproduce the declared stream is assessed in S1.5 and cannot silently
replace it.

### 5.3 Summed regularized loss

For an ordered sampled multiset $\mathcal T$, define

$$
\Delta_{uij}=s_{ui}-s_{uj}
$$

and

$$
L_{\mathrm{BPR}}(\theta)
=
\sum_{(u,i,j)\in\mathcal T}
\log\left(1+\exp\{-\Delta_{uij}\}\right)
+\lambda\lVert\theta\rVert_2^2.
$$

The softplus term is evaluated with a stable `logaddexp`-equivalent calculation. The norm contains
every active entry of $X$, $\eta$, $b$, and, when genre is enabled, $G$. There is one frozen
regularization value for all active parameters. Biases are regularized. There is no user-bias
parameter.

This is a summed objective, not an average over triples. Duplicate sampled triples contribute once
per occurrence. Oracle objective and gradient calculations accumulate all data terms and add
$\lambda\lVert\theta\rVert_2^2$ and $2\lambda\theta$ once. A backend-specific per-event
regularization convention is not assumed equivalent unless S1.5 demonstrates the required
parameter mapping.

### 5.4 Analytic gradients

For one triple, let

$$
h_{uij}=\sigma(-\Delta_{uij})
=\frac{1}{1+\exp(\Delta_{uij})}.
$$

The data-loss contributions are

$$
\frac{\partial \ell_{uij}}{\partial b_i}=-h_{uij},
\qquad
\frac{\partial \ell_{uij}}{\partial b_j}=h_{uij},
$$

$$
\frac{\partial \ell_{uij}}{\partial x_u}
=-h_{uij}(q_i-q_j),
$$

$$
\frac{\partial \ell_{uij}}{\partial \eta_i}
=-h_{uij}x_u,
\qquad
\frac{\partial \ell_{uij}}{\partial \eta_j}
=h_{uij}x_u,
$$

and, for each genre feature $a$,

$$
\frac{\partial \ell_{uij}}{\partial g_a}
=
-h_{uij}\rho(f_{ia}-f_{ja})x_u.
$$

Contributions from repeated parameter indices are summed. After all triple contributions are
accumulated, add $2\lambda$ times each active parameter to its gradient. Finite differences on tiny
problems must verify every active parameter family in `float64`.

Joint pairwise training is nonconvex. A stable loss trace, fixed-seed repeatability, or agreement
between two implementations is not a global-optimality certificate.

### 5.5 Identity and content equivalence

The following equalities are exact score-contract tests:

- with $\rho=0$, the score is $b_i+x_u^\top\eta_i$ regardless of any supplied genre values;
- an item with $f_{ia}=0$ for every $a$ has the identity-only score even when $\rho=1$;
- removing the genre block and setting $\rho=0$ produce the same scores for common parameters; and
- enabling genre never rescales or changes the identity block.

The identity-only implementation omits $G$ entirely. Merely multiplying $G$ by zero while still
regularizing it would change the objective and parameter count, so that is not the controlled
identity path.

### 5.6 Pairwise diagnostics

The pairwise diagnostic record contains:

- configuration ID, seed, rank, regularization, learning rate, feature mode, and backend;
- requested and completed epochs and samples per epoch;
- ordered-triple hash and accepted/rejected draw counts for every epoch;
- summed data loss, regularization term, total loss, and gradient norms;
- parameter norms, maximum absolute parameter entries, and nonfinite checks;
- wall time, peak-memory estimate, and serialized-model bytes; and
- a terminal status chosen from `complete`, `nonfinite`, `sampler_failure`, `resource_limit`,
  or `exception`.

Fixed epochs are used. A secondary backend may expose extra convergence diagnostics, but those
diagnostics do not alter the frozen stopping rule.

## 6. Parameter and scorer schemas

Every model declares its family, configuration ID, training seed where applicable, active feature
mode, factor dimension, map hashes, protocol ID, split-set ID, feature-set ID where applicable,
backend name and version, numerical dtype, and producer-source hash.

The minimum parameter arrays are:

| Family | Required arrays |
| --- | --- |
| Popularity | ascending `item_ids` and exact nonnegative `item_counts` |
| Weighted implicit MF | ascending `user_ids`, ascending `item_ids`, `user_factors[U,k]`, and `item_factors[I,k]` |
| Identity BPR | ascending maps, `user_factors[U,k]`, `identity_factors[I,k]`, and `item_bias[I]` |
| Genre BPR | the identity arrays plus `feature_factors[A,k]`, the ordered genre-feature map, and its feature-set hash |

Stored floating arrays are finite `float32`; IDs and counts are nonnegative `int64`. Array shapes,
ordered-map hashes, semantic hashes, and physical file hashes are recorded. Feature and factor rows
must fail closed on a pure permutation.

The scorer exposes only requested user and item blocks. Popularity broadcasts the requested item
counts. Weighted implicit MF computes $X_{\mathcal U}Q_{\mathcal I}^\top$. Pairwise scoring
reconstructs the requested $q_i$ from the declared identity and genre parameters, then adds $b_i$.
The scorer checks shapes, maps, finiteness, and the frozen maximum block size before allocating the
result. It has no default path that scores and saves the complete user-by-item matrix.

## 7. Serialization contract

Binary parameters use non-pickle NumPy or sparse formats with `allow_pickle=False`. A canonical
JSON manifest records:

- cycle, protocol, interaction, split, feature, configuration, and model IDs;
- model family, backend, version, seed, active features, and numerical policy;
- parameter names, shapes, dtypes, semantic hashes, physical hashes, and byte counts;
- training-input hashes and ordered user, item, and feature-map hashes;
- the complete diagnostic schema and terminal status; and
- producer-source hashes.

Saving uses a staged directory, verifies the staged artifact, and publishes the manifest last. A
partial or nonidentical frozen artifact is not overwritten. Loading verifies the manifest ID, model
ID, exact artifact inventory, access class, path scope, sizes, hashes, dtypes, shapes, maps, and
score semantics before returning a scorer.

A clean-process serialization test must:

1. fit or construct a tiny synthetic model;
2. score a declared user-item block directly from the written equation;
3. save and reload the model;
4. verify byte equality of every parameter array; and
5. reproduce the same requested scores under the frozen single-thread numerical policy.

Real model binaries, full score matrices, and sampled triple streams are regenerated artifacts and
remain outside Git. Compact redacted manifests and aggregate diagnostic tables remain visible to
Git.

## 8. Backend boundary for S1.5

The frozen preferred backends are `implicit` 0.7.2 for weighted implicit factorization and
`lightfm` 1.17 for pairwise feature-sum factorization. S1.4 defines equations, independent oracles,
and artifact contracts only. It does not establish that either package is installed, reproducible,
correctly oriented, or equivalent to this specification in the current Windows environment.

S1.5 must test, before any grid run:

- package matrix orientation and confidence conventions against the dense ALS oracle;
- normal-equation and objective agreement on a tiny problem;
- identity and genre score construction;
- the summed-loss and regularization convention or an explicit parameter mapping;
- deterministic triple-stream handling and controlled-pair equality;
- batched versus direct scoring;
- serialization and clean-process score reproduction;
- pseudo-cold identity suppression on synthetic items; and
- fold-in on synthetic or design users without mutating shared parameters.

If a preferred backend cannot satisfy the contract, the project uses only the prospective,
versioned fallback procedure in the frozen plan. Installing a package, selecting a fallback, or
claiming backend equivalence is not part of S1.4.

## 9. Claim and access boundary

No training trace proves a global optimum for either nonconvex joint estimator. Factor coordinates
and rotations have no identified economic interpretation. Genre lift, if later observed, is an
association in held-out reconstruction and is not a causal effect of genre on preference.

S1.4 mathematical oracles and serialization tests use synthetic data. Public manifest aggregates
may be used to bind schemas and hashes, but S1.4 does not fit a real-data model. Validation is first
used for tuning in S1.6. Design-test targets remain sealed until the validation-selected admission
set is hashed. The pseudo-cold cohort remains reserved for S1.8, and assessment users remain sealed
until S1.10. Stage 2 objectives and bundle outcomes remain unavailable throughout Stage 1
selection.
