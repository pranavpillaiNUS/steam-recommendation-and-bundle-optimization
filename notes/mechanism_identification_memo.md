# Mechanism and identification specification

Frozen: 2026-07-17

Status: internal Gate 0 specification. This document supersedes the 2026-07-14 four-decision
memo as the active project contract. Later changes require a dated, versioned amendment made
before inspecting the affected validation or optimization outcomes.

Backing detail is in `planning.md`, `notes/optimization_models.md`,
`notes/assumptions_and_limitations.md`, and the reproducible mechanism-audit artifacts.

## Finding 1: the original model was mismatched to the market

The cross-moment bundle-size pricing model (CMM) prices a size-based menu: a customer freely
chooses any bundle of size $s$ and pays a price that depends only on $s$. The Steam snapshot
contains fixed, curated bundles whose composition is chosen by the seller. CMM therefore models a
different selling mechanism. It is retained as a documented research pivot, not used by the live
pipeline.

## Finding 2: recommender outputs are latent preference scores

The Stage 1 models produce latent preference scores $s_{ui}$. Held-out ownership reconstruction
can test their ordinal ranking performance, but the data do not identify willingness to pay,
cardinal utility, purchase probability, or a monetary unit.

Stage 2 therefore uses explicit nonnegative pseudo-utility scenarios

$$
v_{ui}^{m}=T_m(s_{ui})\ge 0.
$$

Each $T_m$ is a frozen modeling choice, not an economic calibration. Prices, costs, and objectives
are reported in the corresponding normalized scenario units. A common positive rescaling has a
limited equivariance property, but a nonlinear or user-specific transformation can change sums,
interpersonal comparisons, demand thresholds, and the selected bundle. Decision stability across
the declared transformations is the result.

## Finding 3: the observed offers are SBA-like, not affirmative evidence for SBR

The mechanism audit classifies all 615 observed bundles by cross-referencing each component with
the 32,132-game catalogue and the per-item fields in `bundle_data.json`.

| Static-snapshot classification | Count | Share | High-confidence |
| --- | ---: | ---: | ---: |
| SBA-like separable components | 568 | 92.4% | 513 |
| Affirmative SBR-style exclusivity | 0 | 0.0% | 0 |
| Unclear or coverage-limited | 47 | 7.6% | -- |

Supporting reconciliation:

- all 615 bundles display a standalone price for every recorded component;
- 475 bundles have every component directly confirmed in the individual catalogue;
- 437 of 615 are single-publisher overall; and
- 562 of 615 contain at most 12 components.

This supports only an **SBA-like component-availability description** for the observed snapshot.
It does not identify CP-optimal component prices, historical availability, ownership-adjusted
pricing, indivisible keys, transaction-time menus, or legal control of the proposed products. The
47 unclear cases are mainly coverage-limited non-game media. Zero affirmative SBR evidence is not
proof that Steam has never offered an SBR-like package.

## Primary Layer 2 mechanism

The primary empirical mechanism is CP-anchored Single Bundle with All,
$\mathrm{SBA}^{CP}$. For each pseudo-utility scenario, component prices $p_i^{CP}$ are first
estimated on design users and then held fixed. The seller chooses a feasible fixed bundle
$B$ and normalized bundle price $b$ while every component remains separately available.

For user $u$, define

$$
w_u(B)=\sum_{i\in B}\min\{v_{ui},p_i^{CP}\}.
$$

Under the primary bundle-preferred weak-tie convention, the user chooses the bundle exactly when

$$
w_u(B)\ge b.
$$

This is the finite-panel specialization of the truncated choice condition in *Partition and
Prosper*, equations (13)--(14), with component prices fixed as in its SBA equation (17). It is not
the SBR rule $\sum_{i\in B}v_{ui}\ge b$.

Let

$$
A_u(B)=\sum_{i\in B}(p_i^{CP}-c_i)
\mathbf 1\{v_{ui}\ge p_i^{CP}\}.
$$

The empirical objective is

$$
\widehat\Pi_{SBA}(B,b)
=
\widehat\Pi_{CP}
+
\frac{1}{U}\sum_u
\mathbf 1\{w_u(B)\ge b\}
\left[b-\sum_{i\in B}c_i-A_u(B)\right].
$$

The displaced component-margin term $A_u(B)$ is essential: a bundle sale can cannibalize a
profitable separate sale. These are conditional normalized objectives, not actual Steam revenue.

## SBR benchmark firewall

Single Bundle with the Rest (SBR) removes the products in $B$ from separate sale. Empty SBR equals
component pricing. Grand-bundle SBR equals pure bundling only when $B=N$ is feasible under every
declared pool constraint; in a capacity-only family this requires $C\ge n$. SBR is retained as a
theoretical and empirical benchmark, while SBA is the project's primary empirical model rather
than an identified claim about Steam's mechanism.

The paper's SBR results remain conditional on their stated assumptions. In particular, the normal
tractability theorem requires a positive-diagonal-minus-fixed-rank-PSD covariance decomposition;
the half-purchase result belongs to the normal CP-anchored SBR reformulation; and the hardness,
comparative-statics, and approximation results do not transfer to SBA. Low-rank recommender factors
do not establish the required covariance structure.

## Frozen Gate 0 decisions

1. **Stage 1 ladder and split.** Compare training-only popularity, implicit ALS, identity-only
   pairwise low-rank MF, and the same model with genre metadata. Tags are optional and begin only
   after all four core rungs pass. Use the frozen outer 80/20 design/assessment user split and the
   nested design-user interaction split specified in `planning.md`, followed by design-only
   production refits and frozen assessment-user fold-in.
2. **Interpretation.** Treat model outputs as latent preference scores. Stage 2 uses declared
   pseudo-utility scenarios and reports normalized within-model objectives only.
3. **Mechanism.** Use $\mathrm{SBA}^{CP}$ as the primary empirical model and SBR as a distinct
   theoretical benchmark. The audit supports SBA-like availability, not the CP anchor itself.
4. **Ordering.** Learn and label the relevant SBR theory, implement shared CP/PB/SBR primitives,
   and implement the direct finite-panel SBA evaluator before headline bundle experiments. The
   empirical narrative and decision target remain SBA-primary.
5. **Component prices.** Estimate empirical CP-optimal pseudo-prices on design users, apply the
   frozen CP tie rule, and hold those prices fixed while selecting the SBA composition and price.
6. **Economic conventions.** Use additive nonnegative pseudo-utilities, primary zero pseudo-costs,
   declared weak and strict tie conventions, observed-capacity sensitivities, and metadata-coherent
   pools as feasibility proxies. The primary population is synthetic pre-acquisition preference
   types; any installed-base convention is a separately named sensitivity.
7. **Required finish line.** Exact fixed-composition pricing, exhaustive certification on measured
   feasible instances, an independently checked scalable heuristic, frozen-policy assessment, and
   robustness are core deliverables.
8. **Stretch rule.** No advanced solver, joint component pricing, robust max-min extension, or new
   recommender begins before the required core passes. At most one stretch is selected afterward.
9. **Calendar rule.** Protect the OR-complete core. If time compresses, apply the cut order in
   `planning.md`: remove optional paper proofs, tags, auxiliary review validation, broad bridge
   variants, large-pool scaling, and advanced extensions before any protected exactness,
   assessment, robustness, or reporting requirement.

## Gate 0 exit

Gate 0 is passed internally when this specification, the live notebook/archive map, the
mechanism-audit generator, and the prerequisite claim corrections are present and verified. Its
role is to freeze conventions before Stage 1 tuning; supervisor feedback is nonblocking.
