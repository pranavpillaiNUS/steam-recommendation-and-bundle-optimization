# Optimization models: live theory reference

Last revised: 2026-07-17

Status: governing mathematical specification for the post-meeting pivot. The empirical results
described here have not yet been run unless another project artifact explicitly says otherwise.
Gate 0 is an internal specification freeze and is complete as of 2026-07-17. Work proceeds under
this specification unless the project owner records a prospective, versioned amendment; supervisor
feedback is nonblocking.

## 1. Role of this note and result provenance

The live project has two optimization layers:

1. Layer 1 estimates latent preference scores from implicit ownership using collaborative filtering
   and low-rank matrix factorization, with a controlled genre-metadata extension.
2. Layer 2 maps those scores through declared pseudo-utility scenarios and solves a fixed,
   seller-curated bundle-design problem.

The main Layer 2 specification is **CP-anchored Single Bundle with All**, abbreviated
$\mathrm{SBA}^{CP}$: retain all component products at component-pricing-optimal pseudo-prices,
then add one fixed bundle and choose its composition and price. This is the exact SBA definition in
Section 5, equation (17), of Sun, Li, and Teo, evaluated here on a finite panel of pseudo-consumer
types rather than through the paper's normal upper-bound approximation.

The static mechanism audit supports only an institutional description of Steam as **SBA-like**:
568 of 615 observed bundles have separable-component evidence, 47 are coverage-limited, and none
contains affirmative evidence of SBR exclusivity. The audit does not establish Steam's full
historical mechanism, transaction-time availability, ownership-adjusted pricing, or contractual
control. It also does not identify CP anchoring: fixing component prices at empirical CP-optimal
pseudo-prices is the project's declared modeling convention, not an observed Steam pricing fact.

Single Bundle with the Rest, SBR, is retained as a theory benchmark. Its bundled components are
not separately available, unlike the SBA-like component availability supported by the static
audit. All hardness,
half-purchase-probability, tractability, comparative-statics, and constant-approximation results
from *Partition and Prosper* are labeled as SBR results and are not transferred to SBA.

The pre-pivot cross-moment bundle-size pricing model, CMM, is preserved in Appendix A. It is an
archived learning record, not a live model.

Throughout this note:

| Label | Meaning |
| --- | --- |
| **Project result** | Derived for this UROP's finite-panel CP-anchored SBA model; a proof is given here. |
| **Paper result** | Stated or reproduced from Sun, Li, and Teo under the paper's assumptions. |
| **Modeling assumption** | An explicit bridge needed because the data do not identify the object. |
| **Empirical claim** | Permitted only after the corresponding saved experiment and test gate exist. |

## 2. Identification boundary and notation

### 2.1 From observed ownership to a modeling input

Let $o_{ui}\in\{0,1\}$ be observed ownership and $t_{ui}\ge0$ observed playtime for user $u$
and game $i$. Ownership is implicit feedback: it is not a rating, exposure record, transaction,
or purchase occasion. Playtime is used mainly to change confidence in an observed interaction; it
is not a monetary outcome.

A fitted recommender produces a latent score $s_{ui}$. For scenario $m$, define

$$
v^{m}_{ui}=T_m(s_{ui})\ge0,
$$

where $T_m$ is fitted or specified without using the candidate bundle or assessment objective.
The number $v^m_{ui}$ is a **pseudo-utility**, not an estimate of willingness to pay. Bundle
prices, component prices, costs, and objective values below are in the same assumed normalized
units. They are within-model counterfactual quantities, not dollars or estimates of Steam revenue.

A common strictly increasing transformation preserves a within-user ranking but generally changes
sums and hence bundle choices. A nondecreasing positive-part map can additionally create ties.
A user-specific transformation also changes interpersonal comparisons against one common price.
Accordingly, the model, seed, transformation, and decision split jointly identify a scenario; no
one transformation is declared true.

Pearson correlation is not invariant to arbitrary monotone transformations. Rank correlations are
invariant only under the relevant monotone transformation applied consistently to each variable;
they are not invariant to arbitrary user-specific transformations that reorder users within an
item. Dependence findings must therefore be reported separately for raw behavior, identity-only
scores, hybrid scores, rank-based measures, and named pseudo-utility scenarios.

### 2.2 Decision and assessment samples

Let $\mathcal U_D$ be the design users and $\mathcal U_A$ the assessment users. Shared
recommender parameters, global transformation parameters, component prices, bundle composition,
bundle price, and heuristic settings are selected using $\mathcal U_D$ only. Assessment-user
representations are folded into frozen shared parameters using only their permitted histories.

A frozen policy is evaluated on $\mathcal U_A$ without reoptimizing any price or composition.
This is conditional out-of-sample evaluation of a pseudo-utility decision, not validation against
observed bundle purchases and not automatically a population-consistency result.

### 2.3 Products, pools, and feasibility

Let $N=\{1,\ldots,n\}$ be one preregistered catalogue-coherence pool. It may be
publisher-coherent, developer-coherent, franchise-coherent, or compatibility-coherent. These are
feasibility proxies, not proof that one legal seller controls every right.

A bundle is $B\subseteq N$. To avoid collision with the score matrix $\mathbf S$, bundle
composition is never denoted $S$ in project-specific formulas. Let $z_i=\mathbf1\{i\in B\}$.
The feasible family is

$$
\mathcal F
=
\{\varnothing\}
\cup
\{B\subseteq N:2\le |B|\le C,\ \mathbf H\mathbf z\le\mathbf h\},
$$

where $\mathbf H\mathbf z\le\mathbf h$ represents the frozen capacity, inclusion, exclusion,
base-game/DLC compatibility, and any frozen genre constraints. Exact and heuristic methods must
call the same feasibility predicate. Products outside the modeled pool add the same constant to
every policy and can be omitted from the optimization comparison.

Let $c_i\ge0$ be a normalized pseudo-cost and $c(B)=\sum_{i\in B}c_i$. The primary scenario
sets $c_i=0$, motivated by low digital marginal cost. A positive-cost sensitivity is still an
assumed cost in pseudo-utility units, not an observed accounting cost.

### 2.4 Choice assumptions and market timing

The core model makes the following **modeling assumptions**:

- pseudo-utilities are additive across games;
- utility is quasi-linear in the normalized price;
- all users see the same posted menu and can acquire at most one unit of each item;
- a user may buy any subset of separately offered games and, under SBA, at most one copy of the
  bundle;
- there is no budget constraint, search cost, exposure effect, or bundle-specific interaction
  utility; and
- the primary tie rules are purchase at $v_{ui}=p_i$ and bundle purchase when the bundle and
  separate-component surpluses are equal.

Genre features can improve score prediction; they do not identify complementarity. Adding pairwise
interaction utility would change the choice reduction and exact price theorem and is outside the
required core.

The primary timing convention treats score vectors as synthetic pre-acquisition preference types.
It does not ask the same observed account literally to repurchase its owned library. An
installed-base sensitivity is outside the frozen core; if added later through a logged amendment,
it first sets

$$
v^{m,\mathrm{new}}_{ui}
=
v^m_{ui}\mathbf1\{o_{ui}=0\}
$$

and then recomputes every benchmark and price. That remains a one-price forward-offer scenario,
not identified complete-the-set demand.

## 3. Layer 1 as optimization

Layer 1 is an estimation layer, but its main models are themselves optimization problems.

### 3.1 Implicit-feedback ALS

Let $y_{ui}=\mathbf1\{o_{ui}=1\}$. A confidence specification that distinguishes an owned but
unplayed game from an unobserved pair is

$$
\gamma_{ui}
=
1+\alpha_o o_{ui}
+\alpha_p\min\{\log(1+t_{ui}),\tau\},
\qquad
\alpha_o>0,\ \alpha_p\ge0.
$$

Implicit ALS estimates user factors $\mathbf x_u\in\mathbb R^k$ and item factors
$\mathbf q_i\in\mathbb R^k$ by minimizing

$$
\sum_{u,i}
\gamma_{ui}\bigl(y_{ui}-\mathbf x_u^\top\mathbf q_i\bigr)^2
+
\lambda\left(
\sum_u\|\mathbf x_u\|_2^2+\sum_i\|\mathbf q_i\|_2^2
\right).
$$

The joint problem is nonconvex. With either factor block fixed, each other block solves a convex
weighted ridge-regression problem, so exact alternating updates do not increase the objective.
This establishes an optimization algorithm; it does not give the scores a cardinal economic
interpretation.

### 3.2 Pairwise identity and metadata factorization

For an owned item $i$ and sampled unobserved item $j$, a BPR-style model minimizes

$$
\sum_{(u,i,j)}
-\log \sigma(s_{ui}-s_{uj})
+
\lambda\|\theta\|_2^2.
$$

An identity-only feature-sum model can use an item embedding
$\boldsymbol\eta_i$. Its controlled genre extension uses

$$
\mathbf q_i
=
\boldsymbol\eta_i
+
\sum_a x_{ia}\mathbf f_a,
$$

where $x_{ia}$ is a frozen item-feature matrix and $\mathbf f_a$ is a learned feature
embedding. Identity-only and identity-plus-genre comparisons must hold the loss, split,
regularization protocol, dimension, epochs, sample weighting, and seeds fixed. The metadata lift is
a hypothesis, not a required outcome.

The numerical value of $s_{ui}$ depends on the loss, sampling, regularization, and
parameterization. Held-out Recall and NDCG validate ranking performance only. Layer 2 therefore
starts only after applying the declared $T_m$.

## 4. Component pricing and pure bundling on a finite panel

For this and subsequent sections, fix one scenario $m$ and suppress its superscript. Let
$U=|\mathcal U_D|$ unless another sample is explicitly named.

For item $i$, the finite-panel component-pricing objective is

$$
\widehat r_i(p)
=
(p-c_i)\frac1U\sum_{u=1}^U
\mathbf1\{v_{ui}\ge p\}.
$$

Because a no-sale policy yields zero, restricting to $p\ge c_i$ is without loss. Define

$$
p_i^{CP}
\in
\arg\max_{p\ge c_i}\widehat r_i(p),
\qquad
\widehat\Pi_{CP}
=
\sum_{i\in N}\widehat r_i(p_i^{CP}).
$$

**Proposition P1 (project result: empirical CP price candidates).** An empirical CP optimum has
either no sales or a price equal to a distinct observed pseudo-utility $v_{ui}\ge c_i$.

**Proof.** Take a price with positive demand that is not an observed pseudo-utility. Increasing it
to the smallest observed pseudo-utility among its current buyers leaves the buyer set unchanged
under the weak purchase rule and weakly increases the margin, strictly if the increase is positive.
A zero-demand policy is represented explicitly by no sale. $\square$

Thus CP is computed by sorting distinct values, grouping ties, and scanning complete tied blocks.
A deterministic design-sample tie rule is fixed before assessment evaluation. The primary rule is
the smallest finite CP-optimal candidate; key SBA results are repeated at the largest finite
CP-optimal candidate and, when relevant, the no-sale representative. This sensitivity matters:
different CP-optimal prices can change the truncated bundle values even when CP profit is tied.

No sale is a distinct policy action, not an arbitrarily large finite number. When the maximum CP
objective is zero, the numerical argmax can contain an unbounded interval, so a largest
CP-optimal price need not exist. For the no-sale action define the standalone purchase indicator
and displaced margin as zero, while its contribution to the bundle threshold is $v_{ui}$, the
limit of $\min\{v_{ui},p\}$ as $p\to\infty$. Code must use an explicit sentinel rather than
forming an undefined infinity-times-zero margin.

The pure-bundling benchmark on $N$ is

$$
\widehat\Pi_{PB}
=
\max_{b\ge c(N)}
(b-c(N))
\frac1U\sum_u
\mathbf1\left\{\sum_{i\in N}v_{ui}\ge b\right\}.
$$

Its exact finite-panel price likewise lies at an observed aggregate pseudo-utility threshold or the
no-sale option. PB is a separate benchmark; its components are not offered individually.

## 5. Mechanisms that contain one fixed bundle

| Mechanism | Items in $B$ sold separately? | Component prices | Valid relationship |
| --- | --- | --- | --- |
| CP | Yes; there is no bundle | Optimized item by item | Baseline |
| PB | No | None | All of $N$ or no purchase |
| SBR | No | Outside-$B$ prices optimize separately to CP | Contains CP; contains PB if $N\in\mathcal F$ |
| $\mathrm{SBA}^{CP}$ | Yes | Fixed at $p_i^{CP}$ | Contains CP; does **not** contain PB |
| JSBC | Yes | Jointly optimized with bundle and composition | General one-bundle joint mechanism |
| CMM | Customer constructs any set of a priced size | Size-menu prices | Different, archived mechanism |

The paper's SBA equation (17) is $\mathrm{SBA}^{CP}$: first fix every item at its CP price, then
optimize one bundle and its price. The paper calls SBA a heuristic restriction of JSBC because JSBC
also jointly optimizes component prices. The UROP's main model solves the finite-panel
$\mathrm{SBA}^{CP}$ objective directly. It is not a claim to solve JSBC.

## 6. CP-anchored SBA: choice and objective

### 6.1 Exact customer-choice reduction

For a proposed nonempty $B$ at bundle price $b$, a user can obtain the best separate-component
surplus

$$
U_u^{sep}(B)
=
\sum_{i\in B}(v_{ui}-p_i^{CP})_+.
$$

The bundle surplus is

$$
U_u^{bun}(B,b)
=
\sum_{i\in B}v_{ui}-b.
$$

Outside-$B$ purchases are available under either choice and therefore cancel in this comparison.
Because $U_u^{sep}(B)\ge0$, a bundle that weakly beats the separate option also satisfies
individual participation; no additional bundle-versus-no-purchase constraint is missing.

**Provenance.** The choice comparison is the finite-panel specialization of the SBA customer-choice
relations in Sun, Li, and Teo, equations (13)--(14), after fixing component prices at
$p_i^{CP}$. The pointwise truncation derivation below, the explicit weak-tie convention, and its
finite-panel implementation tests are project-specific; this attribution does not transfer any SBR
theorem or guarantee.

**Proposition P2 (adapted project result: SBA choice equivalence).** Under the primary
bundle-preferred tie rule, user $u$ chooses the bundle if and only if

$$
w_u(B)
:=
\sum_{i\in B}\min\{v_{ui},p_i^{CP}\}
\ge b.
$$

**Proof.** For every item,
$v_{ui}-(v_{ui}-p_i^{CP})_+=\min\{v_{ui},p_i^{CP}\}$. Therefore

$$
U_u^{bun}\ge U_u^{sep}
\iff
\sum_{i\in B}
\left[v_{ui}-(v_{ui}-p_i^{CP})_+\right]
\ge b,
$$

which is the displayed condition. $\square$

The truncation is essential. The SBR rule
$\sum_{i\in B}v_{ui}\ge b$ is wrong for SBA because it ignores the user's separate-component
alternative.

### 6.2 Direct and incremental objectives

Write

$$
d_{ui}:=\mathbf1\{v_{ui}\ge p_i^{CP}\},
\qquad
m_i:=p_i^{CP}-c_i\ge0,
$$

and define the CP profit displaced when user $u$ takes bundle $B$:

$$
A_u(B)
:=
\sum_{i\in B}m_i d_{ui}.
$$

Let $y_u(B,b)=\mathbf1\{w_u(B)\ge b\}$. The direct per-user-average SBA objective is

$$
\begin{aligned}
\widehat\Pi_{SBA}(B,b)
=
\frac1U\sum_u
\Bigg[
&y_u(B,b)\bigl(b-c(B)\bigr)\\
&+(1-y_u(B,b))\sum_{i\in B}m_i d_{ui}
+\sum_{i\notin B}m_i d_{ui}
\Bigg].
\end{aligned}
$$

The first line is the bundle margin for a bundle buyer. The second line preserves component sales
inside $B$ for nonbuyers and component sales outside $B$ for everyone.

**Proposition P3 (project result: incremental SBA decomposition).**

$$
\boxed{
\widehat\Pi_{SBA}(B,b)
=
\widehat\Pi_{CP}
+
\frac1U\sum_u
y_u(B,b)
\left[b-c(B)-A_u(B)\right].
}
$$

**Proof.** Start from CP. For a nonbuyer, nothing changes. For a bundle buyer, replace the CP
margins $A_u(B)$ from bundled items by the bundle margin $b-c(B)$; all outside-$B$ terms
remain unchanged. Averaging gives the formula. $\square$

The term $A_u(B)$ prevents a false conclusion that every bundle sale is wholly incremental. It
records cannibalized component profit user by user.

The main design problem is

$$
\boxed{
\max_{B\in\mathcal F,\ b\ge0}
\widehat\Pi_{SBA}(B,b),
}
$$

with $B=\varnothing$ defined as the no-bundle CP policy. Since $m_i\ge0$, any price below
$c(B)$ gives a nonpositive incremental contribution from every buyer, so $b\ge c(B)$ is without
loss when the no-bundle option is available.

### 6.3 Hand-computed mechanism check

Consider two pseudo-users, two zero-cost items, and

$$
(v_{ui})=
\begin{pmatrix}
10&2\\
2&10
\end{pmatrix}.
$$

Each item has the unique CP price $10$, earning average objective $5$; hence
$\widehat\Pi_{CP}=10$. For $B=\{1,2\}$, both users have $w_u(B)=12$ and
$A_u(B)=10$. At $b=12$, both choose the bundle under the weak tie rule and

$$
\widehat\Pi_{SBA}
=
10+\tfrac12[(12-10)+(12-10)]
=
12.
$$

The gain of $2$ comes from the low pseudo-utility for the item each user would not buy
separately, net of the displaced high-price component sale. These are normalized scenario units,
not currency.

## 7. Exact pricing for a fixed SBA composition

Fix nonempty $B$, fixed CP prices, and the weak bundle-preferred rule. Define the incremental
objective

$$
\Delta_B(b)
=
\frac1U\sum_{u:w_u(B)\ge b}
\left[b-c(B)-A_u(B)\right].
$$

**Proposition P4 (project result: finite exact price set).** A fixed-composition optimum is either
the no-bundle policy or has

$$
b\in\{w_u(B):u=1,\ldots,U\}.
$$

**Proof.** Suppose $b$ has a nonempty buyer set and is not a threshold. Raise $b$ to the
smallest threshold among its current buyers. No buyer leaves under the weak rule, $c(B)$ and
$A_u(B)$ remain fixed, and the objective increases linearly with slope equal to the positive
buyer share. A price above every threshold has no buyers and equals the no-bundle CP baseline.
Thus only observed thresholds and no bundle need be checked. $\square$

Let $t_1>\cdots>t_L$ be the distinct values of $w_u(B)$, and
$G_\ell=\{u:w_u(B)=t_\ell\}$. After adding the entire tied block $G_\ell$, define

$$
M_\ell=\sum_{h\le\ell}|G_h|,
\qquad
H_\ell=\sum_{h\le\ell}\sum_{u\in G_h}A_u(B).
$$

The exact candidate gain is

$$
\Delta_\ell
=
\frac{M_\ell(t_\ell-c(B))-H_\ell}{U}.
$$

Compare every $\Delta_\ell$ with zero and apply a deterministic price tie rule. The primary
mathematical rule returns no bundle when the best gain is exactly zero; among positive equal-gain
thresholds it chooses the smallest price. Outer exact-search ties are then broken by smaller
cardinality and lexicographic item ID. These choices are frozen before assessment because equal
design objectives can have different assessment buyer sets. Numerical zero comparisons use a
homogeneous, scale-aware tolerance recorded in configuration. Never evaluate a partial tied block:
it is not a feasible buyer set under the declared weak rule.

Constructing $w_u(B)$ and $A_u(B)$ costs $O(U|B|)$; sorting costs $O(U\log U)$; the grouped
scan costs $O(U)$. Therefore exact fixed-composition pricing costs

$$
O(U|B|+U\log U)
$$

time and $O(U)$ working memory.

Under a components-preferred bundle tie, purchase requires $w_u(B)>b$. In a continuous price
domain, the best value may be a left-hand limit $t_\ell-\varepsilon$ and need not be attained.
That sensitivity must use a declared price tick, explicitly evaluate left limits, or report a
supremum. It must not reuse Proposition P4's weak-tie certificate unchanged.

## 8. Composition search, certificates, and heuristic status

Substitute the exact fixed-$B$ price into the outer problem:

$$
\widehat\Pi_{SBA}^*
=
\max\left\{
\widehat\Pi_{CP},
\max_{B\in\mathcal F\setminus\{\varnothing\}}
\max_{\ell=1,\ldots,L(B)}
\bigl(\widehat\Pi_{CP}+\Delta_\ell(B)\bigr)
\right\}.
$$

Exhaustive enumeration is an exact certificate when it completes over the declared feasible
family. Its transparent worst-case work is

$$
O\left(
\sum_{k=2}^{C}
\binom nk
[Uk+U\log U]
\right)
$$

before memoization or incremental updates. This exponential enumeration bound is **not** itself a
proof that the SBA composition problem is NP-hard, and the paper's SBR hardness theorem cannot be
used as such a proof.

**Proposition P5 (project result: singleton redundancy on the design sample).** If
$p_i^{CP}$ is CP-optimal, adding a singleton bundle $\{i\}$ cannot improve the design-sample
objective above CP.

**Proof.** If CP selects the no-sale action, any finite singleton price produces
$\widehat r_i(b)\le0$, so it cannot beat the zero no-sale objective. Otherwise
$p_i^{CP}$ is finite. If $b\ge p_i^{CP}$, the component option weakly dominates the duplicate singleton
bundle, aside from a revenue-neutral tie at equality. If $b<p_i^{CP}$, every buyer of the item
uses the lower bundle price, so the seller obtains exactly the component-pricing objective
$\widehat r_i(b)\le\widehat r_i(p_i^{CP})$. Other products are unchanged. $\square$

Thus the live feasible family may exclude singletons without losing a strict design-sample
improvement. It retains $\varnothing$, so an optimizer can always return CP.

For larger pools, the required scalable method is multistart add/drop/swap local search. It must:

- call the same feasibility predicate as exact search;
- reoptimize $b$ exactly after every proposed composition;
- include the no-bundle policy explicitly;
- use starts and stopping rules frozen on a development suite;
- store objective-evaluation counts and complete traces; and
- report best solution found unless an exact certificate or valid bound exists.

The absence of a generic greedy guarantee is not just a missing proof. Define the optimized
incremental gain

$$
G(B)=\max\{0,\max_b\Delta_B(b)\}.
$$

**Proposition P6 (project result: the SBA gain is neither submodular nor monotone).** These
failures occur with two pseudo-users, zero costs, unique CP prices, and the weak bundle tie rule.

**Proof.** For two items, take

$$
(v_{ui})=
\begin{pmatrix}
1&1/4\\
1/4&1
\end{pmatrix}.
$$

Each unique CP price is $1$. Singleton gains are zero by Proposition P5. For
$B=\{1,2\}$, both users have $w_u=5/4$ and $A_u=1$, so price $5/4$ gives
$G(\{1,2\})=1/4$. Hence

$$
G(\{1\})+G(\{2\})=0
<
G(\{1,2\})+G(\varnothing)=1/4,
$$

which violates submodularity.

For nonmonotonicity, add item $3$ with user values $(1,0)$ and unique CP price $1$.
For $B=\{1,2,3\}$, the pairs $(w_u,A_u)$ are $(9/4,2)$ and $(5/4,1)$.
The two threshold gains are $1/8$ at $b=9/4$ and $-1/4$ at $b=5/4$, so

$$
G(\{1,2,3\})=1/8<1/4=G(\{1,2\}).
$$

Thus adding an item can reduce optimized gain. $\square$

Consequently, the usual monotone-submodular cardinality-greedy guarantee does not apply; greedy
search from the empty set can see zero singleton gains even when a profitable pair exists; and an
at-most-$C$ capacity must not be replaced by an exactly-$C$ requirement. Local search remains
heuristic, and empirical gaps on a locked exact suite validate performance only on that suite.

## 9. SBR and other benchmarks

### 9.1 Empirical SBR benchmark

In SBR, items in $B$ are unavailable separately. For fixed CP-optimal prices on items outside
$B$,

$$
\widehat\Pi_{SBR}(B,b)
=
(b-c(B))
\frac1U\sum_u
\mathbf1\left\{\sum_{i\in B}v_{ui}\ge b\right\}
+
\sum_{i\notin B}\widehat r_i(p_i^{CP}).
$$

For fixed $B$, its exact empirical price is an observed threshold
$\sum_{i\in B}v_{ui}$ or no sale, by the same piecewise-linear argument as Proposition P4.
This does not change the fact that SBR models a different menu from SBA.

**Proposition P7 (project result: benchmark relationships).**

1. Empty SBA equals CP, so optimized $\mathrm{SBA}^{CP}$ weakly dominates CP in its own design
   objective.
2. Empty SBR equals CP.
3. If $N\in\mathcal F$ (and therefore $C\ge n$ under the capacity constraint), grand-bundle SBR
   with $B=N$ equals PB; hence optimized SBR nests CP and PB.
4. SBA does not nest PB because its component options remain available. There is no general
   SBA-versus-PB ordering supplied by nesting.
5. Optimized SBA and optimized SBR are not ordered in general.

**Proof.** Substitute $B=\varnothing$ in the SBA and SBR definitions. When $N\in\mathcal F$,
substitute $B=N$ in SBR, which removes every outside component and leaves the PB objective. In
SBA, even $B=N$ retains all component alternatives, so its menu and choice rule differ from PB.

For non-dominance, take zero costs and capacity two. With three equally weighted users
$(2,0),(0,2),(1.1,1.1)$, the unique CP prices are $(1.1,1.1)$; SBA cannot improve its CP
objective $22/15$: its only nontrivial thresholds are $1.1$, which gives negative incremental
gain, and $2.2$, which gives zero. SBR can instead choose the grand bundle at price $2$, sell
to all three types, and earn $2$.
Conversely, with nine equally weighted users (four of type $(4,0)$, four of type $(0,4)$, and
one of type $(3,3)$), the unique CP prices are $(4,4)$. SBA adds the grand bundle at price $6$
and earns $38/9$: CP contributes $32/9$, and the central type contributes the incremental
bundle payment $6/9$. SBR's empty and singleton policies earn at most $32/9$, while its grand
bundle earns $4$ at price $4$; hence its optimum is $4$. Thus each mechanism can outperform
the other. $\square$

### 9.2 Results from *Partition and Prosper*: SBR only

Under the paper's continuous additive valuation model, and under multivariate normal valuations
where specified, the paper obtains the following results. Continuity makes ties probability-zero
in that theorem world. The finite empirical panel is discrete and often tied, so the project's weak
tie rule, complete tied-block scan, and strict-tie sensitivity are new requirements.

For reference, let the paper's random valuation vector satisfy
$\widetilde{\mathbf u}\sim N(\boldsymbol\mu,\boldsymbol\Sigma)$, let
$\mathbf z\in\{0,1\}^n$ select the SBR bundle, and let
$r_i^*=\max_p(p-c_i)\Pr(\widetilde u_i\ge p)$ be item $i$'s CP profit. After subtracting
the constant $\sum_i r_i^*$, normal SBR is

$$
\max_{\substack{\mathbf z\in\{0,1\}^n,\ \mathbf e^\top\mathbf z\le C\\b\ge0}}
\left(b-\mathbf z^\top\mathbf c\right)
\left[
1-\Phi\left(
\frac{b-\mathbf z^\top\boldsymbol\mu}
{\sqrt{\mathbf z^\top\boldsymbol\Sigma\mathbf z}}
\right)
\right]
-\mathbf z^\top\mathbf r^*.
$$

The empty vector is defined separately as the zero-increment CP action, avoiding the $0/0$
normal-demand expression.

For nonempty $\mathbf z$, purchase probability $x$ and price are related by

$$
b
=
\mathbf z^\top\boldsymbol\mu
+\Phi^{-1}(1-x)
\sqrt{\mathbf z^\top\boldsymbol\Sigma\mathbf z}.
$$

Substitution gives the paper's baseline-adjusted reformulation

$$
\max_{\substack{x\in[0,1],\ \mathbf z\in\{0,1\}^n\\
\mathbf e^\top\mathbf z\le C}}
x\,\mathbf z^\top(\boldsymbol\mu-\mathbf c)
+x\Phi^{-1}(1-x)
\sqrt{\mathbf z^\top\boldsymbol\Sigma\mathbf z}
-\mathbf z^\top\mathbf r^*
=
\max_{x\in[0,1]}f(x).
$$

These equations belong to the SBR normal model. In particular, their raw-sum purchase condition,
normal quantile change of variables, and $f(x)$ structure are not SBA formulas.

| Paper result | Exact scope | Permitted use here |
| --- | --- | --- |
| Theorem 1 | The SBR problem that takes a predetermined component-price vector as input is NP-hard even for independent normal valuations; this does not say every fixed vector, especially the CP vector, is hard. | Theory benchmark; not an SBA hardness proof. |
| Proposition 1 and Corollary 1 | In the baseline-adjusted normal SBR reformulation with CP-optimal outside prices, incremental value $f(x)=0$ for $x<1/2$; a nondegenerate optimal SBR bundle has purchase probability at least $1/2$ and mean at least cost. | SBR only; not an SBA demand restriction. |
| Theorem 2 | For some $t\ge0$, normal SBR is solvable using at most $O(n^{\lfloor(K+3)/2\rfloor}C^{\lceil(K+3)/2\rceil})$ univariate concave problems when $\Sigma=(1+t)\operatorname{Diag}(\sigma_i^2)-M(t)$, with $M(t)\succeq0$ of rank $K$ fixed independently of $n$. | Apply only after verifying the decomposition. |
| Corollary 2 / Appendix C | Independence gives at most $O(C^2n)$ univariate concave maximizations by Corollary 2; Appendix C's separate line-sorting implementation runs in $O(Cn^2)$. | SBR only. |
| Proposition 2 | With finite values, equal positive utility margins, equal positive standard deviations, and positive incremental gain, a lower-variance addition is more profitable; a positive-gain optimum minimizes variance among bundles of its size. | Conditional SBR comparative statics, not a general genre theorem. |
| Proposition 3 | If every marginal distribution is MHR, SBR is at least CP and CP is a $1/e$ approximation to the general optimal mechanism, with arbitrary dependence and costs as stated by the paper. | Literature statement only unless assumptions are defended. |
| Theorem 3 | Under the base additive model and arbitrary valuation distributions and costs, $\mathrm{SBR}\le\mathrm{JSBC}\le2\,\mathrm{SBR}$. | SBR-versus-JSBC statement; not an SBA guarantee. |
| Section 5 / equation (17) | SBA fixes component prices at CP and adds one bundle. | This is the definition used by the main empirical model. |
| Appendix E | Develops an SBA upper-bound model and a normal approximation solved by BO/SDP. | Optional stretch comparison, not the exact empirical evaluator. |

These paper results are currently cited with proof sketches, not claimed as project derivations.
One proof will be selected internally and reproduced in full after the required empirical and
project-specific theory core; supervisor feedback may inform that nonblocking selection.

- **Theorem 1.** The appendix reduces PARTITION to the predetermined-price SBR instance. It sets
  capacity $C=n$ and means equal to costs, then chooses the input component prices and independent
  variances so the optimized binary objective reaches its target exactly when a selected subset has
  the required partition sum. This proves hardness of the problem class, not of every fixed price
  vector.
- **Proposition 1 and Corollary 1.** For $x<1/2$,
  $x\Phi^{-1}(1-x)>0$. A covariance bound makes the nonseparable standard-deviation term no
  larger than a sum of item terms, each bounded by its CP monopoly profit; the empty bundle attains
  zero, so baseline-adjusted $f(x)=0$. A nondegenerate optimum must therefore have
  $x\ge1/2$, and a mean below cost would force every non-loss price above the normal mean and
  hence demand below one half.
- **Theorem 2 and Corollary 2.** Proposition 1 restricts the search to $x\ge1/2$. Under the
  diagonal-minus-fixed-rank decomposition, the fixed-$x$ objective is convex in a fixed number of
  additive bundle statistics; an optimum occurs at an extreme point, and a geometric
  $\le C$-set bound gives polynomially many candidates. Each candidate leaves a univariate
  concave price problem. Independence is the $K=0$ case; Appendix C obtains a separate exact
  line-sorting implementation.
- **Proposition 2.** Under its symmetry and positive-gain conditions, same-cardinality alternatives
  have the same mean and displaced CP profit. A normal-hazard/envelope argument shows optimized
  profit decreases with bundle standard deviation once optimal demand exceeds one half, yielding
  the conditional lower-variance comparison.
- **Proposition 3.** Empty-bundle feasibility gives SBR at least CP. For each MHR marginal, the
  cumulative-hazard argument bounds expected item surplus by $e$ times monopoly profit; total
  optimal profit is at most total surplus, giving
  $\mathrm{SBR}\ge\mathrm{CP}\ge\mathrm{OPT}/e$.
- **Theorem 3.** Split JSBC profit into its bundle-plus-outside contribution and separate sales of
  items inside the bundle. The former is bounded by an SBR policy and the latter by CP; optimized
  SBR dominates both, so $\mathrm{JSBC}\le2\,\mathrm{SBR}$, while mechanism inclusion gives the
  reverse lower bound.

For normal SBR, the paper changes variables from price to purchase probability $x$ and solves
the general correlated case by Bayesian optimization over $x\in[1/2,1]$; each evaluation is a
mixed 0-1 second-order-cone problem. That interval restriction follows from the SBR Proposition 1.
It must not be imposed on empirical SBA. A finite BO budget returns the best evaluated solution,
not a global certificate for the overall SBR problem.

### 9.3 Why Layer 1 low rank does not activate the SBR theorem

A raw $k$-factor score matrix has low-rank cross-user item covariance after appropriate
centering, up to explicitly modeled bias directions. A positive diagonal residual appears only if
the project specifies and estimates an idiosyncratic-error or shrinkage model. It is not generated
automatically by matrix factorization. Moreover, a nonlinear pseudo-utility transformation can
destroy the raw low-rank covariance structure.

Theorem 2 requires a **positive diagonal minus** a fixed-rank PSD matrix. A low-rank PSD score
covariance, or an assumed **low-rank plus diagonal** covariance, is not automatically of that form.
If the normal SBR stretch is attempted, the required decomposition, rank, positive-semidefiniteness,
tail behavior, and approximation error must be checked numerically. The paper's normal theorem
also permits negative support, whereas the project's core pseudo-utilities are nonnegative; a
normal fit is therefore an approximation, not a consequence of nonnegativity or low rank.
Otherwise the empirical-panel optimizer remains primary.

## 10. Scaling, cardinalization, and policy evaluation

**Proposition P8 (project result: common positive scaling equivariance).** Fix a transformation
scenario and let $\lambda>0$. If every pseudo-utility and pseudo-cost is scaled by $\lambda$,
then scaling every component and bundle price by $\lambda$:

- leaves every CP, PB, SBR, and SBA choice indicator unchanged;
- scales $w_u(B)$, $A_u(B)$, and every objective by $\lambda$;
- maps each optimal price to a scaled optimal price; and
- preserves the set of optimal compositions and dimensionless within-scenario objective ratios.

**Proof.** All choice inequalities are homogeneous of degree one in utilities and prices. Every
margin and the incremental decomposition in Proposition P3 are also homogeneous of degree one.
The feasible composition family is unchanged. $\square$

This result handles one common multiplicative unit only. It does not justify nonlinear
cardinalization, user-specific normalization, or unscaled costs. If costs are held fixed while only
utilities change, the proposition does not apply. A strict-tie price lattice must also be scaled by
$\lambda$; a fixed absolute tick would break the equivariance.

For scenario $m$, the design policy is

$$
\pi_m^D
=
\left(
\mathbf p_m^{CP,D},B_m^D,b_m^D
\right).
$$

Assessment users evaluate exactly this frozen triple using the direct or incremental SBA formula.
Every assessment CP baseline also uses $\mathbf p_m^{CP,D}$. Reoptimizing component prices,
bundle price, or composition on $\mathcal U_A$ creates an oracle benchmark and is prohibited.

When transferring a composition across transformations that do not share cardinal units, transfer
the composition only. Reprice it on the **target scenario's design users**, freeze that target
price, and then evaluate it on target assessment users. Cross-scenario comparisons use
dimensionless, within-scenario normalization and must identify all-zero cases. They never compare
raw pseudo-price levels across incompatible transformations.

Required robustness axes are model, training seed, transformation, decision split, CP price tie,
bundle tie rule, capacity, pool definition, and cost scenario. The installed-base endowment
convention is outside the frozen core and appears only if added later as a separately logged
sensitivity. If composition is unstable but within-scenario regret is small, report multiple
near-equivalent designs. If both are unstable, narrow the design conclusion.

## 11. Algorithms, tests, and claim discipline

The reference implementation belongs in src/bundle_design.py; notebooks orchestrate it. The
minimum exact API should expose CP, PB, fixed-composition SBR, fixed-composition SBA, exact
composition enumeration, a shared feasibility predicate, and multistart local search.

The required mathematical invariants are:

1. direct SBA profit equals the incremental-over-CP expression;
2. the choice comparison equals the truncated-value rule;
3. CP and bundle threshold scans add complete tied blocks;
4. fixed-$B$ scans match an independent brute-force or dense-grid oracle on hand cases;
5. exact composition search matches an independently written enumerator on small pools;
6. empty SBA and empty SBR equal CP;
7. grand SBR equals PB when $N\in\mathcal F$;
8. no test asserts that SBA nests PB;
9. scaling all utilities, costs, and prices gives Proposition P8;
10. exact and heuristic methods return only bundles accepted by the same constraints; and
11. assessment IDs cannot enter any fitting, price, composition, or heuristic-tuning step.

A completed enumeration may be called an exact optimum over the declared finite feasible family.
A heuristic output is a best solution found. The empirical finite-panel objective is a
sample-average objective conditional on the recommender and pseudo-utility scenario; it is not free
of upstream modeling assumptions and is not real revenue.

## 12. Live proof and implementation checklist

- [x] Specify the CP-anchored SBA menu and distinguish it from SBR, PB, JSBC, and CMM.
- [x] Prove empirical CP threshold candidates (Proposition P1).
- [x] Prove SBA choice equivalence (Proposition P2).
- [x] Prove the direct/incremental objective identity (Proposition P3).
- [x] Prove the exact weak-tie fixed-composition price set and complexity (Proposition P4).
- [x] Prove singleton redundancy and valid benchmark nesting (Propositions P5 and P7).
- [x] Give explicit nonmonotonicity and nonsubmodularity counterexamples (Proposition P6).
- [x] Prove common positive scaling equivariance at its correct strength (Proposition P8).
- [ ] Implement the transparent reference evaluators and hand-computed tests.
- [ ] Benchmark the exact certified region as a function of $n,C,U$.
- [ ] Freeze the heuristic on a development suite and evaluate it once on a locked exact suite.
- [ ] Evaluate every frozen policy on assessment users across the preregistered scenario grid.
- [ ] Reproduce one selected SBR proof from the paper, clearly labeled as reproduced.
- [ ] Attempt the normal SBR BO/conic comparison only if the required structure and solver path are
      verified and it is selected through a logged post-core scope decision as the single stretch
      goal.

## 13. References for the live model

- Sun H, Li X, Teo C-P. *Partition and Prosper: Design and Pricing of Single Bundle*. SBR in
  Sections 3--4; CP-anchored SBA in Section 5 and equation (17); SBA approximation in Appendix E.
- Hu Y, Koren Y, Volinsky C (2008). Collaborative filtering for implicit feedback datasets.
- Koren Y, Bell R, Volinsky C (2009). Matrix factorization techniques for recommender systems.
- Rendle S, Freudenthaler C, Gantner Z, Schmidt-Thieme L (2009). BPR: Bayesian personalized
  ranking from implicit feedback.
- Kula M (2015). Metadata embeddings for user and item cold-start recommendations.
- Krichene W, Rendle S (2020). On sampled metrics for item recommendation.
- Adams WJ, Yellen JL (1976); Schmalensee R (1984); McAfee RP, McMillan J, Whinston MD (1989);
  Bakos Y, Brynjolfsson E (1999). Classic bundling economics, used as context rather than as
  identification of Steam demand.

## Appendix A: archived cross-moment bundle-size pricing model

This appendix preserves the CMM mathematics completed before the 2026-07-11 mechanism pivot.
CMM prices a size menu from which a customer constructs any bundle of the chosen size. It is not
the live Steam model, supplies no input to the SBA optimizer, and supports no claim about actual
Steam prices or revenue. Its proofs remain a learning record and regression reference for the
archived notebooks 06 and 10. CMM's valuation notation below belongs to the paper being
reproduced; it must not be read as identifying the recommender scores as valuations.

### Notation (CMM)

Scalars $x$, vectors $\mathbf{x}$, matrices $\mathbf{X}$. $\mathbf{e}$ is the all-ones
vector, $\mathbf{e}_i$ the $i$th unit vector. $\operatorname{Diag}(\mathbf{x})$ is the
diagonal matrix with $\mathbf{x}$ on the diagonal; $\operatorname{diag}(\mathbf{X})$ is the
vector of diagonal entries. $\operatorname{tr}(\mathbf{X})$ is the trace. For
$\mathbf{A}\succeq 0$, $\mathbf{A}^{1/2}$ is the unique PSD square root. Random quantities
carry a tilde, e.g. $\tilde{\mathbf{u}}$. $\mathbf{X}\succ 0$ means positive definite.

### The bundle size pricing problem

A monopolist sells $n$ heterogeneous products to a population of customers. A customer has a
random valuation vector $\tilde{\mathbf{u}} = (\tilde u_1,\dots,\tilde u_n)^\top \sim F$, with
finite first and second moments. The outside option has deterministic valuation
$\tilde u_0 = 0$.

Valuations are additive across items, and a bundle contains at most one of each item. The
firm offers a set of bundle sizes $S \subseteq [n]$ with $|S| = m$, and a price $p_s$ for each
size $s \in S$. Crucially, the price depends only on the size, not on which items are in the
bundle.

A customer who buys a size-$s$ bundle picks the $s$ items they value most (free disposal of
any negatively valued item). Writing the order statistics
$\tilde u_{(1)} \le \dots \le \tilde u_{(n)}$, the customer's best size-$s$ valuation is the
sum of the top $s$ order statistics:

$$\tilde w_s(\tilde{\mathbf{u}}) = \sum_{k=0}^{s-1} \tilde u_{(n-k)}, \qquad \tilde w_0 = 0.$$

Let $c_s$ be the (deterministic, homogeneous) cost of a size-$s$ bundle. For information goods
such as digital games the marginal cost is near zero, so $c_s \approx 0$. With the population
normalized to mass one, the expected demand for size $s$ is the probability that size $s$ is
the surplus-maximizing choice, and the firm solves

$$
\textbf{(BSP)} \quad
\max_{\mathbf{p}\ge 0} \ \sum_{s\in S} (p_s - c_s)\, q_s^*(\mathbf{p}),
\qquad
q_s^*(\mathbf{p}) = \Pr\!\left( s = \arg\max_{i\in\{0\}\cup S}\{\tilde w_i(\tilde{\mathbf{u}}) - p_i\} \right).
$$

#### Why this is hard

The demand $q_s^*(\mathbf{p})$ requires the joint distribution $G$ of the vector
$\tilde{\mathbf{w}} = (\tilde w_s)_{s\in S}$. Each $\tilde w_s$ is a maximum partial sum of
order statistics of $\tilde{\mathbf{u}}$, and $G$ is a function of $F$ with no usable closed
form in general. Even given $G$, the optimization is a bilevel problem (the customer's argmax
inside the firm's profit). This is the obstacle CMM is designed to bypass.

### The cross-moment model

The idea: do not characterize $G$. Keep only its first two moments and replace $G$ by the
distribution in that moment class that is most favorable to the customer.

Let

$$\boldsymbol{\omega} = \mathbb{E}_G[\tilde{\mathbf{w}}], \qquad \boldsymbol{\Sigma} = \operatorname{Cov}_G(\tilde{\mathbf{w}}),$$

and let
$\Theta = \{\theta : \mathbb{E}_\theta[\tilde{\mathbf{w}}] = \boldsymbol{\omega},\ \operatorname{Cov}_\theta(\tilde{\mathbf{w}}) = \boldsymbol{\Sigma}\}$
be all distributions
matching them. The true $G \in \Theta$. For a given price $\mathbf{p}$, CMM evaluates the
best-case expected customer surplus over $\Theta$:

$$
\textbf{(CMM)} \quad Z^*(\mathbf{p}) = \sup_{\theta\in\Theta}\ \mathbb{E}_\theta\!\left[ \max_{s\in\{0\}\cup S} \{\tilde w_s - p_s\} \right],
$$

and uses the maximizing distribution $\theta^*(\mathbf{p})$ to define the approximate demand

$$
q_s(\mathbf{p}) = \Pr_{\tilde{\mathbf{w}}\sim\theta^*(\mathbf{p})}\!\left( s = \arg\max_{i\in\{0\}\cup S}\{\tilde w_i - p_i\} \right).
$$

Then (BSP) is approximated by **(BSP-CMM)**, the same profit objective with $q_s$ in place of
$q_s^*$. This is the persistency / semi-parametric choice idea of Mishra, Natarajan, Tao, Teo
(2012): the choice probabilities are the "persistency values" of the extremal distribution,
and they are obtained from a semidefinite program.

#### The SDP form

Let $\boldsymbol{\alpha} = \boldsymbol{\omega} - \mathbf{p}$ be the mean surplus vector and
$\boldsymbol{\Pi} = \boldsymbol{\Sigma} + \boldsymbol{\alpha}\boldsymbol{\alpha}^\top$ the
second-moment matrix. CMM (with the outside option eliminated, the reduced form CMM2 in the
paper) is the SDP

$$
Z^* = \max_{\mathbf{x},\mathbf{Y}}\ \operatorname{tr}(\mathbf{Y})
\quad\text{s.t.}\quad
\mathbf{x}\in\Delta_m,\quad
\begin{pmatrix} \boldsymbol{\Pi} & \mathbf{Y}^\top & \boldsymbol{\alpha} \\ \mathbf{Y} & \operatorname{Diag}(\mathbf{x}) & \mathbf{x} \\ \boldsymbol{\alpha}^\top & \mathbf{x}^\top & 1 \end{pmatrix} \succeq 0,
$$

where $\Delta_m = \{\mathbf{x}\in\mathbb{R}^m_+ : \mathbf{e}^\top\mathbf{x}\le 1\}$. The optimal
$x_i^*$ is the choice probability of size-$i$ bundle, i.e. $q_i$. The outside option takes the
slack $1 - \mathbf{e}^\top\mathbf{x}$.

### Main results

#### Theorem 1: demand as a concave maximization in the simplex

Assume $\boldsymbol{\Sigma}\succ 0$. Then the CMM demand is

$$
\mathbf{q}(\mathbf{p}) = \arg\max_{\mathbf{x}\in\Delta_m} \left\{ (\boldsymbol{\omega} - \mathbf{p})^\top\mathbf{x} + f(\mathbf{x}) \right\}, \qquad f(\mathbf{x}) = \operatorname{tr}\!\left( \big(\boldsymbol{\Sigma}^{1/2}\mathbf{S}(\mathbf{x})\boldsymbol{\Sigma}^{1/2}\big)^{1/2} \right),
$$

where $\mathbf{S}(\mathbf{x}) = \operatorname{Diag}(\mathbf{x}) - \mathbf{x}\mathbf{x}^\top$.

$f$ is strongly concave on $\Delta_m$ (Ahipasaoglu, Li, Natarajan 2018). So the demand is the
unique solution of a strongly concave program.

The full proof (reproduced from the paper's Appendix A) is below. It has two parts: Lemma 3
reduces the CMM SDP with the outside option (CMM1) to the one without it (CMM2), and then CMM2
collapses to the concave simplex program. Both are written out in full.

##### Lemma 3: the SDP reduction CMM1 $\equiv$ CMM2

Write the SDP of the previous section, but now keeping the outside option as an explicit
alternative of dimension $m+1$ (subscript $0$ marks the augmented objects):

$$
\textbf{(CMM1)}\quad Z^* = \max_{\mathbf{x}_0,\mathbf{Y}_0}\operatorname{tr}(\mathbf{Y}_0)
\ \text{s.t.}\ \mathbf{x}_0\in\Delta^=_{m+1},\
\begin{pmatrix} \boldsymbol{\Pi}_0 & \mathbf{Y}_0^\top & \boldsymbol{\alpha}_0 \\ \mathbf{Y}_0 & \operatorname{Diag}(\mathbf{x}_0) & \mathbf{x}_0 \\ \boldsymbol{\alpha}_0^\top & \mathbf{x}_0^\top & 1 \end{pmatrix}\succeq 0,
$$

with $\Delta^=_{m+1}=\{\mathbf{x}\in\mathbb{R}^{m+1}_+:\mathbf{e}^\top\mathbf{x}=1\}$ and the
outside option's surplus deterministically zero ($\tilde a_0=\alpha_0=0$). The reduced form is
CMM2, the SDP of the previous section, with $\Delta_m=\{\mathbf{x}\in\mathbb{R}^m_+:\mathbf{e}^\top\mathbf{x}\le 1\}$.

**Lemma 3.** CMM1 and CMM2 have the same optimal value, and their optimal $\mathbf{x}$ agree.

**Proof.** *(CMM1 $\Rightarrow$ CMM2.)* The CMM2 matrix is the principal submatrix of the CMM1
matrix obtained by deleting the outside option's row and column, hence is PSD; and
$\operatorname{tr}(\mathbf{Y}_0)=\operatorname{tr}(\mathbf{Y})$ because the outside option
contributes $\alpha_0=0$ to the relevant diagonal entry. So any CMM1 solution is CMM2-feasible
with the same objective.

*(CMM2 $\Rightarrow$ CMM1, the completion.)* Take a CMM2 optimal $\mathbf{x}=(x_1,\dots,x_m)$.
A Schur complement on the trailing $1$ of the CMM2 block gives
$\bigl(\begin{smallmatrix}\boldsymbol{\Sigma}&\hat{\mathbf{Y}}^\top\\\hat{\mathbf{Y}}&\mathbf{S}(\mathbf{x})\end{smallmatrix}\bigr)\succeq0$,
where $\hat{\mathbf{Y}}=\mathbf{Y}-\mathbf{x}\boldsymbol{\alpha}^\top$ and we used
$\boldsymbol{\Pi}=\boldsymbol{\Sigma}+\boldsymbol{\alpha}\boldsymbol{\alpha}^\top$. The matching
CMM1 Schur complement is the *partial* matrix (the entries marked $?$ are the unspecified
cross-moments between $\boldsymbol{\Sigma}$ and the outside option, $x_0=1-\mathbf{e}^\top\mathbf{x}$):

$$
\mathbf{Q}=\begin{pmatrix} \boldsymbol{\Sigma} & \hat{\mathbf{Y}}^\top & ? \\ \hat{\mathbf{Y}} & \mathbf{S}(\mathbf{x}) & -x_0\mathbf{x} \\ ? & -x_0\mathbf{x}^\top & x_0-x_0^2 \end{pmatrix}.
$$

The trailing block $\mathbf{S}(\mathbf{x}_0)=\bigl(\begin{smallmatrix}\mathbf{S}(\mathbf{x})&-x_0\mathbf{x}\\-x_0\mathbf{x}^\top&x_0-x_0^2\end{smallmatrix}\bigr)$ is PSD, because for any
$\mathbf{z}\in\mathbb{R}^{m+1}$,
$\mathbf{z}^\top\mathbf{S}(\mathbf{x}_0)\mathbf{z}=\sum_{i=0}^m x_i z_i^2-\bigl(\sum_{i=0}^m x_i z_i\bigr)^2=\operatorname{Var}(\tilde z)\ge0$,
where $\tilde z$ takes value $z_i$ with probability $x_i$. So every fully specified principal
submatrix of $\mathbf{Q}$ is PSD, i.e. $\mathbf{Q}$ is partial PSD. The specified entries define
a graph $G$ on vertices $\{1,\dots,m\}\cup\{1',\dots,m'\}\cup\{\nu\}$ (the $\boldsymbol{\Sigma}$
rows, the $\mathbf{S}$ rows, and the last row), in which $\{1,\dots,m,1',\dots,m'\}$ is a clique
and $\{\nu,1',\dots,m'\}$ is a clique, the only missing edges being the $?$ pair. The ordering
$(\nu,1',\dots,m',1,\dots,m)$ is a perfect elimination ordering, so $G$ is chordal (Rose 1970),
and a partial PSD matrix with a chordal pattern admits a PSD completion (Grone et al. 1984). Undo
the Schur complement on the completed matrix to get a feasible CMM1 point with the same
$\operatorname{tr}(\mathbf{Y})$. Hence the two values are equal. $\square$

This is the step that goes beyond Ahipasaoglu et al. (2018): their argument needs the covariance
*including* the outside option to be nonsingular, which fails here because the outside option is
deterministically zero. The chordal completion supplies the missing cross-moments instead. The
overlapping pattern of $\mathbf{Q}$ (compare Padmanabhan et al. 2019, whose pattern is
non-overlapping) is what forces the particular elimination ordering above.

##### Proof of Theorem 1

By Lemma 3 work with CMM2. A Schur complement on its trailing $1$ turns it into the two-stage
problem

$$
Z^*=\max\{\boldsymbol{\alpha}^\top\mathbf{x}+f(\mathbf{x}):\mathbf{x}\in\Delta_m\},\qquad
f(\mathbf{x})=\max_{\hat{\mathbf{Y}}}\operatorname{tr}(\hat{\mathbf{Y}})\ \text{s.t.}\
\begin{pmatrix}\boldsymbol{\Sigma}&\hat{\mathbf{Y}}^\top\\\hat{\mathbf{Y}}&\mathbf{S}(\mathbf{x})\end{pmatrix}\succeq0,
$$

where $\hat{\mathbf{Y}}=\mathbf{Y}-\mathbf{x}\boldsymbol{\alpha}^\top$ so that
$\operatorname{tr}(\mathbf{Y})=\boldsymbol{\alpha}^\top\mathbf{x}+\operatorname{tr}(\hat{\mathbf{Y}})$,
which is where the linear term $\boldsymbol{\alpha}^\top\mathbf{x}$ comes from. $\mathbf{S}(\mathbf{x})$
is diagonally dominant, hence PSD, so the inner problem is a genuine SDP. With
$\boldsymbol{\Sigma}\succ0$ we have $\operatorname{range}(\mathbf{S}(\mathbf{x}))\subseteq\operatorname{range}(\boldsymbol{\Sigma})$,
and the trace-maximizing coupling has the closed form (Dowson and Landau 1982; Olkin and
Pukelsheim 1982; Shapiro 1985)

$$
\hat{\mathbf{Y}}^{*\top}=\boldsymbol{\Sigma}\,\mathbf{S}(\mathbf{x})^{1/2}\Bigl(\mathbf{S}(\mathbf{x})^{1/2}\boldsymbol{\Sigma}\mathbf{S}(\mathbf{x})^{1/2}\Bigr)^{1/2\dagger}\mathbf{S}(\mathbf{x})^{1/2},
$$

with $\dagger$ the Moore-Penrose pseudoinverse (this is exactly the pseudoinverse the code uses
in `cmm_grad_f`). Substituting and using the cyclic property of the trace, the pseudoinverse
identity, and the symmetry of $\boldsymbol{\Sigma}^{1/2}$, $\mathbf{S}(\mathbf{x})^{1/2}$,

$$
f(\mathbf{x})=\operatorname{tr}\!\Bigl(\bigl(\mathbf{S}(\mathbf{x})^{1/2}\boldsymbol{\Sigma}\mathbf{S}(\mathbf{x})^{1/2}\bigr)^{1/2}\Bigr)=\operatorname{tr}\!\Bigl(\bigl(\boldsymbol{\Sigma}^{1/2}\mathbf{S}(\mathbf{x})\boldsymbol{\Sigma}^{1/2}\bigr)^{1/2}\Bigr),
$$

the last equality because $\mathbf{A}\mathbf{B}$ and $\mathbf{B}\mathbf{A}$ share nonzero
eigenvalues (here $\mathbf{A}=\boldsymbol{\Sigma}^{1/2}$, $\mathbf{B}=\mathbf{S}(\mathbf{x})\boldsymbol{\Sigma}^{1/2}$).
This is the claimed objective. $f$ is strongly concave on $\Delta_m$ (Ahipasaoglu et al. 2018),
so the maximizer $\mathbf{q}(\mathbf{p})$ is unique. $\square$

#### The price-demand bijection

On the interior
$\operatorname{int}(\Delta_m) = \{\mathbf{x} : \mathbf{e}^\top\mathbf{x} < 1,\ \mathbf{x} > 0\}$,
with $\mathbf{T}(\mathbf{q}) = \boldsymbol{\Sigma}^{1/2}\mathbf{S}(\mathbf{q})\boldsymbol{\Sigma}^{1/2}$,
the gradient of $f$ is

$$
\nabla f(\mathbf{q}) = \tfrac{1}{2}\operatorname{diag}\!\left( \boldsymbol{\Sigma}^{1/2}\mathbf{T}(\mathbf{q})^{-1/2}\boldsymbol{\Sigma}^{1/2} \right) - \boldsymbol{\Sigma}^{1/2}\mathbf{T}(\mathbf{q})^{-1/2}\boldsymbol{\Sigma}^{1/2}\mathbf{q}.
$$

- **Lemma 1.** If $\boldsymbol{\Sigma}\succ 0$ then $\|\nabla f\|\to\infty$ as $\mathbf{x}$
  approaches the boundary of $\Delta_m$. So the optimum of Theorem 1 lies in the interior, and
  the first-order condition holds with equality.

  *Proof.* The boundary has two faces, $x_i=0$ and $\mathbf{e}^\top\mathbf{x}=1$; the first is
  handled as in Theorem 5 of Ahipasaoglu et al. (2018), so take a sequence of interior points
  approaching $\mathbf{x}_0$ with $\mathbf{e}^\top\mathbf{x}_0=1$ along direction $\mathbf{e}_i$.
  At such $\mathbf{x}_0$, $\mathbf{T}(\mathbf{x}_0)=\boldsymbol{\Sigma}^{1/2}\mathbf{S}(\mathbf{x}_0)\boldsymbol{\Sigma}^{1/2}$
  has rank $m-1$ with smallest eigenvalue $0$ and eigenvector $\boldsymbol{\Sigma}^{-1/2}\mathbf{e}$,
  since $\mathbf{T}(\mathbf{x}_0)\boldsymbol{\Sigma}^{-1/2}\mathbf{e}=\boldsymbol{\Sigma}^{1/2}\mathbf{S}(\mathbf{x}_0)\mathbf{e}=\mathbf{0}$
  (as $\mathbf{S}(\mathbf{x}_0)\mathbf{e}=\mathbf{x}_0-\mathbf{x}_0(\mathbf{e}^\top\mathbf{x}_0)=\mathbf{0}$).
  A first-order perturbation of the smallest eigenvalue gives
  $\lambda_1(\mathbf{T}(\mathbf{x}_0-\delta\mathbf{e}_i))=\delta/(\mathbf{e}^\top\boldsymbol{\Sigma}^{-1}\mathbf{e})+o(\delta)$.
  Since $f(\mathbf{x})=\sum_j\sqrt{\lambda_j(\mathbf{T}(\mathbf{x}))}$ and the other eigenvalues
  stay bounded away from zero, the directional derivative
  $\lim_{\delta\to0^+}\bigl(f(\mathbf{x}_0)-f(\mathbf{x}_0-\delta\mathbf{e}_i)\bigr)/\delta$ is
  dominated by $-\sqrt{\delta/(\mathbf{e}^\top\boldsymbol{\Sigma}^{-1}\mathbf{e})}/\delta\to-\infty$
  (the square root is concave, so the remaining terms are $O(1)$). Hence $\|\nabla f\|\to\infty$. $\square$
- The first-order condition of Theorem 1 is $\mathbf{p} - \boldsymbol{\omega} = \nabla f(\mathbf{q})$, i.e.

  $$\mathbf{p} = \boldsymbol{\omega} + \nabla f(\mathbf{q}).$$

- **Lemma 2.** The map $H:\mathbf{p}\mapsto\mathbf{q}$ is a bijection from $\mathbb{R}^m$ onto
  $\operatorname{int}(\Delta_m)$. So we may treat the choice probabilities $\mathbf{q}$ as the
  decision variable and recover prices afterwards.

  *Proof.* Strong concavity of $f$ on $\Delta_m$ together with Lemma 1 gives a unique interior
  maximizer $\mathbf{q}$ of Theorem 1 for every $\mathbf{p}\in\mathbb{R}^m$, so $H$ is injective.
  Surjectivity is immediate from $\mathbf{p}=\boldsymbol{\omega}+\nabla f(\mathbf{q})$: any
  $\mathbf{q}\in\operatorname{int}(\Delta_m)$ is the image of that $\mathbf{p}$. Hence $H$ is a
  bijection. $\square$

#### Profit reparametrized, and Theorem 2: concavity

Substitute $\mathbf{p} = \boldsymbol{\omega} + \nabla f(\mathbf{q})$ into the profit
$\pi = \sum_{s}(p_s - c_s)q_s = \mathbf{q}^\top(\mathbf{p} - \mathbf{c})$:

$$
\pi(\mathbf{q}) = \mathbf{q}^\top(\boldsymbol{\omega} - \mathbf{c}) + \tfrac{1}{2}\mathbf{q}^\top\operatorname{diag}\!\left(\boldsymbol{\Sigma}^{1/2}\mathbf{T}(\mathbf{q})^{-1/2}\boldsymbol{\Sigma}^{1/2}\right) - \mathbf{q}^\top\boldsymbol{\Sigma}^{1/2}\mathbf{T}(\mathbf{q})^{-1/2}\boldsymbol{\Sigma}^{1/2}\mathbf{q}.
$$

**Theorem 2.** Assume $\boldsymbol{\Sigma}\succ 0$. Then $\pi(\mathbf{q})$ is concave on
$\operatorname{int}(\Delta_m)$.

So the BSP problem under CMM is

$$\max_{\mathbf{q}\in\Delta_m}\ \pi(\mathbf{q}),$$

a concave maximization, solvable to global optimality with off-the-shelf solvers, after which
$\mathbf{p}^* = \boldsymbol{\omega} + \nabla f(\mathbf{q}^*)$ gives the optimal price menu. This
is the whole payoff: an apparently intractable bilevel order-statistics problem becomes a
single convex program once the inputs are reduced to $(\boldsymbol{\omega}, \boldsymbol{\Sigma})$.

The full proof (reproduced from Appendix A) rests on one operator-concavity lemma.

**Lemma 6.** For $\mathbf{x},\mathbf{y}\in\Delta_m$ and $\lambda\in[0,1]$, with
$\mathbf{T}(\mathbf{x})=\boldsymbol{\Sigma}^{1/2}\mathbf{S}(\mathbf{x})\boldsymbol{\Sigma}^{1/2}$,

$$\mathbf{T}(\lambda\mathbf{x}+(1-\lambda)\mathbf{y})^{1/2}\succeq\lambda\,\mathbf{T}(\mathbf{x})^{1/2}+(1-\lambda)\,\mathbf{T}(\mathbf{y})^{1/2}.$$

*Proof.* Expanding $\mathbf{S}$,

$$
\mathbf{S}(\lambda\mathbf{x}+(1-\lambda)\mathbf{y})=\lambda\mathbf{S}(\mathbf{x})+(1-\lambda)\mathbf{S}(\mathbf{y})+\lambda(1-\lambda)(\mathbf{x}-\mathbf{y})(\mathbf{x}-\mathbf{y})^\top\succeq\lambda\mathbf{S}(\mathbf{x})+(1-\lambda)\mathbf{S}(\mathbf{y}),
$$

so conjugating by $\boldsymbol{\Sigma}^{1/2}$ gives $\mathbf{T}(\lambda\mathbf{x}+(1-\lambda)\mathbf{y})\succeq\lambda\mathbf{T}(\mathbf{x})+(1-\lambda)\mathbf{T}(\mathbf{y})$.
The matrix square root is operator monotone and operator concave on PSD matrices (Horn and
Johnson 1990), hence
$\mathbf{T}(\lambda\mathbf{x}+(1-\lambda)\mathbf{y})^{1/2}\succeq(\lambda\mathbf{T}(\mathbf{x})+(1-\lambda)\mathbf{T}(\mathbf{y}))^{1/2}\succeq\lambda\mathbf{T}(\mathbf{x})^{1/2}+(1-\lambda)\mathbf{T}(\mathbf{y})^{1/2}$. $\square$

**Proof of Theorem 2.** Since $\pi(\mathbf{q})=\mathbf{q}^\top(\boldsymbol{\omega}-\mathbf{c})-g(\mathbf{q})$
with $g(\mathbf{q})=-\mathbf{q}^\top\nabla f(\mathbf{q})$ and the first term linear, it suffices
to show $g$ is convex. Writing $\mathbf{T}=\mathbf{T}(\mathbf{q})$ and using
$\operatorname{Diag}(\mathbf{q})-\mathbf{q}\mathbf{q}^\top=\mathbf{S}(\mathbf{q})$ together with
$\operatorname{tr}\!\bigl(\boldsymbol{\Sigma}^{1/2}\mathbf{S}(\mathbf{q})\boldsymbol{\Sigma}^{1/2}\mathbf{T}^{-1/2}\bigr)=\operatorname{tr}(\mathbf{T}^{1/2})$,

$$
g(\mathbf{q})=-\tfrac12\mathbf{q}^\top\operatorname{diag}\!\bigl(\boldsymbol{\Sigma}^{1/2}\mathbf{T}^{-1/2}\boldsymbol{\Sigma}^{1/2}\bigr)+\mathbf{q}^\top\boldsymbol{\Sigma}^{1/2}\mathbf{T}^{-1/2}\boldsymbol{\Sigma}^{1/2}\mathbf{q}=-\tfrac12\operatorname{tr}(\mathbf{T}^{1/2})+\tfrac12\mathbf{q}^\top\boldsymbol{\Sigma}^{1/2}\mathbf{T}^{-1/2}\boldsymbol{\Sigma}^{1/2}\mathbf{q}.
$$

Therefore the epigraph $\{(\mathbf{q},t):g(\mathbf{q})\le\tfrac12 t\}$ equals

$$
\Bigl\{(\mathbf{q},t):\ t+\operatorname{tr}(\mathbf{T}^{1/2})-\mathbf{q}^\top\boldsymbol{\Sigma}^{1/2}\mathbf{T}^{-1/2}\boldsymbol{\Sigma}^{1/2}\mathbf{q}\ge0\Bigr\}=\Bigl\{(\mathbf{q},t):\begin{pmatrix} t+\operatorname{tr}(\mathbf{T}^{1/2}) & \mathbf{q}^\top\boldsymbol{\Sigma}^{1/2}\\ \boldsymbol{\Sigma}^{1/2}\mathbf{q} & \mathbf{T}^{1/2}\end{pmatrix}\succeq0\Bigr\},
$$

the equality by a Schur complement on the matrix-fractional term. Introducing a slack
$\mathbf{Z}$, this epigraph is described by three constraints: (i) the linear inequality
$t+\operatorname{tr}(\mathbf{T}(\mathbf{q})^{1/2})\ge z$; (ii) the PSD constraint
$\bigl(\begin{smallmatrix} z & \mathbf{q}^\top\boldsymbol{\Sigma}^{1/2}\\ \boldsymbol{\Sigma}^{1/2}\mathbf{q} & \mathbf{Z}\end{smallmatrix}\bigr)\succeq0$;
and (iii) $\mathbf{T}(\mathbf{q})^{1/2}\succeq\mathbf{Z}$. Constraint (ii) is affine in
$(z,\mathbf{q},\mathbf{Z})$, hence convex. In (i), $\operatorname{tr}(\mathbf{T}(\mathbf{q})^{1/2})$
is concave in $\mathbf{q}$ by Lemma 6, so the inequality defines a convex set. For (iii), the set
$\{(\mathbf{q},\mathbf{Z}):\mathbf{T}(\mathbf{q})^{1/2}\succeq\mathbf{Z}\}$ is convex: for two of
its members and $\lambda\in[0,1]$, Lemma 6 gives
$\mathbf{T}(\lambda\mathbf{q}_1+(1-\lambda)\mathbf{q}_2)^{1/2}\succeq\lambda\mathbf{T}(\mathbf{q}_1)^{1/2}+(1-\lambda)\mathbf{T}(\mathbf{q}_2)^{1/2}\succeq\lambda\mathbf{Z}_1+(1-\lambda)\mathbf{Z}_2$.
The intersection of three convex sets is convex, so the epigraph of $g$ is convex, $g$ is convex,
and $\pi$ is concave on $\operatorname{int}(\Delta_m)$. $\square$

This is the non-obvious step: computing CMM choice probabilities was already known to be convex,
but that alone does not make the *pricing* problem convex; Theorem 2 establishes it via the
operator concavity of $\mathbf{q}\mapsto\operatorname{tr}(\mathbf{T}(\mathbf{q})^{1/2})$.

#### Note on assumptions

$\boldsymbol{\Sigma}\succ 0$ held in all of the paper's numerical tests. If $\boldsymbol{\Sigma}$ is singular, adding $\epsilon\mathbf{I}$ produces a regularized
positive-definite approximation. The theorem then applies to that perturbed input, not literally
to the original singular instance; the perturbation and its sensitivity must be reported. The convexity result relies on
homogeneous price sensitivity (normalized to one); with heterogeneous price sensitivity the
problem is not generally convex.

### Single-size case (Corollary 1), derived in full

This is the clean, self-contained case. Take $S = \{i\}$, so $m = 1$ and the decision is a
scalar choice probability $x\in[0,1]$. Let $\omega = \omega_i$, $\sigma = \sigma_i$ (so
$\boldsymbol{\Sigma} = \sigma^2$), and $c = c_i$.

With $m = 1$, $\mathbf{S}(x) = x - x^2$ and $\boldsymbol{\Sigma}^{1/2} = \sigma$, so

$$f(x) = \big(\sigma^2 (x - x^2)\big)^{1/2} = \sigma\sqrt{x - x^2}.$$

**Price-demand map.** The demand maximizes $(\omega - p)x + \sigma\sqrt{x-x^2}$. Its
first-order condition $\omega - p + \sigma\frac{1-2x}{2\sqrt{x-x^2}} = 0$ rearranges to

$$p^{\mathrm{CMM}} = \omega + \sigma\,\frac{1-2x}{2\sqrt{x-x^2}}.$$

For comparison, the multinomial probit (MNP) price-demand map is
$p^{\mathrm{MNP}} = \omega + \sigma\,\Phi^{-1}(1-x)$, where $\Phi$ is the standard normal CDF.
CMM uses $\frac{1-2x}{2\sqrt{x-x^2}}$ as a closed-form proxy for $\Phi^{-1}(1-x)$; the two
curves are very close, which is the basis of the paper's Figure 1 validation.

**Profit first-order condition.** The profit is $\pi(x) = (p - c)x$. Substitute the price map:

$$\pi(x) = (\omega - c)x + \sigma\,\underbrace{\frac{x(1-2x)}{2\sqrt{x-x^2}}}_{g(x)}.$$

Differentiate $g$. Write $g(x) = \tfrac{1}{2}\,h(x)\,k(x)$ with $h = x - 2x^2$ (so
$h' = 1 - 4x$) and $k = (x - x^2)^{-1/2}$ (so $k' = -\tfrac{1}{2}(x-x^2)^{-3/2}(1-2x)$). Then

$$
g'(x) = \tfrac{1}{2}(h'k + hk') = \tfrac{1}{2}(x-x^2)^{-3/2}\Big[(1-4x)(x-x^2) - \tfrac{1}{2}(x-2x^2)(1-2x)\Big].
$$

Expanding the bracket: $(1-4x)(x-x^2) = x - 5x^2 + 4x^3$ and
$\tfrac{1}{2}(x-2x^2)(1-2x) = \tfrac{1}{2}x - 2x^2 + 2x^3$, whose difference is
$\tfrac{1}{2}x - 3x^2 + 2x^3 = \tfrac{1}{4}x(4x^2 - 6x + 1)$. Using
$(x-x^2)^{3/2} = x(1-x)\sqrt{x-x^2}$,

$$g'(x) = \frac{x(4x^2 - 6x + 1)}{4\,(x-x^2)^{3/2}} = \frac{4x^2 - 6x + 1}{4(1-x)\sqrt{x-x^2}}.$$

So $\pi'(x) = (\omega - c) + \sigma g'(x) = 0$ gives the optimal single-size demand $x$ as the
root of

$$\boxed{\ \omega_i - c_i + \sigma_i\,\frac{4x^2 - 6x + 1}{4(1-x)\sqrt{x-x^2}} = 0\ }$$

which is Corollary 1. Solve it by one-dimensional root finding (e.g. scipy brentq on $(0,1)$),
then read off the price from the price-demand map. Running this over every size $i\in[n]$ and
keeping the best gives the optimal single-size policy, a complete and cheap fallback to the
full convex program.

### Archive interpretation and errata

The archived implementation transformed an ownership-derived score panel and then applied CMM to
its order-statistic moments. That exercise is not part of the live dependency graph. In particular:

- the transformed panel was model-dependent and was not identified willingness to pay;
- the attempted observed-price anchor failed, so no dollar calibration was established;
- the differential-evolution empirical menu was a best solution found, not a certified optimum or
  assumption-free ground truth;
- notebook 10 found that the CMM menu attained 69--97 percent of the best empirical menu found on
  its selected item sets, while the real partial sums were highly skewed;
- the decisive reason for retirement is the selling-mechanism mismatch, not merely the observed
  two-moment approximation error; and
- no CMM demand, moment, menu, price, or result is reused by the live CP-anchored SBA model.

### Phase-0 reproduction checklist

- [x] Re-derive the single-size results above from scratch and match the boxed equation.
- [x] Reproduce Lemma 3 (the SDP reduction) and Theorem 1 from Appendix A (written out above).
- [x] Reproduce Theorem 2 (concavity of $\pi$) from Appendix A (written out above, via Lemma 6).
- [x] Implement the single-size root-find and check the optimal $x$, $p$, profit by hand on a
      tiny example (`optimize_single_size`; `tests/test_bundle_pricing.py`).
- [x] Implement the convex program and confirm it matches a brute-force search on a small
      instance. The pricing problem is solved in the native $\mathbf{q}$ variables with scipy
      (`optimize_convex`); cvxpy/SCS is used for the demand SDP (eq. 12, `cmm_demand_sdp`) as an
      independent cross-check. Agreement is asserted in the tests and shown in notebook 06.
- [x] Reproduce the CMM-vs-MNP single-size comparison (the paper's Figure 1; notebook 06).

### References for Appendix A

- Li X, Sun H, Teo C-P. Convex Optimization for the Bundle Size Pricing Problem.
- Mishra VK, Natarajan K, Tao H, Teo C-P (2012). Choice prediction with semidefinite
  optimization when utilities are correlated. IEEE TAC.
- Ahipasaoglu SD, Li X, Natarajan K (2018). A convex optimization approach for computing
  correlated choice probabilities with many alternatives. IEEE TAC.
- Natarajan K, Teo C-P (2017). On reduced semidefinite programs for second order moment
  bounds. Math. Programming.
- Chu CS, Leslie P, Sorensen A (2011). Bundle-size pricing as an approximation to mixed
  bundling. American Economic Review.
- Abdallah T, Asadpour A, Reed J (2017). Large-scale bundle size pricing.
- Adams WJ, Yellen JL (1976). Commodity bundling and the burden of monopoly. QJE.
- McAfee RP, McMillan J, Whinston MD (1989). Multiproduct monopoly, commodity bundling, and
  correlation of values. QJE.
- Bakos Y, Brynjolfsson E (1999). Bundling information goods. Management Science.
- Vandenberghe L, Andersen MS (2015). Chordal graphs and semidefinite optimization.
- Grone R, Johnson CR, Sa EM, Wolkowicz H (1984). Positive definite completions of partial
  Hermitian matrices. Linear Algebra Appl. (chordal PSD completion, used in Lemma 3.)
- Rose DJ (1970). Triangulated graphs and the elimination process. (Perfect elimination
  ordering; chordality in Lemma 3.)
- Dowson D, Landau B (1982); Olkin I, Pukelsheim F (1982); Shapiro A (1985). Closed-form
  trace-maximizing coupling used for the inner SDP in the Theorem 1 proof.
- Horn RA, Johnson CR (1990). Matrix Analysis. (Operator monotonicity / concavity of the matrix
  square root, used in Lemma 6.)
- Padmanabhan D, Natarajan K, Murthy K (2019). Exploiting partial correlations in
  distributionally robust optimization. (Non-overlapping completion pattern, contrasted in Lemma 3.)
