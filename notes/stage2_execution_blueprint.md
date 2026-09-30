# Stage 2 execution blueprint and design review

Reviewed: 2026-09-29. Status: **planning only; implementation has not begun**.
Execution links reconciled: 2026-09-30.

Stage 1 completion and improvement precede Stage 2. This document
turns Sections 11 to 17 of [planning.md](../planning.md) into an execution contract to freeze later.
No Stage 2 inputs, policies, solver results, or assessment outcomes were generated for this review.
The earlier study pause through November 2026 remains a calendar note; a later request to begin is
still needed. Proposed defaults below are explicit recommendations, not an already frozen protocol.

The historical mechanism specification and [theory note](optimization_models.md) are retained.
P1 to P8 have project proofs there. P9, the implementation tests, numerical certificates, and a
complete reproduced SBR paper proof remain future deliverables. The new clarifications below do not
change historical Stage 1 metrics, fitted parameters, transformations, or admission decisions.

## What Stage 2 can establish

The question is: under declared additive pseudo-utilities, which feasible fixed bundle and posted
price improve normalized seller margin when components remain available at fixed component prices?
The primary mechanism is CP-anchored Single Bundle with All, abbreviated SBA. The answer is
conditional on the model, transformation, user panel, pool, costs, constraints, and tie convention.

The current design has several strengths: it includes the no-bundle alternative, subtracts
displaced component margin, uses exact empirical price thresholds, distinguishes complete search
from heuristics, and evaluates design-selected policies on separate users. These choices make a
useful operations research study possible even without purchase records.

The main empirical limitation remains identification. A held-out user is represented by scores
inferred from permitted ownership history, and those scores become the outcome used to evaluate a
menu. Successful assessment demonstrates transfer of a policy across modeled user types. It does
not establish real demand, willingness to pay, future acquisition, welfare, or realized revenue.
Better ownership ranking does not validate the cardinal scale needed for bundle optimization.

Use synthetic pre-acquisition preference types as the primary interpretation. Do not count an
existing library as a purchase occasion or literally ask owners to repurchase it. A future
installed-base experiment must mask owned items, refit all design-side prices and policies, and
state its treatment of owned DLC and base games. It remains a separate modeled sensitivity.

## Review findings and required decisions

| Finding | Consequence | Required action before implementation or outcomes |
| --- | --- | --- |
| Four transformations preserve different aspects of scores | Similar rankings can produce different prices and compositions | Keep every transformation; choose exposition independently of gain |
| Within-user transforms use the full warm catalogue | Transforming only pool columns silently changes the scenario | Transform full-catalogue row blocks, then select pool columns |
| Only ALS passed historical admission | A model-family versus decision-quality comparison is not currently available | Bind the final Stage 1 release; retain only qualified core inputs |
| A 128-user aggregate assessment-score peek occurred | The panel is not pristine for all prior analysis | Carry the release-audit disclosure and log every later access |
| CP price ties can change bundle demand | Equal CP margin does not identify a unique SBA input | Freeze complete anchor vectors and a coverage rule |
| Different pool IDs can share the same games | Splitting by ID alone gives weak heuristic validation separation | Group related pools before development/locked assignment |
| Complete floating-point enumeration can leave close objectives unresolved | Search completeness and numerical proof are different | Record certificate tier and numerical ambiguity |
| Many seeds, scenarios, and overlapping pools reuse users | Naive pooling overstates independent evidence | Register summary weights and shared-user resampling |
| Broad factorial experiments can exhaust the budget | Missing cells invite selective reporting | Freeze a core grid, one-axis sensitivities, and explicit stop rules |

## Entry contract from Stage 1

Stage 2 must choose one complete Stage 1 release before any bundle outcomes are seen. The current
historical option is `s1-v2-20260814`, with one admitted ALS specification, three production seeds,
four frozen transformations, and 12,585 folded-in assessment users. A prospective Stage 1
improvement may replace it only after its own evidence and interface are complete. Comparing
releases remains separately labeled; a new cycle ID cannot erase prior test exposure.

The new [Stage 1 development runner](../src/stage1_successor.py),
[bounded runtime](../src/stage1_runtime.py), and
[minibatch trainer](../src/stage1_training.py) are additive development tools. Their
`configs/stage1_successor.json` configuration and validation outputs do not automatically qualify
a new Stage 2 model. In particular, the new BPR training objective changes the regularization
convention and needs its own tuning and evaluation. Bind either the complete historical release
or a subsequently completed successor release; do not mix one release's fitted parameters with
another's transformation parameters, catalogue map, or admission evidence.

The input manifest must bind the model and seed, training and fold-in manifests, ordered item and
user maps, transformation parameters, numerical dtypes, source hashes, and data provenance. Validate
all dependencies when loading a cached run. Require the same item order and scoring semantics for
design and assessment. Missing raw or protected inputs stop a scientific run; public manifest
verification alone cannot recreate them.

Pool scoring needs careful resource handling. Global transformation parameters come from their
frozen design fitting sample. Within-user percentiles, means, and standard deviations use the
complete production catalogue, even when the optimizer needs only a small pool. Bound memory by
scoring full-catalogue blocks of users, transforming each block, and retaining pool columns. Test
that an item's transformed value is identical when requested through two different pools.

For the first protocol, use the existing fixed 5,000 design evaluation users as the proposed
optimization panel. This is an explicit finite-panel choice that keeps repeated exact solves
tractable; it does not claim to optimize all design users. Freeze the ordered IDs through the
protected manifest, and use equal user weights. Keep all eligible assessment users for frozen
evaluation. A larger design-panel sensitivity is optional and must be registered before its
policy outcomes; do not choose panel size by which one delivers the largest gain.

No new assessment scoring should occur while Stage 1 is being improved. At future Stage 2 entry,
freeze the protocol before any further assessment diagnostics, including harmless-looking value
summaries. Assessment access is limited to complete frozen-policy evaluation and registered
uncertainty calculations. Record the historical aggregate peek rather than describing the panel
as never accessed.

## Menus and exact price primitives

All values and costs below are nonnegative and finite. Let `p_i` be the design-selected CP anchor,
`c_i` the pseudo-cost, `B` a composition, and `b` its price. Users may buy at most one unit of an
item, have additive quasi-linear utility, no budget limit, and the declared tie rule.

| Mechanism | Available menu | Weak-tie bundle threshold | Design objective |
| --- | --- | --- | --- |
| CP | Every item separately | Not applicable | Sum of itemwise margins `(p_i-c_i) Pr(v_i >= p_i)` |
| PB | Whole pool as one bundle | `sum_i v_ui` | `(b-c(N)) Pr(sum_i v_ui >= b)` |
| SBR | Bundle plus separate items outside it | `sum_(i in B) v_ui` | Bundle margin plus CP margin outside `B` |
| SBA | Bundle plus every separate item | `w_u(B) = sum_(i in B) min(v_ui,p_i)` | CP margin plus `mean[1(w_u >= b) (b-c(B)-A_u(B))]` |

Here `A_u(B)` is the CP margin that user `u` would contribute on bundle items through separate
purchases. It must be subtracted when that user switches to the bundle. Products outside the pool
cancel only under the additive, no-budget, common-menu assumptions; complementarity, budgets, or
ownership-adjusted prices would require a new choice model.

CP scans distinct observed item values at or above cost plus an explicit no-sale policy. PB and
fixed-composition SBR scan distinct total values plus no sale. Fixed-composition SBA scans distinct
truncated totals plus no bundle. Sort complete equality blocks, then update buyer count and
displaced margin. The SBA gain at threshold `t` is

```text
(buyer_count * (t - bundle_cost) - displaced_margin_of_buyers) / user_count.
```

Between thresholds, the buyer set is constant and this expression increases with price whenever
there is a buyer. Raising a price to its smallest buying threshold therefore suffices under weak
ties. An empty buyer set is dominated or matched by the baseline. This is the project-specific
finite-price argument; it does not depend on normal valuations or an imported SBR approximation.

The no-sale policy must remain structural on assessment users. A numerical price just above the
design maximum is not equivalent: an assessment user could exceed it. For a CP no-sale anchor use
`Q_ui = v_ui` and `R_ui = 0` directly. Do not calculate infinity times zero. All-zero instances
must return the declared baseline and leave zero-denominator ratios undefined.

Component equality means buy; primary bundle equality means take the bundle. Return no bundle on
zero incremental gain. For positive-gain ties, use the smallest price, then smallest composition,
then canonical item order. Keep exact ties separate from numerical near-ties and report both.

For the strict bundle-tie sensitivity, specify a price lattice `b = k * delta`. Candidate prices
are the greatest nonnegative tick strictly below each threshold, plus no bundle. Handle thresholds
exactly on a tick with verified arithmetic; ordinary floating-point division followed blindly by
`ceil` can select the wrong tick. Register `delta` as a fixed fraction of a positive design-only
scenario scale, with an all-zero fallback. Scale the tick whenever values and costs scale. A
continuous strict-tie supremum is not an executable policy.

The existing P9 proof obligation is straightforward: the feasible composition family is finite;
each nonempty member has a complete finite candidate price set; the empty member supplies CP;
and selecting the maximum over their union gives a global optimum for that declared instance.
Its implementation obligation additionally requires complete counts, correct arithmetic, and the
same feasibility predicate everywhere. Incomplete enumeration fails that obligation.

## Pools, feasibility, costs, and anchors

Construct pools from raw publisher, developer, franchise, or compatibility metadata. Normalize
aliases and preserve missingness and manual-override reasons. Exclude non-games and incompatible
products through a single documented rule. Unknown legal control remains an interpretation limit.
Do not select pools using favorable correlations or bundle gains. Save excluded pools and reasons.

All mechanisms use a common registry of item order and constraints. Bundle composition is empty
or has size two through capacity, with the registered compatibility constraints. CP and PB are
benchmarks with their own menus: PB can be infeasible under the capacity imposed on SBA/SBR, so
label that difference instead of implying a feasible-policy dominance comparison. An observed
Steam composition is comparable only when every component is covered and the composition is
feasible. Do not silently drop unavailable items; that creates a different bundle.

The minimum descriptive bridge uses raw co-ownership and admitted identity-only ALS dependence.
Genre-aware BPR failed admission, so it cannot be required as a production input. If shown from an
archived fit, label it diagnostic and keep it outside the core downstream scenarios. Freeze at most
two primary matched-control statistics before their differences are inspected. Use design users
for matching diagnostics and the bridge.

Primary costs remain zero. A proposed positive-cost sensitivity is
`c_i = 0.1 * median_positive_design_value_i`, with zero when an item has no positive values.
Compute the scale separately for each scenario and seed using its fixed design panel. Record the
formula and resulting costs before optimization. These are assumed scenario costs, not observed
seller expenses. This rule respects common positive scaling; holding dollar costs fixed across
arbitrary pseudo-utility transformations would not.

Record all CP-optimal itemwise threshold representatives and any optimal no-sale representative.
The primary vector takes the smallest positive-demand optimum, falling back to no sale. Required
representative sensitivities use the largest threshold vector and a no-sale-preferred vector.
Mixed itemwise combinations can matter. Count their Cartesian product; exhaust it only within a
registered anchor budget, otherwise apply an outcome-independent sample and state that coverage is
partial. Extreme vectors alone do not certify worst-case robustness over all anchors.

## Execution packages and acceptance gates

Each package begins only after a future request to start Stage 2. Estimated effort is deliberately
not a completion promise: runtime pilots determine the search frontier.

| Package | Work and proposed code boundary | Acceptance evidence | If it fails |
| --- | --- | --- | --- |
| S2.0 inputs and protocol | Final Stage 1 binding; `candidate_pools.py`; protected user maps; pool groups; core grid; costs; ties; budgets; access ledger | Content-derived protocol and instance identities preceding affected outcomes | Resolve missing metadata or inputs; retain exclusions |
| Bridge/Gate 3 | Notebook 11, matched controls, design-only raw and score dependence | Balance, coverage, registered statistics, descriptive wording | Mark unmatched cases; narrow bridge claims |
| S2.1–S2.3 semantics | `bundle_design.py` pure CP/PB/SBR/SBA APIs and an independently written tiny menu oracle | Direct/reduced choice and objective agreement; P1–P9 tests; no-sale and strict-grid edge cases | Correct semantics before real-pool optimization |
| S2.4 exact search | Reference enumeration, counts, numerics, checkpoint identities | Per-instance completeness and numerical certificate tier; all timeouts retained | Reduce declared scope, never silently subsample the same instance |
| S2.5 scalable search | Multistart add/drop/swap with exact repricing; baseline searches | Frozen settings and outputs before locked exact results; separate SBA/SBR gap reports | Headline exact pools only; larger results exploratory |
| S2.6 policies | Design-select CP, PB, SBR, SBA and feasible observed-composition prices | Complete policy hashes including costs, component anchors, prices and tie rules | No assessment access for incomplete policies |
| S2.7 assessment | Apply all frozen policies; paired resampling; registered robustness | Full grid and exclusions, negative outcomes retained, no reoptimization | Report loss or fragility; no assessment-selected replacement |
| S2.8 release | Tables, trace/certificate ledgers, claim index, synthetic reproduction | Clean regeneration and public evidence checks | Publish limitations and unfinished gates honestly |

Before S2.0 closes, assign exact pilot pools and freeze an outcome-independent rule for later
development and locked suites. Group pools that share a publisher/franchise or exceed a registered
item-overlap threshold, then assign connected groups together. Publish remaining cross-group
overlap. Pilot groups never enter development or locked suites. Exact runtime/completion evidence
may guide the size frontier; pilot objectives and compositions may not guide pool eligibility.

Development exact results may guide heuristic choices. Locked exact objectives, compositions, and
certificates remain sealed while the heuristic runs on their registered inputs. Freeze settings,
run the locked heuristic, hash its outputs, and then open exact results once. Keep timeouts as
unresolved comparisons. Report the number of independent catalogue groups in addition to the
number of scenario instances. Many variants of one pool do not supply independent validation.

The heuristic includes no bundle and starts of size at least two because singleton gains can be
zero while a pair helps. Compare it against an equal objective-evaluation budget random search and
a simple greedy baseline; also report wall time and memory. Use separate locked gates for SBA and
SBR. The provisional median 5% and 95th-percentile 20% incremental-gain-loss thresholds in
planning.md must be finalized before locked results, with all-zero and negligible-gain rules.
No theoretical approximation guarantee follows from passing this empirical gate.

## Numerical certificates

Separate three output classes rather than using one `exact` flag:

1. `certified_exact`: complete enumeration plus exact comparisons in the declared input arithmetic,
   or outward objective bounds proving the selected optimum. This certifies the declared finite
   panel, not the unknown true valuations or the original real-valued model before rounding.
2. `enumerated_numerical`: all feasible compositions and price candidates evaluated, but numerical
   uncertainty prevents a rigorous ordering. Report the best computed value, uncertainty interval,
   and possible winners. Call this exhaustive numerical evaluation.
3. `best_found`: search ended early or used a heuristic. Report incumbent, evaluations, termination
   reason and a bound only when one has actually been derived.

Tiny synthetic oracle cases should use integer or rational values so exact comparisons are easy.
For real score panels, freeze float64 inputs and deterministic summation order, then register a
method for conservative error bounds or higher-precision resolution of close candidates. An
exhaustive run with an arbitrary equality tolerance is not a mathematical proof. Keep the best
computed objective even if a canonical near-tie representative has slightly lower margin, and
record that difference. Do not merge distinct price thresholds to satisfy a tolerance.

The certificate records instance/protocol hashes, mechanism, users, item order, full feasibility
definition, expected and visited composition counts, price counts, optimizer and numeric status,
CP baseline, best/runner-up values, tied and near-optimal sets, canonical policy, resource usage,
and environment. Count feasible compositions through an independent check on small instances.
Store large tie sets in a streamed sidecar with a count and digest rather than consuming unbounded
memory. A timeout or failed count cannot be relabeled as an optimum.

## Experiment grid and assessment summaries

Use a compact grid agreed before outcomes. The proposed core crosses the admitted model's three
seeds and four transformations with each primary pool, its registered primary capacity, primary
anchor, weak bundle tie, and zero costs. The model family remains fixed to ALS unless a completed
new Stage 1 release admits another family. Choose the exposition transformation for a scientific
reason; global robust softplus is a proposed presentation default because it uses one global
mapping, but all four results remain equally visible and none is an identified monetary scale.

Run nearby capacities, alternative anchors, strict ties, and positive costs as one-axis
sensitivities around every primary scenario. Use a small preregistered factorial on representative
exact pools to check interactions. Freeze all cell IDs in advance and retain failed/infeasible
rows. Independent cost or tie choices must not be selected after observing which increases gain.
Do not promise that a smaller panel's exact optimum certifies a larger panel's instance.

For each policy report design and assessment mean margin, SBA-minus-frozen-CP difference, demand,
displaced component margin, price, bundle size, selected items, numerical/search status, and
resources. Assessment policies retain the design-selected component prices, bundle composition,
bundle price, costs, and tie rules. CP and all benchmarks are equally frozen. Repricing on
assessment is oracle optimization and changes the estimand.

Show each seed separately. Within a pool and transformation, summarize using equal seed weights
and dimensionless paired gains. Across transformations, show signs, range, composition overlap,
and target-scenario regret; never average raw pseudo-prices or objectives across incompatible
units. Average performance of three separately optimized policies is an algorithm-level summary,
not a single deployable menu. Selecting one seed after assessment is prohibited. An ensemble menu
would need a separate design-side definition frozen before assessment.

Use the same resampled assessment-user indices for every policy and seed in a paired bootstrap.
Keep fitted parameters and policies fixed. A confidence interval on the equal-seed mean then
reflects conditional user-panel variation, not uncertainty over all possible training seeds or
recommender estimation. Report three-seed ranges separately. Cross-pool summaries require explicit
pool weights and the same shared-user resampling; overlapping pools are separate counterfactuals
and cannot be added into portfolio revenue. State whether a summary describes registered pools
or a wider catalogue population; the present design supports the former.

Choose at most two primary assessment contrasts before access. The primary is SBA minus CP within
each registered scenario. Label the full robustness grid descriptive or register simultaneous
inference if making a multiple-scenario significance claim. Do not count a favorable subset of
transformations as confirmation. Negative assessment differences remain valid outcomes.

For cross-scenario comparison transfer only composition. Reprice it on target-design users under
the same mechanism, freeze that target policy, then assess. Only an exact target reference supports
nonnegative design regret; a heuristic reference yields a signed best-found gap. Assessment
differences can be negative even with an exact design reference. Report no-bundle and all-zero
cases explicitly. Instability with negligible regret means many alternatives are nearly tied;
instability with material regret means the modeled recommendation depends strongly on assumptions.

## Final decision register and stopping rule

The future protocol still needs concrete values for: primary pools and capacities; grouping and
suite-assignment rules; exact time and memory budgets; heuristic start/restart/evaluation budgets;
anchor-combination budget; strict-tie tick scale; numerical error resolution; bootstrap replicates
and seed; primary contrasts; and final core/sensitivity cell IDs. Freeze these using metadata,
synthetic checks, or allowed development evidence as appropriate, before their protected outcomes.
An incomplete field means S2.0 is not yet complete.

Stop expansion once the reference semantics pass, the exact frontier and failures are recorded,
each claimed heuristic has a locked validation result, all registered policies are frozen and
assessed, and required robustness and reproduction are complete. A failed heuristic gate narrows
scope. A negative SBA assessment result is publishable. No new model or transformation is selected
to rescue it. Advanced solvers, joint component pricing, complementarity, robust optimization,
installed-base demand, and monetary calibration remain later work.

## Primary source and attribution boundary

Sun, Li, and Teo's [Partition and Prosper: Design and Pricing of Single
Bundle](https://pubsonline.informs.org/doi/10.1287/opre.2022.0465), *Operations Research* 73(4),
1983–2001 (2025), is the governing paper reference. The publisher abstract confirms that its
polynomial-time result concerns SBR with multivariate normal values and a positive diagonal minus
fixed-rank positive-semidefinite covariance structure. Its predetermined-price hardness result
also concerns SBR. These claims do not supply an SBA complexity result or a guarantee for the
empirical local search proposed here.

This review checked that primary publication record, not a new full-text proof reproduction.
Equation-level attribution remains recorded in the existing project theory note and must be
checked against the exact paper version during S2.2. The finite-panel price arguments and proposed
execution controls above are project derivations and design choices. No normal-model guarantee
follows from low-rank recommendation factors or from applying the four nonlinear transformations.
