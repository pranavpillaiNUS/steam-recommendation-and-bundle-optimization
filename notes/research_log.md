# Research log

## 2026-05-31 - First pass on the Steam data

Goal for this session: get hands dirty with the Steam bundle dataset, figure out what is
actually in it, and see what research questions it can support. Dr Li's instruction was that
the direction becomes clear only after exploring the data.

What I did:

1. Inventoried data/raw (notebook 00). Six files. The JSON files are line-delimited Python
   dicts, not real JSON, so everything is parsed with ast.literal_eval. Confirmed there is no
   user-to-bundle purchase file. Only user-to-game ownership.

2. Cleaned the bundle pricing data (notebook 01) into bundle_df (615 bundles) and
   bundle_items_df (3525 bundle-item rows). Found that bundle_price equals the sum of item
   standalone prices exactly, so the implied discount and the reported discount agree for
   every bundle (discount_mismatch = 0). Mean bundle has 5.7 items and a 24.5% discount. 29
   bundles have no discount.

3. Flattened the ownership panel (notebook 02) into user_items_df: 70,912 users, 10,978
   games, ~5.1M ownership rows after dropping duplicate pairs. 36% of ownership rows have zero
   playtime.

4. Built bundle demand proxies (notebook 03) since there are no direct purchases. For each
   bundle: users_own_all, users_own_any, avg_playtime_overlap. Important finding: only about
   half of all bundle items appear in the panel, so only 238 of 615 bundles have full
   coverage. users_own_all is only trustworthy for those 238. Added panel_coverage as a column
   and filter on it.

5. First analytical step (notebook 05). Regressed log demand proxy on discount rate, bundle
   size, and log component price (full-coverage bundles, descriptive only). Discount rate and
   price both positive and significant; size adds little once price is controlled. Then set up
   a simple counterfactual pricing model and derived a closed-form revenue-maximising discount
   delta* = 1 - 1/b from the discount semi-elasticity b. With b around 2.1 this gives delta*
   around 0.53. Treated as a method demonstration, not a recommendation.

Key findings so far:
- No bundle purchases; ownership-based proxies are the only option.
- Pricing data is clean and internally consistent.
- Panel coverage is the main data-quality constraint on the demand proxy.
- There is a usable, if non-causal, positive relationship between discount and demand.

Decisions:
- Primary direction is A then C: describe observed bundle pricing, then build and optimise a
  counterfactual pricing model.
- Keep Steam as the main dataset for now; revisit Nielsen only if access comes through.

Next session:
- Bring genre and playtime into the demand model.
- Sketch an item-level valuation proxy from playtime.
- Write the structural revenue objective for a single bundle and connect it to the three
  optimization papers.

## 2026-06-02 - Overlap-adjusted attribution (notebook 04)

Goal: sharpen the users_own_all proxy. It overcounts because when bundles share games, a user
who owns the union is counted toward every compatible bundle. 278 of 615 bundles overlap, so
this is widespread.

What I did:

1. Formalised the fix as minimum-cost set cover. For each user, the bundles they could have
   bought are exactly those whose games are all in their library; attribute their ownership to
   the cheapest reconstruction (those bundles plus solo buys) at snapshot prices.

2. Checked feasibility first. A user can own up to 140 candidate bundles, but the candidate
   bundles split into overlap-connected components and the largest component is only size 12
   (2^12 subsets), so exact brute force per component is safe. Only 173 users have any
   overlapping candidates. Money is handled in integer cents so ties are exact.

3. Built notebook 04. Main output mincost_attributed_count per bundle, plus a soft-logit
   robustness appendix (temperatures 0.5/1/2, stress 5) that is a sensitivity check, not a
   behavioural model. All sanity checks pass, including that candidate_ownership_count
   reproduces nb03 users_own_all exactly.

Key findings:
- The overlap correction is small in this catalogue: of ~3,900 reassigned units, only ~370 are
  from overlap. The rest is dominated by one zero-discount bundle (BioShock Triple Pack, 6951
  -> 3475.5) where the cost rule cannot tell a bundle buy from three solo buys.
- So users_own_all was not badly inflated by overlap. That is itself a useful, honest result:
  the simple proxy is mostly fine, and the cases where it is not are identifiable (high
  contested_candidate_share, or zero discount).
- This is attribution under snapshot prices, not identification of purchases. Trusted on
  full-coverage bundles only.

Next session:
- Decide how nb05 should treat zero-discount bundles (flag rather than halve).
- Otherwise proceed to the valuation model; this attribution can later validate it.

## 2026-06-08 - Rewrote notebook 05 as a proper bridge

Goal: notebook 05 was written before notebook 04, so it was out of step with the chain it is
supposed to close. It regressed on nb03's raw users_own_all, ignored nb04's overlap-adjusted
mincost_attributed_count, did not flag zero-discount bundles (it even put the zero-discount
BioShock pack in its example list with the inflated 6951 count), and ended on a constant
semi-elasticity counterfactual that gives the same revenue-maximising discount for every
bundle. The task was to make it the correct stepping stone from the EDA to the optimization
stage.

What I did:

1. Switched the demand proxy to nb04's mincost_attributed_count, with users_own_all and
   users_own_any kept only as a robustness panel. Added a short diagnostic showing the two
   proxies disagree on 49 of the 238 full-coverage bundles, with the largest changes on
   zero-discount or high-contested bundles (BioShock 6951 -> 3475.5, Interstellar 79 -> 5).

2. Kept the descriptive regression but on the attributed proxy, with HC3 robust standard
   errors (the residuals are non-normal) and a zero-discount indicator so all 238 bundles
   stay in the fit while the four zero-discount bundles' attribution tie is controlled for
   rather than silently halved (nb04's guidance). Discount stays positive and significant
   (coef ~2.3) and stable across all three proxies; size is now slightly negative once price
   is controlled.

3. Dropped the closed-form delta* counterfactual and its figure. Replaced them with a section
   on why a reduced-form discount response cannot price a bundle (non-causal, proxy not sales,
   one delta* for all), and a prose bridge to the bundle-size pricing optimization (latent
   valuations -> dollar calibration -> the order-statistic moment interface -> the concave
   CMM program, with the single-size root-find as fallback), pointing at
   notes/optimization_models.md and planning.md. Rewrote the next-actions list to follow
   planning.md stages 0 to 4. Removed the now-stale 05_counterfactual_examples.csv and
   05_counterfactual_revenue.png.

Key findings:
- The proxy switch matters for a minority of bundles (49 of 238), and the descriptive
  discount-demand relationship is robust to which proxy is used, which is itself a reassuring
  consistency result.
- The reduced-form regression is a descriptive endpoint, not a pricing tool; the pricing
  claim has to come from the structural valuation model, which is the next stage.

Next session:
- Begin planning.md stage 0: stand up src/ and tests/ and implement the single-size root-find
  and the convex program against synthetic data before any Steam valuations.

## 2026-06-13 - BSP optimizer (stage 0) and the start of the preference engine (stage 1)

Goal: build the bundle-size pricing optimizer of paper 1 (the cross-moment model) and validate
it on synthetic data, then start the latent-factor preference model that will eventually feed
it. This was scoped and re-scoped against a detailed technical review before any code was
written, so the design avoids a list of subtle traps (recorded in the plan).

What I did:

1. Wrote the full proofs of Theorems 1 and 2 (plus Lemmas 1, 2, 3, 6) into
   notes/optimization_models.md, reproduced from the paper's Appendix A: the SDP reduction by
   chordal PSD completion (Lemma 3), the reduction of the demand SDP to the concave simplex
   program (Theorem 1), and the operator-concavity argument for the profit (Theorem 2).

2. Built src/bundle_pricing.py and tests/test_bundle_pricing.py. The module keeps the CMM model
   objective (uses only the moments omega, Sigma) strictly separate from the empirical objective
   (uses the full per-user valuation sample). Design points worth recording: the multi-size
   optimizer solves the concave program in the native q variables with an interior margin
   (softmax is only a diagnostic, since it is not concave in the logits); the outer profit
   gradient is numerical, not a mislabeled analytic one; the demand has two implementations
   (the simplex program and the eq. 12 SDP) that cross-check, with solver-independent feasibility
   diagnostics; covariance regularization is relative, not an absolute epsilon; and the
   eigenvalue policy raises on materially negative eigenvalues but clips float-error ones. 17
   unit tests pass.

3. Built 06_bsp_synthetic_validation.ipynb. It reproduces the paper's Figure 1 (single-size CMM
   vs MNP, max profit gap about 1.5%), confirms the convex optimum matches a brute-force grid
   oracle and the simplex demand matches the SDP, and runs two findings: the CMM optimum recovers
   97 to 99% of the model-free empirical optimum across distributions, and the BSP advantage over
   separate selling rises as valuations become negatively correlated (1.53x at rho = -0.3 down to
   1.05x at rho = +0.6), the Adams-Yellen / McAfee-McMillan-Whinston story.

4. Started stage 1. src/valuation.py builds the sparse ownership matrices (binary preference, a
   log-playtime weighted matrix, and an observed-only confidence matrix; the baseline confidence
   of one is never materialized) and fits SVD/NMF, with a bounded scoring API and an optional
   genuine Hu-Koren-Volinsky implicit-ALS wrapper. 07_game_features.ipynb builds a game-level
   table anchored on the 10,978 panel games, pre-aggregating bundles to one row per game before
   joining (the in-bundle reconciliation is exact: 1394 = 1394). 08_valuation_model.ipynb fits
   the factor model with a leakage-safe per-user leave-two-out split.

Key findings:
- The optimizer is numerically sound: every internal cross-check (convex vs grid, simplex vs SDP,
  single-size root vs direct optimization, gradient vs finite difference) agrees.
- The preference model beats the popularity baseline where it matters: SVD (k=32) gives Recall@10
  0.339 vs 0.195 and far better catalogue coverage (0.051 vs 0.007), while popularity has the
  higher AUC (0.962) - a clean illustration of why AUC is misleading when positives are rare
  among thousands of items, so top-K metrics are primary. 7.2% of users (fewer than 3 games) are
  excluded from ranking and reported, not hidden.
- Mean preference score per bundle correlates with users_own_all on the 238 full-coverage bundles
  (Spearman 0.71), but this is internal concordance, not independent validation: both come from
  the same ownership matrix.

Honest caveat carried forward: the stage 1 outputs are latent preference scores, not dollar
valuations. CMM needs w_s - p_s on a common scale, and that calibration is the next stage and the
weakest link in the chain.

Next session:
- Stage 2: calibrate preference scores to dollars via the censored/threshold price link, build
  the order-statistic moment bridge on real item sets, and regularize the partial-sum covariance,
  then run the BSP optimizer on real Steam bundles (stage 3).

## 2026-06-17 - Stage 2 calibration: the price anchor fails, and why that is fine

Goal: turn the Stage 1 preference scores into dollar valuations (Route A: the affine link
v = a + b s with the scale set by a censored-price anchor) and wire them into the moment
bridge, so the BSP optimizer can run on real item sets.

What I did:

1. Built src/calibration.py and tests/test_calibration.py (12 tests, 29 in the suite). The
   module has three price-anchor estimators (an aggregate link regression of the
   ownership rate on per-game mean score and price; the cleaner per-game quantile anchor,
   where price_g should equal the (1 - r_g) quantile of the game's user-score distribution;
   and a nonlinear rate-matching refinement), a transparent normalize_calibration fallback,
   the per-user valuation builder with the free-disposal floor, and a valuation_moments
   wrapper onto the existing bridge in bundle_pricing.py.

2. Ran the anchors on the real factors and game features. All fail the sign test: conditional
   on preference, ownership rises with price (aggregate price-coef t = +6.4; quantile anchor
   slope b = -20.5, Spearman(price, threshold quantile) = -0.24). So the censoring logic
   (ownership reveals v >= price) does not hold across games. The cause is the quality and
   exposure confound: expensive games are higher quality and more marketed, so they are owned
   more, not less. The standalone price therefore cannot identify the dollar scale here.

3. Confirmed numerically that this is not fatal: the headline economics are scale invariant.
   With zero marginal cost, scaling every valuation by a constant leaves the
   BSP-vs-separate-selling and pure-bundling profit ratios exactly unchanged
   (BSP/separate = 0.7698 at lambda = 1, 10, 100 on the demo bundle). So the multiplicative
   scale is set by a transparent normalization and swept; only the free-disposal floor
   location and the cross-good correlation structure affect the ratios.

4. Built 09_valuation_calibration.ipynb (runs clean under nbconvert): the failed anchors and
   the figure, the scale-invariance check, the normalization, face validity (per-game
   valuation is mechanically ~ownership at Spearman 0.996, the one independent signal playtime
   is weakly positive at 0.168), the moment bridge and its conditioning (Ledoit-Wolf restores
   Sigma > 0), and a location sensitivity sweep. Saved outputs/tables/valuation_calibration.csv.
   This shifts the planned data-driven notebooks down: bundle-size pricing is now 10 and the
   counterfactual is 11.

Key findings:
- The standalone price is endogenous (positively related to ownership), so it cannot anchor
  the valuation scale. This is the precise, quantified version of the discount-endogeneity
  caveat already in assumptions_and_limitations.md.
- Scale invariance rescues the core research questions: the uplift ratios and the correlation
  sweep do not need a credible dollar scale. Only the location (swept) and the correlation
  structure (from the factors) matter for them.
- For the EA Racing Pack demo bundle, BSP/separate stays below one across the whole location
  sweep: the five racing games are positively correlated in value, the textbook case where
  bundling does not help. A reassuring sign the machinery reproduces the economics.

Honest caveat carried forward: the dollar scale is an assumption, not a measurement. Absolute
prices and any comparison to the observed Steam bundle price are sensitivity analyses under the
swept scale, never point claims.

Next session:
- Stage 3 (notebook 10): run optimize_convex and optimize_single_size on real full-coverage
  item sets, cross-check against optimize_empirical_schedule, and report the size-price menu.

## 2026-07-04 - Stage 2 reviewed, Stage 3 built: BSP on real item sets (notebook 10)

Goal: check the Stage 2 calibration for correctness, then run the BSP optimizer on real
full-coverage item sets (Stage 3, notebook 10) with the simulation optimizer as the
cross-check.

What I did:

1. Reviewed notebook 09 and src/calibration.py. The anchor tests, the scale-invariance check
   (Ledoit-Wolf is scale-equivariant, so the ratios are exactly invariant), and the moment
   bridge wiring are all correct. One genuine issue found and fixed: normalize_calibration's
   docstring claimed b_scale is cosmetic for the ratios, but because the location re-anchors
   to the price median at the new spread, b_scale moves the free-disposal floor in score units
   (-a/b) and therefore moves the ratios (pure/separate goes 0.75 / 0.77 / 0.87 across b_scale
   0.5 / 1 / 2 on the demo bundle). The correct statement: ratios depend on the normalization
   only through -a/b, and the exactly invariant transformation is the joint rescaling
   (a, b) -> (lambda a, lambda b). Docstrings, notes, and caveat 30 updated; caveats 32 to 34
   added.

2. Made the empirical path in src/bundle_pricing.py fast enough for the full panel: vectorized
   the per-user choice rule (the Python loop is kept as a reference implementation and the two
   are pinned together by tests), let evaluate_schedule accept precomputed partial sums, gave
   the DE optimizer a warm start (x0), and promoted the separate-selling and pure-bundling
   benchmarks from notebook 09 inline code into src with hand-computed tests. Suite is 38
   tests, all passing. Full-panel menu evaluation is ~14 ms, so the simulation optimizer runs
   on real bundles in about a minute each.

3. Built and ran 10_bundle_size_pricing.ipynb: six full-coverage bundles (Half-Life Anthology,
   BioWare, EA Racing, Command & Conquer, F1 Franchise, Sakura; sizes 4 to 9) plus the curated
   top-8-owned set. Policies: CMM menu (model and realized kept separate), the DE empirical
   menu (searched on 20k users seeded at the CMM prices, evaluated on all 70,912), the
   single-size fallback, separate selling, pure bundling, and the observed Steam price. Saved
   outputs/tables/bundle_size_pricing.csv and three figures.

Key findings:
- The bundling economics depend on the item set exactly as the theory says: BSP/separate is
  0.73 to 0.94 for the small franchise packs (positively correlated valuations), about 1.00
  for Half-Life, 1.01 to 1.02 for the larger bundles, and 1.03 on the curated set, which is
  the one strict menu win over both pure bundling (0.97) and separate selling. EA Racing
  reproduces notebook 09's 0.7698 exactly.
- The convex-vs-simulation check failed informatively, which is the methodological finding of
  the stage: the CMM menu realizes only 69 to 97% of the empirical optimum (notebook 06's
  synthetics gave 97 to 99%), and the model value misjudges its own menu by -14% to +52%. The
  diagnosis pins it on distribution shape, not machinery: at the CMM prices the model expects
  39% of users to walk away and 53% actually do; a Gaussian population with identical
  (omega, Sigma) realizes within 5% of the model value; the real partial sums have skewness
  9 to 18. Two moments cannot represent a mass of near-zero-value users plus a thin high-value
  tail. Per the fallback ladder, the economics ride on the simulation optimizer, and the CMM
  degradation is reported as a finding, not hidden.
- The observed-price benchmark moves one-for-one with the assumed scale (obs/separate 0.59 to
  0.36 across b_scale 0.5 to 2), which is exactly why it stays a sensitivity.

Next session:
- Stage 4 (notebook 11): the cross-bundle counterfactual and the correlation sweep through the
  factor covariance, using the empirical optimizer as the BSP number and reusing notebook 10's
  benchmark machinery.

## 2026-07-11 - Met Dr. Li at NUS, went through progress for notebooks 03 to 09

Goal: present the temperature-weighted soft-attribution sensitivity, the minimum-cost demand proxy,
and the preliminary ranking work. We discussed Recall@K and AUC for ranking held-out games among
the training catalogue. Dr Li highlighted that the cross-moment model applies to a specific menu in
which customers can choose any bundle of a posted size. The Steam snapshot instead contains fixed
compositions. The meeting therefore established a mechanism mismatch: I had focused on CMM's
tractability and empirical behavior without first validating that its selling mechanism matched the
institutional setting.

What was recommended as a next step:
- Dr Li asked me to overhaul the project spine around collaborative filtering and low-rank matrix
  factorization, using Netflix as a reference point. Unlike that example, the Steam archive also
  contains game metadata such as genre, which supports a controlled content-feature ablation.

## 2026-07-13 - Initial pivot proposal: CMM out, latent preference feeding single-bundle design

Goal: act on the 2026-07-11 steer. Retire the cross-moment bundle-size pricing model (CMM) as the
spine, re-center the empirical half on collaborative filtering and low-rank matrix factorization
(the Netflix route, plus the genre data Steam has and Netflix did not), and switch the optimization
endpoint to a mechanism that fits Steam. This session was scaffolding only, the plan and the theory
and the notebook 05 bridge, before any new modeling.

Why the pivot, precisely:
- CMM prices a size-based menu where the customer freely chooses any bundle of size $s$. The
  snapshot instead records fixed curated compositions, so CMM modeled a different mechanism.
  Notebook 10 had already shown the symptom: CMM achieved 69%--97% of an uncertified empirical
  search on one transformed score panel. That comparator was neither model-free nor a valuation
  benchmark, and the binding problem remained the mechanism mismatch.
- The initial replacement proposal drew on *Partition and Prosper* but conflated SBR and SBA. The
  2026-07-14 and 2026-07-17 corrections below separate them: CP-anchored SBA is the project's
  empirical model and does not nest pure bundling; SBR is a different benchmark, and its hardness,
  tractability, comparative statics, and guarantees do not transfer to SBA.

The corrected architecture is two layers plus a bridge: Stage 1 compares collaborative-filtering
and matrix-factorization models, with genre entering only as a controlled ablation; the bridge is a
descriptive dependence analysis; and Stage 2 designs one fixed bundle from frozen pseudo-utility
scenarios derived from latent preference scores.

What I did:
1. Rewrote planning.md around the two-layer spine (the keep/rewrite/archive table, the bridge
   interface, stages A to E, the CF/MF reading list, the fallback ladder, verification, and
   CPBSD/SBA as optional extensions).
2. Refocused notes/optimization_models.md. The CMM proofs are preserved in full under an "Archived"
   banner (they were a real learning exercise and are the record of the pivot). The live content is
   now a Layer 1 "matrix factorization as optimization" section and the full Partition and Prosper
   theory reference (the SBR formulation and its CP/PB nesting, Theorem 1 the NP-hard fixed-price
   subproblem, Proposition 1 the half-probability structure, Theorem 2 the polynomial-time
   condition, Proposition 2 the comparative statics, the BO-plus-conic solver, and the 1/e and 1/2
   guarantees), grounded in the paper, with the full proofs left as a learning checklist.
3. Re-pointed notebook 05's Section 5 bridge from CMM to the two-layer spine (markdown only, no
   re-run needed). Reframed the CMM caveats in notes/assumptions_and_limitations.md as the pivot
   rationale and added caveats 35 to 39 for the pivoted spine.

Key decisions and findings:
- The overhaul is contained. The data stage (00 to 04) and the descriptive regression (05) stand.
  Layer 1 is an extension of existing work (notebooks 07 and 08, src/valuation.py), not a rebuild.
  Only the CMM-specific notebooks (06 synthetic validation, 10 real-data run) are archived, kept as
  the documented pivot record.
- I initially recorded a "bonus" that the calibration problem stops being load-bearing because the
  bundling ratios are scale-invariant. The 2026-07-14 review corrected this (see the next entry):
  scale invariance is real but insufficient, because the data and ranking evaluation validate only
  the ordinal ordering the scores induce, not a cardinal utility; the cardinalization T (loss- and
  specification-dependent) mapping scores to pseudo-utilities is itself a first-class sensitivity,
  since a monotone map preserves rankings but not sums. A later correction removed the claim that
  correlation is cardinalization-free: common positive scaling preserves Pearson correlation, but a
  nonlinear monotone transformation can change dependence measures.
- One structural flag, confirmed against the paper: a low-rank factor score covariance does not by
  itself verify Theorem 2's positive-diagonal-minus-fixed-rank-PSD condition. Adding a positive
  diagonal residual would still require an explicit decomposition check rather than an assumed sign
  pattern. Therefore the theorem is not activated by recommender low rank. The planned finite-panel
  optimizer must earn exact certificates through completed exhaustive enumeration on declared small
  instances; no such SBA certificate existed at this date.
- Rigor note: the log already dated the meeting correctly as 2026-07-11, so there was no date error
  to fix (the "June" was only a verbal slip in discussion).

Next session:
- This historical proposal was superseded before fitting: no model ordering is predetermined, the
  existing notebook 08 split is not the frozen Stage 1 protocol, and tags remain optional until the
  required four-rung ladder is complete.
- Housekeeping: update the README notebook map, and add the CF/MF literature (Koren-Bell-Volinsky,
  Hu-Koren-Volinsky, Rendle BPR, Kula LightFM, Bakos-Brynjolfsson) to the Drive folder.

## 2026-07-14 - Second review: identification corrections and the bundle-mechanism audit

Goal: a detailed methodological review (the strongest yet) flagged that the pivot scaffolding, while
right to drop CMM, still overclaimed in the same family the project tries to avoid. Split the
response into two tracks: mechanism-independent corrections applied now, and a working mechanism
specification settled by the data audit and later internal freeze.

Corrections applied (planning.md, optimization_models.md, the caveats, this log, README, notebook 05):
1. MF outputs are latent preference scores, not identified valuations. Renamed the "valuation
   engine" to the latent preference model and "V" to the preference-score matrix S. The optimizer
   runs on explicitly normalized pseudo-utilities v~ = T(s) and is reported as a within-model
   counterfactual, never as Steam willingness-to-pay or revenue.
2. Scale invariance was overstated. It is real but insufficient: the data and ranking evaluation
   validate only the ordinal ordering, not a cardinal utility, so the cardinalization T (loss- and
   specification-dependent) is itself a sensitivity, since a monotone map preserves rankings but not
   sums. Removed "calibration is no longer load-bearing" everywhere and added the
   normalization-stability experiment.
3. "Model-free ground truth" was a misnomer (the panel is a model output, greedy search is not a
   global optimum). The replacement empirical sample-average optimizer must later be benchmarked
   against completed exact enumeration on small pools; "best solution found" is reserved for
   uncertified results.
4. Layer 1 no longer predetermines "popularity < CF < hybrid": the metadata lift is a hypothesis to
   be tested by an identity-only versus +genre ablation with everything else fixed, using exact
   full-catalogue ranking (sampled metrics can reverse orderings). Playtime is confidence, not an
   item feature. Added planned candidate-pool feasibility, go/no-go gates, and a
   decision-quality-versus-prediction check.

The bundle-mechanism audit (outputs/tables/bundle_mechanism_audit.csv) classified all 615 bundles by
cross-referencing each component against 32,135 catalogue records (32,132 distinct nonblank
application IDs) and the per-item prices in bundle_data.json:
- Every bundle (615/615) shows a standalone price for every component.
- 568/615 (92%) have SBA-like static component-availability evidence; 475 have complete catalogue
  confirmation and 513 are high-confidence. 0/615 show affirmative evidence of SBR-style
  exclusivity. A static snapshot cannot prove SBR absent, and the 47/615 unclear rows are
  coverage-limited rather than evidence of exclusivity.
- Ownership-adjusted ("complete the set") and indivisible-key packages are not observable in a static
  snapshot. 77% of SBA-like bundles are single-publisher; 562/615 have <= 12 components.
- This documents that notebook 04's cheapest-reconstruction proxy treats displayed component prices
  as available reconstruction options. It does not identify the full historical selling mechanism.

Drafted notes/mechanism_identification_memo.md for technical discussion. On 2026-07-17 it was
superseded by the active nine-decision internal specification; supervisor feedback became
nonblocking.

Conclusion: CP-anchored SBA is the project's declared primary empirical model; SBR stays as the
theoretical benchmark; ownership-adjusted pricing cannot be identified from this data.

Next session:
- This historical next-step note is superseded by the 2026-07-17 internal freeze below. The next
  binding item is S1.0, before any new Stage 1 fitting.

## 2026-07-17 - Internal specification freeze and continuation

Goal: convert Gate 0 to an internal specification freeze, validate the adaptation of *Partition and
Prosper*, and continue through the binding plan without changing the scientific boundaries.

Decision:
- The project proceeds under the current frozen working specification. Gate 0 is an internal
  specification-and-cleanup gate. The historical 2026-07-11 meeting record remains unchanged.
- The four-decision mechanism memo was superseded by a nine-decision internal specification that
  freezes the Stage 1 ladder and split, latent-score interpretation, SBA-primary/SBR-benchmark
  distinction, CP anchoring, economic conventions, finish line, stretch rule, and cut order.

Paper check:
- Section 3, equation (3), defines SBR with products inside the selected bundle unavailable
  separately.
- Section 5, equation (17), defines SBA by retaining the CP menu and adding one selected bundle.
- Equations (13)--(14) give the truncated choice condition used by the finite-panel SBA model.
- The governing theory note does not transfer SBR hardness, half-purchase, tractability,
  comparative-statics, nesting, or approximation guarantees to SBA. The primary empirical model
  remains CP-anchored SBA; SBR remains a benchmark.

Work queued at freeze time:
1. finish the reproducible mechanism-audit generator and prerequisite notebook/archive corrections;
2. verify Gate 0 internally; and
3. stop before freezing the S1.0 split, evaluation, hyperparameter, and admission protocol.

Claim boundary: this governance change authorizes continuation; it does not turn latent scores into
valuations, identify actual demand or revenue, establish historical Steam mechanisms, or validate
any bundle policy against observed purchases.

Prerequisite execution:
- Added a deterministic generator and independent reconciliation tests for the 615-row mechanism
  audit. The existing reviewed CSV was preserved byte-for-byte; the manifest distinguishes its
  row-order hash from the canonical generated-order hash and records the identification boundary.
- Added archive errata to notebooks 06 and 10 and an archive README. Their code and rendered
  outputs remain historical evidence and are not inputs to the live pipeline.
- Corrected notebook 05's zero-discount interpretation to coefficient $1.483$, HC3 standard error
  $1.904$, and $p=0.436$; removed the unsupported $R^2$ decomposition; added the SBA bridge; and
  saved the 15-row HC3 table for three specifications.
- Updated the README, theory note, assumptions, data dictionary, module descriptions, and tests so
  the live/archived boundary is explicit. No file was staged or committed.

Verification: all three touched notebooks pass `nbformat.validate`; the mechanism audit reconciles
all 615 rows and every cell; notebook 05's saved coefficients match an independent HC3 refit; and
the repository test suite passes 45/45 tests. The one SciPy optimizer warning arises in an archived
CMM stability test and is not a failure.

Stopping point: Gate 0 and prerequisite actions 1--4 in the binding sequence are complete. The next
untouched item is S1.0. No frozen split, Stage 1 configuration, new Stage 1 fit, or Stage 1 result was
created in this pass.

## Where we are at

I started by inspecting the Steam data rather than assuming a research question. I cleaned the
bundle data into a table with bundle id, size, component prices, final prices, and discounts,
and confirmed there are no direct user-bundle purchases, so I built demand proxies from game
ownership and playtime. The main limitation is that the ownership panel covers only about half
the bundle items, so the clean demand signal exists for 238 bundles. On those, higher
discounts and higher component prices are associated with more owners. I then sharpened the
ownership proxy in notebook 04 by attributing each user's library to the cheapest
bundle-plus-solo reconstruction, which removes the double-counting when bundles overlap; the
honest result is that overlap barely changes the proxy here, so the simpler users_own_all is
mostly fine. In notebook 05 I then ran a descriptive regression of that attributed proxy on
discount, size, and price, which closes the descriptive stage: the discount-demand
association is positive, significant, and robust to the proxy choice, but it is non-causal and
cannot price a bundle on its own. So notebook 05 ends with the bridge to the next stage, replacing
the reduced-form proxy with latent preference estimation.

The first optimization half used the cross-moment bundle-size pricing model in notebooks 06 and 10,
alongside a latent-factor prototype in notebook 08 and a failed monetary anchor in notebook 09.
Those artifacts predate the frozen evaluation protocol. They are retained as evidence of the pivot,
not as Stage 1 results or live optimization inputs.

As of the 2026-07-17 internal freeze, the live spine has two stages. Stage 1 must complete the full
popularity, implicit-ALS, identity-only pairwise-MF, and controlled identity-plus-genre ladder under
the leakage-safe, tie-aware, multi-seed protocol; metadata lift is an empirical question. Stage 2
will transform the resulting latent preference scores only through frozen pseudo-utility scenarios
and optimize one fixed bundle under CP-anchored SBA. SBR remains a different, theory-only benchmark.
The 568/615 SBA-like audit count describes component availability in one static snapshot; it neither
identifies CP anchoring nor establishes Steam's historical mechanism. The immediate next step is
S1.0, and it has deliberately not begun.

## 2026-07-18 - S1.0 Stage 1 protocol freeze

Planning step: S1.0, before any dependency experiment or new model fit.

Frozen artifacts:
- `configs/preference_models.json` fixes the four-rung ladder, three seeds, finite ALS and pairwise
  grids, fixed epochs/iterations, confidence and negative-sampling rules, genre-only controlled
  ablation, fold-in rules, numerical failure policy, and memory/wall-time budgets.
- `configs/ranking_evaluation.json` fixes the 80/20 activity-stratified outer split, numeric ID and
  duplicate contract, warm-support rules, capacity-aware validation/test holdouts, 5,000-user
  full-catalogue evaluation sample, 300-item pseudo-cold cohort, exact tied-block metrics,
  validation tie-break, paired bootstrap, and validation-only Stage 2 admission rule.
- `outputs/modeling/stage1_protocol_manifest.json` records semantic configuration hashes, upstream
  hashes, the pre-Stage-1 repository baseline, environment, and all 24 ALS plus four pairwise
  configurations. Its protocol ID is
  `00b18d784ee34196e34a90c354fb03f45fa025082039eb7a90cc662b23a22f6f`.

Identification and access boundary: the target remains held-out ownership reconstruction. Scores
are latent preference scores only. Design-test outcomes remain sealed until validation admission is
hashed; assessment IDs are unavailable to tuning and their histories remain sealed until S1.10;
Stage 2 objectives and bundle outcomes are unavailable throughout Stage 1 selection.

Verification: configuration validation, semantic hashing, namespace separation, exact
stratum counts, and input-permutation invariance pass on synthetic tests. No Stage 1 model has been
fitted and no validation, design-test, assessment, or bundle outcome has been inspected. The next
permitted work is S1.1's canonical sparse interaction and ID contract, followed by S1.2's actual
split manifests.

Pass-condition reconciliation: this entry records the S1.0 rule and configuration freeze. The
literal S1.0 pass condition in `planning.md` also names concrete split manifests. Those materialized
outer-user, nested-edge, evaluation-user, and pseudo-cold artifacts remain S1.2 evidence and are not
claimed here. S1.2 must instantiate them from the S1.1 canonical data contract under the frozen
rules, without reopening S1.0.

## 2026-07-29 - S1.1 canonical sparse interaction contract

Planning step: S1.1, before any outer-user or nested-edge split.

Artifacts:
- `src/interactions.py` enforces exact nonnegative decimal IDs, deterministic duplicate collapse by
  fieldwise maximum playtime, canonical sparse storage, exact ordered-map alignment, held-out
  removal from ownership and both playtime inputs, and physical plus semantic artifact hashes.
- `src/stage1_interaction_artifacts.py` generates and verifies the canonical sparse artifacts without
  constructing a dense user-by-item object or reading any protected outcome.
- `outputs/modeling/stage1_interaction_manifest.json` records the complete load audit, map and matrix
  semantics, upstream and artifact hashes, item reconciliation, activity and support summaries,
  and storage diagnostics. Its interaction-set ID is
  `58d6c59169b2bcf6b23e5ba67fc9e003da69cca2508857ac46c72f0c729ac721`.

Input audit:
- The frozen interaction source contains 5,094,082 rows. Exactly 2,221,650 rows satisfy the frozen
  unpadded-decimal user and item ID rule; they form 2,221,650 unique ownership edges, so no duplicate
  excess rows remain after the upstream notebook's deduplication.
- The remaining 2,872,432 rows have noncanonical user identifiers, including username-form
  identifiers. They are excluded and counted under the S1.0 `exclude_and_count` rule. There are no
  missing IDs, invalid item IDs, or negative or nonfinite playtime values among the loaded rows.
  This is a material loss of source coverage caused by the already-frozen numeric user-ID contract,
  not an outcome-driven filter or a post hoc rule change.
- The retained matrix contains 43,802 ascending numeric users and the full 10,978-item metadata
  universe. Of those items, 9,733 have eligible interaction support and 1,245 are retained as
  explicit zero-support metadata-only columns. No eligible interaction item is absent from the
  metadata universe.

Diagnostics: each ownership and playtime CSR matrix has shape $43{,}802 \times 10{,}978$, 2,221,650
stored entries, and density 0.00462018. The three sparse matrices and two ID maps occupy 54,283,476
bytes under the declared array-storage measure, compared with 5,770,738,512 bytes for three dense
float32 matrices plus the same maps.

Verification: repeated full generation reproduced the same audit and interaction-set ID. The
read-only `--check-only` path verifies current upstream hashes, artifact hashes, semantic CSR
hashes, ordered-map hashes, saved-matrix counts and diagnostics, manifest identity, and the audit
reconciliation equations; it does not rescan the raw interaction CSV. The full repository suite
passes 115 tests. This includes interaction row-permutation invariance, duplicate-collapse,
held-out preference and confidence removal, sparse-memory, serialization, zero-support catalogue,
and deliberately permuted item-row alignment failures. The one warning is the previously known
SciPy optimizer warning in an archived bundle-pricing stability test.

Access boundary: no validation, design-test, assessment, Stage 2, candidate-pool, or bundle outcome
was accessed. No model was fitted. The next permitted step is S1.2, which materializes the frozen
outer-user and nested-edge splits from this interaction contract.

## 2026-07-29 - S1.2 frozen outer and nested splits

Planning step: S1.2, after the S1.1 canonical interaction contract and before any S1.3 feature work
or Stage 1 model fit.

Two deterministic implementation clarifications were recorded before the final protected
publication. A game's primary genre is the lexicographically first unique label after HTML
unescaping, Unicode NFKC normalization, whitespace collapse, and blank removal. When proportional
genre quotas for a pseudo-cold support band are fractional, Hamilton largest-remainder
reconciliation is used with canonical lexical ties. These clarifications fill deterministic details
left implicit by S1.0; they do not alter the frozen support bands, candidate requirements, cohort
sizes, or model-selection rules, and they were made without inspecting any protected outcome.

Artifacts:
- `src/splits.py` implements the exact activity-band outer split, transactional capacity-aware
  validation/test assignment, proportional evaluation-user sampling, deterministic primary-genre
  construction, and genre-stratified pseudo-cold sampling. Nullable access labels fail closed,
  caller row indices do not affect assignments, and a valid zero-evaluable cohort is supported.
- `src/stage1_split_artifacts.py` constructs the complete split state from the S1.1 sparse contract,
  removes both held-out edges from ownership and both playtime matrices, writes physically separated
  permitted, sealed, mask-only, and reserved artifacts, and exposes hash-bound scoped loaders.
  Publication is cycle-scoped and staged with the public manifest moved last, and an existing
  nonidentical or partial frozen cohort is never overwritten.
- `outputs/modeling/stage1_split_manifest.json` is the tracked redacted manifest. It records the
  frozen rules, numeric code maps and sentinels, aggregate reconciliation, access classes, producer
  hashes, physical hashes, semantic array hashes, and split-set ID
  `0656e84b1fd9a59d51f5cf52d8001bb82b9fa50a1870115236d0b7ba50c0ec71`. Its complete public
  manifest ID is `52d3d91a36c2979403b9d4b9753fd819428cf614db50dec6f44c4d126cba95b3`.
  It contains no protected user or item identifiers.

Cohort and edge reconciliation:
- The 43,802 canonical users and 2,221,650 canonical ownership edges split into 6,722 excluded
  low-activity users with 14,131 edges and 37,080 eligible users with 2,207,519 edges.
- The eligible cohort contains 29,666 design users with 1,764,950 edges and 7,414 assessment users
  with 442,569 edges. User and edge totals reconcile exactly, and assessment users never enter the
  nested split or shared-parameter training contract.
- All 29,666 design users are evaluable under the deterministic feasible scan. The nested roles are
  1,699,520 training edges, 29,666 validation edges, 29,666 sealed design-test edges, and 6,098
  nonwarm excluded edges. The warm catalogue contains 6,721 items. Its minimum pre-holdout design
  support is 5, and its realized minimum post-holdout training support is 4, above the frozen
  threshold of 3. This is a deterministic greedy feasible result, not a maximum-cardinality claim.
- The evaluation sample contains exactly 5,000 design evaluable users, with activity-band counts
  584, 1,202, 1,242, 1,145, 634, and 193. The reserved pseudo-cold cohort contains exactly 100 items
  in each of the three frozen support bands, for 300 unique genre-covered items.

Leakage and access boundary: the tuning-facing validation target contains coordinates only. Its
playtime values are isolated in an S1.7 diagnostic-only artifact. Validation ranking can mask each
user's other held-out positive only through an opaque mask operation that does not return the
sealed test coordinates. Design-test targets remain sealed until the validation admission manifest
is hashed; assessment IDs remain sealed until S1.10; outer and nested audit files are not loaded by
tuning APIs. Stage 2 objectives and bundle outcomes remain unavailable. No validation metric,
design-test result, assessment outcome, model score, or bundle outcome was generated or inspected.

Verification: 26 focused split tests cover permutation invariance, exact quotas, nullable access
failures, zero-evaluable cohorts, transactional holdouts, preference and confidence leakage,
synthetic end-to-end state construction, public-manifest redaction, scoped-loader behavior,
validation masking, corruption rejection, and no-overwrite behavior. A full independent
`--check-only` reconstruction reproduced every assignment, aggregate, semantic hash, physical hash,
split-set ID, and manifest ID without rewriting the publication. The full repository suite passes
133 tests. Its one warning is the previously known SciPy optimizer warning in an archived
bundle-pricing stability test. S1.2 is complete. Work stops here; S1.3 has not begun.

## 2026-07-30 - S1.3 frozen identity and genre features

Planning step: S1.3, after the frozen S1.2 split and before any estimator implementation or model
fit.

Artifacts and contract:
- `src/features.py` builds an exact float32 CSR identity block and a separate genre block aligned to
  the S1.1 signed-int64 item map. Each observed genre receives weight $1/|G_i|$; an item without a
  genre has a zero-content row. Identity remains at unit weight, the two blocks are never jointly
  normalized, and a model-facing row projection retains the complete frozen feature-column map.
- The controlled model views are identity alone and identity plus genre. The metadata flag appends
  the genre block without changing the identity rows or coefficients. Price, bundle membership,
  ownership popularity, playtime, tags, publisher, and developer are absent from predictive item
  features; publisher and developer remain candidate-pool feasibility fields only.
- `src/stage1_feature_artifacts.py` verifies the S1.0 protocol, S1.1 interaction set, S1.2 split set,
  catalogue hash, full item map, and permitted design-training item map. It publishes through a
  staged, manifest-last, no-overwrite path and supports exact read-only regeneration.
- `outputs/modeling/item_feature_manifest.json` is the tracked redacted manifest. Its feature-set ID
  is `27219b9e0f61417c7dbf4530f36a34a70603ebb7db1986f44648030f75b89056`, and its complete
  manifest ID is `d8947689491853aa685556077c2b633ffa377e235bbf640113ffe5f7509f153e`.
  Binary matrices, item IDs, feature names, and their serialization manifest remain under the
  ignored cycle-scoped protected directory.
- `outputs/tables/stage1_feature_coverage.csv` records aggregate full-catalogue and warm-training
  coverage for every genre without publishing item identifiers.

Reconciliation: the full feature map contains 10,978 items and matches both S1.1 item-map hashes.
The identity block is $10{,}978 \times 10{,}978$ with 10,978 unit entries. The genre block is
$10{,}978 \times 21$ with 21,559 entries: 8,658 items are genre-covered and 2,320 use the declared
zero-content row. The S1.2 warm projection contains 6,721 items, matches its frozen array hash, and
has 5,567 genre-covered and 1,154 zero-content rows. Every one of the 21 genres occurs on warm
training items; the least-supported occurs on five. It follows without opening the reserved cohort
that no genre can occur only in pseudo-cold items.

Verification: focused tests cover canonical IDs, pure-permutation rejection, conservative token
normalization, exact equal weights, zero-content rows, identity/genre toggle equivalence, warm-row
projection, sparse format and dtype enforcement, semantic and physical tamper detection, duplicate
catalogue rows, LF-stable coverage output, partial-publication cleanup, and refusal to overwrite an
existing publication. Independent `--check-only` reconstruction reproduced the feature and
manifest IDs and verified every public and protected hash without rewriting any file. The full
repository suite passes 144 tests; its one warning is the previously documented SciPy optimizer
warning in an archived bundle-pricing stability test.

Access boundary: only pre-model catalogue metadata, the S1.1 canonical item map, and the permitted
S1.2 design-training item map were consumed. Validation targets, sealed design-test targets,
assessment IDs and histories, the reserved pseudo-cold cohort, model scores, Stage 2 objectives,
candidate-pool results, and bundle outcomes were not accessed. No model was fitted. S1.3 is
complete; S1.4 is next and S1.5 dependency/backend work has not begun.

## 2026-07-30 - S1.4 backend-neutral estimator specification

Planning step: S1.4, after the frozen S1.3 feature hierarchy and before the S1.5 dependency,
backend-equivalence, and fold-in feasibility spike.

Prospective implementation clarifications were recorded before any model fit or outcome access:
- The $t_{ui}$ in the frozen ALS confidence equation is `playtime_forever`. Ownership remains the
  binary preference target, playtime remains a confidence modifier only, and an owned but unplayed
  edge receives confidence $1+\alpha_o>1$.
- The pairwise objective is a sum over sampled triple occurrences. The one regularization value
  applies to all active user, identity, bias, and genre-factor parameters, so its gradient is
  $2\lambda\theta$. Inactive genre parameters are absent from the identity model and excluded from
  its penalty.
- Identity uses fixed $\rho=0$ and identity plus genre uses fixed $\rho=1$. The genre matrix contains
  only S1.3's genre block; identity is already represented by $\eta_i$ and is not duplicated inside
  the metadata matrix.
- Pairwise positives are sampled uniformly from canonical training edges with replacement.
  Negatives are sampled uniformly from the warm catalogue, rejecting training positives only. A
  SHA-256-namespaced NumPy PCG64 stream uses a fixed scalar draw order, continues across epochs, and
  is shared exactly between the controlled identity and genre runs. Users with no possible negative
  fail before sampling.

Artifacts and implementation:
- `notes/preference_model_specification.md` is the written mathematical contract. It distinguishes
  fixed-block convex ALS solves from the jointly nonconvex factorization, forbids global-optimum
  claims, defines the pairwise score and summed regularized loss, and declares parameter,
  diagnostic, scorer, and later serialization schemas.
- `src/preference_model.py` implements exact integer popularity counts; lifetime-playtime observed
  confidence; the sparse WRMF objective; exact user and item normal equations; float64 Cholesky
  solves with the frozen jitter sequence and residual diagnostics; fixed-iteration reference ALS
  with float32 stored factors and post-block objectives; byte-bounded factor scoring; feature-sum
  item vectors and scores; stable BPR loss and accumulated analytic gradients; the continuing
  deterministic triple sampler and hash; and non-pickle float32 parameter round trips.
- `src/stage1_estimator_spec.py` binds the written equations and implementation to the S1.0
  protocol, S1.1 interaction set, S1.2 split set, S1.3 feature set, frozen model configuration,
  numerical policy, source hashes, parameter names, diagnostic fields, and access boundary.
- `outputs/modeling/stage1_estimator_spec_manifest.json` is the tracked public freeze. Its
  specification ID is `e39a3993985a5bcf723c0ea8aa0f87e9986845b24e8492635cc570d09b69571c`,
  and its complete manifest ID is
  `df196b35a3c50e29c92a3d23340f113f163b0cdcb966f34086d173c57ee4354f`.

Independent verification:
- A tiny dense all-pair oracle matches the sparse WRMF objective, and dense confidence-matrix
  systems match every sparse user and item normal equation.
- Exact block updates reproduce direct linear solves, record residuals and jitter, and decrease the
  fixed-block objective on the oracle. Fixed-iteration diagnostics record initial, post-user, and
  post-item objectives, runtimes, and final float32 score semantics.
- Central finite differences match the BPR gradients for user factors, identity factors, genre
  factors, and item biases. Repeated users and items accumulate correctly; extreme margins remain
  finite; dense and CSR feature paths agree.
- $\rho=0$ and zero-feature paths reproduce the identity-only scores, losses, shared gradients, and
  active regularization exactly. The centered score-matrix rank obeys the declared $k$ bound.
- Popularity uses exact `int64` training counts. Pair and block scores enforce explicit bounds.
  Triple sampling reproduces an independent namespaced PCG64 oracle, rejects every training
  positive, fails on a full-catalogue user, continues across calls, and has a canonical ordered hash.
- Temporary synthetic float32 parameters save and reload without pickle and reproduce scores
  exactly. The public manifest regenerates exactly and refuses a nonidentical overwrite. The full
  repository suite passes 187 tests; its one warning remains the documented SciPy optimizer warning
  in an archived bundle-pricing stability test.

Backend and access boundary: the current environment still has neither `implicit` nor `lightfm`
installed. Neither preferred backend is claimed oriented correctly, deterministic, serializable,
fold-in safe, or equivalent to these equations. Those are S1.5 questions; no fallback was activated
and requirements were unchanged. S1.4 used synthetic numerical oracles only. No real model was
fitted, no validation target or metric was accessed, and sealed design-test, assessment, reserved
pseudo-cold, Stage 2, candidate-pool, and bundle outcomes remained unopened. S1.4 is complete. Work
stops here before S1.5.

## 2026-08-14 - Prospective identity correction and complete Stage 1 closeout

Objective: audit the whole repository before real fitting, correct any binding data-contract issue,
and finish S1.5--S1.12 without violating the validation, design-test, assessment, pseudo-cold, or
Stage 2 access boundaries.

Identity audit and prospective cycle:
- The original notebook and `s1-v1-20260718` interaction path treated `user_id` as the account ID.
  A raw audit found 31,625 records with nonnumeric display aliases, plus leading-zero and int64-
  overflow values. The same records all contain a valid numeric `steam_id`. Even numeric-looking
  `user_id` is not reliably the account ID.
- The mapping from `user_id` to `steam_id` is functional. Its sole reverse collision is between two
  inactive zero-item records, so it does not change any interaction. Deduplicating by
  `(steam_id,item_id)` retains exactly 5,094,082 edges and 70,912 active users, versus only 2,221,650
  edges and 43,802 users in the superseded v1 contract.
- A second audit found 58,647 duplicate identity-item groups. Only ten retained edges differ between
  keep-first and the declared fieldwise-maximum playtime rule, but v2 rebuilds the table from raw data
  so the contract is literal as well as numerically close.
- This correction was made before any real fit or protected metric was opened. The prospective live
  cycle is `s1-v2-20260814`, source-set ID
  `9058bed498e13232c938820afdaf0f004a6bba895ebe1ebb541fac4ed8f397b9`.

Regenerated foundations:
- Protocol ID: `179e4861df905aaae8344104cb4fd598924f073ada3026b9ab22a8c739e7aafc`.
- Interaction-set ID: `8e86e07e04c003d2fabff87432cc84ca26ee2da08e08c3e83888b094fca8e82a`;
  aligned ownership and playtime CSR matrices are 70,912 by 10,978 with 5,094,082 stored edges.
- Split-set ID: `6e326b169b3f7499cca66a52c168d2bb5ab978331c4d925fdd22e87fc4aa047f`.
  The eligible outer sample has 50,351 design and 12,585 assessment users; 7,976 low-activity users
  are excluded. The nested design split has 3,951,166 training edges, 50,351 validation positives,
  50,351 sealed design-test positives, 8,902 warm items, 5,000 fixed evaluation users, and 300
  reserved pseudo-cold items.
- Feature-set ID: `b05bd1856a65e8e4cb10805adf2e4aa01db7fe8d6a376b36c677af6c236b4244`.
  The full 10,978-item identity block and 21-column genre block align exactly; 8,658 catalogue items
  and 7,226 warm items have genre content.
- Estimator specification ID:
  `fa3795fc5e7c549375ddd9d9258004b11af24c4e30651c87711e88a36ad627a3`.

S1.5 backend and fold-in spike:
- Installed and pinned `implicit==0.7.2`. Its ALS orientation, objective behavior, deterministic
  operation, serialization, clean reload, bounded scoring, and closed-form fold-in passed the
  predeclared complete-case checks.
- LightFM 1.17 failed to build in this Windows Python environment with a package setup error. Before
  validation, a hashed prospective amendment activated the independently checked NumPy feature-sum
  BPR implementation. The implementation matches independent finite-difference and explicit
  sampled-gradient oracles and has deterministic triple streams and serialization.
- The 12-check spike passed under manifest ID
  `5392d8393b9e15ef3d05dccfd3c4d220668fedf6893c11fdf9d6b4d1e1095238`.

S1.6 validation and admission:
- Ran all 88 frozen rows: popularity, 24 ALS configurations by three seeds, four identity-BPR
  configurations by three seeds, and the selected identity-plus-genre BPR configuration by three
  seeds. No failed or unlogged fit was removed.
- Validation selected `als__k064__reg0p05__ao20__ownership_only`. Across seeds its validation
  NDCG@20 is about 0.203 and Recall@20 about 0.429. The selected BPR configurations were retained in
  the ledger but did not qualify against popularity.
- Admission ID `f535e24b11e3c0f707e9ee6e28d0d2b37c8cba7e0183eac9eaa2eacb2a1e16e0`
  was hashed before the sealed design-test coordinates were opened. Only implicit ALS was admitted.

S1.7--S1.9 one-time design test and Gate 1:
- Popularity: NDCG@20 0.135350, Recall@20 0.2524.
- Admitted ALS, three-seed mean: NDCG@20 0.206089, Recall@20 0.4196. The paired NDCG@20 improvement
  is 0.070739 with frozen 95% bootstrap interval [0.062947, 0.078603]. The seed-specific
  improvements range from 0.069517 to 0.072019.
- Identity BPR averages NDCG@20 0.133170 and identity-plus-genre BPR averages 0.077396. Neither is
  admitted. The result says that this genre encoding did not improve this predictive task; it is
  not a causal statement about genre.
- The pseudo-cold diagnostic covers 30,077 positive edges across 300 temporarily withheld items.
  Identity-only is correctly unavailable. The genre content-only model is computable but
  underperforms popularity, so the evidence is explicitly narrow rather than generalized as a
  cold-start claim.
- Selection was not changed after test access. Gate 1 passes under manifest ID
  `b95e89ac2bce59cd9630f59104f6cbd2bea6244d77093bf583c381cf37d7856e`.

S1.10 production and fold-in:
- Refit the three admitted ALS seeds on 4,051,868 restored warm design edges.
- Folded all 12,585 assessment users into each seed using their permitted histories. Every run has
  12,585 `complete`, zero `insufficient_history`, and zero `solver_stopped` statuses. Shared item
  parameters are byte-semantic unchanged by fold-in, and clean reload reproduces bounded scores.
- Production manifest ID:
  `c8b76f330e382dc74a3b67361bc763cc4030f329d399e95e7926dd60b23e5ba1`.

S1.11 pseudo-utilities and Gate 2:
- Froze four deterministic nonnegative transformations separately for every production seed over
  the ordered 8,902-item production catalogue: global shift/q90 scale, global robust softplus,
  within-user midrank percentile, and positive-part user standardization.
- Parameters use the frozen 5,000 design users and full warm catalogue. Assessment score blocks were
  used only for bounded diagnostics. Dense user-catalogue matrices are not persisted; downstream
  pool values are generated on demand from hash-bound parameters.
- All diagnostics are finite and nonnegative. Gate 2 passes under manifest ID
  `3c0714a65cc9807d9a6a3be5688b1f932b0dc83af3e83609536e99a95353f87b`.
  These are pseudo-utility scenarios, not WTP, price, purchase probability, or comparable welfare.

S1.12 evidence and verification:
- The evidence assembler verifies 264 referenced artifact hashes, writes the evidence summary,
  complete aggregate and segment tables, seed contrasts, runtime/resource evidence, a ranking
  figure, and the mathematical appendix. Evidence manifest ID:
  `9c0d5b48059cfbecad0d0c9fd2da8a025dc57942104dd181e308d338b07b6650`.
- The complete repository suite passes 204 tests. The sole warning is the already documented SciPy
  quasi-Newton warning in an archived bundle-pricing stability test.
- `python -m src.stage1_pipeline` re-verifies the complete dependency graph and returns
  `status: complete`. No Stage 2 objective, candidate-pool outcome, or bundle-design result was used
  anywhere in model selection or Gate 2.

Interpretation for Dr Li: the completed predictive result is simple and defensible. On this frozen
Steam ownership task, ownership-only ALS substantially beats popularity; the pairwise models and
genre extension do not. That negative result is part of the evidence, not something tuned away.
The downstream optimizer therefore receives three seed-specific ALS score interfaces and four
declared normalizations. It does not receive dollar valuations.

Next session:
- Freeze the outcome-independent candidate-pool registry and build notebook 11's descriptive bridge.
- Do not expand or retune Stage 1 after seeing any Stage 2 output. A changed model, feature, or
  transformation requires a new prospective cycle.

## 2026-08-14 - Public release audit

Goal: check whether the Stage 1 repository can be read and verified from a public clone without
publishing raw or user-level data.

What I changed:

1. Rewrote the README around the current question, Stage 1 result, evaluation design, and next
   stage. The long notebook-by-notebook catalogue is now replaced by links to the detailed notes.
2. Removed saved Steam account details, review text, profile links, and local machine paths from
   notebook outputs. Raw records and user-level artifacts remain ignored.
3. Added a standard-library public verifier. It checks the evidence manifest, all public hashes,
   run-log inventories, semantic IDs, safe paths, and cross-manifest links while allowing only raw
   and protected references to be absent.
4. Added seven verifier tests, a Python 3.10 CI workflow, exact direct package versions, stable line
   endings for hash-bound files, and short provenance, citation, contribution, and security files.
5. Recorded the post-freeze issues that should be fixed in the next scientific cycle instead of
   editing source or configurations already bound into the Stage 1 evidence.

Checks:

- 211 tests pass; the only warning is the existing SciPy warning from the archived pricing test.
- The public verifier checks 12 manifests, 88 validation logs, 3 production logs, 11 top-level
  outputs, 147 public references, and 413 private references.
- A clean public-copy run passes with the raw and protected files absent.
- A Windows-style fresh clone preserves the frozen hashes.
- The proposed public file set contains no known user identifiers, profile URLs, machine paths,
  credentials, or files from the ignored raw/protected directories.

Before release:

- confirm the redistribution terms for the derived tables and figures;
- choose the code license with the project owner and supervisor;
- publish from a sanitized Git history;
- review and add the currently untracked v2 files; and
- set the final Git author name and email before committing.
