# Assumptions and limitations

Last revised: 2026-08-14

Things to keep honest about when writing this up.

## Data

1. **No bundle purchases.** The dataset has user-to-game ownership, not user-to-bundle
   purchases. Every demand number here is a proxy built from ownership. Owning all games in a
   bundle is not the same as having bought the bundle: the user may have bought the games
   separately, on sale, or through a different bundle.

2. **Panel coverage.** Only about half of the 2,798 distinct bundle items appear in the
   Australian ownership panel. So:
   - 238 of 615 bundles have full coverage (every item in the panel),
   - 70 have zero coverage,
   - the rest are partial.
   For any bundle missing even one item from the panel, users_own_all is mechanically biased
   toward zero. We only trust users_own_all on full-coverage bundles.

3. **Australian panel.** The ownership data is a panel of Australian users, not a random
   global sample. Tastes, prices, and ownership rates may not generalise.

4. **Snapshot prices.** bundle_price, bundle_final_price, and discount are a single snapshot.
   Steam discounts change constantly. We do not see the price history or the discount the
   user actually faced when they bought.

5. **bundle_price is sum of parts by construction.** bundle_price equals the sum of item
   discounted_price exactly, so the implied and reported discounts always agree. The
   discount_mismatch check is an internal-consistency test, not a comparison of two
   independent price sources.

6. **Playtime is noisy.** 36% of ownership rows have zero playtime. Zero can mean never
   played, idle, or never launched. Playtime is a weak engagement signal, not value.

7. **Digital goods.** These are games, so marginal cost is near zero and bundling is natural.
   This is cleaner theoretically than grocery, but conclusions are specific to
   low-marginal-cost digital goods.

## Demand proxy

8. **Size bias.** Larger bundles need a user to own more games to count toward users_own_all,
   so the metric falls with bundle size for mechanical reasons. Compare within size or control
   for it.

9. **Substitution ignored.** The proxy counts ownership of a bundle's games in isolation and
   ignores that owning game X may substitute for or complement game Y.

## Counterfactual pricing (notebook 05)

10. **Not causal.** The discount coefficient is cross-sectional. Publishers choose discounts
    strategically, often discounting weaker bundles more, so the estimate is confounded and
    may even carry the wrong sign relative to a true causal demand curve. Everything in
    notebook 05 is descriptive plus an illustrative optimisation.

11. **Constant semi-elasticity is too simple.** The exp(b * delta) demand form gives the same
    revenue-maximising discount for every bundle, which is clearly an artefact. Real demand
    curvature should vary across bundles.

12. **Proxy stands in for quantity.** The revenue calculation uses a demand proxy where it
    should use bundle sales. The level of revenue is not meaningful; only the shape of the
    response to discount is being illustrated.

## Min-cost attribution (notebook 04)

13. **Attribution, not identification.** Notebook 04 does not recover true bundle purchases. It
    reconstructs the cheapest way a rational consumer could have acquired their owned games
    (bundles plus solo buys) and credits the bundles in that reconstruction. It is a
    cost-based attribution proxy, not a revealed purchase path.

14. **Snapshot prices are not the prices faced.** This is the central caveat. A user who
    bought a game on a flash sale may have a true cheapest path unrelated to the snapshot
    cheapest path. Notebook 04 only ever sees one price snapshot.

15. **Rationality.** The rule assumes perfect cost minimisation at those snapshot prices. Real
    buyers buy across time, on impulse, as gifts, or without knowing a bundle exists.

16. **Upward bias toward bundles.** Because bundle_final_price <= sum of component prices for
    all 615 bundles, a fully-owned bundle almost always beats buying its games solo, so
    attribution leans toward bundles. The exception is the solo-price choice below.

17. **Solo price = min across bundles.** The 5 items with inconsistent prices use the cheapest
    observed standalone price. For 2 bundles this makes buying the games solo cheaper than the
    bundle, so those bundles can lose credit. Minor, but documented.

18. **Zero-discount ties dominate the reassignment.** A bundle with no discount costs exactly
    the same as buying its games solo, so the cost rule is indifferent and splits the credit
    (e.g. BioShock Triple Pack, 6951 -> 3475.5). In this catalogue the genuine overlap
    correction is small (~370 of ~3,900 reassigned units); the rest is this tie effect. It is
    not the same as overlap and is reported separately. For downstream demand work,
    zero-discount bundles should be flagged rather than silently halved.

19. **Bundle type and ownership adjustment are not observed.** Steam supports both
    [Complete The Set and Must Purchase Together](https://partner.steamgames.com/doc/store/application/bundles)
    bundle types. The historical snapshot does not identify which transaction-time rule applied.
    Notebook 04 prices each recorded offer as a whole bundle and therefore cannot reconstruct
    ownership-adjusted checkout prices or restrictions.

20. **No timestamps.** Ownership has no dates, so bundle and game availability over time is
    ignored.

21. **Panel incompleteness understates eligibility.** Requiring every bundle item to appear in
    a user's observed library means a real bundle purchase is missed whenever the user appears
    not to own a game they actually own. This is why attribution is trusted only on
    full-coverage bundles (panel_coverage == 1.0), the same rule as users_own_all in nb03.

## The 2026-07-13 pivot: CMM retired, spine moved to a latent preference model plus single-bundle design

The cross-moment bundle-size pricing model (CMM) was retired as the project spine on 2026-07-13.
CMM prices a size-based menu in which the customer freely chooses any bundle of size $s$; Steam's
audited snapshot instead contains fixed, curated offers, so CMM modeled a mechanism the observed
system does not contain. Notebook 10 had already shown the two-moment approximation degrading on
the skewed transformed score panel, but the binding problem is the mechanism, not the approximation.

The bundle-mechanism audit is complete: 568 of 615 observed bundles have evidence that components
are separately available, 47 are coverage-limited, and none has affirmative evidence of SBR
exclusivity. This supports only an **SBA-like component-availability description** of the static
snapshot. It does not identify historical availability, ownership-adjusted pricing, or the use of
CP-optimal component prices. CP anchoring is a declared project convention, not an audited Steam
fact. The internally frozen 2026-07-17 specification therefore uses CP-anchored SBA as the primary
empirical model and SBR only as a theoretical benchmark. Supervisor feedback is nonblocking.

The empirical half is re-centered on collaborative filtering and low-rank matrix factorization,
with genre introduced as a controlled metadata ablation rather than a presumed improvement.

The caveats that follow are retained. Read them with this map:
- Caveats 22, 25 to 28, and 31 (preference scores are not valuations, the evaluation metrics and the
  AUC trap, the concordance limits, and the failed price anchor) carry forward to the new spine.
- Caveats 29 and 30 describe the archived CMM normalization exercise; only the common-scaling
  lesson survives in the live project, at the narrower strength stated in caveat 35.
- Caveats 23, 24, and 32 to 34 now describe archived work (notebooks 06 and 10, the CMM validation
  and its real-data run) and are the record of why the project pivoted, not live claims.
- The live caveats for the pivoted spine are 35 to 42, at the end of this file.

## Preference estimation and the archived BSP optimizer (notebooks 06, 07, 08)

22. **Preference scores are not valuations.** This is the central caveat of Stage 1. The
    recommender produces ownership-derived latent preference scores, not willingness to pay,
    consumer surplus, purchase probabilities, or identified cardinal utility. Stage 2 applies
    frozen nonnegative pseudo-utility transformations as explicit modeling scenarios. Those
    transformations are not calibrations, and their normalized prices and objectives are not
    dollars, price recommendations, or estimates of Steam revenue.

23. **Notebook 06 validates the optimizer, not the economics.** The synthetic experiments confirm
    the optimizer is numerically correct (it matches the brute-force oracle, the SDP form, and the
    paper's Figure 1) and they probe the bundling-vs-correlation hypothesis in a controlled
    design. They use synthetic valuations, so they say nothing about Steam demand; their only role
    is to separate optimizer bugs from later modeling errors.

24. **CMM is an approximation.** The cross-moment model keeps only the first two moments of the
    bundle-size valuations and uses the best-case distribution in that moment class. On synthetic
    data its realized objective reaches about 97 to 99% of the best empirical menu found, but that
    comparator is a numerical search result rather than a universal ground truth. CMM is an
    approximation, not an identity; cases where it underperforms are kept, not tuned away.

25. **Evaluation metrics and the AUC trap.** The preference model is judged by a leakage-safe
    per-user leave-two-out split (popularity from training only, factors fit after holdout
    removal, training positives masked, separate validation and test holdouts). Top-K metrics
    (Recall@K, NDCG@K, coverage) are primary; AUC is reported but secondary, because with one
    positive among thousands of unobserved items AUC is inflated and can rank a popularity
    baseline above a model that is clearly better on top-K. Users with fewer than three owned
    games (7.2%) are excluded from ranking and that share is reported.

26. **Concordance with notebook 03 is not independent validation.** The correlation between mean
    preference score and users_own_all on full-coverage bundles (Spearman about 0.71) is internal
    concordance: both quantities are functions of the same ownership matrix, so agreement is
    reassuring face validity but cannot establish construct validity.

27. **Game features retain, not resolve, ambiguity.** Notebook 07 anchors on the panel's games,
    pre-aggregates bundle membership to one row per game, keeps a price-disagreement flag where a
    game's snapshot price differs across bundles, and stores availability flags rather than
    dropping games with missing catalogue or bundle data. Missingness is preserved, not imputed.

## Archived attempted calibration to dollars (notebook 09)

28. **The standalone price does not identify the valuation scale.** The intended Route A anchor
    was the censoring link: ownership reveals v >= price, so price should set the dollar scale.
    On this data it fails the sign test in every form (aggregate price coefficient t = +6.4;
    per-game quantile anchor slope negative, Spearman(price, threshold quantile) = -0.24).
    Conditional on preference, expensive games are owned more, not less, because price is
    endogenous to quality and exposure (the quantified version of caveat 10). So no positive
    dollar scale is identified from price, and the calibration cannot be read as measured
    willingness to pay.

29. **The dollar scale is an assumption, made tolerable by scale invariance.** Because marginal
    cost is near zero, scaling every valuation by a constant scales prices and revenues but
    leaves the BSP-vs-separate-selling and pure-bundling profit ratios, and the correlation-sweep
    uplift, exactly unchanged (verified in notebook 09). The scale is therefore set by a
    transparent normalization (match the per-game valuation spread and median to the price
    distribution) and swept. Absolute dollar prices and any comparison to the observed Steam
    bundle price hold only under that assumed scale and are reported as sensitivity, never as
    point claims.

30. **The free-disposal floor location does affect the ratios and is swept.** With v = max(a + b
    s, 0) = b * max(s + a/b, 0), the ratios depend on the normalization only through the floor
    location in score units (-a/b); the overall factor b cancels. Both sensitivity knobs move
    that location: a_shift directly, and b_scale too, because the location re-anchors to the
    price median at the new spread. So b_scale is not cosmetic for the ratios; notebook 09
    sweeps the location and reports the floor fraction alongside the uplift, and notebook 10
    sweeps both knobs. The cross-good correlation structure that drives the bundling economics
    comes from the latent factors and is preserved by the affine map.

31. **Face validity is mostly internal.** The archived per-game transformed score is almost a
    monotone function of ownership (Spearman 0.996) because the preference model is fit on
    ownership, so that agreement is mechanical, not evidence. The one semi-independent signal,
    playtime, is only weakly positive (Spearman 0.168) and is itself noisy (36% of ownership rows
    have zero playtime). Neither diagnostic identifies valuation level or cardinal utility.

## Bundle-size pricing on real item sets (notebook 10)

32. **The CMM two-moment approximation degrades on the transformed score panel.** On the seven
    archived Stage 3 item sets the CMM menu realizes only 69 to 97% of the best empirical menu
    found, against 97 to 99% on notebook 06's synthetic distributions. The CMM model value is not
    a bound in either direction: it overstates its own menu's realized profit by up to 52% and
    understates it on one set. The diagnosis in notebook 10: the calibrated partial sums have
    skewness 9 to 18, and a Gaussian control with identical moments realizes within about 5% of
    the model value, so the archived discrepancy is associated with the two-moment reduction on a
    heavily skewed distribution rather than the validated optimizer, SDP, or moment bridge. No live
    SBA conclusion relies on this archived CMM comparison.

33. **The empirical BSP comparator is a search result, not a certificate.** The differential-
    evolution menu is searched on a fixed 20,000-user subsample (seeded at the CMM prices) and
    evaluated on the full panel; there is no optimality proof. It is the best menu found under that
    archived procedure, so no directional claim about its gap to a true optimum is justified.

34. **The observed-price benchmark is scale-dependent.** Comparing the observed Steam bundle
    price against valuations on an assumed dollar scale moves one-for-one with that scale
    (notebook 10 shows obs/separate swinging from 0.59 to 0.36 across b_scale 0.5 to 2 on the
    demo bundle), so it is reported only as a sensitivity, never as a claim about Steam's
    actual pricing.

## The pivoted spine: preference-model ladder and single-bundle design (revised 2026-07-17)

35. **Cardinalization and normalization are a key sensitivity, not a solved problem.** Qualified
    recommender models produce latent preference scores, not identified valuations, and the standalone price
    still fails to identify the dollar scale (caveat 28 holds). Scale-invariance of the profit
    ratios (multiplying every pseudo-utility, pseudo-cost, and price by one constant) is real but insufficient:
    the implicit-feedback data and ranking evaluation validate only the ordinal ordering the scores
    induce, not a cardinal utility, so the numerical cardinalization is model-dependent (loss,
    regularization, specification). Since a strictly increasing map preserves rankings but not sums
    ($\sum_i T(s_i)$ is not generally $T(\sum_i s_i)$), the choice of cardinalization $T$ that maps
    scores to pseudo-utilities changes which bundles clear which prices and therefore the optimal
    design. The optimization is run in explicitly normalized pseudo-utility units as a
    within-model counterfactual (never as identified dollars or revenue), and the bundle choices are
    stress-tested across several declared cardinalizations (the normalization-stability experiment).
    Cross-user dependence is not invariant to arbitrary per-user monotone transforms: such a rule
    can reorder users within an item. Pearson dependence also changes under nonlinear mappings.
    Rank dependence is preserved only under appropriate monotone transformations applied
    consistently to the variable whose cross-user order is being measured. The bridge must therefore
    report raw behavior, identity-only scores, hybrid scores, rank-based measures, and named
    pseudo-utility scenarios separately rather than call one dependence reading cardinalization-free.

36. **Genre and content features are noisy metadata, not ground truth.** Steam genres and tags are
    coarse, publisher-assigned, and inconsistent across games, and some panel games lack catalogue
    metadata entirely (caveat 27). A hybrid model that folds them in inherits that noise. The
    content contribution must be measured on the leakage-safe holdout (Recall@K, NDCG@K), including
    sparse and pseudo-cold item segments, and reported as a gain, null result, or loss relative to
    the controlled identity-only model rather than assumed.

37. **The genre / dependence reading is descriptive, not causal or independent.** Dependence among
    model scores or pseudo-utilities, grouped by genre, is partly computed from the same factor model
    that produced those quantities. Within-genre versus cross-genre differences are internal
    associations (the concordance caveat 26 applies), not evidence that genre causes preference or
    that a particular bundle will improve actual purchases.

38. **SBA exactness and heuristic evidence have different certificate status.** The empirical
    objective is conditional on the Layer 1 model, pseudo-utility transform $T$, additive choice
    assumptions, costs, and tie convention. Under CP-anchored SBA, a user takes bundle $B$ at price
    $b$ under the primary weak-tie rule when
    $\sum_{i\in B}\min\{v_{ui},p_i^{CP}\}\ge b$; the objective must also subtract the component
    margin displaced for that buyer. The raw-sum rule $\sum_{i\in B}v_{ui}\ge b$ belongs to SBR,
    not SBA. For a fixed composition, scanning complete threshold blocks gives an exact price.
    Exhaustively enumerating the entire declared finite feasible family and pricing every composition
    exactly gives a global finite-instance certificate. On larger pools, multistart add/drop/swap
    search with exact repricing is a heuristic. Locked-suite gaps measure its performance on that
    suite but do not certify a new large-pool instance; its output is the **best solution found**
    unless an exact certificate or valid bound is available.

39. **SBR tractability is conditional, not activated by recommender low rank.** The polynomial
    result in *Partition and Prosper* applies to the paper's normal SBR model only when, for some
    $t\ge0$, the covariance has the required positive-diagonal-minus-PSD form with the subtracted
    matrix of fixed rank (independence is a special case). A low-rank factor-score covariance does
    not by itself establish that decomposition; a positive diagonal residual exists only if a
    separate residual or shrinkage model is specified and estimated, and nonlinear pseudo-utility
    transforms can destroy the raw covariance structure. The theorem may be invoked only after its
    distributional and matrix conditions are verified. The general BO/conic method is an optional
    normal-SBR stretch, and a finite BO run returns the best evaluated solution rather than a global
    certificate. The primary empirical SBA optimizer does not rely on either normal-SBR result.

40. **The audit supports SBA-like availability, not CP anchoring or a full historical mechanism.**
    The completed audit (outputs/tables/bundle_mechanism_audit.csv) finds separately available
    components for 568 of 615 observed bundles, 47 coverage-limited cases, and no affirmative SBR
    exclusivity. It is a static availability audit, not evidence of transaction-time menus,
    complete-the-set pricing, or CP-optimal component prices. CP anchoring is therefore an explicit
    modeling convention. SBR remains a different benchmark in which bundled items are unavailable
    separately. Empty SBR reproduces CP, while grand-bundle SBR reproduces PB only when $B=N$ is
    otherwise feasible and capacity permits it; in the capacity-only benchmark this requires
    $C\ge n$. No SBR nesting, hardness, tractability, or approximation result is transferred to SBA.
    Additive pseudo-utility remains a simplification rather than an identified fact about
    substitution or complementarity among games.

41. **Candidate pools are metadata-based feasibility proxies.** The optimizer is restricted to
    frozen publisher-, developer-, franchise-, compatibility-, or co-promotion-coherent pools rather
    than the full 10,978-game catalogue. Those restrictions reduce obviously implausible searches;
    they do not prove common legal control, licensing authority, contractual permission, or actual
    commercial implementability. Every inclusion, exclusion, normalization rule, and coverage
    limitation must remain auditable.

42. **Recommender evaluation must be exact, and prediction is not decision quality.** Sampled
    ranking metrics (a held-out positive against a handful of random negatives) can reverse model
    orderings systematically (Krichene and Rendle 2020), so evaluation ranks each held-out positive
    against the full eligible catalogue with explicit expected treatment of score ties. This is
    held-out ownership reconstruction in a static panel, not prediction of future purchases.
    Separately, a model with better Recall@K need not yield more stable bundle designs; predictive
    quality and downstream decision stability are reported as distinct objects.

## What would fix the big ones

- Transaction-level choices linked to the offers, component availability, prices, exposure, and
  timing each user actually faced, including rejected offers rather than purchases alone.
- Credible exogenous pricing variation or a defensible identification design that addresses price,
  quality, and exposure endogeneity.
- Actual bundle and component transactions with acquisition timing and ownership-adjusted prices,
  supporting a structural demand model whose assumptions can be tested against observed choices.
