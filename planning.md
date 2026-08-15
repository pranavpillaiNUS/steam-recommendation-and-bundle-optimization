# UROP technical plan: from the CMM pivot to preference estimation and fixed-bundle design

Last revised: 2026-08-14

Current status: Gate 0 and every Stage 1 step S1.0 to S1.12 are complete under the prospective
`s1-v2-20260814` Steam-ID cycle. Gate 1 and Gate 2 pass; only implicit ALS was admitted. Production
refits and fold-in are complete for all three seeds and 12,585 assessment users. The next binding
step is the outcome-independent candidate-pool registry and notebook 11, followed by Gate 3. No
Stage 2 objective or bundle outcome was accessed during Stage 1 selection or scenario freezing.

Post-freeze review note (2026-08-14): the cycle is an internally prospective, hash-bound record,
not an external preregistration. Scientific inputs and evidence remain unchanged. Wording errata,
the aggregate assessment-score peek, public/private verification boundaries, runtime-label issue,
and next-cycle engineering amendments are recorded in `notes/stage1_release_audit.md`. That audit
controls when a historical prospective paragraph below is stale.

## 1. What this document is for

This is my main technical reference for the rest of the project. It records:

- how the project moved from the cross-moment bundle-size pricing model, CMM, to the current two-stage direction;
- what parts of the earlier work still stand and what parts are now only archive material;
- the exact purpose, data, models, splits, metrics, and pass conditions for Stage 1;
- the interface that converts Stage 1 ranking scores into declared pseudo-utility scenarios;
- the exact CP-anchored Single Bundle with All, SBA, problem used in Stage 2;
- the proofs, algorithms, tests, certificates, experiments, and robustness checks needed to finish;
- the order in which protected outcomes may be opened;
- what the project can and cannot claim; and
- the current stopping point and next actions.

This file is both a plan and a record of the research design. It should be possible to read it from the beginning to understand why the direction changed, or to jump to a numbered stage and see what has to be implemented and checked.

The machine-readable configurations and frozen manifests control the current Stage 1 cycle when a prose summary disagrees with them. The main files are:

- `configs/cycles/s1-v2-20260814/preference_models.json`;
- `configs/cycles/s1-v2-20260814/ranking_evaluation.json`;
- `configs/cycles/s1-v2-20260814/pseudo_utility_scenarios.json`; and
- the complete public dependency graph in `outputs/modeling/cycles/s1-v2-20260814/`.

`notes/research_log.md` records what was actually completed and when. `notes/preference_model_specification.md` and `notes/optimization_models.md` hold the mathematical details. Later changes to a frozen choice must be prospective, dated, versioned, and made before looking at the outcome affected by that choice.

## 2. Short version of the new direction

The live project has two main stages and one bridge between them.

```text
Raw Steam snapshot
  -> cleaned bundle, catalogue, ownership, and playtime data
  -> descriptive bundle evidence in notebooks 00 to 05
  -> Stage 1: estimate and evaluate latent user-game preference scores
  -> freeze qualified model-by-seed score interfaces
  -> map scores into declared nonnegative pseudo-utility scenarios
  -> freeze catalogue-coherent candidate pools and build notebook 11
  -> Stage 2: design and price one fixed bundle under CP-anchored SBA
  -> compare with CP, PB, SBR, and observed compositions
  -> assess frozen policies on held-out users and report robustness
```

Stage 1 asks whether collaborative filtering reconstructs held-out ownership better than popularity, and whether genre adds anything when it is the only feature change. Stage 2 asks which fixed bundle should be added to a component-pricing menu under each declared pseudo-utility scenario.

The final output is not a Steam price recommendation. It is a data-driven optimization study in normalized scenario units. Its value comes from a careful link between prediction, explicit modeling assumptions, exact and heuristic optimization, and held-out decision evaluation.

## 3. How the project changed direction

### 3.1 What the data exploration established

The project started by inspecting the Steam Video Game and Bundle Data rather than assuming that the dataset contained purchase records.

The main descriptive facts were:

- 615 observed bundles and 3,525 bundle-item rows;
- 70,912 users, 10,978 games, and 5,094,082 deduplicated ownership rows in the original flattened panel;
- no file joining a user ID directly to a bundle ID;
- one static price snapshot rather than transaction-time prices;
- only 238 of 615 bundles with complete coverage in the ownership panel; and
- 36.3 percent of ownership rows with zero lifetime playtime.

Because direct bundle purchases are absent, notebooks 03 and 04 construct ownership-based attribution proxies. `users_own_all` counts users who own all observed components. The minimum-cost reconstruction then attributes a library to the cheapest combination of bundles and solo items at snapshot prices.

This correction was useful, but its result was mostly negative. Out of 3,885 reassigned units, about 367 came from actual bundle overlap. Most of the remaining change came from a zero-discount tie, especially the BioShock Triple Pack. The correct interpretation is that overlap does not greatly change the simple proxy in this snapshot, not that bundle purchases have been identified.

Notebook 05 closes the descriptive branch. On the 238 full-coverage bundles, the primary HC3 regression gives a positive discount association, a negative size association, and a positive component-price association, with an R-squared of about 0.079. The zero-discount coefficient has p = 0.436. These are cross-sectional associations using an ownership-derived proxy. They are not causal demand estimates and cannot identify an optimal discount.

### 3.2 The original CMM direction

The first optimization direction used the cross-moment model for bundle-size pricing. In that model, the seller posts a price `p_s` for each bundle size `s`. A customer may choose any set of `s` items, so the relevant value is the sum of the customer's top `s` item values.

For user `u`, let

$$
W_{us}=\text{sum of the top }s\text{ item values for user }u.
$$

The customer chooses the size that maximizes `W_us - p_s`, including the outside option. CMM summarizes the joint distribution of these partial sums using their mean vector and covariance matrix, then solves a tractable problem in demand-share coordinates.

This was mathematically useful work. The project:

- derived and documented the main CMM reductions and proofs;
- implemented the simplex and SDP demand formulations;
- checked gradients and eigenvalue handling;
- matched a brute-force oracle on small cases;
- reproduced the paper's synthetic figures;
- compared the CMM solution with a direct sample-average price search; and
- studied how the advantage of bundling changed with dependence.

That work remains in `src/bundle_pricing.py`, `tests/test_bundle_pricing.py`, notebook 06, notebook 10, and Appendix A of `notes/optimization_models.md`.

### 3.3 Why CMM is no longer the project spine

The binding problem is a mechanism mismatch.

CMM lets a customer construct an arbitrary collection after choosing a size. The observed Steam bundles have seller-chosen, fixed compositions. A price menu for sizes 1, 2, 3, and so on is therefore not the same decision as choosing one particular set of games and its bundle price.

There was also a secondary empirical problem. On the synthetic distributions in notebook 06, the CMM menu realized about 97 to 99 percent of the best empirical menu found. On the seven Steam-derived item sets in notebook 10, it realized about 69 to 97 percent. The real partial-sum distributions were highly skewed, so their first two moments did not describe them well. The empirical comparator was still a numerical search rather than a certificate, so this result is evidence about approximation quality, not a model-free truth.

The mechanism mismatch is enough to retire CMM even if its numerical approximation had been perfect. The approximation result helps explain why continuing to tune CMM would also have been unhelpful.

### 3.4 Why the old monetary calibration also stopped

The first preference prototype produced SVD and NMF scores. Notebook 09 then tried to identify an affine dollar scale from standalone prices and ownership rates.

The proposed censoring argument required higher prices, conditional on preference, to reduce ownership. The data gave the opposite sign:

- aggregate price coefficient t-statistic about +6.37;
- quantile-anchor slope about -20.46; and
- Spearman correlation between price and the ownership-threshold score quantile about -0.243.

Price is confounded with quality, popularity, exposure, and release conditions. The snapshot therefore does not identify a positive conversion from a latent score to dollars. The retained affine normalization was an assumption, not a calibration.

This matters beyond the common unit. Multiplying every utility, price, and cost by one positive constant only changes the common scale. A nonlinear monotone transformation can preserve rankings while changing sums, bundle thresholds, common-price comparisons across users, and the selected bundle. Cardinalization must therefore be an explicit sensitivity in the new direction.

### 3.5 Mechanism audit and the new choice model

The mechanism audit cross-referenced all 615 bundles with the individual-game catalogue and the displayed component prices.

Its findings were:

- 568 bundles have SBA-like component-availability evidence;
- 47 bundles are unclear, mainly because of catalogue coverage;
- 0 bundles have affirmative evidence of SBR-style exclusivity;
- all 615 display a standalone price field for every recorded component;
- 475 have complete catalogue confirmation; and
- 513 of the SBA-like classifications are high-confidence under the audit rule.

This supports a static description in which bundle components remain available separately. It does not identify historical menus, transaction-time availability, complete-the-set pricing, legal control, or component-pricing-optimal prices.

The internally frozen primary model is therefore CP-anchored SBA. First estimate component-pricing-optimal pseudo-prices on design users. Hold those prices fixed. Then choose one fixed bundle and its normalized price while every component remains separately available.

SBR stays as a different benchmark. Under SBR, games placed in the bundle are removed from separate sale. Its choice rule, nesting properties, hardness results, and normal-model theorems do not transfer to SBA.

### 3.6 What is retained and what is retired

The following work remains part of the live evidence base:

- notebooks 00 to 05 and their cleaned tables;
- the mechanism audit and its provenance manifest;
- the failed price-anchor diagnostics as identification evidence;
- the sparse interaction and feature foundations;
- the theoretical lessons about dependence and bundling, when restated for the correct mechanism; and
- the numerical and research lessons from the CMM attempt.

The following work is retained only as archive or prototype material:

- notebook 06 and `src/bundle_pricing.py` as the CMM validation record;
- notebook 10 and `outputs/tables/bundle_size_pricing.csv` as the real-data CMM record;
- `src/calibration.py` and `valuation_calibration.csv` as the failed monetary-anchor record;
- notebook 08's SVD and NMF leaderboard; and
- `preference_factors.npz`, which is not a frozen Stage 1 model artifact.

None of those archived outputs may be used as a live Stage 1 winner or as an input to the new bundle optimizer.

## 4. Final research questions

### 4.1 Main question

Can leakage-safe collaborative filtering, with genre used as a controlled metadata extension, produce useful latent preference rankings from the Steam ownership panel, and what fixed curated bundle should be selected under CP-anchored SBA when those rankings are mapped through explicit pseudo-utility scenarios?

### 4.2 Supporting questions

1. How do training-only popularity, weighted implicit ALS, identity-only pairwise matrix factorization, and identity-plus-genre matrix factorization compare when every target is ranked against the complete 6,721-item warm catalogue?
2. Does genre help on warm items, sparse items, or pseudo-cold items when every other training choice is held fixed?
3. How do observed bundles compare with matched catalogue-coherent alternatives in raw co-ownership, preference dependence, genre concentration, popularity balance, and reach?
4. Is there a tension between user-facing coherence and seller-facing diversification?
5. How sensitive are bundle choices to the score model, training seed, pseudo-utility transformation, candidate pool, capacity, costs, and tie rules?
6. For each declared scenario, which fixed bundle and normalized price maximize the empirical CP-anchored SBA objective?
7. On which pool sizes can the global finite-instance optimum be certified by complete enumeration?
8. How close does a locked scalable heuristic come to those certified optima?
9. Does a policy selected on design users retain its advantage over frozen CP when evaluated on assessment users without reoptimization?
10. Does better ranking accuracy translate into more stable or better downstream bundle decisions?

### 4.3 Intended contributions

The intended completed project has four contributions.

1. An empirical contribution: a controlled and leakage-aware comparison of implicit-feedback models on a large sparse Steam panel.
2. A descriptive contribution: a matched study of observed and alternative bundles using raw behavior, identity-only scores, and genre-aware scores.
3. An optimization contribution: a correct finite-panel CP-anchored SBA formulation, exact pricing for a fixed composition, exact small-pool search, and a scalable heuristic with measured gaps.
4. A methodological contribution: a direct study of how ordinal prediction, cardinalization assumptions, and model uncertainty affect downstream decisions.

### 4.4 Claims that are outside the project

The data do not support claims about:

- individual willingness to pay;
- actual Steam bundle purchases;
- causal discount elasticity;
- current or historical Steam demand curves;
- dollar-optimal Steam prices;
- actual or expected Steam revenue;
- causal effects of genre;
- legal authority to create a proposed bundle; or
- a global optimum on a large pool unless a complete certificate or valid bound exists.

## 5. Core definitions and interpretation rules

### 5.1 Data objects

- `o_ui` is a binary ownership indicator. Ownership is implicit positive feedback, not a rating, purchase occasion, or exposure record.
- `t_ui` is playtime. In the live Stage 1 models it changes confidence on an observed ownership edge. It is not a cardinal value target.
- An unobserved user-item pair is treated as missing implicit feedback, not as a known dislike.
- The Australian-user panel is a selected historical snapshot, not a representative current global population.

### 5.2 Stage 1 objects

- `s_ui` is a latent preference score from a fitted model.
- Stage 1 validates the ordering induced by `s_ui` through held-out ownership reconstruction.
- A score is not a utility, purchase probability, WTP estimate, or monetary quantity.
- Each model is identified by its family, configuration, training seed, split set, feature set, training data, software version, and serialized parameters.

### 5.3 Stage 2 objects

- `v_ui^m = T_m(s_ui) >= 0` is a pseudo-utility under scenario `m`.
- `T_m` is a declared modeling choice. It is not a calibration to a true utility scale.
- `N` is one frozen catalogue-coherence candidate pool.
- `B` is one fixed bundle selected from `N`.
- `c_i` is a nonnegative pseudo-cost. The primary setting is `c_i = 0`.
- `p_i^CP` is a component-pricing-optimal pseudo-price estimated on design users.
- `b` is the normalized bundle price.
- Every price and objective is reported in the same normalized units as the scenario that produced `v_ui^m`.

### 5.4 User samples

- Design users are used for model selection, production shared-parameter fitting, global transformation fitting, component pricing, bundle selection, bundle pricing, and heuristic development.
- Assessment users receive folded-in representations from frozen shared parameters. They evaluate complete frozen policies only.
- Assessment data cannot be used to select a different price, composition, transformation, model, or heuristic setting.

### 5.5 Mechanisms

- Component pricing, CP: each item is offered separately and priced independently.
- Pure bundling, PB: the whole modeled pool is offered only as one grand bundle.
- Single Bundle with All, SBA: one fixed bundle is added while its components remain separately available.
- CP-anchored SBA: the component prices are first set to their CP optima and then held fixed while the bundle is designed.
- Single Bundle with the Rest, SBR: items placed in the bundle are removed from separate sale. This is a benchmark, not the primary empirical mechanism.
- CMM bundle-size pricing: a customer chooses an arbitrary set of a posted size. This is the archived mechanism.

## 6. Rules that apply to the whole project

### 6.1 No outcome leakage

The order of access is binding:

1. freeze the protocol, data, splits, features, and estimator equations;
2. use design-training data to fit configurations;
3. use validation only for model selection and Stage 2 admission;
4. hash the admission decision;
5. open the design-test outcomes once and run the preregistered pseudo-cold evaluation, without using either result to replace the validation-selected configurations;
6. refit permitted shared parameters on design users;
7. fold in assessment users without changing shared parameters;
8. freeze pseudo-utility scenarios and metadata-based candidate-pool membership before inspecting any bundle objective;
9. complete the preregistered notebook 11 comparisons without changing pool membership to improve their results;
10. freeze the Stage 2 instance registry, exact suites, heuristic protocol, and all tie and feasibility rules;
11. use design users to select complete Stage 2 policies; and
12. evaluate the frozen policies on assessment users without reoptimization.

If a protected result changes a rule that should have been fixed earlier, the old cycle is closed and a new prospective cycle is created. The result cannot be silently reused to tune the rule.

### 6.2 Exactness language

- A completed scan of all valid price thresholds for a fixed composition is an exact fixed-composition price result.
- A completed enumeration of every feasible composition, with exact repricing, is a global optimum for that declared finite instance.
- A local-search or differential-evolution result without a bound is the best solution found.
- Runtime evidence does not prove computational complexity.
- Exponential enumeration does not by itself prove NP-hardness.
- A theorem about SBR does not become a theorem about SBA.

### 6.3 Scale and cardinalization

A common positive scaling of all pseudo-utilities, prices, and costs rescales the objectives and preserves choices and optimal compositions. This only handles the choice of a common unit.

A nonlinear monotone mapping can preserve within-user ranking and still change:

- sums across items;
- a common price comparison across users;
- buyer thresholds;
- component prices;
- the selected composition; and
- the measured advantage over a benchmark.

User-specific mappings also impose an interpersonal location or scale convention. Every transformation must state that convention. The final result is stability or fragility across scenarios, not discovery of the one true transformation.

### 6.4 Reproducibility and artifact rules

- Large score matrices are not materialized or saved.
- Model scoring is done in bounded user and item blocks.
- IDs, sparse matrices, feature rows, and per-user metric rows have explicit order contracts.
- Public manifests contain rules, counts, schemas, and hashes, but not protected identifiers.
- Protected IDs and matrices remain in cycle-scoped ignored directories.
- Saved model parameters use fixed numeric dtypes and do not rely on pickle.
- Every headline row carries the model, seed, split, transformation, pool, cost, tie, and code identity needed to reproduce it.
- A failure, NaN, timeout, or retry is logged. There is no unrecorded retry with a more favorable seed.

## 7. Full dependency graph and gates

```text
Completed descriptive data work, notebooks 00 to 05
  -> Gate 0 mechanism and identification freeze, complete
  -> S1.0 protocol freeze, complete
  -> S1.1 canonical interactions, complete
  -> S1.2 protected outer and nested splits, complete
  -> S1.3 identity and genre features, complete
  -> S1.4 backend-neutral estimator specification, complete
  -> S1.5 backend and fold-in feasibility spike, complete
  -> S1.6 validation tuning and pre-test admission hash, complete
  -> S1.7 one-time exact design-test ranking against the complete warm catalogue, complete
  -> S1.8 controlled genre and pseudo-cold evaluation, complete
  -> S1.9 Gate 1 closeout and downstream claim freeze, complete
  -> S1.10 production refits and assessment fold-in, complete
  -> S1.11 pseudo-utility scenarios and Gate 2, complete
  -> S1.12 Stage 1 evidence package, complete
  -> candidate-pool registry and notebook 11
  -> Gate 3 descriptive bridge
  -> S2.0 Stage 2 preregistration
  -> S2.1 to S2.4 formulation, proofs, exact code, and certificates
  -> S2.5 locked heuristic validation
  -> Gate 4 optimizer validity
  -> S2.6 frozen-policy assessment
  -> S2.7 robustness and decision quality
  -> S2.8 Stage 2 evidence package
  -> Gate 5 empirical conclusion
  -> report, presentation, clean reproduction, and Gate 6
```

Stage 2 objectives must not be used to choose a Stage 1 model or transformation. Notebook 11 and synthetic Stage 2 implementation may be prepared in parallel after the Stage 1 interface is fixed, but real pool optimization waits for the scenario and pool manifests.

## 8. Stage 1 purpose and estimand

Stage 1 estimates latent preference scores from sparse implicit feedback. Its empirical target is reconstruction of held-out ownership within the retained Australian-user snapshot.

The target is deliberately narrow. It asks whether a model places one held-out owned game near the top of the complete frozen 6,721-item warm catalogue after the user's training positives and other holdout are masked. It does not ask whether the user will buy the game in the future.

Stage 1 must produce:

- one canonical user, item, interaction, and feature contract;
- a frozen outer design and assessment split;
- nested design-user training, validation, and design-test roles;
- a training-only popularity baseline;
- weighted implicit ALS;
- identity-only pairwise feature-sum matrix factorization;
- the same pairwise model with genre as the only controlled change;
- exact tie-aware metrics against the complete 6,721-item warm catalogue for all three frozen seeds of every stochastic specification, with popularity fitted once;
- a separate pseudo-cold experiment;
- a validation-selected and hashed Stage 2 admission set;
- design-only production refits;
- a tested assessment-user fold-in path;
- a bounded and reproducible scorer; and
- a frozen model-by-seed-by-transformation interface for Stage 2.

Alignment, leakage, masking, tie, confidence, nonfinite-score, or fold-in failures block Stage 2. Genre failing to help does not block Stage 2.

## 9. Completed Stage 1 foundations

The S1.0--S1.4 subsections below preserve the original `s1-v1-20260718` foundation as a research
record. That cycle was superseded before any real fit or protected-outcome access when a raw identity
audit established that `user_id` is not the account identifier. The controlling live counts and IDs
are the prospective `s1-v2-20260814` completion record in section 10.9 and Appendix B. No v1 fit or
result was promoted into v2.

### 9.1 S1.0: frozen protocol

S1.0 was frozen on 2026-07-18. It fixed the full model ladder, split rules, seeds, grids, metrics, resource limits, tie rules, uncertainty method, and Stage 2 admission rule before a new real model was fitted.

The interpretation is fixed: all fitted outputs are latent preference scores for held-out ownership reconstruction.

#### Model grid

| Family | Frozen settings |
| --- | --- |
| Popularity | Exact design-training ownership count, no fitted parameters |
| Implicit ALS | Factors 32 or 64; regularization 0.05 or 0.2; `alpha_o` 20 or 40; three confidence schemes; 12 fixed iterations |
| Identity BPR | Factors 32 or 64; regularization 0.0001 or 0.001; learning rate 0.05; 12 fixed epochs; 1,000,000 triples per epoch |
| Identity plus genre BPR | Same selected BPR settings, shared initialization, and complete triple stream as identity-only; genre block is the only controlled toggle |

The three ALS confidence schemes are:

1. ownership only: `alpha_p = 0`, `tau = 0`;
2. log playtime: `alpha_p = 2`, `tau = 14`; and
3. capped log playtime: `alpha_p = 2`, `tau = 5`.

This gives 24 ALS configurations and four BPR hyperparameter configurations before training seeds. The training seeds are:

```text
104729, 130363, 155921
```

The numerical policy uses float32 parameter storage, float64 accumulation, one BLAS thread for controlled numerical work, and a linear-solve jitter sequence of `0`, `1e-8`, and `1e-6`. An unlogged retry is not allowed. A nonfinite seed invalidates the configuration in the current cycle.

The resource limits are:

- 8 GiB peak memory;
- 2,700 seconds per fit;
- 43,200 seconds total tuning time; and
- 256 MiB saved model size per seed.

The ranking configuration fixes Recall and NDCG at 10 and 20, with mean NDCG@20 as the primary selection metric. It also fixes the exact tie rule, 64 MiB maximum score block, 5,000 evaluation users, 2,000 paired bootstrap replicates, and bootstrap seed 314159.

The design test is sealed until the validation admission manifest is hashed. Assessment identifiers and histories are sealed until S1.10. Stage 2 objectives and bundle outcomes are unavailable during Stage 1 selection.

### 9.2 S1.1: canonical sparse interactions

The live Stage 1 interaction contract is stricter than the original notebook 02 table.

IDs must be exact, nonnegative, unpadded decimal values that fit in signed int64. Rows with invalid identifiers are excluded and counted. Negative or nonfinite playtime fails the build. Duplicate user-item rows collapse to one ownership edge with the maximum nonnegative value of each playtime field.

The saved contract has three aligned float32 CSR matrices:

- binary ownership;
- lifetime playtime; and
- two-week playtime.

All three share the same user order, item order, shape, and stored pattern. The user and item arrays are ascending numeric int64. Metadata-only catalogue items remain as explicit zero-support columns. An interaction item missing from the metadata universe is a failure.

The source and retained counts are:

| Quantity | Count |
| --- | ---: |
| Source ownership rows | 5,094,082 |
| Retained canonical edges | 2,221,650 |
| Excluded noncanonical-user rows | 2,872,432 |
| Retained users | 43,802 |
| Full item universe | 10,978 |
| Items with retained support | 9,733 |
| Metadata-only zero-support items | 1,245 |

The 56.4 percent row loss is material. It comes from the already frozen numeric user-ID rule, including username-form IDs, rather than from an outcome-based filter. Final reporting must show this as a coverage and representativeness limitation.

The sparse contract occupies about 54.3 MB under the recorded storage measure, compared with about 5.77 GB for three dense float32 matrices plus the ID maps.

### 9.3 S1.2: outer users and nested holdouts

Users with fewer than five retained ownership edges are excluded from the model-evaluation cohort. The six frozen activity bands are `[5, 10)`, `[10, 25)`, `[25, 50)`, `[50, 100)`, `[100, 200)`, and `[200, infinity)`. Within each band, users are sorted by the frozen SHA-256 namespace and then by user ID. The first `floor(0.20 * band size)` users are assessment users and the remainder are design users.

The outer split is:

| Cohort | Users | Edges before nested holdout |
| --- | ---: | ---: |
| Excluded low activity | 6,722 | 14,131 |
| Design | 29,666 | 1,764,950 |
| Assessment | 7,414 | 442,569 |

The warm catalogue requires design-user support of at least five before holdout and at least three after holdout. An evaluable design user must have at least three warm edges before the nested assignment. The assignment processes users in the frozen hash order and considers the test role before the validation role. It assigns one distinct test positive and one distinct validation positive only when item-support capacity remains feasible.

The realized nested split is:

| Role | Count |
| --- | ---: |
| Design-training edges | 1,699,520 |
| Validation positives | 29,666 |
| Sealed design-test positives | 29,666 |
| Nonwarm excluded edges | 6,098 |
| Warm items | 6,721 |

All 29,666 design users are evaluable in the realized greedy scan. The minimum post-holdout warm-item support is four. This is a deterministic feasible result, not a maximum-cardinality theorem.

The primary warm-ranking evaluation sample contains exactly 5,000 design users. It is formed by taking the hash-smallest users within each outer activity band and reconciling the proportional quotas deterministically. It is used for validation and, after the admission set is frozen, design-test. Primary metrics rank against the complete 6,721-item warm catalogue.

The pseudo-cold cohort contains 300 genre-covered items:

- 100 with design support in `[5, 20)`;
- 100 with support in `[20, 100)`; and
- 100 with support in `[100, 500)`.

Pseudo-cold selection is hash-stratified by normalized primary genre inside each support band. Primary genre is the lexicographically first unique label after HTML unescaping, Unicode NFKC normalization, whitespace collapse, and blank removal. Fractional genre quotas are reconciled by Hamilton largest remainder with canonical lexical ties.

Protected artifacts are separated by access class. Tuning code may load design-training matrices, validation coordinates, and the evaluation-user sample. It may mask each user's other holdout through an opaque operation, but it does not receive the sealed test coordinate. Public manifests contain hashes and aggregates rather than protected IDs.

### 9.4 S1.3: identity and genre features

The full item map has 10,978 rows in exact canonical item order.

The identity block is a 10,978 by 10,978 float32 sparse identity matrix. The genre block has 21 columns. Each observed genre on an item receives weight `1 / number of observed genres for that item`. Missing genre produces a zero-content row rather than a learned missing token.

Identity rows have weight 1 and the genre block has weight 1. The blocks are not jointly row-normalized. The vocabulary comes only from the pre-model game-feature catalogue. Genre tokens are HTML-unescaped, Unicode NFKC-normalized, whitespace-collapsed, blank-filtered, and deduplicated before the equal weights are assigned.

The frozen feature counts are:

| Quantity | Full catalogue | Warm projection |
| --- | ---: | ---: |
| Items | 10,978 | 6,721 |
| Genre-covered items | 8,658 | 5,567 |
| Zero-content items | 2,320 | 1,154 |
| Genre columns | 21 | 21 retained |

The full genre block has 21,559 nonzero values. Every genre occurs on at least five warm items, so no genre is learnable only from the reserved pseudo-cold cohort.

The predictive feature blocks exclude:

- price;
- bundle membership;
- popularity;
- playtime;
- tags;
- publisher;
- developer; and
- downstream pool membership.

Playtime remains an interaction-confidence input. Publisher and developer are reserved for later feasibility-pool construction, not prediction in the core genre ablation.

### 9.5 S1.4: backend-neutral estimator mathematics

S1.4 fixed the equations and synthetic reference implementations without fitting a real model.

#### Popularity

For item `i`, define the exact design-training count

$$
d_i=\sum_u o_{ui}^{train},
$$

so every user receives the popularity score

$$
s_{ui}^{pop}=d_i.
$$

It has no trainable parameters. Ties are expected and must be handled by the common exact-tie metric rule.

#### Weighted implicit ALS

The preference target is binary ownership. Here `t_ui` is `playtime_forever`; `playtime_2weeks` is outside the headline confidence grid. For an observed edge,

$$
\gamma_{ui}
=1+\alpha_o o_{ui}
+\alpha_p\min\{\log(1+t_{ui}),\tau\}.
$$

The unobserved baseline confidence is one and is kept implicit. An owned but unplayed game receives confidence `1 + alpha_o`, not one.

The model minimizes

$$
\sum_{u,i}
\gamma_{ui}
\left(o_{ui}-x_u^\top q_i\right)^2
+\lambda
\left(
\sum_u\|x_u\|_2^2+
\sum_i\|q_i\|_2^2
\right).
$$

The joint problem is nonconvex. With one factor block fixed, each update is a convex ridge problem. The reference code evaluates the exact objective, builds user and item normal equations, performs controlled Cholesky solves, and records objective and residual diagnostics after every block.

The small NumPy implementation is a mathematical oracle. It is not yet the production full-data backend.

#### Pairwise feature-sum BPR

For user `u` and item `i`, the item representation is

$$
h_i=\eta_i+\rho F_iG,
$$

and the score is

$$
s_{ui}=b_i+x_u^\top h_i.
$$

Here:

- `x_u` is the user factor;
- `eta_i` is the identity factor;
- `b_i` is an item bias;
- `F_i` is the frozen genre row;
- `G` contains genre-factor parameters; and
- `rho = 0` for identity-only and `rho = 1` for identity plus genre.

There is no user bias. Identity is not duplicated inside `F_i`.

For a sampled triple `(u, i, j)`, positive `(u, i)` is drawn uniformly from canonical design-training edges with replacement. One negative proposal `j` at a time is drawn uniformly from the warm catalogue with replacement until it is not one of user `u`'s training positives. There is one accepted negative per positive. The summed loss is

$$
\mathcal L
=\sum_{(u,i,j)}
\log\left(1+\exp\{-(s_{ui}-s_{uj})\}\right)
+\lambda\|\theta\|_2^2.
$$

Here `theta` contains every active user factor, item-identity factor, item bias, and genre factor. Item biases are regularized. Inactive genre parameters are absent from the identity model and are not included in its penalty.

The reference code implements a stable loss, exact accumulated gradients, deterministic continuing PCG64 triple streams, bounded scoring, and pickle-free float32 parameter serialization. Every scalar proposal draw, including a rejected negative, advances the stream. The same namespaced stream continues across epochs rather than restarting.

The frozen sampler rejects training positives only. A validation or design-test positive is still unobserved by the trainer, so it may be drawn as a negative sample. This is an imperfect implicit-feedback convention, but changing it would reveal protected targets to training. The convention must be identical for identity-only and identity-plus-genre runs and must be stated when the results are interpreted.

When the common parameter arrays are held fixed, the identity-only and genre paths are required to agree exactly when `rho = 0` or the content block is zero. This is an implementation invariant. It does not mean that separately trained identity and genre models must give the same user, identity, bias, or zero-content-item scores, because training with genre active can change their shared parameters.

#### What S1.4 does not yet provide

S1.4 does not provide:

- a production `implicit` backend adapter;
- a production LightFM adapter;
- an Adagrad BPR training loop for the full data fallback;
- real-data convergence or resource evidence;
- popularity or ALS model serialization under a complete public model contract;
- assessment fold-in;
- a tuning runner;
- a validation leaderboard;
- an admission manifest; or
- a clean-process end-to-end scorer.

The estimator manifest correctly records `real_model_fit: false`.

## 10. Stage 1 execution specification and completion record

Sections 10.1--10.8 preserve the prospective execution specification used for the completed cycle.
They are historical requirements, not a current to-do list. Section 10.9 records the realized v2
completion; Stage 2 is the remaining live work.

### 10.1 S1.5: backend and fold-in feasibility spike

At execution time, this was the next binding step.

At the time of the spike, the intended packages were `implicit==0.7.2` for ALS and
`lightfm==1.17` for the pairwise feature model, and neither was installed. Installation alone was
not a pass; each backend had to be checked against the frozen equations and artifact rules. The v2
completion record below documents native `implicit` and the prospective NumPy BPR fallback.

The spike must test:

1. orientation of user and item matrices;
2. exact meaning of confidence weights;
3. agreement with a tiny explicit objective and score oracle;
4. direct versus batched score equality;
5. deterministic behavior under the fixed seeds and thread policy;
6. fixed iteration or epoch counts;
7. parameter dtype and serialization;
8. score reproduction after a clean reload;
9. continuation and equality of the complete BPR positive-negative triple stream;
10. identity-only versus zero-content equivalence with common parameters held fixed;
11. suppression of identity factors and bias for pseudo-cold items;
12. ALS fold-in with frozen item factors;
13. pairwise user-only fold-in with frozen item and genre parameters;
14. deterministic construction and reproduction of the pairwise fold-in set `T_u`;
15. proof that fold-in does not mutate shared parameters;
16. runtime and peak memory at planned dimensions; and
17. clean failure for an insufficient-history user.

Use synthetic users or design users treated as pseudo-new, using permitted design-training histories only. Do not open validation coordinates or diagnostics, assessment IDs or histories, sealed design-test targets, or the reserved pseudo-cold cohort during this spike.

If a preferred package cannot match the contract on Windows, time-box the environment work. Then finish the relevant NumPy production fallback, test it against the small mathematical oracle and the full artifact contract, and record a prospective amendment before using it. The fallback must keep the same objective, feature-sum score, split, negative rule, controlled genre toggle, and artifact contract. Do not silently replace BPR with a different loss or tune the fallback on protected outcomes.

#### S1.5 outputs

Planned outputs are:

- `outputs/modeling/stage1_backend_spike_manifest.json`;
- a small backend-equivalence results table;
- serialized synthetic model fixtures;
- runtime and memory measurements; and
- focused backend and fold-in tests.

S1.5 passes only when the intended production training and fold-in route works on a small complete case.

### 10.2 S1.6: fit and select on validation

Run the frozen grid with design-training data only.

For ALS, fit all 24 configurations for all three training seeds. For the pairwise family, fit the complete four-configuration identity-only grid for all three seeds. Use the frozen selection rule to choose one common identity configuration, then run identity plus genre with the same dimension, regularization, learning rate, epoch count, initialization, positive stream, negative stream, and seeds. The only controlled change is whether the frozen genre block is active. There is no separately tuned genre model in the current cycle.

Validation ranking uses the same exact complete-warm-catalogue evaluator and masking rules described in S1.7. The difference is its role: validation selects configurations and the admission set, while design-test is opened once afterward to estimate performance without another round of selection.

For every attempted fit, save:

- configuration ID;
- model family and seed;
- upstream set IDs;
- start and completion status;
- software and backend version;
- parameter count;
- iteration or epoch diagnostics;
- objective or loss trace;
- solve jitter and residual information where relevant;
- triple-stream hash for BPR;
- runtime and peak memory;
- artifact hashes; and
- any failure or invalidation reason.

The common selection order is:

1. mean NDCG@20, descending;
2. mean Recall@20, descending;
3. trainable parameter count, ascending;
4. latent dimension, ascending; and
5. configuration ID, lexicographic.

Metrics are averaged arithmetically across training seeds for selection. Runtime is reported but is not a tie-break. Fixed iterations and epochs are used, so there is no model-specific early stopping advantage.

After validation selection, apply the predeclared Stage 2 admission rule and hash the validation-selected admission set before opening the design test.

### 10.3 S1.7: one-time exact design-test ranking

The evaluator in this section is first run on validation during S1.6. After the validation-selected Stage 2 admission set and its upstream identities are hashed, S1.7 opens the sealed design-test targets once. Only the already selected configurations are evaluated on design-test. Those results may narrow the claim, but they do not start another tuning round or replace a selected configuration.

For one target item, let:

- `g` be the number of eligible candidates with score strictly greater than the target; and
- `e` be the size of the exact score-tied block containing the target.

Under uniform random ordering inside that exact tied block,

$$
\mathbb E[\operatorname{Recall@}K]
=\frac{\min\{\max(K-g,0),e\}}{e},
$$

and

$$
\mathbb E[\operatorname{NDCG@}K]
=\frac{1}{e}
\sum_{r=g+1}^{\min(g+e,K)}
\frac{1}{\log_2(r+1)}.
$$

Score equality is exact floating-value equality. A numerical tolerance must not merge different score levels. This is especially important for popularity and zero-vector fallbacks, which can create large tied blocks.

For validation, the candidate mask removes training positives and the other held-out positive while leaving the validation target eligible. For design-test, it removes training positives and the validation positive while leaving the test target eligible.

The primary evaluation uses the complete frozen warm catalogue. Score at most 128 users and 4,096 items per configured scoring block, while also enforcing the 64 MiB byte limit. Smaller blocks are allowed when the scorer or masks require them. Sampled negatives may be used during training or as a clearly secondary speed diagnostic, but not for the leaderboard.

Store one row per evaluated user, model, seed, split, and metric. The row must include the user ID or a bound row hash so paired comparisons cannot accidentally combine different user orders.

Report:

- Recall@10 and Recall@20;
- NDCG@10 and NDCG@20;
- expected target rank;
- expected catalogue coverage at 20;
- expected top-one-percent concentration at 20;
- user activity segment;
- target item support segment;
- held-out target played versus owned-but-unplayed status;
- metadata coverage;
- seed-wise results; and
- runtime and peak memory.

All segment labels must come from training-only information. User activity is the number of that user's design-training warm positives. Target support is the target item's design-training ownership count. The planned top-one-percent popularity set is the first 68 items after sorting the 6,721 warm items by descending design-training count and then ascending canonical item ID. Because the current JSON names this diagnostic but does not encode its boundary tie rule, this clarification must be copied into the versioned evaluator configuration or manifest before validation metrics are opened. These definitions and any displayed bin edges are fixed before metric comparisons.

Played status is loaded only from the reserved original-playtime diagnostic for the held-out edge at the stage when that diagnostic is permitted. It is a post-ranking subgroup label. It cannot affect fitting, configuration selection, admission, masking, or the primary leaderboard.

At a top-K score boundary, assign every tied item its exact fractional inclusion probability. Use those probabilities for expected exposure, expected catalogue coverage, and concentration. A stable hash may render an example recommendation list, but it does not determine a reported metric.

Use a 95 percent paired percentile user bootstrap with 2,000 replicates and seed 314159 for personalized-versus-popularity and genre-versus-identity differences. For an admission contrast, first calculate each user's paired metric difference for each of the three seeds, then average that user's differences across seeds. Bootstrap the 5,000 user-level averages, not 15,000 user-seed rows. Report every seed-specific contrast and the seed range separately. This interval measures conditional user-sampling variation and does not treat training seeds as independent users.

Completion note: `src/ranking.py` and `tests/test_ranking.py` now provide the tested primitives, and
the cycle-scoped runners integrate them with model, item, user, mask, and row-alignment checks. The
v2 public files still need to be deliberately committed before repository release.

### 10.4 S1.8: genre and pseudo-cold questions

The warm-item controlled question is simple: does adding the frozen genre block change ranking when the training setup is otherwise the same?

Report the paired aggregate difference and the item-support segments even if the result is zero or negative. Do not add new features until something wins. A genre result is predictive and descriptive, not causal.

For the pseudo-cold experiment:

1. use the already frozen 300-item cohort;
2. remove all design interactions for those items from collaborative training;
3. retain their genre rows;
4. require an evaluable warm history for each user;
5. suppress the cold item's identity embedding and item bias;
6. derive candidate representations from content only;
7. rank each cold positive against the complete 300-item cold catalogue;
8. mask the user's other known cold positives; and
9. mark identity-only BPR as unavailable and use only the frozen nonpersonalized fallback, design-training item support measured before cold removal.

Never leave a random or untrained identity vector in the cold comparison. The cold result must demonstrate that no collaborative identity information reaches the held-out item.

Historical execution note: at this point in the protocol the cohort existed, but cold-training
removal and scoring had not yet been performed. Section 10.9 records the completed diagnostic.

### 10.5 S1.9: Gate 1 closeout and downstream claim freeze

S1.9 records and audits the admission set that was already selected from validation and hashed at the end of S1.6, before design-test access. It does not recalculate admission after seeing design-test or pseudo-cold results. Those later results affect the strength of the claim attached to an admitted model, not which configuration replaces it.

For a personalized model to be noninferior to popularity:

- the lower 95 percent paired NDCG@20 difference bound must be at least -0.005; and
- the mean Recall@20 difference must be at least -0.01.

For genre to be noninferior to identity-only:

- the lower 95 percent paired NDCG@20 difference bound must be at least -0.005; and
- the mean Recall@20 difference must be at least -0.01.

At most three personalized families enter Stage 2. The admission manifest records the validation-selected set and its upstream identities before the design test is opened. The design test estimates generalization and may narrow the claim. It cannot be used to replace the winner with another configuration from the same cycle.

Hybrid victory is not required. If genre loses, report the negative result and continue with the qualified identity model. If no personalized family qualifies, retain the best validation personalized model only under the configured narrowed methodological label and avoid describing its downstream score panel as strongly validated preference.

Gate 1 requires:

- all required ladder models;
- all three frozen seeds finite for every valid stochastic specification, with popularity evaluated once;
- correct masks and exact-tie metrics;
- complete validation selection logs;
- a pre-test admission hash;
- one-time design-test results;
- warm and pseudo-cold conclusions at their correct strength; and
- reproducible model artifacts and scorers.

### 10.6 S1.10: production refit and assessment fold-in

The superseded pre-correction instruction expected 1,758,852 warm production edges. In the live v2
execution, the corresponding restored design count is 4,051,868, as recorded in section 10.9. In
both cases the binding rule is the same: preserve every Gate 1 artifact, restore the design
holdouts, do not carry a temporary pseudo-cold removal into production, and do not recompute Gate 1
metrics from the production fit.

Freeze the production shared parameters before assessment personalization.

For ALS with frozen item factors `Q`, fold in assessment user `u` by solving

$$
x_u
=\left(Q^\top C_uQ+\lambda_x I\right)^{-1}
Q^\top C_uo_u.
$$

For pairwise MF, freeze item identity factors, genre factors, and item biases. Solve only

$$
\min_{x_u}
\sum_{(i,j)\in\mathcal T_u}
\log\left(
1+\exp\{-x_u^\top(q_i-q_j)-(b_i-b_j)\}
\right)
+\lambda_x\|x_u\|_2^2.
$$

With positive regularization this user-only problem is strictly convex. The frozen solver is L-BFGS-B with tolerance `1e-8`, maximum 250 iterations, and zero-vector initialization.

The current frozen configuration does not yet fully define `T_u`. S1.5 must fix its positive-history source, negative candidate catalogue, rejection rule, pairs or negatives per positive, sample weights, deterministic seed namespace or exact deterministic construction, and objective scaling before any assessment identifier or history is opened. S1.10 then binds that rule into the production and fold-in manifests. It cannot be chosen after inspecting assessment scores or objectives.

If an assessment ranking positive is reserved, remove it from that user's fold-in history. A user with insufficient history receives the frozen zero-user-vector fallback and is counted. Do not drop users because their outcome is inconvenient.

Verification must prove byte or hash equality of every shared parameter before and after fold-in. Scores must reproduce in a clean process.

### 10.7 S1.11: pseudo-utility scenarios and Gate 2

Notebook 09 will be rewritten around the identification failure and the new interface. The failed monetary anchors stay as evidence. Dollar valuation language, CMM moments, and `valuation_calibration.csv` leave the live dependency graph.

For every admitted model and seed, construct a small frozen set of nonnegative transformations. The current plan names four candidate families:

1. a common positive shift and scale with parameters fitted globally on design users;
2. a logistic or softplus mapping with global robust location and scale;
3. a within-user percentile mapping over the exact ordered catalogue named in the scenario manifest; and
4. positive-part user standardization.

Historical execution note: these names were not yet fully specified at this point. S1.11 then froze
the exact equations, estimators, tie convention, clipping bounds, and fallbacks in the cycle-scoped
scenario configuration before any bundle objective was inspected.

For the current warm production models, the default Stage 2 eligible universe is the frozen ordered 6,721-item warm map. The full 10,978-item metadata map is retained for alignment, but its 4,257 nonwarm rows do not automatically enter a core Stage 2 pool. The 300 pseudo-cold items are a separate evaluation cohort and also do not automatically enter Stage 2. A future genre-only nonwarm scenario needs its own prospective transformation, scoring, and eligibility contract.

Every candidate pool is intersected with the scenario's eligible item map, and all exclusions are counted. Cross-model comparisons either use the same ordered item universe or are recorded as different Stage 2 instances. A user-specific rule may use that user's complete score vector over the named scenario universe. It may not use one candidate pool, an observed bundle outcome, or an assessment objective to set its transformation parameters.

For each transformation, record:

- its domain and range;
- proof or test of nonnegativity and finiteness;
- whether it strictly preserves, weakly preserves, or changes score ties;
- whether its parameters are global or user-specific;
- the design data used to fit global parameters;
- the interpersonal location and scale assumption;
- behavior for missing or constant score vectors;
- treatment of unseen users and items;
- common-scaling behavior; and
- source model, seed, split, catalogue, code, and parameter hashes.

Optional review-label and playtime checks are only auxiliary face validity. They cannot become a new supervised target or independent ground truth.

Planned outputs are:

- `src/pseudo_utility.py`;
- `tests/test_pseudo_utility.py`;
- `configs/cycles/s1-v2-20260814/pseudo_utility_scenarios.json`;
- `outputs/tables/09_price_anchor_diagnostics.csv`;
- `outputs/tables/09_pseudo_utility_diagnostics.csv`; and
- `outputs/modeling/cycles/s1-v2-20260814/pseudo_utility_scenarios_manifest.json`.

Gate 2 passes when every admitted model-seed-transformation scenario is deterministic, finite, nonnegative, fully identified by hashes, and frozen before any bundle objective is inspected. No scenario is described as the true utility scale.

### 10.8 S1.12: Stage 1 evidence package and stop rule

The final Stage 1 package must contain:

- protocol, interaction, split, feature, estimator, training, ranking, admission, production-refit, fold-in, and pseudo-utility manifests;
- the complete model leaderboard;
- paired validation and design-test contrasts;
- warm support and user-activity segments;
- pseudo-cold results;
- runtime and memory tables;
- per-user metric rows with verified alignment;
- production model artifacts and a bounded scorer smoke test;
- a mathematical appendix for ALS, BPR, low-rank scoring, and convex fold-in; and
- a short claim ledger stating what each result can support.

Once the required identity and genre comparison and pseudo-utility scenarios are complete, stop expanding the recommender. Tags, review-text NLP, LightGCN, transformers, graph recommenders, and a larger model zoo cannot delay Stage 2.

### 10.9 Stage 1 completion record: `s1-v2-20260814`

The identity correction was made prospectively, before fitting. Every raw record has a valid numeric
`steam_id`; using it retains 70,912 active users, all 5,094,082 deduplicated ownership edges, and all
10,978 games. Duplicate playtime was rebuilt from raw input with the declared fieldwise-maximum rule.
The protected split contains 50,351 design users and 12,585 assessment users after excluding 7,976
low-activity users. It has 3,951,166 warm training edges, 50,351 validation positives, 50,351 sealed
design-test positives, 8,902 warm items, a fixed 5,000-user evaluation sample, and a 300-item
pseudo-cold cohort.

The S1.5 spike passed 12 numerical, serialization, reload, scorer, and fold-in checks. Native
`implicit` 0.7.2 was accepted for ALS. LightFM 1.17 could not be installed in the frozen Windows
environment, so a hashed prospective amendment activated the independently tested NumPy BPR route
before validation. The full S1.6 ledger contains 88 rows: popularity, 72 ALS fits, 12 identity-BPR
fits, and three identity-plus-genre BPR fits, across the frozen seeds.

Validation selected one configuration per family and froze admission before the design-test was
opened. Only `als__k064__reg0p05__ao20__ownership_only` passed admission. On the one-time design
test, the three ALS seeds average NDCG@20 0.206089 and Recall@20 0.4196; popularity scores 0.135350
and 0.2524. The paired ALS-minus-popularity NDCG@20 difference is 0.070739 with frozen 95% bootstrap
interval [0.062947, 0.078603]. Identity BPR averages NDCG@20 0.133170 and is not admitted. The
controlled genre extension averages 0.077396 and is not admitted. This is a predictive ablation,
not evidence that genres cause ownership.

The pseudo-cold diagnostic evaluates 30,077 positive edges over 300 temporarily withheld items. The
genre content-only model is available there but underperforms popularity; identity-only prediction is
correctly labeled unavailable. Gate 1 nevertheless passes because genre is not required to win and
the downstream admission set was already fixed from validation.

S1.10 refit all three admitted ALS seeds on 4,051,868 restored warm design edges. All 12,585
assessment users folded in successfully for every seed, with zero insufficient-history or
solver-stopped statuses and no mutation of shared item factors. S1.11 froze four deterministic,
nonnegative transformations over the 8,902-item production catalogue: global shifted robust scaling,
global robust softplus, within-user midrank percentile, and positive-part user standardization. Gate
2 passes before any Stage 2 bundle objective was opened. The transformations are scenarios, not
identified WTP or money.

S1.12 verifies 264 artifact references, writes the model card/evidence summary, aggregate and segment
tables, seed contrasts, runtime/resource evidence, a ranking figure, and the mathematical appendix.
The complete repository suite passes with 204 tests. `python -m src.stage1_pipeline` re-verifies the
entire dependency graph idempotently and returns `status: complete`.

## 11. The bridge between Stage 1 and Stage 2

Stage 1 gives a set of qualified score models. Stage 2 needs pseudo-utilities and catalogue-coherent candidate pools. Notebook 11 is the descriptive bridge that checks how raw behavior and model-derived dependence look inside those pools before optimization.

### 11.1 Frozen Stage 1 to Stage 2 interface

Each downstream scenario must have an ID that binds:

- the admitted model family and configuration;
- the training seed;
- the protocol, interaction, split, and feature set IDs;
- the production shared-parameter artifact;
- the fold-in rule and solver;
- the ordered eligible user and item maps;
- the pseudo-utility transformation and its parameters;
- the score and pseudo-utility dtypes;
- the score-block memory limit; and
- the source and artifact hashes.

Stage 2 asks for bounded user-by-pool blocks from this interface. It does not read an unversioned notebook variable or the legacy `preference_factors.npz` file.

### 11.2 Candidate-pool registry

The candidate-pool registry is shared by notebooks 11 and 12. Pool membership is frozen before looking at matched-control or bundle-design outcomes.

Candidate pools may come from:

- one publisher's catalogue;
- one developer's catalogue when publisher data are weak;
- one franchise or series;
- one compatible base-game and DLC family; or
- one documented co-promoted set.

The current `game_features.csv` does not contain all publisher, developer, franchise, or compatibility fields required here. `src/candidate_pools.py` must therefore extend the catalogue extraction from the raw Steam metadata and keep the relevant raw labels, normalized keys, missingness, and manual overrides auditable.

The registry must record:

- stable pool ID;
- pool type and construction rule;
- source entity or franchise key;
- item ID and canonical item position;
- inclusion or exclusion status;
- inclusion or exclusion reason;
- base-game and DLC compatibility fields;
- metadata and score coverage;
- observed Steam bundle links;
- possible capacity values;
- normalization and alias rules;
- manual overrides with reasons; and
- upstream file and code hashes.

Planned files are:

- `src/candidate_pools.py`;
- `tests/test_candidate_pools.py`;
- `outputs/tables/candidate_pool_definitions.csv`;
- `outputs/tables/candidate_pool_items.csv`; and
- `outputs/modeling/candidate_pool_manifest.json`.

Pools may overlap. Pool membership cannot depend on which items have favorable score correlation, pseudo-utility, or optimizer results. A pool is a metadata-based feasibility proxy. It is not proof that one legal seller controls every item.

### 11.3 Notebook 11 evidence layers

Every important bundle-structure comparison should be repeated at three levels:

1. Raw behavior: binary co-ownership, Jaccard, lift, popularity-adjusted association, and playtime co-engagement where available.
2. Identity-only scores: dependence from a model that did not receive genre.
3. Identity plus genre scores: the same analysis after genre features enter.

If a genre pattern appears only in the genre-aware score model, describe it as model-imposed or model-amplified. Agreement with raw behavior and identity-only scores is stronger descriptive evidence.

For each observed or matched bundle, calculate:

- size and active capacity;
- publisher, developer, franchise, and compatibility coherence;
- genre concentration and number of genres;
- popularity mean, dispersion, and head-tail balance;
- release-window dispersion;
- pairwise raw co-ownership association;
- pairwise score Pearson and rank dependence under named specifications;
- aggregate pseudo-utility dispersion only under a named transformation;
- reach, such as the share of users with a high score for at least one component; and
- metadata and ownership coverage.

The descriptive question is whether there is a frontier between coherence and diversification. Similar games may be easier to market jointly and more likely to appeal to the same user. Less positively dependent preferences may allow more aggregation and price discrimination. The notebook tests this tension without assuming which side wins.

### 11.4 Matched controls

Do not compare an observed bundle with arbitrary random games from the whole catalogue. Generate controls inside the same frozen pool and match or stratify on:

- publisher or developer;
- bundle size;
- item-popularity profile;
- release period;
- price availability;
- base-game and DLC compatibility; and
- metadata coverage.

Use several stored seeds and multiple controls per observed bundle. Save match balance before interpreting an outcome. If adequate controls cannot be found, mark the observed bundle unmatched rather than relaxing a constraint after seeing the result.

Planned notebook 11 outputs are:

- `outputs/tables/11_bundle_structure.csv`;
- `outputs/tables/11_matched_controls.csv`;
- `outputs/tables/11_match_balance.csv`; and
- `outputs/tables/11_bundle_structure_results.csv`.

### 11.5 Statistical analysis for notebook 11

The final comparison statistics are not frozen yet. Before opening observed-versus-control differences, choose and record no more than two primary statistics. Reasonable candidates are the matched difference in raw co-ownership association and the matched difference in score dependence, with identity-only and genre-aware results kept separate rather than pooled.

The analysis rules are:

- report balance and the number of usable matched sets before outcome differences;
- use a user bootstrap for quantities estimated from user score panels;
- use a matched-set bootstrap or a declared within-set permutation interval for observed-versus-control differences;
- do not call the permutation calculation randomization inference unless an exchangeable assignment mechanism is justified;
- keep raw, identity-only, genre-aware, rank-based, and named pseudo-utility results in separate columns;
- if many bundle-level hypotheses are displayed, use an explicit multiplicity adjustment or label them exploratory;
- save all seeds and the complete resampling specification; and
- describe results as associations because observed Steam bundles may have been promotional, completion, or catalogue-clearing products rather than optimized designs.

### 11.6 Gate 3

Gate 3 passes when:

- the candidate-pool registry is frozen without outcome-based membership;
- matching is reproducible;
- balance is adequate or its failure is reported;
- any pattern that appears only in the genre-aware model is labeled model-imposed or model-amplified and is not used to alter pool membership or claimed as independent raw evidence;
- raw, identity, genre, rank, and named pseudo-utility measures are not mixed together; and
- the language remains descriptive rather than causal or profit-optimal.

After Gate 3, the data, estimation, and descriptive bridge are ready for the first full report draft. The optimizer is still required to finish the project.

## 12. Stage 2 inputs, feasible sets, and behavioral assumptions

### 12.1 Scenario-specific pseudo-utility panel

Fix one admitted model, seed, transformation, and user sample. Suppress the scenario superscript and write the nonnegative pseudo-utility as `v_ui`.

The same Stage 2 code runs separately for every registered scenario. It must not combine raw numerical prices or objectives from transformations with incompatible units.

### 12.2 Design and assessment users

Let `U_D` be the design users and `U_A` the assessment users. In every design-sample objective below, write `U = |U_D|`. Assessment evaluation replaces this denominator with `|U_A|` and never reoptimizes the policy.

Metadata and compatibility rules determine candidate-pool membership before score or objective outcomes are opened. A preregistered design-coverage rule may decide whether that already frozen pool is eligible for one experiment, but design data cannot add or remove items to improve an outcome.

Design users determine:

- production shared preference parameters;
- global transformation parameters;
- preregistered experiment-eligibility checks that depend on design coverage;
- component prices;
- bundle composition;
- bundle price;
- exact-suite and heuristic settings; and
- every complete benchmark policy.

Assessment users are folded into frozen shared parameters from permitted histories. They evaluate complete frozen policies only.

Here `r_u(pi; v_u)` means user `u`'s realized normalized seller margin under the complete frozen policy `pi` and that user's scenario-specific pseudo-utility vector. Products outside the modeled pool stay at their frozen component prices under every compared policy, so their common margin is omitted from both optimization and the reported difference.

The main Stage 2 assessment estimand is

$$
\widehat\Delta_A
=\frac{1}{|U_A|}
\sum_{u\in U_A}
\left[
r_u(\pi_D^{SBA};v_u)
-r_u(\pi_D^{CP};v_u)
\right].
$$

This is a conditional normalized margin difference between two design-selected policies. It is not observed purchase validation or dollar revenue. A negative held-out difference stays negative. Choosing a new policy after seeing assessment outcomes would be oracle reselection.

### 12.3 Candidate pool and feasible family

Let

$$
N=\{1,\ldots,n\}
$$

be one frozen pool, and let `z_i` indicate whether item `i` is in the bundle. The feasible family is

$$
\mathcal F
=\{\varnothing\}
\cup
\left\{
B\subseteq N:
2\le |B|\le C,
Hz\le h
\right\}.
$$

`Hz <= h` may contain:

- maximum capacity;
- required or excluded items;
- mutually incompatible products;
- base-game and DLC rules;
- minimum metadata coverage;
- a frozen genre rule if one is included; and
- any pool-specific feasibility condition registered before outcomes.

Singletons are excluded because Proposition P5 shows that a singleton SBA offer cannot strictly improve over CP on the design sample. The same conclusion holds for SBR because a singleton SBR policy only reprices that one item while every other item stays at CP, and the original item price was already CP-optimal. The SBR corollary still needs an explicit test. The empty set remains feasible so the optimizer can correctly choose no bundle.

The modeled pool is the decision universe, not necessarily the whole 10,978-game catalogue. Products outside `N` remain available at the same frozen component prices and contribute the same constant to every policy, so the Stage 2 formulas omit that constant.

Exact search and every heuristic call one shared feasibility function. A notebook cannot apply a slightly different interpretation of the constraints.

Capacity is based on the observed bundle-size distribution and nearby registered sensitivities. The audit found 562 of 615 bundles with at most 12 components, but this does not mean 12 is automatically the correct capacity for every pool.

### 12.4 Costs and timing

Let `c_i >= 0` be the normalized pseudo-cost and

$$
c(B)=\sum_{i\in B}c_i.
$$

The primary scenario sets all `c_i = 0`, reflecting low digital marginal cost. The implementation still accepts nonnegative costs, and one small positive-cost sensitivity may be registered. These are assumed costs in scenario units, not accounting estimates.

The primary interpretation treats each score vector as a synthetic pre-acquisition preference type. It does not literally ask an observed owner to buy the same games again.

An installed-base sensitivity is outside the frozen core. If added later through a dated amendment, first define

$$
v_{ui}^{new}
=v_{ui}\mathbf 1\{u\text{ does not own }i\},
$$

then recompute every CP price and every CP, PB, SBR, and SBA policy. This is still a one-price forward-offer scenario, not identified complete-the-set demand.

### 12.5 Behavioral assumptions

The core Stage 2 model assumes:

- additive pseudo-utility across games;
- quasi-linear utility in the normalized price;
- one common posted menu for all users;
- at most one unit of each item;
- no budget constraint;
- no search or exposure effect;
- no bundle-specific complementarity term;
- no strategic timing; and
- the declared component and bundle tie rules.

Genre may help prediction or pool construction, but a genre feature does not identify complementarity. Adding pairwise bundle interaction utility would change the customer-choice reduction, price theorem, and search problem. It is outside the core.

## 13. The four Stage 2 mechanisms

### 13.1 Component pricing

For item `i`, price `p`, and cost `c_i`, define the design-sample component objective

$$
\widehat r_i(p)
=(p-c_i)
\frac{1}{U}
\sum_{u=1}^U
\mathbf 1\{v_{ui}\ge p\}.
$$

Then

$$
p_i^{CP}
\in
\arg\max_{p\ge c_i}\widehat r_i(p),
$$

and

$$
\widehat\Pi_{CP}
=\sum_{i\in N}\widehat r_i(p_i^{CP}).
$$

The primary component tie rule is purchase at equality, `v_ui >= p_i`.

An empirical finite optimum with positive demand occurs at a distinct observed pseudo-utility at or above cost. Also include an explicit no-sale policy. No sale is a sentinel, not a very large number.

The CP routine returns every optimal observed-threshold candidate and an explicit no-sale representative when no sale is optimal. It does not try to enumerate an unbounded interval of zero-demand numerical prices. If at least one positive-demand observed threshold is optimal, the primary representative is the smallest such price. Key SBA results are repeated at the largest optimal observed-threshold candidate when one exists, and at the no-sale representative when it is also optimal. If no positive-demand threshold is optimal, the canonical policy is no sale. Equal CP profit does not imply equal SBA results because the truncated bundle values can change with the anchor.

### 13.2 Pure bundling

Pure bundling offers the whole pool as one grand bundle and does not offer its components separately.

$$
\widehat\Pi_{PB}
=\max_{b\ge c(N)}
(b-c(N))
\frac{1}{U}
\sum_u
\mathbf 1\left\{
\sum_{i\in N}v_{ui}\ge b
\right\}.
$$

Its exact finite-panel price is an observed total-pseudo-utility threshold or no sale. PB is calculated as a separate benchmark. If `N` is not feasible under the SBR capacity and constraints, SBR does not nest this PB benchmark.

### 13.3 Single Bundle with the Rest

Under SBR, items in `B` are removed from separate sale. Items outside `B` remain available at their CP prices.

$$
\widehat\Pi_{SBR}(B,b)
=(b-c(B))
\frac{1}{U}
\sum_u
\mathbf 1\left\{
\sum_{i\in B}v_{ui}\ge b
\right\}
+\sum_{i\notin B}
\widehat r_i(p_i^{CP}).
$$

For fixed `B`, the exact empirical SBR price is an observed raw-sum threshold or no sale.

Empty SBR equals CP. Grand-bundle SBR equals PB only when `N` belongs to the full feasible family. SBR can be evaluated as an empirical benchmark, but all imported hardness, half-purchase, covariance, tractability, and approximation results remain SBR-specific.

### 13.4 CP-anchored Single Bundle with All

CP-anchored SBA keeps every component available at its fixed CP price and adds one bundle `B` at price `b`.

The mechanism is the project's finite-panel implementation of the SBA model in Sun, Li, and Teo, Section 5, especially equation (17). The truncated customer comparison specializes their equations (13) and (14). The explicit weak-tie convention, empirical objective, finite-panel proofs, and tests below are project-specific. This attribution does not transfer any SBR theorem to SBA.

For one user and proposed bundle, the best separate-component surplus inside `B` is

$$
U_u^{sep}(B)
=\sum_{i\in B}(v_{ui}-p_i^{CP})_+.
$$

The bundle surplus is

$$
U_u^{bun}(B,b)
=\sum_{i\in B}v_{ui}-b.
$$

Products outside `B` cancel from the comparison because they remain available under both choices.

Define the truncated bundle threshold

$$
w_u(B)
=\sum_{i\in B}
\min\{v_{ui},p_i^{CP}\}.
$$

Under the primary bundle-preferred weak tie, user `u` takes the bundle exactly when

$$
w_u(B)\ge b.
$$

The proof uses

$$
v_{ui}-(v_{ui}-p_i^{CP})_+
=\min\{v_{ui},p_i^{CP}\}.
$$

The raw-sum condition used by SBR is wrong for SBA because it ignores the user's option to buy profitable components separately.

Define the CP margin displaced for user `u` when that user takes the bundle:

$$
A_u(B)
=\sum_{i\in B}
(p_i^{CP}-c_i)
\mathbf 1\{v_{ui}\ge p_i^{CP}\}.
$$

Let

$$
y_u(B,b)=\mathbf 1\{w_u(B)\ge b\}.
$$

The direct objective is

$$
\begin{aligned}
\widehat\Pi_{SBA}(B,b)
=\frac{1}{U}\sum_u
\Bigg[
&y_u(B,b)(b-c(B))\\
&+(1-y_u(B,b))A_u(B)\\
&+\sum_{i\notin B}
(p_i^{CP}-c_i)
\mathbf 1\{v_{ui}\ge p_i^{CP}\}
\Bigg].
\end{aligned}
$$

The equivalent incremental form is

$$
\boxed{
\widehat\Pi_{SBA}(B,b)
=\widehat\Pi_{CP}
+\frac{1}{U}
\sum_u
y_u(B,b)
\left[b-c(B)-A_u(B)\right].
}
$$

`A_u(B)` is essential. Without it, every bundle sale would be treated as new revenue even when it replaces profitable component purchases.

The main design problem is

$$
\max_{B\in\mathcal F,\ b\ge0}
\widehat\Pi_{SBA}(B,b).
$$

Empty `B` is the CP policy. Optimized SBA therefore weakly dominates CP on its own design sample. SBA does not contain PB because the component alternatives remain available. SBA is also not JSBC because the component prices are not jointly reoptimized.

### 13.5 No-sale sentinel and tie rules

The following conventions must be fixed in code and configuration before optimization outcomes are viewed.

1. Component purchase occurs at `v_ui >= p_i`.
2. CP scans complete equal-value blocks.
3. No sale is a distinct policy sentinel.
4. If item `i` uses the no-sale sentinel, set its SBA threshold contribution to `v_ui` and its displaced-margin contribution to zero by an explicit branch.
5. Never evaluate `infinity * 0`.
6. Bundle purchase occurs at `w_u(B) >= b` in the primary specification.
7. Fixed-composition pricing adds complete equal-threshold buyer blocks.
8. For SBA, if the best incremental gain is exactly zero, return no bundle.
9. For SBA, among equal positive-gain prices, choose the smallest bundle price.
10. Among equal composition objectives, choose no bundle on zero gain, then smaller cardinality, then lexicographic item ID.
11. Store all tied optima even though one canonical optimum is reported.
12. Use a homogeneous, scale-aware tolerance for objective equality and numerical zero only.
13. Never use a tolerance to merge different price thresholds.
14. For PB, choose no sale when maximum profit is zero; among equal positive-profit prices, choose the smallest observed total-value threshold.
15. For SBR, choose empty SBR, which is CP, when maximum incremental gain is zero; among equal positive-increment prices, choose the smallest observed raw-sum threshold.

A component-preferred strict bundle tie uses `w_u(B) > b`. In a continuous price domain its best value may be an unattained left limit. An executable sensitivity therefore needs a preregistered price tick and its own exact candidate and correctness statement. A theoretical supremum is not a frozen policy and cannot enter assessment evaluation.

## 14. Stage 2 theory and proof status

`notes/optimization_models.md` is the detailed theory reference. The current project-specific results are P1 to P8. They are already written and proved. P9 and one full reproduced SBR paper proof are still pending.

### 14.1 P1: empirical CP price candidates

For an item with positive demand, a finite-panel CP optimum occurs at a distinct observed pseudo-utility at or above cost. Raising a price to the smallest value among its current buyers preserves demand and weakly increases margin. The no-sale boundary is represented separately.

Implementation consequence: sort distinct values, add whole tied blocks, compare their objectives with no sale, return all optimal observed-threshold representatives, and keep no sale as a separate sentinel when it is optimal.

### 14.2 P2: SBA choice equivalence

Under the weak bundle tie, comparing the bundle with the best set of component purchases is equivalent to checking

$$
\sum_{i\in B}\min\{v_{ui},p_i^{CP}\}\ge b.
$$

Implementation consequence: the reduced threshold rule must agree with a direct evaluator that constructs the original menu alternatives.

### 14.3 P3: direct and incremental objective identity

The direct SBA objective equals CP plus the bundle contribution net of displaced component margin.

Implementation consequence: direct and incremental functions must agree on hand-built and randomized finite cases. This catches sign and cannibalization mistakes.

### 14.4 P4: exact price set for a fixed composition

For fixed nonempty `B` under weak ties, the optimal bundle price is either no bundle or one of the distinct observed thresholds `w_u(B)`.

Let

$$
t_1>t_2>\cdots>t_L
$$

be the distinct thresholds. After adding the complete block at `t_l`, define

$$
M_l=\text{number of current buyers}
$$

and

$$
H_l=\sum_{u:w_u(B)\ge t_l}A_u(B).
$$

The incremental candidate objective is

$$
\Delta_l
=\frac{M_l(t_l-c(B))-H_l}{U}.
$$

The exact scan costs

$$
O(U|B|+U\log U)
$$

time and `O(U)` working memory.

### 14.5 P5: singleton redundancy

A singleton SBA bundle cannot strictly improve over the component-pricing optimum on the same design sample. For singleton SBR, the bundle is just a replacement price for one component, so itemwise CP optimality gives the same redundancy conclusion. The live feasible family may therefore exclude bundles of size one for both mechanisms while retaining the empty CP policy.

### 14.6 P6: gain is not monotone or submodular

Let

$$
G(B)=\max\{0,\max_b\Delta_B(b)\}.
$$

The existing two-user counterexamples show that `G` is neither submodular nor monotone.

Consequences:

- ordinary monotone-submodular greedy guarantees do not apply;
- a greedy search starting from the empty set can see zero singleton gain and still miss a profitable pair;
- adding an item can reduce the optimized gain; and
- an at-most-capacity constraint cannot be replaced by an exactly-capacity constraint.

Local search remains a heuristic whose quality must be measured against exact cases.

### 14.7 P7: valid benchmark relationships

The valid relationships are:

1. Empty SBA equals CP.
2. Empty SBR equals CP.
3. Grand-bundle SBR equals PB only when the full pool is feasible.
4. SBA does not nest PB.
5. Optimized SBA and optimized SBR have no general ordering.

The theory note contains explicit finite examples in both directions for the last point.

### 14.8 P8: common positive scaling

If every pseudo-utility, pseudo-cost, component price, and bundle price is multiplied by the same positive constant, all choice indicators remain the same, every objective scales by that constant, and the set of optimal compositions is preserved.

This does not apply to:

- nonlinear transformations;
- user-specific normalization;
- utilities scaled while costs remain fixed; or
- a strict-tie price lattice that is not scaled.

### 14.9 P9: exhaustive enumeration correctness, pending

The remaining project proposition should state:

Under the weak-tie specification, if the finite feasible family is completely enumerated and every nonempty composition is priced exactly using P4, then the best returned policy is globally optimal for that declared finite-panel CP-anchored SBA instance.

The proof is short but must still be written formally. The feasible family is finite. P4 gives a complete candidate price set for every nonempty composition. The empty composition supplies CP. Comparing all of these policies under the frozen outer tie rule is therefore complete.

This statement does not cover a strict continuous-price policy or an incomplete enumeration.

### 14.10 Paper-result firewall

The following rules are binding:

- The paper's SBR NP-hardness result is not an SBA hardness proof.
- Exponential enumeration is not by itself an NP-hardness proof.
- The SBR half-purchase result is not an SBA demand restriction.
- SBR nesting of CP and, when feasible, PB does not transfer to SBA.
- The paper's SBR approximation results do not give an SBA approximation ratio.
- Low-rank recommender factors do not establish the paper's positive-diagonal-minus-fixed-rank covariance condition.
- A low-rank-plus-diagonal covariance model is not automatically the required diagonal-minus-low-rank structure.
- Nonlinear pseudo-utility transformations can change the covariance structure.
- Bayesian optimization with a finite budget returns the best evaluated solution, not a global certificate.

Maintain a theorem registry with:

- result ID;
- statement;
- mechanism;
- assumptions;
- provenance;
- proof status;
- code dependency;
- verification test; and
- permitted final claim.

One selected SBR theorem from *Partition and Prosper* must be reproduced in full after the project-specific core is complete. It remains clearly labeled as a reproduced SBR result.

## 15. Stage 2 algorithms and implementation plan

The live implementation belongs in `src/bundle_design.py`, with independent oracles and invariants in `tests/test_bundle_design.py`. Notebooks orchestrate the tested code and display results. They do not contain the only copy of the optimization logic.

### 15.1 Pool-sized precomputation

For one scenario and pool, precompute

$$
Q_{ui}=\min\{v_{ui},p_i^{CP}\}
$$

and

$$
R_{ui}
=(p_i^{CP}-c_i)
\mathbf 1\{v_{ui}\ge p_i^{CP}\}.
$$

Then

$$
w_u(B)=\sum_{i\in B}Q_{ui}
$$

and

$$
A_u(B)=\sum_{i\in B}R_{ui}.
$$

Materialize only the required user-by-pool blocks. Do not build or save a dense full-catalogue pseudo-utility matrix.

For a no-sale CP sentinel, branch to `Q_ui = v_ui` and `R_ui = 0`.

### 15.2 Reference functions

Implement pure, notebook-independent functions for:

1. exact empirical CP pricing;
2. exact empirical PB pricing;
3. a naive direct SBA menu evaluator from the original alternatives;
4. the reduced direct SBA objective;
5. the incremental SBA objective;
6. exact fixed-composition SBA pricing;
7. exact fixed-composition SBR pricing;
8. a direct SBR objective evaluator;
9. the shared feasibility predicate;
10. an intentionally simple tiny-instance enumerator;
11. full exact composition enumeration parameterized by SBA or SBR; and
12. mechanism-parameterized multistart local search.

The naive direct menu evaluator is an independent semantic oracle. It must not simply call the same `Q` and `R` reduction as the optimized evaluator.

### 15.3 Exact fixed-composition SBA price

For a proposed `B`:

1. construct `w_u(B)` and `A_u(B)`;
2. sort users by `w_u(B)` in descending order;
3. group exact equal thresholds;
4. add the complete tied block;
5. update buyer count and cumulative displaced margin;
6. evaluate the candidate gain at that threshold;
7. compare all thresholds with the zero-gain no-bundle option; and
8. apply the frozen price tie rule.

Do not evaluate partial tied blocks. Store buyer count, threshold, cumulative displaced margin, gain, and chosen status for diagnostic cases.

### 15.4 Exact composition search

The transparent reference algorithm takes a mechanism ID, enumerates every `B` in `F`, prices it with the matching exact SBA or SBR routine, and applies that mechanism's outer tie rule. SBA and SBR results have separate instance and certificate identities.

A simple worst-case work expression is

$$
\sum_{k=2}^{C}
{n\choose k}
\left[
Uk+U\log U
\right].
$$

This describes transparent enumeration work. It is not a complexity lower bound.

Only after the reference implementation passes may an accelerated depth-first search, cached sum update, or memoized evaluator be used. Every accelerated result must match the reference result on certifiable cases.

The exact search must count both expected and visited feasible compositions. A timeout or incomplete count is not a certificate.

### 15.5 Exact certificates

Every completed exact instance records:

- instance ID;
- mechanism ID;
- protocol, model, seed, transformation, pool, cost, constraint, and tie IDs;
- ordered item map and upstream hashes;
- number of design users, pool size, and capacity;
- expected and visited feasible-composition counts;
- candidate prices evaluated;
- termination status;
- CP baseline;
- best and runner-up objectives;
- canonical optimum and all tied optima;
- selected item IDs;
- normalized price and demand;
- displaced component margin;
- incremental gain;
- runtime and peak memory;
- hardware and software versions; and
- numerical diagnostics.

Only a completed enumeration of the declared feasible family with exact fixed-composition pricing is called a global finite-instance optimum.

Benchmark exact completion across `n`, `C`, and `U` to map the measured certified region. Do not assume or promise a universal 20 to 25 item cutoff. If an instance times out, keep it in the runtime record.

### 15.6 Scalable heuristic

The required scalable method is multistart add, drop, and swap local search with exact repricing after every proposed move. The search framework may be shared, but SBA and SBR call different objective and pricing functions, keep separate traces, and receive separate locked gap summaries. Passing the SBA heuristic gate does not validate SBR.

Required properties are:

- the same feasibility predicate as exact search;
- explicit inclusion of the no-bundle policy;
- deterministic tie handling;
- fixed random seeds;
- deduplicated starts;
- complete accepted and rejected move traces;
- exact repricing after each move;
- fixed objective-evaluation and runtime budgets; and
- best-solution-found language unless a certificate or bound exists.

The start library should include:

- observed Steam compositions where applicable;
- low-dependence pairs or sets;
- genre-diverse sets;
- popularity-balanced sets;
- high-reach sets; and
- random feasible sets.

Compare with equal-objective-evaluation-budget random feasible search and a simple greedy-add baseline. This shows whether multistart neighborhood search adds value.

If the SBR version does not have enough separately certified locked cases or fails its own frozen gate, report SBR only on exact pools. Any larger-pool SBR output is then labeled exploratory best-found and cannot be used as a validated comparison with SBA.

### 15.7 Development and locked exact suites

S2.0 first freezes pilot pool IDs and an outcome-independent rule for assigning later pool IDs to the development and locked suites. Pilot pool IDs are excluded from both later suites. S2.4 uses the pilots to measure the approximate exact-search frontier, then applies that frozen rule and hashes the actual development and locked instance IDs before heuristic tuning begins. Pool IDs cannot occur in both suites.

The exact engine writes locked-instance objectives, compositions, and certificates to sealed artifacts. The heuristic may know the registered inputs, but it cannot load those sealed results. A locked instance remains in the registry if exact search times out; it is not dropped because it is inconvenient.

The two roles are:

- a development exact suite; and
- a locked exact-validation suite.

Use the development suite for neighborhood choice, starts, restart count, stopping rule, cache design, deterministic ties, and budget selection. Freeze and hash these choices, run the heuristic on the locked inputs, and hash its outputs before opening locked exact objectives, compositions, or gaps.

The provisional diversity target is at least 20 certified locked instances from at least five pools, two capacities, two transformations, and two model-seed specifications for each mechanism that will receive a validated scalable-search claim. Timeouts remain visible but do not count as certified cases for percentile gap summaries. If the available exact inventory cannot support the target, report every case individually and avoid broad percentile claims.

For every locked case report:

- exact objective match;
- canonical composition match;
- recovery of the exact no-bundle decision;
- absolute objective gap;
- scale-normalized absolute gap;
- relative incremental-gain loss when well-defined;
- composition Jaccard overlap;
- normalized price difference;
- runtime;
- peak memory; and
- objective-evaluation count.

For a locked exact timeout, mark every exact-dependent comparison as unavailable and keep the runtime, search count, and termination status. Do not invent a zero gap or silently remove the row.

Define the homogeneous instance scale

$$
S^*=\max\{|\hat\Pi_{CP}|,|\hat\Pi^*|\}.
$$

If `S* = 0`, both CP and the exact optimum are zero. Label the instance `all_zero`, leave scale-normalized and relative ratios undefined, and test exact no-bundle recovery using absolute objectives only. All-zero cases remain in recovery counts but are excluded from ratio percentiles. Evaluate the nontrivial-gain condition only when `S* > 0`.

Do not hide a weak bundle search by dividing its gap by the large total SBA objective. Do not divide by incremental gain when the exact gain is zero or negligible.

The provisional gate, to be finalized using development cases only, is:

- recover every exact no-bundle decision;
- call a gain nontrivial when `(Pi* - Pi_CP) / S* >= 0.005`;
- on nontrivial cases, median relative incremental-gain loss at most 5 percent;
- 95th-percentile relative incremental-gain loss at most 20 percent;
- median absolute gap at most 0.0025 times `S*`;
- 95th-percentile absolute gap at most 0.01 times `S*`; and
- no unexplained catastrophic failures.

Calculate these summaries separately for SBA and SBR. Do not pool the two mechanisms to make one of their gates pass.

If the locked gate fails, restrict headline claims to exact pools or label larger-pool outputs exploratory. Improving the heuristic after locked-suite exposure requires a new versioned validation suite.

### 15.8 Required Stage 2 tests

At minimum, tests must cover:

- P2 choice equivalence on hand-built and randomized panels;
- direct versus incremental P3 equality;
- CP threshold scans versus an independent brute-force price oracle;
- PB and SBR price scans versus independent oracles;
- fixed-B SBA scans versus an independent brute-force price oracle;
- full SBR enumeration versus a separately written tiny oracle;
- exact equal-threshold block handling;
- no-sale and zero-demand cases;
- displaced profitable component sales;
- weak and strict tie variants;
- zero and positive costs;
- user and item reordering;
- the P5 SBA singleton case and its SBR repricing corollary;
- the P6 counterexamples;
- P7 benchmark relationships and nonrelationships;
- P8 common scaling;
- P9 exhaustive correctness on tiny cases;
- exact search versus a separately written enumerator;
- accelerated versus reference enumeration;
- exact and heuristic feasibility agreement;
- separate SBA and SBR heuristic-gap calculations without cross-mechanism pooling;
- correct no-bundle recovery;
- deterministic results under fixed seeds; and
- a guard that assessment identifiers cannot enter fitting or policy selection.

## 16. Stage 2 execution sequence

### 16.1 S2.0: preregister the optimization contract

Before opening locked exact-suite gaps, headline bundle outcomes, or assessment objectives, freeze:

- admitted Stage 1 model and seed IDs;
- production refit and fold-in manifests;
- pseudo-utility scenario IDs;
- eligible users and catalogue;
- candidate-pool registry and stable item order;
- one shared feasibility predicate;
- capacities and every active constraint;
- cost scenarios;
- CP representative rule;
- component purchase tie;
- bundle purchase tie;
- the strict-tie sensitivity's price tick and exact candidate-price rule;
- price and outer objective tie rules;
- numerical tolerances;
- exact pilot pool IDs and the frontier-measurement protocol;
- the outcome-independent rule and seed for assigning pool IDs to development and locked suites;
- sealed-output access rules for locked exact results;
- heuristic start library;
- heuristic tuning space;
- development selection rule;
- runtime and objective-evaluation budgets; and
- the registered headline and sensitivity grid.

Create content-derived instance IDs linking all of these choices. The primary exposition scenario must be chosen for a reason independent of which scenario gives the largest objective improvement.

Do not create an unnecessarily large Cartesian grid. Use a declared core grid, one-axis sensitivities, and a small full factorial on representative exact pools.

Planned configuration and registry files are:

- `configs/layer2_evaluation.json`;
- `configs/bundle_design.json`; and
- `outputs/tables/12_instance_registry.csv`.

S2.0 passes when every input and experiment identity predates the protected objective it governs.

### 16.2 S2.1: freeze the direct model and assumptions

Write the CP, PB, SBR, and CP-anchored SBA menus in the code, theory note, notebook, and report with the same notation.

Implement both:

- the direct customer-menu evaluator; and
- the truncated-value reduced evaluator.

Confirm that they agree on hand calculations before any headline pool is solved. Include no-sale sentinels, displaced component margin, zero and positive costs, and the complete tie rules.

S2.1 passes when the menu definitions, assumptions, code docstrings, and tests agree.

### 16.3 S2.2: finish proofs and provenance

Tasks are:

- audit P1 to P8 against their assumptions and tests;
- write and prove P9;
- finish the theorem registry;
- reproduce one selected SBR proof in full;
- check every imported paper statement against its actual mechanism; and
- create a claim-to-proof and claim-to-test index.

S2.2 passes when the selected SBR proof is reproduced and labeled, and no theorem or complexity statement is used outside its assumptions.

### 16.4 S2.3: exact primitives and independent oracles

Implement the reference API in `src/bundle_design.py` and an independent tiny oracle in the tests. Use synthetic panels first.

The pass condition is agreement among:

- direct and reduced customer choice;
- direct and incremental SBA objectives;
- threshold scans and independent price enumeration;
- exact composition search and a separately written tiny enumerator; and
- every required invariant in Section 15.8.

### 16.5 S2.4: exact enumeration and certified region

Run the preregistered pilot instances from the frozen pool registry first. Enumerate every feasible composition, reprice exactly, and issue one certificate per completed pilot.

Benchmark the effect of:

- number of users `U`;
- pool size `n`;
- capacity `C`;
- constraint density;
- number of price thresholds; and
- caching or reference implementation choice.

Use pilot runtime and completion evidence, not pilot objective values or selected compositions, to map the measured exact frontier. Apply the frozen S2.0 suite-assignment rule, hash nonoverlapping development and locked instance IDs, and only then run their exact jobs. Development results may be opened for heuristic work. Locked objectives, compositions, and certificates remain sealed until the heuristic and its locked outputs are frozen. Keep all timeouts and incomplete cases in the eventual frontier report.

S2.4 passes when composition counts reconcile, exact and independent implementations agree, and the certified region is based on measurements.

### 16.6 S2.5: freeze and validate the heuristic

Develop the multistart add, drop, and swap search on the development suite only. Freeze its starts, neighborhoods, improvement rule, seeds, restart count, stopping rule, budget, cache behavior, and tie handling.

Run the frozen heuristic on the registered locked inputs without loading their sealed exact results. Hash the heuristic outputs, then open the locked exact records once. Apply the frozen gate and report every required gap, composition, timeout, and computational measure.

If the gate fails, either:

- keep the final optimization claims inside the exact region; or
- show the larger results as exploratory best solutions found.

Do not retune on locked cases under the same validation version.

### 16.7 S2.6: design and frozen-policy assessment

For every registered instance, use design users to produce complete policies for:

- CP;
- PB;
- empirical SBR, using a certified exact optimum where enumeration completes and a separately validated SBR best solution found elsewhere;
- CP-anchored SBA, using a certified exact optimum where enumeration completes and the locked heuristic's best solution found elsewhere; and
- an observed Steam composition repriced under the CP-anchored SBA menu in that scenario, where coverage permits.

If the SBR heuristic has not passed its separate gate, the SBR bullet applies only to exact pools. An observed composition evaluated under SBR is a different benchmark policy and must receive a separate ID rather than being mixed with the SBA repricing.

Freeze:

- transformation;
- component price vector and no-sale sentinels;
- composition;
- bundle price;
- costs;
- constraints;
- tie rules;
- heuristic settings;
- certificate class; and
- policy hash.

Assessment users evaluate these frozen policies without reoptimization. This applies to every benchmark, including CP, PB, SBR, and the observed composition.

For each policy report:

- design objective;
- assessment objective;
- increment over the frozen CP policy;
- demand share;
- displaced component margin;
- selected items;
- capacity use;
- normalized price;
- design-to-assessment gap;
- exact or best-found status;
- runtime; and
- objective-evaluation count.

Use a paired assessment-user bootstrap for the primary SBA-minus-CP difference and other registered frozen-policy contrasts. Keep the policy fixed in every bootstrap replicate. This is conditional panel-resampling uncertainty. It does not include model estimation, transformation selection, or bundle-selection uncertainty.

Observed Steam dollar prices are not comparable with normalized pseudo-prices. Only the observed composition is transferred, and its scenario price is selected on design users.

### 16.8 S2.7: robustness and decision quality

Keep these sources of variation separate:

1. assessment-user sampling for a fixed policy;
2. design-user selection variation from resampling and reoptimizing;
3. model family;
4. training seed;
5. pseudo-utility transformation;
6. interpersonal normalization;
7. CP anchor tie;
8. bundle tie;
9. capacity;
10. pool definition;
11. cost scenario;
12. optional installed-base convention; and
13. dependence perturbation.

The required Gate 5 robustness grid is:

- every admitted model family;
- all three frozen training seeds for every valid stochastic specification;
- every declared pseudo-utility transformation;
- every optimal CP anchor representative required by Section 13.1;
- the primary weak bundle tie and the registered tick-based strict specification;
- the preregistered nearby capacities;
- every registered core pool definition; and
- every declared core cost scenario.

The paired assessment-user bootstrap for each frozen policy is also required. A different outer user split is optional and, if run, is a new versioned cycle. The installed-base convention, a copula model, robust max-min design, and broader dependence variants remain post-core options. A simple marginal-preserving permutation experiment is part of the recommended full project but is cut before the required grid if time is short.

The primary cycle uses the one frozen S1.2 outer split. A different outer split is a separately versioned sensitivity cycle, not a quiet resplit after seeing the result.

For cross-scenario comparisons, transfer composition only. Reprice that composition on the target scenario's design users, freeze the target price, and then evaluate it on the target assessment users. Never transfer a raw numerical price between transformations. Keep the mechanism fixed during the transfer: SBA compositions are repriced under SBA and SBR compositions under SBR.

On an exact target instance `t`, let `pi_D^{t,*}` be the certified target-scenario design optimum. Let `B_s` be a composition selected in source scenario `s`, and let `b_D^t(B_s)` be its exact target-scenario design reprice. Define

$$
S_D^t
=\max\left\{
|\widehat\Pi_{CP,D}^t|,
|\widehat\Pi_D^t(\pi_D^{t,*})|
\right\},
$$

and, when `S_D^t > 0`,

$$
\operatorname{Regret}_D^{s\to t}
=
\frac{
\widehat\Pi_D^t(\pi_D^{t,*})
-\widehat\Pi_D^t(B_s,b_D^t(B_s))
}{S_D^t}.
$$

This is defined inside the target scenario, so its numerator never mixes units across transformations. If `S_D^t = 0`, label the comparison all-zero and leave the normalized regret undefined. If the target policy is only heuristic best-found, report the analogous signed normalized reference gap and do not call it regret against a global optimum.

On assessment users, keep both policies frozen and report the signed target-scenario performance difference. It may be negative because assessment sampling can reverse the design ordering. Do not force it to be a nonnegative regret.

Report:

- item selection frequency;
- bundle-size and genre mix;
- price and demand dispersion;
- no-bundle frequency;
- pairwise composition Jaccard overlap;
- tied and near-optimal sets on exact pools;
- cross-scenario design regret; and
- frozen assessment performance.

Composition instability with negligible regret means that several designs are nearly equivalent. Instability with material regret means that the recommendation is fragile.

Prediction and decision quality are different. Compare each qualified model's NDCG and Recall with its downstream composition, regret, assessment performance, and stability. A ranking winner need not be a decision winner.

For a dependence experiment, preserve empirical item marginals while independently permuting item columns or using a separately declared copula. Use several permutations and a null distribution. Independence does not imply that bundling gain must be zero, and a permutation exercise is not causal evidence.

### 16.9 S2.8: Stage 2 evidence package and stop rule

Stage 2 stops when:

- P1 to P9 are linked to proofs and tests;
- the selected SBR proof is reproduced and labeled;
- the direct and reduced evaluators agree;
- the exact region is measured and certified;
- the locked heuristic benchmark is reported as a pass or failure for every mechanism claimed at scale;
- all headline policies are selected on design users;
- assessment evaluation is frozen and paired;
- required robustness axes are complete;
- every result row carries complete identities and certificate status;
- a clean run regenerates the headline outputs; and
- the claim ledger points from every conclusion to its table, figure, proof, and test.

Do not start joint component pricing, an advanced normal-model solver, robust max-min design, or another recommender while these requirements remain unfinished.

## 17. Stage 2 output contract

Use one canonical set of outputs rather than several files with overlapping meanings.

### 17.1 Pool and instance files

- `outputs/tables/candidate_pool_definitions.csv`;
- `outputs/tables/candidate_pool_items.csv`;
- `outputs/modeling/candidate_pool_manifest.json`;
- `outputs/tables/12_instance_registry.csv`; and
- `outputs/modeling/12_exact_suite_manifest.json`.

### 17.2 Optimization and certificate files

- `outputs/certificates/12_exact_certificates.jsonl`;
- `outputs/tables/12_algorithm_benchmark.csv`;
- `outputs/tables/12_bundle_design_results.csv`;
- `outputs/tables/12_selected_bundle_items.csv`;
- `outputs/tables/12_observed_bundle_comparison.csv`;
- `outputs/tables/12_frozen_policy_evaluation.csv`; and
- `outputs/traces/bundle_search/*.jsonl`.

`12_bundle_design_results.csv` contains design-selected policies by mechanism. `12_frozen_policy_evaluation.csv` contains those same policy IDs evaluated on both design and assessment users. This avoids using two files as competing sources of the selected policy.

### 17.3 Robustness files

- `outputs/tables/13_assessment_bootstrap.csv`;
- `outputs/tables/13_decision_stability.csv`;
- `outputs/tables/13_cross_scenario_regret.csv`;
- `outputs/tables/13_dependence_experiments.csv`;
- optional `outputs/tables/13_near_optimal_sets.csv`; and
- optional `outputs/tables/13_robust_design.csv` if the single stretch is activated.

`13_cross_scenario_regret.csv` records the mechanism, source and target scenario IDs, source composition, target-design reprice, target reference policy and certificate class, target scale, raw design difference, normalized exact regret or signed best-found reference gap, all-zero flag, and signed frozen assessment difference.

Every result row must contain or reference:

- instance ID;
- policy ID;
- mechanism ID;
- user split;
- model family and configuration;
- training seed;
- transformation;
- pool;
- constraint specification;
- capacity;
- cost scenario;
- CP representative rule;
- component-purchase tie rule;
- bundle-choice tie rule;
- mechanism-specific price tie rule;
- outer composition tie rule;
- strict-tie tick ID or `not_applicable`;
- numerical-tolerance specification;
- upstream hashes;
- code version; and
- certificate class.

## 18. Notebook and source-code map

Notebook numbers remain stable so that the research history stays visible.

| Notebook or module | Role in the final project | Current status |
| --- | --- | --- |
| Notebook 00 | Raw file inventory and proof that no user-bundle purchase file exists | Complete |
| Notebook 01 | Bundle pricing and item cleaning | Complete |
| Notebook 02 | Original flattened ownership panel | Complete |
| Notebook 03 | Ownership-based bundle demand proxies | Complete, with historical next-step prose |
| Notebook 04 | Minimum-cost overlap attribution | Complete, with minor rough notes to clean |
| Notebook 05 | Descriptive regression and bridge to the live direction | Complete |
| Notebook 06 | CMM synthetic validation | Archived in place |
| Notebook 07 | Base game-feature table feeding S1.3 | Existing; final version must also support pool metadata extraction |
| Notebook 08 | Legacy SVD/NMF prototype, retained separately from the source-module Stage 1 implementation | Explicitly legacy; not a live result |
| Notebook 09 | Failed monetary anchor and historical calibration record | Explicitly archived; live pseudo-utility interface is frozen in source/config |
| Notebook 10 | CMM on Steam-derived item sets | Archived in place |
| Notebook 11 | Candidate-pool descriptive bridge and matched controls | Planned |
| Notebook 12 | Exact and heuristic CP, PB, SBR, and SBA design experiments | Planned |
| Notebook 13 | Frozen-policy assessment, robustness, and decision quality | Planned |
| `src/valuation.py` | Legacy SVD, NMF, and optional ALS foundation | Retained prototype |
| `src/calibration.py` | Failed monetary-anchor functions | Archived |
| `src/bundle_pricing.py` | CMM bundle-size pricing | Archived |
| `src/mechanism_audit.py` | Reproducible static mechanism audit | Live and complete |
| `src/stage1_protocol.py` | Protocol validation and freeze | Live and complete |
| `src/interactions.py` and interaction artifacts | Canonical sparse interaction contract | Live and complete |
| `src/splits.py` and split artifacts | Protected deterministic splits | Live and complete |
| `src/features.py` and feature artifacts | Identity and genre feature contract | Live and complete |
| `src/preference_model.py` and `src/stage1_backend.py` | Backend-neutral equations, independent oracles, tested ALS/BPR implementations, serialization, and fold-in | Live and complete |
| `src/ranking.py` | Exact bounded-memory, tie-aware full-catalogue metric primitives | Live and complete |
| `src/stage1_validation.py`, `src/stage1_gate1.py`, and `src/stage1_production.py` | Frozen validation selection, sealed design-test gate, production refit, and assessment fold-in | Live and complete |
| `src/pseudo_utility.py` and `src/stage1_pseudo_utility_freeze.py` | Score-to-scenario interface and Gate 2 | Live and complete |
| `src/stage1_evidence.py` and `src/stage1_pipeline.py` | Evidence graph and idempotent Stage 1 verifier | Live and complete |
| `src/candidate_pools.py` | Frozen pool construction | Planned |
| `src/bundle_design.py` and `tests/test_bundle_design.py` | Live Stage 2 mechanisms, algorithms, and independent oracles | Planned |

The final notebooks should call tested source modules rather than reimplementing the same equations inside cells.

## 19. Gate summary

| Gate | Purpose | Pass condition | Status on 2026-08-14 |
| --- | --- | --- | --- |
| Gate 0 | Freeze mechanism, interpretation, core ladder, and claim boundaries | Mechanism memo, audit, archive map, and internal specification agree | Complete |
| Gate 1 | Establish credible Stage 1 ranking evidence | Frozen selection, exact metrics, design test, pseudo-cold result, model artifacts, and honest claim | Pass; implicit ALS only |
| Gate 2 | Freeze the Stage 1 to Stage 2 utility interface | Deterministic nonnegative scenarios with explicit assumptions and hashes | Pass; four scenarios, three seeds |
| Gate 3 | Freeze candidate pools and descriptive bridge | Outcome-independent pools, reproducible matching, balanced or qualified comparisons | Pending |
| Gate 4 | Establish optimizer correctness and computational validity | Proofs, exact oracles, certificates, measured frontier, and mechanism-specific locked heuristic results | Pending |
| Gate 5 | Freeze empirical conclusion | Assessment and robustness support the stated conclusion or narrow it | Pending |
| Gate 6 | Define the project as finished | Clean reproduction, report, appendix, presentation, and claim index | Pending |

### 19.1 Gate 1 does not require genre to win

A null or negative genre result is valid. The gate checks whether the comparison was controlled and whether at least one downstream score model has a defensible methodological label.

### 19.2 Gate 2 does not identify true utility

The gate checks whether the transformation assumptions are explicit and reproducible. It does not select one transformation as economically true.

### 19.3 Gate 4 distinguishes certificates from search results

Exact finite instances and heuristic large instances have different claim status. A failed heuristic gate is reported and narrows scope rather than being tuned away.

### 19.4 Gate 5 permits a negative held-out result

If a design-selected SBA policy does not beat frozen CP on assessment users, that is the result. The assessment set is not used to choose a replacement.

## 20. Reproducibility plan

### 20.1 Current non-artifact-mutating verification commands

From the repository root:

```powershell
python -m src.stage1_protocol --check-only
python -m src.stage1_interaction_artifacts --check-only
python -m src.stage1_split_artifacts --check-only
python -m src.stage1_feature_artifacts --check-only
python -m src.stage1_estimator_spec --check-only
python -m src.mechanism_audit --check-only --compare-to outputs/tables/bundle_mechanism_audit.csv
python -m pytest -q
```

These verify the current frozen research artifacts without intentionally rewriting them. Normal Python or pytest cache files may still be created by the environment, so this is not a claim of literal filesystem read-only execution.

### 20.2 Raw-data reproduction

The raw dataset is several gigabytes and is not tracked. A clean full-data reproduction must:

1. obtain the exact Kaggle archive;
2. place the expected files under `data/raw`;
3. record their SHA-256 hashes;
4. run notebook 01 and either retain the explicitly labeled legacy notebook-02 descriptive path
   or build a new versioned descriptive-v2 Steam-ID table;
5. run notebooks 03, 04, and 05 in dependency order against the declared descriptive version;
6. run notebook 07, or its final source-module replacement, to regenerate `game_features.csv`;
7. regenerate the mechanism audit;
8. regenerate the Stage 1 interaction, split, feature, and estimator artifacts;
9. run the cycle-scoped model and ranking pipeline in a new staging/rebuild cycle;
10. regenerate pseudo-utility and candidate-pool manifests;
11. run Stage 2 through the registered instances; and
12. compare all public outputs and manifests.

Some old notebook cells accumulate into dictionaries when rerun in the same kernel. Final reproduction uses Restart and Run All rather than selective cell reruns.

### 20.3 Environment

`requirements-frozen.txt` now records the audited direct package versions used for the completed
Stage 1 environment. It is not a complete transitive, OS, hardware, or BLAS lock. A new expensive
training cycle must additionally save:

- Python version;
- operating system;
- NumPy, SciPy, pandas, scikit-learn, and pytest versions;
- `implicit` and LightFM versions if used;
- BLAS library and thread settings;
- CVXPY and solver versions for archived checks or an optional stretch;
- CPU, RAM, and hardware identifiers; and
- a lock file or explicit frozen environment export.

The production run must not rely only on a developer machine's existing package state.

### 20.4 Synthetic smoke test

The final repository should include one command that needs no restricted or large data and demonstrates:

- a tiny interaction build;
- one small preference-model fit or oracle;
- bounded scoring;
- a pseudo-utility transformation;
- CP pricing;
- fixed-composition SBA pricing;
- exact enumeration; and
- a frozen-policy assessment calculation.

This smoke test is not evidence for the Steam result. It is a quick check that the complete data-to-decision interface is wired correctly.

### 20.5 Final reproducibility index

Create a table that maps every report claim to:

- source data;
- configuration;
- model or policy ID;
- proof if relevant;
- test;
- output table;
- figure;
- notebook section; and
- regeneration command.

## 21. Main risks and how they affect interpretation

### 21.1 Data risks

- There are no user-bundle purchases or rejected offers.
- Ownership does not reveal acquisition route, timing, exposure, or liking.
- Prices are a snapshot rather than the prices users faced.
- Only 238 bundles have complete ownership-panel coverage.
- The superseded v1 display-ID contract would have removed 56.4 percent of ownership rows. The
  live v2 Steam-ID correction retains all 5,094,082 deduplicated ownership edges; the remaining
  risk is historical-panel representativeness, not that row loss.
- The panel is Australian and historical.
- Playtime is noisy and selected after ownership.
- The catalogue and genre fields are incomplete.

These risks limit population and economic interpretation. They do not prevent a carefully labeled reconstruction and within-model optimization study.

### 21.2 Estimation risks

- ALS and BPR are jointly nonconvex.
- Even after all three seeds are run, their variation will not prove a global optimum.
- An unobserved item is not a negative or proof of exposure.
- Popularity creates large exact ties.
- LightFM may not match the frozen sampler, regularization, bias, or fold-in contracts.
- The total tuning budget may be tight for all ALS and pairwise runs.
- 1,154 warm items have zero genre content.
- The pseudo-cold cohort is selected and contains only 300 items.

### 21.3 Utility-interface risks

- Ranking performance only validates order.
- User-specific transformations impose interpersonal assumptions.
- Monotone transformations can change sums and bundle choices.
- No price-based monetary anchor passed its validity check.
- A model can rank well and still produce unstable bundle decisions.

These are reasons to report a scenario grid and decision stability rather than one preferred cardinalization.

### 21.4 Optimization risks

- CP anchoring is a modeling convention, not an audited Steam fact.
- Additivity and quasi-linearity are assumptions.
- Candidate pools do not establish legal control.
- Exact enumeration may become slow at modest `n` and `C`.
- SBA gain is not monotone or submodular.
- A local optimum has no automatic approximation guarantee.
- CP tie ambiguity can change the SBA bundle.
- Strict bundle ties can create unattained continuous-price suprema.
- An assessment loss can reverse the design result.

### 21.5 Engineering and governance risks

- Protected files are logically separated but physically present on the local machine.
- Reproducibility depends on exact upstream raw-data hashes.
- Direct dependency versions are frozen, but the transitive environment, OS, hardware, and BLAS
  stack are not fully locked.
- The ranking runner is complete, but the full v2 public evidence set remains untracked until the
  release commit is deliberately assembled.
- Public verification is now strict; several protected-stage cached runners still need stronger
  dependency revalidation in a prospective cycle.
- Archived and descriptive notebooks retain historical execution states; their role and known
  identity erratum must remain explicit.

The answer is explicit manifests, scoped loaders, no-clobber publication for new artifacts, clean-process tests, and a final notebook cleanup after the source pipeline is stable.

## 22. Fallbacks and scope control

### 22.1 Stage 1 fallbacks

- If `implicit` fails its contract, extend the small NumPy ALS oracle into a production implementation, test its objective, scoring, serialization, resource use, and fold-in path, then activate it through a prospective amendment.
- If LightFM fails its contract, implement the missing NumPy Adagrad feature-sum BPR training loop, test its loss, sampler, scoring, serialization, resource use, and fold-in path, then activate it through a prospective amendment.
- Do not replace the loss, sampler, or feature equation under the word equivalent without testing and documenting the change.
- If genre gives no lift, keep the negative result and use the qualified identity model as primary.
- If no personalized model qualifies against popularity, narrow the Stage 1 claim and treat Stage 2 as a methodological downstream exercise.
- If the tuning budget is infeasible, record the issue before outcome access and revise the cycle prospectively. Do not skip slow configurations after seeing partial metrics.

### 22.2 Stage 2 fallbacks

- If pool metadata are weak, restrict the registry to adequately documented pools.
- If exact search is slow, reduce the declared certified region using measured runtime and retain every timeout in the evidence.
- If too few locked exact cases exist, show every case and avoid percentile generalization.
- If the heuristic fails its locked gate, report exact-only headline results and exploratory large-pool results.
- If CP anchor ambiguity changes the bundle, report the ambiguity rather than choosing the favorable representative.
- If strict ties lack an attainable continuous optimum, use the preregistered tick or keep the calculation as a theory diagnostic.
- If cardinalization is unstable, report near-equivalent choices and regret rather than selecting the most favorable transform.
- If the normal SBR structure or solver cannot be verified, omit that stretch.

### 22.3 Minimum project that must be protected

The minimum complete project includes:

- the four-rung Stage 1 ladder;
- exact full-warm-catalogue evaluation;
- the controlled genre result;
- assessment fold-in;
- at least two clearly different pseudo-utility scenarios if the full four cannot be completed;
- the frozen candidate-pool registry;
- a focused notebook 11 bridge;
- correct CP-anchored SBA choice and objective;
- exact CP and fixed-composition SBA pricing;
- exact certification on small pools;
- one multistart heuristic checked against exact cases;
- frozen assessment evaluation;
- model, seed, transformation, capacity, and tie sensitivity;
- report, tests, manifests, and clean reproduction.

### 22.4 Cut first if time is short

Cut in this order:

1. full reproduction of additional paper proofs beyond one selected SBR proof;
2. tag features and any extra recommender family;
3. review-label auxiliary validation;
4. broad matched-control variants;
5. large-pool scaling beyond the required demonstration;
6. extensive dependence or copula experiments; and
7. every advanced solver or robust-design stretch.

Do not cut:

- the correct SBA mechanism;
- exact fixed-composition pricing;
- an independent direct oracle;
- small-instance certificates;
- a locked heuristic check;
- frozen assessment evaluation;
- cardinalization sensitivity; or
- the final write-up and reproducibility package.

### 22.5 Optional work after the core

After every required gate passes, choose at most one stretch:

- scenario-robust max-min SBA;
- jointly optimized SBA component prices;
- the normal-model SBR Bayesian-optimization and conic benchmark, after verifying every assumption; or
- a deeper dependence or copula experiment.

Tags are disabled in the current Stage 1 cycle. Adding tags later requires a prospective cycle and must not delay Stage 2.

## 23. Final report structure

The report should follow the actual research story.

### 23.1 Introduction

- Product bundling problem and Steam setting.
- What the static dataset contains and does not contain.
- Final two-stage research question.
- Summary of contributions and nonclaims.

### 23.2 Data and descriptive evidence

- Bundle and ownership tables.
- Coverage problem.
- Demand proxies and minimum-cost attribution.
- Descriptive regression.
- Mechanism audit.

### 23.3 The CMM attempt and pivot

- Original size-menu mechanism.
- Mathematical and numerical work completed.
- Monetary-anchor failure.
- Real-panel approximation result.
- Mechanism mismatch and reason for retirement.
- What the pivot demonstrates about model selection and institutional fit.

### 23.4 Stage 1 estimation

- Estimand and access boundaries.
- Canonical data and split contract.
- Popularity, ALS, identity BPR, and genre BPR.
- Exact ranking and uncertainty.
- Warm and pseudo-cold results.
- Production refit and assessment fold-in.
- Limits of latent scores.

### 23.5 Pseudo-utility interface and descriptive bridge

- Failed dollar anchor.
- Declared transformations and assumptions.
- Candidate pools.
- Observed versus matched bundle structure.
- Coherence and diversification evidence.

### 23.6 Stage 2 model and theory

- CP, PB, SBR, and CP-anchored SBA menus.
- Customer-choice reduction.
- Cannibalization-adjusted objective.
- P1 to P9 and theorem provenance.

### 23.7 Algorithms and computational evidence

- Exact component and bundle pricing.
- Exact composition search and certificates.
- Measured exact region.
- Heuristic design and locked gaps.

### 23.8 Decision results

- Design-selected policies.
- Frozen assessment evaluation.
- Prediction versus decision quality.
- Model, seed, transformation, pool, capacity, cost, and tie robustness.

### 23.9 Limitations and conclusion

- Identification limits.
- Data selection and coverage.
- Conditional meaning of normalized objectives.
- Exact versus best-found claims.
- Stable conclusion or clearly stated fragility.

The final report appendix should contain full proofs, the theorem registry, additional model diagnostics, exact certificates, heuristic traces, complete robustness tables, environment details, and the claim index.

## 24. Current position and immediate next actions

### 24.1 Completed

- Data inventory and descriptive notebooks 00 to 05.
- CMM work preserved as archive evidence.
- Mechanism audit and Gate 0.
- S1.0 protocol freeze.
- S1.1 canonical interactions.
- S1.2 protected outer and nested splits.
- S1.3 identity and genre features.
- S1.4 estimator equations and synthetic oracles.
- Prospective Steam-ID correction and regenerated S1.0--S1.4 v2 dependency chain.
- S1.5 backend, serialization, bounded scorer, and fold-in spike.
- S1.6 full validation ledger and pre-test admission hash.
- S1.7 exact one-time design-test evaluation.
- S1.8 controlled genre and pseudo-cold diagnostics.
- S1.9 Gate 1 closeout; implicit ALS only admitted.
- S1.10 three-seed production refits and assessment fold-in.
- S1.11 four pseudo-utility scenarios and Gate 2.
- S1.12 evidence package, 264-reference integrity verification, and 204-test suite.
- P1 to P8 in the Stage 2 theory note.

### 24.2 Not completed

- Candidate-pool registry.
- Notebook 11.
- Live Stage 2 implementation.
- P9.
- Exact certificates or heuristic evidence.
- Notebook 12 or 13.
- Frozen-policy assessment.

### 24.3 Immediate sequence

1. Freeze the outcome-independent candidate-pool registry.
2. Build notebook 11's descriptive dependence and matched-control bridge from the admitted ALS score
   interfaces without changing Stage 1.
3. Close Gate 3 and preregister Stage 2 instance, policy, tie, cost, and evaluation rules.
4. Implement and independently test the CP-anchored SBA mechanisms and exact certificates.
5. Validate any heuristic on a locked exact-solvable suite before larger-pool use.

Stage 1 is now read-only evidence. Any new recommender family, feature block, transformation, or
selection rule requires a new prospective cycle and cannot replace the completed result after seeing
Stage 2 outcomes.

## 25. Review checkpoints and evidence package

These are review points, not permission gates. Technical feedback can change later work only through a prospective, dated amendment made before the affected protected result is opened.

| Checkpoint | When | What I should bring | What should be settled |
| --- | --- | --- | --- |
| A: specification | Completed | Mechanism memo, architecture, SBA choice equation, model ladder, claims and nonclaims, backend-spike evidence | Archived Stage 1 design decision |
| B: preference evidence | After Gate 1 | Leakage tests, complete leaderboard, paired intervals, support segments, pseudo-cold result, identity-versus-genre ablation, admission manifest | Confirm the already frozen admission set and decide how strongly each admitted model can be described |
| C: bridge and theory | After Gate 3 | Pool registry, match balance, raw-versus-score dependence results, direct SBA formulation, P1 to P9 draft status | Whether pool constraints, descriptive claims, and theory provenance are adequate before headline optimization |
| D: optimizer validity | After Gate 4 | Unit-test summary, hand-worked menu example, certificate ledger, measured exact frontier, locked heuristic gaps and runtimes | Whether large-pool claims must remain exact-only or may include validated best-found results |
| E: full draft | After Gate 5 | Frozen assessment results, robustness tables, claim index, complete draft, proof appendix, archive narrative, reproduction checklist | Exposition and necessary robustness only, not a new model or a new policy selected from assessment results |

The final evidence package should include:

- one technical report telling the CMM-to-SBA research story;
- a proof appendix and theorem-provenance registry;
- a Stage 1 model card and a Stage 2 instance and certificate ledger;
- one main prediction table, one exact-versus-heuristic table, and one frozen assessment table;
- measured runtime and memory evidence;
- a synthetic end-to-end demonstration and the full-run manifest;
- a one-page claim-to-evidence index; and
- a short contribution note stating exactly what I derived, implemented, tested, and wrote.

The remaining work follows the gates: candidate pools and notebook 11, exact Stage 2 code and proofs,
locked heuristic validation, frozen assessment and robustness, then final reproduction and writing.
Once the submission deadline and review dates are known, each block needs a start date, internal stop
date, and cut decision. Stage 1 model expansion is closed.

## Appendix A. Archived CMM record

The CMM branch stays in the repository because it records substantial mathematical work and the reason the project changed direction.

Archived files include:

- `src/bundle_pricing.py`;
- `tests/test_bundle_pricing.py`;
- `src/calibration.py`;
- `tests/test_calibration.py`;
- `notebooks/06_bsp_synthetic_validation.ipynb`;
- `notebooks/09_valuation_calibration.ipynb` in its current historical form;
- `notebooks/10_bundle_size_pricing.ipynb`;
- `outputs/tables/valuation_calibration.csv`;
- `outputs/tables/bundle_size_pricing.csv`; and
- the CMM appendix in `notes/optimization_models.md`.

Notebook 06 and notebook 10 retain their original numbers and paths. Commit `3918b2b5afe88b88e8b8a6ce57533cc14d66d5a3` is the archive comparison baseline named in `notebooks/archive/README.md`.

Permitted archive claims are:

- the CMM implementation passed its stated synthetic and numerical cross-checks;
- the size-menu mechanism is different from fixed curated bundling;
- the attempted monetary anchor failed its sign tests;
- the real score-derived partial sums were much more skewed than the synthetic controls; and
- the CMM menu achieved 69 to 97 percent of the uncertified empirical menu search on the seven archived item sets.

Prohibited uses are:

- treating the legacy SVD result as the frozen Stage 1 winner;
- treating the affine normalization as a dollar calibration;
- treating differential evolution as a certified optimum;
- consuming a CMM output in the live SBA pipeline;
- transferring a CMM or SBR theorem to SBA; or
- presenting an archived objective as Steam revenue.

## Appendix B. Frozen Stage 1 identities

| Object | ID |
| --- | --- |
| Cycle | `s1-v2-20260814` |
| Source set | `9058bed498e13232c938820afdaf0f004a6bba895ebe1ebb541fac4ed8f397b9` |
| Protocol | `179e4861df905aaae8344104cb4fd598924f073ada3026b9ab22a8c739e7aafc` |
| Interaction set | `8e86e07e04c003d2fabff87432cc84ca26ee2da08e08c3e83888b094fca8e82a` |
| Split set | `6e326b169b3f7499cca66a52c168d2bb5ab978331c4d925fdd22e87fc4aa047f` |
| Feature set | `b05bd1856a65e8e4cb10805adf2e4aa01db7fe8d6a376b36c677af6c236b4244` |
| Estimator specification | `fa3795fc5e7c549375ddd9d9258004b11af24c4e30651c87711e88a36ad627a3` |
| Backend spike | `5392d8393b9e15ef3d05dccfd3c4d220668fedf6893c11fdf9d6b4d1e1095238` |
| Validation admission | `f535e24b11e3c0f707e9ee6e28d0d2b37c8cba7e0183eac9eaa2eacb2a1e16e0` |
| Gate 1 | `b95e89ac2bce59cd9630f59104f6cbd2bea6244d77093bf583c381cf37d7856e` |
| Production | `c8b76f330e382dc74a3b67361bc763cc4030f329d399e95e7926dd60b23e5ba1` |
| Gate 2 | `3c0714a65cc9807d9a6a3be5688b1f932b0dc83af3e83609536e99a95353f87b` |
| Evidence | `9c0d5b48059cfbecad0d0c9fd2da8a025dc57942104dd181e308d338b07b6650` |

These IDs form the current dependency chain:

```text
protocol
  -> interactions
  -> splits
  -> features
  -> estimator specification
  -> completed training, admission, ranking, and Gate 1 artifacts
  -> completed production, fold-in, pseudo-utility, Gate 2, and evidence artifacts
  -> future Stage 2 instances and policies
```

## Appendix C. Final definition of done

The project is complete only when all of the following are true.

### Research

- The CMM pivot is explained accurately.
- Stage 1 and Stage 2 answer the final research questions.
- Negative and unstable results are retained.
- Claims stay inside the data and model boundaries.

### Mathematics

- P1 to P9 are correct and linked to assumptions.
- The selected SBR proof is reproduced and labeled.
- Complexity and exactness claims match the implemented algorithms.

### Code

- The required source modules and tests exist.
- The complete suite passes in the frozen environment.
- Exact and independent oracles agree.
- Model and policy artifacts reproduce after reload.
- Protected access rules are enforced by the normal APIs.

### Evidence

- Stage 1 leaderboard, pseudo-cold result, and fold-in evidence exist.
- Pseudo-utility scenarios are frozen.
- Candidate pools and notebook 11 are complete.
- Exact certificates and heuristic gaps exist.
- Frozen assessment and robustness results exist.

### Submission

- A clean full-data run regenerates the headline evidence.
- The report and appendices are complete.
- The presentation uses the same claim boundaries.
- Every important claim is traceable to a proof, test, table, or figure.
