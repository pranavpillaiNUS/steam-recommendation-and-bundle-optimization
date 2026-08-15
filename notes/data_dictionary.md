# Data dictionary

Steam Video Game and Bundle Data, obtained from the
[Kaggle mirror](https://www.kaggle.com/datasets/pypiahmad/steam-video-game-and-bundle-data)
on 2026-06-05. The archive SHA-256 is
`fd23836a5450db0543b4a47b730da1373e20032f8c71d3971e3195392138de0d`.
The primary source is the [UCSD Steam dataset page](https://cseweb.ucsd.edu/~jmcauley/datasets.html#steam_data),
and the associated bundle paper is Pathak, Gupta, and McAuley (2017),
["Generating and Personalizing Bundle Recommendations on Steam"](https://doi.org/10.1145/3077136.3080724).

Kaggle's dataset API labels mirror version 1 as `Apache 2.0`, while the upstream UCSD page states
citation requirements but no license. The mirror uploader's authority over every upstream
component has not been established. Raw files, user-level artifacts, reviews, identifiers, and
fitted user parameters therefore remain untracked. The MIT license covers original project code
and documentation but does not grant rights to third-party data. Permission to redistribute the
tracked derived tables should be confirmed separately before public release. Raw files were
downloaded and unzipped into `data/raw`. All JSON files in this dataset are line-delimited
Python-style dicts (single quotes), so they are parsed with `ast.literal_eval`, not `json.load`.

## Raw files (data/raw)

| file | size | one record is | key fields |
|------|------|---------------|------------|
| bundle_data.json | 0.8 MB | One bundle | bundle_id, bundle_name, bundle_url, bundle_price, bundle_final_price, bundle_discount, items |
| australian_users_items.json | 527 MB | One user and their owned games | user_id, steam_id, user_url, items_count, items |
| australian_user_reviews.json | 25 MB | One user and their reviews | user_id, user_url, reviews |
| steam_games.json | 20 MB | One game | id, app_name, title, genres, tags, price, discount_price, publisher, developer, release_date, specs |
| steam_reviews/steam_new.json | 4.1 GB | One review | username, product_id, hours, products, date, text, early_access, page |
| archive.zip | 1.4 GB | The original zipped download | (not used directly) |

Notes:
- There is no file linking `user_id` to `bundle_id` in this archive. The source paper reports
  87,565 user-bundle interactions from a richer graph, but this project does not possess those
  labels. `users_own_all` is a newly constructed upper-bound ownership-compatibility proxy, not a
  recovered purchase label and not a reproduction of the paper's graph.
- Bundle prices are strings like "$73.86"; bundle_discount is a string like "10%".
- In bundle_data, each item's price field is `discounted_price` (the game's standalone store
  price), not a bundle-specific price.
- `bundle_price` equals the sum of item `discounted_price` exactly for all 615 bundles.

## bundle_data.json fields

| field | type | example | meaning |
|-------|------|---------|---------|
| bundle_id | str | "450" | Steam bundle id |
| bundle_name | str | "Dharker Studio 2015 Complete" | Display name |
| bundle_url | str | http://store.steampowered.com/bundle/450/ | Store link |
| bundle_price | str | "$73.86" | List price = sum of item standalone prices |
| bundle_final_price | str | "$66.46" | Price actually paid for the bundle |
| bundle_discount | str | "10%" | Reported discount percentage |
| items | list | [...] | The games in the bundle |
| items[].item_id | str | "326950" | Steam appid |
| items[].item_name | str | "Sword of Asumi" | Game name |
| items[].discounted_price | str | "$8.99" | Game standalone store price |
| items[].genre | str | "Adventure, Indie, RPG" | Comma-separated genres |
| items[].item_url | str | http://store.steampowered.com/app/326950 | Store link |

## australian_users_items.json fields

| field | type | meaning |
|-------|------|---------|
| user_id | str | Display alias; not a stable account key |
| steam_id | str | Stable numeric Steam account key used by live Stage 1 |
| items_count | int | Number of games the user owns |
| items[].item_id | str | Steam appid |
| items[].item_name | str | Game name |
| items[].playtime_forever | int | Total minutes played |
| items[].playtime_2weeks | int | Minutes played in the last two weeks |

## Derived tables (outputs/tables)

### bundle_df.csv (one row per bundle, 615 rows)
Built in 01_bundle_pricing_eda.

| column | meaning |
|--------|---------|
| bundle_id | Bundle id |
| bundle_name, bundle_url | Name and store link |
| price | bundle_price parsed to float (list price) |
| final_price | bundle_final_price parsed to float |
| discount | bundle_discount parsed to a 0-1 decimal |
| n_items | Number of items in the bundle |
| component_price_sum | Sum of item prices (equals price by construction) |
| abs_discount | component_price_sum - final_price |
| implied_discount_rate | abs_discount / component_price_sum |
| discount_mismatch | True if reported and implied discount differ by more than 5pp (0 cases) |

### bundle_items_df.csv (one row per bundle-item pair, 3525 rows)
Built in 01_bundle_pricing_eda.

| column | meaning |
|--------|---------|
| bundle_id | Bundle the item belongs to |
| item_id | Steam appid |
| item_name | Game name |
| item_price | discounted_price parsed to float |
| genre | Comma-separated genres (present on ~90% of items) |
| item_url | Store link |

### user_items_df.csv (one row per user-game pair; legacy descriptive artifact)
Built in 02_user_item_interactions using the original display-alias/keep-first rule. About 5.1M
rows, 70,912 users, and 10,978 games. Live Stage 1 does not use this identity contract: it uses
numeric `steam_id` and fieldwise-maximum duplicate consolidation in
`src/stage1_source_interactions.py`.

A 2026-08-14 audit found that the two paths have exactly the same 5,094,082 ownership edges.
Only 10 playtime rows change, by +217 lifetime minutes and +203 two-week minutes in total. All
ownership-only outputs in notebooks 03--05 and all live Stage 1 results are unaffected. A future
descriptive-v2 cycle should rebuild the legacy tables under new artifact names rather than silently
overwrite them.

| column | meaning |
|--------|---------|
| user_id | User identifier |
| item_id | Steam appid |
| item_name | Game name |
| playtime_forever | Total minutes played |
| playtime_2weeks | Minutes played in the last two weeks |

### bundle_demand_proxy.csv (one row per bundle, 615 rows)
Built in 03_bundle_demand_proxy.

| column | meaning |
|--------|---------|
| bundle_id | Bundle id |
| n_bundle_items | Number of items |
| items_in_panel | How many of the bundle's items appear in the ownership panel |
| panel_coverage | items_in_panel / n_bundle_items |
| users_own_all | Number of panel users who own every item in the bundle |
| users_own_any | Number of panel users who own at least one item |
| avg_playtime_overlap | Average total playtime of owned bundle games, among users who own any |
| n_items, final_price, implied_discount_rate | Merged from bundle_df |
| bundle_name | Merged from bundle_df |

Key caveat: users_own_all is only a clean demand signal when panel_coverage == 1.0 (238 of
615 bundles). See assumptions_and_limitations.md.

### bundle_mincost_attribution.csv (one row per bundle, 615 rows)
Built in 04_mincost_bundle_attribution. Attributes each user's ownership to the cheapest
reconstruction (bundles plus solo buys) at snapshot prices, to undo the overcounting in
users_own_all when bundles share games. A cost-based attribution proxy, not purchases.

| column | meaning |
|--------|---------|
| bundle_id, bundle_name, n_items, final_price, component_price_sum, implied_discount_rate | From bundle_df |
| panel_coverage | Merged from bundle_demand_proxy |
| candidate_ownership_count | Users who own every item in the bundle; equals nb03 users_own_all (the upper-bound compatibility count) |
| mincost_attributed_count | Credit after resolving overlap by min-cost reconstruction; fractional when optimal reconstructions tie |
| attribution_rate_given_candidate | mincost_attributed_count / candidate_ownership_count |
| candidate_users_with_overlap | Candidate users who sit in an overlap component of size >= 2 (the genuinely contested ones) |
| contested_candidate_share | candidate_users_with_overlap / candidate_ownership_count |
| attributed_revenue_proxy | final_price * mincost_attributed_count |
| soft_mincost_attributed_tau_0_5, _tau_1, _tau_2 | Soft-logit robustness at temperatures 0.5, 1, 2 dollars (sensitivity check, not behavioural) |

Key caveats: cost-based attribution under one price snapshot, not identification of
purchases; trusted on full-coverage bundles only; zero-discount bundles get their credit
split with the all-solo path. See assumptions_and_limitations.md (items 13-21).

### 05_descriptive_regressions.csv (one row per coefficient and specification)

Built in 05_initial_research_questions. Stores the complete HC3 coefficient table for the
primary overlap-adjusted demand proxy and the two declared proxy sensitivities.

| column | meaning |
|--------|---------|
| specification | Stable specification identifier |
| dependent_variable | Logged ownership-based proxy used on the left-hand side |
| proxy_label | Human-readable proxy description |
| term | Intercept or regressor name |
| coefficient | OLS coefficient estimate |
| hc3_std_error | HC3 heteroskedasticity-robust standard error |
| ci_95_lower, ci_95_upper | HC3 95% confidence-interval endpoints |
| p_value | Two-sided p-value computed from the HC3 result |
| nobs | Number of bundle observations |
| r_squared | Ordinary OLS $R^2$ for the specification |

These are descriptive cross-sectional associations using ownership-derived proxies. They are not
causal discount effects, demand elasticities, or inputs to the live optimizer.

### game_features.csv (one row per game, 10,978 rows)
Built in 07_game_features. Anchored on the distinct item_ids in user_items_df (the games the
latent-factor model can estimate), then left-joined with bundle membership (pre-aggregated to
one row per game) and Steam catalogue metadata. Missingness is retained, not imputed.

| column | meaning |
|--------|---------|
| item_id | Steam appid (the game universe is the panel's games) |
| ownership_count | Number of panel users who own the game (popularity) |
| playtime_mean, playtime_median, playtime_total | playtime_forever stats across owners (minutes) |
| nonzero_playtime_share | Share of owners with any recorded playtime |
| bundle_count | Number of distinct bundles containing the game |
| item_price_median | Median standalone snapshot price across the bundles containing it |
| item_price_nunique | Number of distinct snapshot prices seen across those bundles |
| price_disagreement | True if the snapshot price differs across bundles (item_price_nunique > 1) |
| genres, tags | Comma-separated Steam catalogue genres and tags |
| steam_price | Catalogue price from steam_games.json (string or number, kept raw) |
| release_date | Catalogue release date |
| in_bundle | bundle_count > 0 |
| has_steam_metadata | Genres or tags present from the catalogue |
| standalone_price | Bundle snapshot price if available, else the catalogue price (dollars) |
| has_price | standalone_price is present |

### bundle_mechanism_audit.csv (one row per observed bundle, 615 rows)

Built reproducibly from `bundle_data.json` and `steam_games.json` by
`src/mechanism_audit.py`. It records static evidence about whether bundle components are also
listed or priced separately in the snapshot. The companion
`bundle_mechanism_audit_manifest.json` records input, code, existing-artifact, canonical-order,
and identification-boundary hashes; the preserved CSV is verified cell-for-cell by bundle ID
without being reordered.

| column | meaning |
|--------|---------|
| bundle_id, bundle_name | Stable bundle identifier and display name |
| n_items | Number of recorded components |
| n_items_in_catalogue, catalogue_coverage | Components found in the individual-game catalogue and their share |
| standalone_price_coverage | Share of components with a recorded standalone-price field |
| n_distinct_publishers | Distinct normalized publishers among catalogue-matched components |
| publisher_coherent, developer_coherent | Whether matched nonmissing entities are coherent under the declared audit rule |
| n_missing_item_id | Components without a usable item identifier |
| mechanism_class | `B_SBA_like` or `E_unclear`; no row contains affirmative SBR evidence |
| ownership_adjusted, indivisible | `not_observable` because the snapshot lacks these fields |
| confidence | Static-evidence confidence label |
| evidence | Concise row-level explanation of the classification |

The audit supports an SBA-like component-availability description for 568 rows and marks 47 as
coverage-limited. It does not prove Steam's historical mechanism, identify CP-optimal prices, or
establish legal or commercial bundling authority.

### valuation_calibration.csv (archived, one row)

Built in the pre-pivot 09_valuation_calibration notebook. It records the attempted affine monetary
anchor and the diagnostics that reject that interpretation. It is retained as historical evidence
and is not a live pseudo-utility contract or an input to the fixed-bundle optimizer.

| column | meaning |
|--------|---------|
| method | The calibration used (normalization, since the price anchors fail the sign test) |
| a, b, sigma | The affine link v = a + b * score (dollars) and the implied noise scale |
| link | Link CDF for the censoring model (probit) |
| b_scale, a_shift | The sensitivity knobs swept downstream (scale multiplier, dollar location shift) |
| n_priced_games | Number of priced games used in the calibration cross-section |
| frac_floored | Fraction of calibrated per-item valuations hit by the free-disposal zero floor |
| anchor_probit_valid | Whether the aggregate censored-price anchor has the right (negative) price sign (False here) |
| anchor_probit_price_t | t-statistic of the price coefficient in the aggregate fit (positive = wrong sign) |
| anchor_quantile_valid | Whether the per-game quantile anchor has a positive price slope (False here) |
| anchor_quantile_slope_b | The price-on-score-quantile slope (negative = wrong sign) |
| anchor_quantile_spearman | Spearman correlation of price and the ownership-threshold score quantile |

Key caveat: the dollar scale is not identified from price because price is confounded with quality
and exposure. The archived transformation is an assumption that failed its intended identification
test; no live result may call its output willingness to pay, calibration, or actual revenue. See
assumptions_and_limitations.md.

### bundle_size_pricing.csv (archived, one row per item set, 7 rows)

Built in the pre-pivot 10_bundle_size_pricing notebook. The historical table contains six
full-coverage bundles (sizes 4 to 9) plus a curated top-8-owned set. Every policy was evaluated on
an ownership-derived, assumed transformed score panel. The table is neither model-free nor in an
identified monetary unit and supplies no input to the live SBA pipeline.

| column | meaning |
|--------|---------|
| set | Bundle id, or CURATED for the top-8-owned priced set |
| name | Bundle name |
| n | Number of games in the item set (the menu offers sizes 1..n) |
| sigma_cond | Condition number of the Ledoit-Wolf partial-sum covariance |
| cmm_model | CMM model value of its own optimal menu (what the two-moment model expects) |
| cmm_realised | Realized profit of the CMM menu on the panel |
| single_size_realised, single_best_size | Best one-size-only policy (Corollary 1): realized profit and the size |
| de_realised | Realized objective of the best empirical differential-evolution menu found; not a certificate |
| separate | Separate-selling benchmark (each game priced independently) |
| pure | Pure-bundling benchmark (one price for the grand bundle) |
| obs_price_realised | Grand bundle at the observed Steam price (scale-dependent sensitivity; NaN for CURATED) |
| bsp | Best realized size-price policy found (max of the four feasible policies) |
| bsp_over_separate, pure_over_separate | The headline scale-invariant ratios |
| cmm_real_over_de | CMM menu realized over the best empirical menu found in that archived run |
| cmm_model_gap | (cmm_model - cmm_realised) / cmm_model, the model's optimism about its own menu |

Key caveat: these are archived CMM diagnostics under a retired size-menu mechanism. The
two-moment approximation was evaluated against an uncertified empirical search on a model-derived
panel. No column is a live Steam pricing or revenue result. See assumptions_and_limitations.md
(items 32-34).

## Stage 1 v2 public evidence

Cycle directory: `outputs/modeling/cycles/s1-v2-20260814/`.

| Artifact | Public content and role |
| --- | --- |
| `stage1_protocol_manifest.json` | Frozen grids, seeds, metrics, resource rules, and config bindings |
| `stage1_source_manifest.json` | Steam-ID source contract, aggregate counts, and raw/protected hashes |
| `stage1_interaction_manifest.json` | Canonical CSR schemas, aggregate support counts, and semantic IDs |
| `stage1_split_manifest.json` | Split rules, aggregate cohort counts, access classes, and protected hashes |
| `item_feature_manifest.json` | Aligned identity/genre schemas and aggregate coverage |
| `stage1_estimator_spec_manifest.json` | ALS/BPR equations, parameter schemas, and numerical conventions |
| `stage1_training_manifest.json` | Complete 88-run ledger, selection, aggregate metrics, and artifact hashes |
| `stage1_validation_admission_manifest.json` | Validation-only admission decisions frozen before design-test access |
| `stage1_gate1_manifest.json` | Design-test, segment, pseudo-cold, and claim-boundary closeout |
| `stage1_production_manifest.json` | Three production refits, assessment fold-in diagnostics, and protected hashes |
| `pseudo_utility_scenarios_manifest.json` | Frozen transformation parameters and Gate 2 status |
| `stage1_evidence_manifest.json` | Top-level public evidence graph and validation/production run-log inventory |
| `stage1_*leaderboard.csv`, `stage1_*segments.csv` | Aggregate public ranking results; no user rows or identifiers |
| `validation_runs/*.json`, `production_runs/*.json` | Per-run aggregate logs with protected artifact references |

The public directory contains no user map, held-out coordinate, per-user metric row, dense score
matrix, or fitted factor array. Those objects are deliberately ignored under
`outputs/modeling/protected/` and can be checked only by an authorized full-local-state verifier.
