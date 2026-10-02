# Stage 1 scientific audit and improvement plan

Audit date: 2026-09-30. Frozen evidence cycle: `s1-v2-20260814`.

Stage 1 has a defensible completed result: the selected ownership-only ALS model improves
warm-item ownership reconstruction over popularity on the declared panel. It does not yet
establish that ALS is the strongest practical model for these data, that the comparison optimizes
every family adequately, or that its scores measure economic value. The most useful next work is
to strengthen the optimization and evaluation process, then run a separately identified successor
experiment when the exact private inputs are available.

This audit reads the public source, configurations, run logs, manifests, and aggregate tables.
The numeric summaries below are arithmetic aggregations of the published rows. They are not new
fits or a recomputation from user records. The current workspace lacks the exact raw and protected
artifacts needed for a Steam retraining run. New code has numerical and synthetic verification,
but **no improved Steam ranking score is claimed**. The frozen evidence and its admission decision
remain the historical result. See the [model card](stage1_model_card.md),
[release audit](stage1_release_audit.md), and [mathematical appendix](stage1_v2_mathematical_appendix_readable.md).

## What the completed experiment establishes

The source contains 70,912 active users and 5,094,082 distinct ownership edges. The outer split
uses users with at least five raw ownership edges: 50,351 design users and 12,585 assessment
users. The ranking experiment evaluates one held-out target per user for a fixed 5,000-user
design sample against the 8,902-item warm catalogue, after masking training positives and the
other holdout. Results average the three frozen training seeds where applicable.

| Family | NDCG@20 | Recall@20 | Catalogue coverage@20 | Top 1% item concentration@20 |
| --- | ---: | ---: | ---: | ---: |
| Popularity | 0.135350 | 0.252400 | 1.21% | 99.97% |
| Identity BPR | 0.133170 | 0.272533 | 4.41% | 94.60% |
| Identity + genre BPR | 0.077396 | 0.169667 | 5.46% | 69.87% |
| Ownership-only ALS | **0.206089** | **0.419600** | **19.63%** | **54.33%** |

Source: [design-test leaderboard](../outputs/modeling/cycles/s1-v2-20260814/stage1_design_test_leaderboard.csv).
Coverage is the expected fraction of catalogue items included across this evaluation panel.
Concentration is the fraction of recommendation exposure assigned to the 90 most popular training
items, the top 1 percent of the 8,902 warm items rounded up. Neither measure estimates sales, relevance outside observed ownership, or consumer welfare.

ALS gains 0.070739 NDCG@20 and 0.167200 Recall@20 over popularity. The recorded paired user
bootstrap interval for the NDCG difference is [0.062947, 0.078603]. It is conditional on the
snapshot, panel, fitted seeds, and protocol. Seed-specific differences range from 0.069517 to
0.072019. These are useful, distinct descriptions of uncertainty. Neither covers future users,
future games, a temporal shift, or arbitrary retraining randomness.

The BPR model has higher Recall@20 than popularity but slightly lower NDCG@20. That means the
top-20 hit rate and placement within the top 20 tell different stories. Its failed validation
admission remains correct under the declared primary metric and rule. Substituting Recall after
seeing this pattern would change the research question after the outcome was known.

## Design strengths worth retaining

1. Stable numeric Steam IDs and fieldwise maximum duplicate consolidation avoid ambiguous display
   names. Held-out edges are removed from both binary preference and playtime confidence.
2. Validation chooses configurations and the admission manifest is fixed before the design-test
   evaluation. The assessment partition is excluded from recommendation tuning.
3. Every candidate is scored in the warm catalogue. Exact ties use expected metrics, so popularity
   is not helped or harmed by arbitrary array order. The importance of distinguishing sampled
   metrics from full ranking is supported by Krichene and Rendle's
   [evaluation paper](https://arxiv.org/abs/1912.02263).
4. Stochastic comparisons use the same seeds and paired user contrasts. Users, rather than
   duplicated user-seed rows, are the unit of bootstrap resampling.
5. The objective, solver, confidence equation, masks, seeds, tie convention, and model provenance
   are explicit. The independent equation oracles are particularly useful for future optimization.
6. The public evidence verifier distinguishes integrity verification from model recomputation.
   The cycle's internal hash records are described accurately as internal records, without a claim
   of independent preregistration.
7. Four score transformations are frozen as pseudo-utility scenarios. The project already
   recognizes that ownership rankings do not identify willingness to pay.

## Where the result is strong and where it is weak

The activity segments use **warm training ownership counts**, computed after the two held-out
edges are removed. They are not the raw activity bands used to assign the outer user split.
Consequently, the `-1` segment below five training games is valid even though outer eligibility
requires five raw games. A successor report should name these two quantities separately.

| Warm training games | Users | Popularity NDCG@20 | ALS NDCG@20 | ALS minus popularity |
| --- | ---: | ---: | ---: | ---: |
| Fewer than 5 | 192 | 0.206338 | 0.257685 | +0.051346 |
| 5–9 | 406 | 0.220015 | 0.353097 | +0.133082 |
| 10–24 | 930 | 0.170317 | 0.289702 | +0.119385 |
| 25–49 | 1,131 | 0.159549 | 0.240312 | +0.080763 |
| 50–99 | 1,170 | 0.108081 | 0.173010 | +0.064929 |
| 100–199 | 795 | 0.076526 | 0.092896 | +0.016371 |
| At least 200 | 376 | 0.057640 | 0.053520 | −0.004120 |

Source: [design-test segments](../outputs/modeling/cycles/s1-v2-20260814/stage1_design_test_segments.csv).
These descriptive differences have no published segment confidence intervals. The final row
does not prove that ALS is worse for all heavy users. It is a reason to examine validation-defined
activity effects and a preregistered popularity/ALS mixture in future work, with tuning confined
to design validation. The highest activity group is hard for both models. Saying popularity
predicts these users well in absolute terms would overstate the observed NDCG of 0.0576.

The item support result is an even larger limitation:

| Target item training support | Targets | Popularity NDCG@20 | ALS NDCG@20 |
| --- | ---: | ---: | ---: |
| 5–19 | 9 | 0 | 0 |
| 20–99 | 59 | 0 | 0 |
| 100–499 | 350 | 0 | 0.013965 |
| At least 500 | 4,582 | 0.147698 | 0.223823 |

The last group supplies 91.64% of the test targets. Both models have no top-20 hits for the 68
targets with fewer than 100 training owners. The overall ALS gain is therefore largely evidence
about popular items, despite its much broader recommendation coverage. A future report should
give macro averages across declared support bands alongside the original user average, with
intervals and counts. Changing the primary metric to emphasize the tail would be a new protocol
decision, not a correction to the frozen result.

ALS also improves both played and unplayed target groups: mean NDCG is 0.195304 versus 0.131060
for played targets, and 0.230253 versus 0.144962 for unplayed targets. Playtime confidence losing
the validation comparison is consistent with an ownership reconstruction objective. It does not
show that playtime is irrelevant to engagement. Those are different targets and need separate
evaluations.

Genre is absent for 1,676 of 8,902 warm items (18.83%), and 1,051 of 5,000 test targets (21.02%)
have zero content rows. The feature vocabulary also includes software categories and “Free to
Play”. It is not a clean taxonomy of game tastes. A missing row supplies no content information
and should not be treated as evidence that genre predicts dislike. Keep coverage and vocabulary
audits in any content experiment. See the
[feature coverage table](../outputs/modeling/cycles/s1-v2-20260814/stage1_feature_coverage.csv).

## Model and optimization audit

### ALS: credible winner of a narrow grid

The selected configuration is 64 factors, ownership alpha 20, regularization 0.05, and 12 exact
alternating least-squares iterations. Those are the maximum factor count and the minimum alpha
and regularization in the grid. This makes an expanded search reasonable, but a boundary winner
is not evidence that continuing in that direction must help.

The best validation mean NDCG@20 is 0.202810. Keeping 64 factors and alpha 20 while changing
regularization from 0.05 to 0.2 yields 0.202768, a difference of only 0.000043. The evidence is
much stronger for ownership confidence and 64 versus 32 factors than for the precise winning
regularization. The best 32-factor ownership model scores 0.193754. The best 64-factor capped
playtime and log-playtime models score 0.188492 and 0.186165. All are validation results, not
new test comparisons. See the
[validation leaderboard](../outputs/modeling/cycles/s1-v2-20260814/stage1_validation_leaderboard.csv).

The winning seed 104729 still decreases its training objective from 0.028912 to 0.028829 during
the final iteration. This does not prove a useful ranking gain from more iterations. Record
validation learning curves at a small, declared set of checkpoints and choose the stopping rule
using validation. Tune factor count, alpha, regularization, and iteration budget together through
a bounded staged search instead of an unrestricted Cartesian expansion.

### BPR: correct declared gradient, insufficient evidence of competitive optimization

The frozen NumPy fallback accumulates the summed gradient over one million triples and then
takes **one** AdaGrad update. Twelve epochs therefore mean twelve updates. It does not implement
one update per triple or per minibatch. The mathematics and recorded implementation agree, so
this finding does not invalidate the archived result. It substantially limits a general
ALS-versus-BPR conclusion. Rendle and colleagues' original
[BPR paper](https://arxiv.org/abs/1205.2618) describes stochastic gradient training with sampled
triples. The frozen fallback uses a materially different update schedule.

For the selected BPR seed 104729, the sampled loss before the last two updates is approximately
182,320 then 175,935. The batches differ, and there is no fixed probe or validation learning
curve establishing convergence. A plateau has not been demonstrated. At initialization, the
item-bias gradient norm is about 18,223 while the user and item-factor norms are about 60.
Coordinate AdaGrad changes the effect of those magnitudes, so these numbers alone do not prove
a causal explanation for the popularity-heavy rankings.

The successor implementation in [stage1_training.py](../src/stage1_training.py) now provides
deterministic minibatch AdaGrad with an update after every batch, updates only touched user/item
rows, and logs update counts, training triple hashes, and a fixed training probe. Positives are
sampled uniformly from training edges. Negatives reject training positives only. Held-out labels
must not be supplied to improve the negative sampler. The independent probe uses a separate RNG
namespace and cannot change optimization triples.

Its objective is explicit: mean sampled logistic loss plus mean squared norms of the sampled
user, positive item, negative item, and both item biases, plus a global genre-factor penalty when
content is active. It differs from the frozen sum-loss/global-L2 equation. Old regularization
values must be retuned. No numeric comparison should silently treat these penalty conventions as
equivalent. The public entry point `minibatch_loss_and_gradients` makes this equation inspectable.

The new [trainer tests](../tests/test_stage1_training.py) passed 23 checks on Python 3.10.11:
unregularized gradients against the independent frozen oracle, regularized gradients against
finite differences, repeated-row accumulation, deterministic fitting and streams, training-only
negative rejection, partial-batch update counts, toy training-probe improvement, zero-content
equivalence, malformed inputs, and nonfinite updates. These establish implementation behavior
on synthetic inputs, not Steam accuracy or production-scale speed.

### Genre: preserve the controlled ablation, add a separate tuned benchmark

The frozen genre model inherits the identity model's selected factors, regularization, learning
rate, sampling stream, and epoch budget. This is a useful controlled toggle. It is not an
independently tuned estimate of the best content model. Adding shared feature parameters can
change the optimization dynamics and useful regularization, even with matched streams.

A successor should report two clearly named experiments: the same-hyperparameter genre toggle
for attribution, and a separately tuned genre model for achievable predictive accuracy. Permit a
small declared grid for feature weight and, if implemented and verified, separate feature
regularization. Do not add unverified modern tags to a historical experiment without documenting
their acquisition time and possible information leakage.

The existing pseudo-cold popularity comparator uses support from before cohort removal. It is
an oracle-like reference. A new item would not have that signal. Ranking occurs within a selected
300-item genre-covered cohort, not the complete real new-item candidate problem. Retain the
diagnostic label. A successor content experiment should compare against a uniform tied ranking
and a model using only information genuinely available after removal, and report item coverage
and user history eligibility. This remains a recommendation for a new experiment.

### Broaden baselines before adding a large model family

Use popularity, the reproduced ALS baseline, and an item-neighborhood model as the minimum
ladder. An optional EASE linear item model is a credible additional benchmark: Steck's
[original paper](https://arxiv.org/abs/1905.03375) gives a closed-form sparse-feedback model and
evaluates it against several recommendation approaches. Its result is motivation to test here,
not evidence of superiority on this Steam snapshot.

At 8,902 items a single dense float64 item-by-item array occupies 633,964,832 bytes, about 605 MiB.
A solve can require several such arrays plus workspaces. EASE therefore needs a declared memory
and time gate. It should not be added by assuming the Gram matrix is the whole memory cost.
Stronger simple baselines, a fair BPR optimizer, and a better ALS search have clearer immediate
value than starting an unbounded neural-model search.

## Evaluation and statistical improvements

The evaluation estimates reconstruction of randomly held-out ownership within this historical
snapshot. Absence is an unobserved interaction, not an observed dislike. Ownership may result
from gifts, bundles, promotions, and unequal exposure. Purchase timing is not recovered by a
random split. Temporal deployment claims require defensible event times or another dataset.

The capacity-aware split protects item training support and assigns test before validation.
Thus the held-out targets are conditioned on warm support and capacity constraints. Report the
result for that estimand and retain the split audit. In a successor, inspect validation/test
support and activity composition on design data to quantify this selection. A leave-one-out
metric also captures only one target per user. A separately specified multi-positive design
evaluation can test sensitivity without replacing the published metric after seeing results.

The bootstrap used for validation admission is computed after choosing the best configuration
on those validation users. It is an operational admission screen, not a selection-adjusted
confidence statement. The one-time design test provides a more appropriate performance report
for the frozen selected policy. For a successor, define the primary paired comparison and an
absolute improvement threshold before running selection. Report user uncertainty separately
from seed variability. Three seeds do not estimate a broad seed distribution precisely.

Use validation-selected finalists with all predeclared seeds, report every failed or timed-out
run, and avoid choosing a lucky seed. Reserve broader seed replication or an extra data split
for a declared final comparison. Keep secondary segments descriptive unless a multiplicity-aware
or hierarchical inferential plan is specified in advance. An ordinary user bootstrap need not
pretend to represent the stratified sampling design. A future population-directed interval
should preserve the activity strata and declare its target weights.

The current three-seed mean of ranking metrics is not the metric of a score-averaging ensemble.
If Stage 2 will consume averaged scores, evaluate that exact aggregation prospectively using
validation and document the order of seed averaging and nonlinear score transformation. They
are different operations. Any adaptive activity mixture or score ensemble is another policy to
freeze and evaluate, not a free consequence of averaging seed-level metrics.

## Honest successor experiment sequence

1. Preserve all v2 files and their hashes. Create a new cycle, a prospective config, an access
   ledger, and isolated output directories. Record that its choices are informed by this audit
   and the already opened v2 results.
2. Require the exact authorized raw and protected inputs before any Steam fit. Verify schema,
   IDs, row/column maps, masks, and train-only feature provenance. Report missing artifacts
   clearly. Do not manufacture replacement data or label a synthetic fit as Steam evidence.
3. Run the numerical and synthetic pipeline checks, reject stale/corrupt caches, and benchmark
   resource use on a declared design-only sample. The 88 frozen validation runs sum to about
   8,554 seconds, but that historical number is not a future runtime guarantee.
4. Reproduce the existing ALS configuration on the permitted development inputs as an anchor.
   Search bounded ALS factor/alpha/regularization/iteration settings. Use declared staged
   screening and finalist seed replication. Keep ownership-only confidence as the anchor.
5. Evaluate the new BPR optimization across a bounded batch-size/learning-rate/regularization/
   update-budget grid, with fixed probe curves and full-catalogue validation. Log total updates
   and accepted triples alongside epochs. Compare the controlled genre toggle and tuned genre
   benchmark separately. Add simple neighborhood and optional resource-gated linear baselines.
6. Select only on design validation. Report core ranking, support/activity segments, coverage,
   concentration, seed variability, runtime, peak memory, and archive size. A change must earn
   its additional cost or solve a declared robustness gap. A negligible validation difference
   is not compelling evidence of improvement.
7. Treat the old design test as historical evidence. It is already opened and cannot become a
   fresh confirmation after this audit guides the next model. Reusing its results for debugging
   or comparison must be labelled exploratory. A new hash namespace or a reshuffle of the same
   snapshot does not erase earlier access. Strong new confirmation requires genuinely unseen
   data. Any newly sealed design subset must disclose its previous training and inspection
   exposure and support only the appropriate internal robustness claim.
8. Keep Stage 2 assessment users out of Stage 1 optimization. The prior aggregate diagnostics on
   the first 128 assessment users are a documented peek, not an unused assessment set. Do not
   use that peek to tune score transformations, a new model, candidate pools, or bundle policies.
   A successor Stage 1 decision must be frozen before any further assessment access.
9. Refit and publish a successor production interface only after the declared selection and
   evidence review. Any replacement of the Stage 1 input to Stage 2 needs an explicit new model
   and scenario manifest, without changing the historical v2 admission claim.

## Priorities and completion criteria

| Priority | Work | Completion evidence |
| --- | --- | --- |
| P0 | Preserve evidence and outcome access boundaries | Frozen verifier passes, new work has separate source/output identity, no further assessment access |
| P0 | Make new runs fail clearly on absent/corrupt inputs | Readable preflight, strict dependency/hash checks, corruption tests |
| P1 | Repair practical BPR optimization | Minibatch implementation with equation checks and train diagnostics, full Steam validation still required |
| P1 | Expand ALS search and stopping choices within a budget | Prospective grid/checkpoints, validation ledger, finalist seed comparisons |
| P1 | Measure weak segments and simple alternatives | Training-support/activity definitions, counts, simple baseline comparison, coverage and concentration |
| P1 | Make runtime and memory claims enforceable | Separate fit/rank/serialization/total timings, peak memory and bounded score blocks |
| P2 | Strengthen content and cold diagnostics | Shared-parameter ablation plus tuned benchmark, valid post-removal comparators, metadata provenance |
| P2 | Strengthen generalization evidence | Declared evaluation target and genuinely unseen data or explicitly limited internal replication |

Stage 1 can be improved now at the level of implementation, diagnostics, reproducibility, and
prospective experiment design. Claiming a better recommender requires the missing private data,
the actual validation experiment, and a defensible evidence label. The existing ALS result is
the current measured winner until that work produces new evidence. The
[Stage 1 closeout](stage1_closeout.md) records which of these items were implemented, deferred, or
left as proposals. Stage 2 remains a blueprint.
Its execution and economic interpretation are governed by
my local working plan and the [execution blueprint](stage2_execution_blueprint.md).
