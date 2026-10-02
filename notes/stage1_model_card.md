# Stage 1 model card

## Status and intended use

Stage 1 is complete under cycle `s1-v2-20260814`. I compare implicit-feedback recommenders on
held-out Steam ownership and then freeze the score transformations needed by the planned
bundle-design stage.

I use the admitted model only as a research input to the declared Stage 2 scenarios. It is not a
production recommender and does not estimate willingness to pay, purchase probability, consumer
surplus, interpersonal welfare, or monetary revenue.

The cycle uses an internal, hash-bound workflow record. It was not externally preregistered or
independently timestamped before I produced the results. I will commit and tag the Stage 2 protocol
before opening any bundle objective so that its sequence can be checked from Git history.

## Data and evaluation population

| Quantity | Value |
| --- | ---: |
| Active users | 70,912 |
| Deduplicated ownership edges | 5,094,082 |
| Catalogue games | 10,978 |
| Warm evaluation games | 8,902 |
| Design users | 50,351 |
| Assessment users | 12,585 |
| Fixed evaluation sample | 5,000 design users |
| Training seeds | 3 |

The source is a historical Australian-user Steam snapshot. The evaluation uses different held-out
validation and design-test edges for the same fixed design-user sample. It measures warm-user,
full-catalogue interaction reconstruction, not temporal generalization, new-user performance, or
performance on a current global Steam population.

## Frozen model comparison

Validation selected one configuration per family. Only the ownership-only 64-factor implicit ALS
configuration (`regularization=0.05`, `alpha=20`) passed the predeclared admission rule.

| Model | Seeds | Design-test NDCG@20 | Design-test Recall@20 | Admitted |
| --- | ---: | ---: | ---: | :---: |
| Popularity | deterministic | 0.135350 | 0.2524 | baseline |
| Identity BPR | 3 | 0.133170 | 0.2725 | no |
| Identity + genre BPR | 3 | 0.077396 | 0.1697 | no |
| Ownership-only implicit ALS | 3 | **0.206089** | **0.4196** | **yes** |

For ALS versus popularity, the paired mean NDCG@20 difference is 0.070739. The frozen 95 percent
percentile user-bootstrap interval is [0.062947, 0.078603], and the three seed-specific differences
range from 0.069517 to 0.072019. This interval is conditional on the fixed snapshot, evaluation
users, trained seeds, and protocol. It is not population-level or training-seed uncertainty.

The negative genre result is an implementation- and protocol-specific predictive ablation. It does
not show that genre is generally unhelpful or causally reduces preference.

## Evaluation contract

- Training positives and the other held-out positive are masked for each target.
- Every target is ranked against the complete 8,902-item warm catalogue.
- Exact score ties receive expected Recall, NDCG, rank, coverage, and concentration contributions.
- Configuration selection uses validation only. The design test is opened after the admission
  manifest is frozen.
- Three fixed seeds are averaged for stochastic families. Paired inference resamples users, not
  user-seed rows.
- Production restores design holdouts, refits the admitted model, and folds in assessment users
  while keeping shared item parameters fixed.

## Pseudo-utility interface

Four deterministic, finite, nonnegative transformations are frozen in
`configs/cycles/s1-v2-20260814/pseudo_utility_scenarios.json`. They are explicit cardinalization
scenarios, not calibrations. A monotone transformation can preserve rankings while changing bundle
sums and the eventual design, so Stage 2 must report stability across the scenario grid rather than
select a favorable transformation after seeing outcomes.

After production, aggregate score summaries were computed for the first 128 assessment users. No
bundle objective or policy outcome was accessed, but the assessment panel is not untouched. The
[release audit](stage1_release_audit.md) records this peek, and any Stage 2 access record must
carry it forward.

## Public verification and local reproduction

From a privacy-safe public clone:

```text
python -m src.stage1_public_verify
python -m pytest -p no:cacheprovider --strict-markers -q
```

The first command verifies the public evidence manifest, aggregate manifests, run logs, tables,
figure, source/config bindings, sizes, hashes, semantic identifiers, and cross-manifest IDs. It
explicitly reports raw and protected references that cannot be checked in a public clone.

`python -m src.stage1_pipeline` is a separate full-local-state verifier. It requires the exact raw
input plus ignored protected matrices, identifiers, per-user metrics, and fitted parameters from a
completed run. It is not currently a clean-clone bootstrap command. Public integrity verification
must therefore not be described as independent model recomputation.

## Evidence

- Frozen summary: `outputs/modeling/cycles/s1-v2-20260814/stage1_evidence_summary.md`
- Complete evidence graph: `outputs/modeling/cycles/s1-v2-20260814/stage1_evidence_manifest.json`
- Design-test results: `outputs/modeling/cycles/s1-v2-20260814/stage1_design_test_leaderboard.csv`
- Segment results: `outputs/modeling/cycles/s1-v2-20260814/stage1_design_test_segments.csv`
- Admission decision: `outputs/modeling/cycles/s1-v2-20260814/stage1_validation_admission_manifest.json`
- Mathematical appendix: `notes/stage1_v2_mathematical_appendix.md`
- [Rendered mathematical appendix](stage1_v2_mathematical_appendix_readable.md), for GitHub and
  VS Code preview. The original above stays unchanged for evidence verification.
- Post-freeze audit and amendments: `notes/stage1_release_audit.md`

## Stage 1 closeout and next step

The [Stage 1 closeout](stage1_closeout.md) of 2026-09-30 follows a
[scientific audit](stage1_scientific_audit.md) and an additive improvement round. That round
implemented and tested successor tooling on synthetic data. It did not fit a new Steam model, so the
result above stands unchanged. The ALS release described here remains the only interface eligible
for Stage 2 until a successor release completes its own selection, production, and scenario
evidence.

I have not begun Stage 2. Before further assessment access, I will freeze and commit the
candidate-pool registry, instance suite, pseudo-costs, tie convention, policy
set, search budgets, and assessment protocol. Only complete policies frozen on design users may then
be evaluated on assessment users.
