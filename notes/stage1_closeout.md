# Stage 1 closeout

Closeout date: 2026-09-30. Historical cycle: `s1-v2-20260814`. Development cycle:
`s1-dev-20260929`.

Final review and verification updated: 2026-10-01.

This note completes step 4 of the Stage 1 sequence in [planning.md](../planning.md#243-stage-1-improvement-sequence-closed-out-2026-09-30).
It gives an explicit account of what the 2026-09-29 improvement round measured, what it only
implemented or proposed, the resource limits that apply, and the one model interface that a future
Stage 2 may use. Stage 2 has not started and this note does not start it.

## Summary

- The frozen `s1-v2-20260814` result is unchanged. No manifest, configuration, run log, table, or
  figure bound into its evidence graph was edited. The public verifier still returns `status: ok`
  with evidence manifest `9c0d5b48059cfbecad0d0c9fd2da8a025dc57942104dd181e308d338b07b6650`.
- No new Steam model was fitted, validated, or tested. The exact private design inputs are absent
  from this workspace, so every improvement below has numerical and synthetic verification only.
- The engineering amendments recorded in the [release audit](stage1_release_audit.md) are now
  addressed by additive modules, deferred with a stated reason, or superseded. The frozen modules
  stay as they are because editing them would reopen the cycle.
- No scientific model, feature block, transformation, or selection rule was adopted. The minibatch
  BPR objective and the extra ALS recipes remain exploratory proposals.
- The only model interface eligible for Stage 2 is the historical ownership-only implicit ALS
  release described at the end of this note.

## Development cycle and access record

The development cycle is `s1-dev-20260929`, with source cycle `s1-v2-20260814`. Its choices are
informed by the [scientific audit](stage1_scientific_audit.md) and by the already opened v2
results, including the one-time design test. Any result it produces is labeled exploratory. A new
cycle ID does not make the historical design test unseen again.

| Access | When | Scope | Status |
| --- | --- | --- | --- |
| v2 validation targets | 2026-08-14 | 5,000 fixed design users, model selection | Historical, recorded in the v2 manifests |
| v2 design test | 2026-08-14 | Opened once after the admission hash | Historical, cannot serve as fresh confirmation |
| v2 assessment scores | 2026-08-14 | Aggregate score summaries for the first 128 assessment users | Historical peek, disclosed in the release audit |
| `s1-dev-20260929` protected reads | 2026-09-29 to 2026-09-30 | None | The private inputs are absent here |
| Assessment users | 2026-09-29 to 2026-09-30 | None | No further access during Stage 1 improvement |

The successor fitting and evaluation loaders consume an explicit allowlist of ten design inputs: design-training
matrices and ID maps, validation targets, the evaluation-user sample, the opaque other-holdout mask,
and the genre block with its item map. It refuses any listed input whose access class mentions
assessment, design test, or audit-only use, and they never open a bundle outcome. Preflight also runs
the public verifier, which may integrity-hash any present raw or protected reference, including
assessment files. Those byte reads do not fit, score, or analyze assessment users. Before a worker
loads any of them for fitting or evaluation, it appends one line to `access_ledger.jsonl` in its protected
cycle directory. Each line names the job, the time, and every input path, hash, and access class.
The ledger distinguishes analytical access, design-test scoring, and use of the opaque other-holdout
mask. It records an intended load, including attempts that later fail. The file is ignored by Git
like the rest of the protected tree, so this table is the public record.

## Measured changes

There are no measured changes to any Stage 1 result. The only measurements from this round are
verification results:

| Check | Result |
| --- | --- |
| `python -m pytest -q --strict-config --strict-markers` | 329 collected, 326 passed, 3 skipped (private artifacts), Python 3.10.11 on Windows, final review 2026-10-01 |
| GitHub Actions on Ubuntu for commit `d02b0e5` | Success |
| `python -m src.stage1_public_verify` | `status: ok`, 12 manifests, 88 validation and 3 production logs, 147 public references |
| `python -m src.stage1_successor preflight` | Exit 2, `private_inputs_missing`, all 10 design inputs listed |
| `run_smoke`, the `smoke` command, under the test suite | Synthetic fit, archive, reload, bounded ranking, four transforms, cache reuse, corruption rejection |

The 118 successor tests cover the runtime, the minibatch trainer, the process supervisor, and the
runner. They establish implementation behavior on synthetic inputs, not Steam accuracy or
production-scale speed.

## Engineering amendments

Each row maps one amendment from the [release audit](stage1_release_audit.md#engineering-amendments-for-the-next-cycle)
to its current status.

| Amendment | Status | Where |
| --- | --- | --- |
| Cached runners must recheck every dependency, with stale-cache and corruption tests | Addressed for successor runs | `VerifiedRunStore` rechecks the specification, every dependency hash including all `src/*.py`, every output, and its own hash. Completed runs are immutable and a dependency change during a run aborts it. |
| The evidence assembler skips missing recorded paths | Public side addressed, successor deferred | `stage1_public_verify` fails on every missing public path. A successor assembler waits for successor Steam results. |
| Pseudo-utility diagnostics touched 128 assessment users | Disclosed, not repeated | Successor scenario diagnostics are synthetic only and the input allowlist excludes assessment artifacts. |
| A dense 5,000 by 8,902 float64 score allocation | Addressed | `fit_global_parameters_disk` computes exact quantiles through a size-checked memory map. `iter_pseudo_utility_rows` holds one catalogue row at a time. |
| Runtime labels and unenforced resource ceilings | Addressed, unmeasured at Steam scale | Run summaries separate fit, serialization, ranking, scenario, and total time. `run_supervised` enforces wall time and sampled process-tree RSS. The saved-model cap is enforced. |
| Configured item batching was not two-dimensional | Addressed | `FactorScoreSource` bounds users, items, and bytes before allocation. Block-shape invariance and exact ties across blocks are tested. |
| Identity BPR archives an undeclared empty `feature_factors` array | Addressed in the successor schema | Identity BPR declares a zero-row tensor and genre BPR requires at least one row. |
| `game_features.csv` lacks a small semantic genre input with raw provenance | Deferred | Building it needs the raw metadata, which is absent. |
| Fold-in bounds and assessment ID binding | Bounds addressed, binding deferred | `validated_fold_in_triples` checks bounds first. Binding assessment IDs and row order belongs to Stage 2 entry because assessment access is closed. |
| Estimator-specific archive validation and nonpositive polling intervals | Addressed | `load_bound_parameter_archive` checks the trusted hash, family, ordered maps, fields, shapes, dtypes, finiteness, and expanded size. `run_supervised` rejects nonpositive or NaN intervals. The legacy `stage1_resource_monitor.py`, which no current module imports, is superseded, not edited. |
| One explicit cycle context and a synthetic end-to-end test | Partly addressed | `CycleContext` refuses v1 and v2 IDs. The synthetic test covers fitting through cache and corruption checks. Selection, production, and evidence assembly follow a real selection and are deferred. |
| Check-only runners traceback in a public clone | Addressed by the successor preflight | `preflight` exits 2 with a readable report. The frozen runners are unchanged: the protocol, interaction, split, and feature check-only commands raise `FileNotFoundError` for the untracked `outputs/tables/user_items_df.csv`, the mechanism audit raises it for `data/raw/bundle_data.json`, and `src.stage1_pipeline` raises `FileExistsError` because the public source manifest is present without its protected source output. |

Successor runs also record the environment detail that planning Section 20.3 asks for. The cache
specification binds the Python, operating system, and key package versions plus a hash of every
installed distribution's name and version. Each result records that same distribution list, CPU
and memory totals, and the BLAS libraries without local file paths. A detected software environment
change during execution prevents completion, so the exported list cannot silently disagree with
the cache specification.

Final verification confirmed the historical counts, metrics, manifests, and Stage 2
eligibility. The review also identified the ledger wording and environment consistency issues above.
No scientific proposal or Stage 2 blueprint was changed during this review.

## Unmeasured proposals

These come from the priorities table in the [scientific audit](stage1_scientific_audit.md#priorities-and-completion-criteria).
Each needs the private design inputs before it can produce evidence.

| Priority | Proposal | Current state |
| --- | --- | --- |
| P1 | Practical BPR optimization | The minibatch AdaGrad trainer is implemented and tested. Its penalty convention differs from v2, so regularization must be retuned. The batch size, learning rate, regularization, and update-budget grid is not yet declared. |
| P1 | Wider ALS search and a stopping rule | Three single-seed recipes exist: the v2 anchor, 30 iterations, and 128 factors. A staged grid, validation checkpoints, and a finalist seed rule are not yet declared. |
| P1 | Weak segments and simple baselines | Expected coverage and top-item concentration are implemented. Activity and support segment tables, support-band macro averages, and an item-neighborhood baseline are not implemented. EASE stays optional behind a memory gate. |
| P1 | Enforceable runtime and memory | Implemented as ceilings. Steam-scale use is unmeasured. |
| P2 | Content and cold diagnostics | A controlled genre-toggle recipe exists. A separately tuned genre benchmark, post-removal cold comparators, and a raw-provenance genre input are not started. |
| P2 | Generalization | This needs genuinely unseen data, which the snapshot does not provide. |

A successor report must name its two activity quantities separately. The raw outer activity band
assigns the design and assessment split. Warm training ownership after both holdouts are removed
defines the evaluation segments.

Before any Steam validation run is used for selection, a prospective, hash-bound configuration must
declare the search grid, checkpoints, seeds and finalist rule, the primary paired contrast with its
absolute improvement threshold, the total budget, and the failure policy. Selection uses design
validation only. A negligible validation difference is not evidence of improvement.

## Resource limits

| Quantity | Value | Kind |
| --- | --- | --- |
| Historical v2 validation ledger | About 8,554 seconds for 88 runs, within the 43,200-second budget | Measured, 2026-08-14 |
| Successor job wall time | 2,700 seconds | Ceiling |
| Successor process-tree RSS | 8 GiB, sampled every 0.25 seconds | Ceiling |
| Saved model | 256 MiB | Ceiling |
| Score and candidate-mask block | 64 MiB, at most 128 users by 4,096 items | Ceiling |
| Opaque validation mask batch | 8 MiB | Ceiling |
| Disk scratch for exact quantiles | 512 MiB in the synthetic path, sized as 8 bytes per user-item score | Ceiling |

The ceilings are enforced but they are not measurements. RSS supervision samples the process
tree, so it can miss short peaks and it is not an operating-system memory reservation. Memory-mapped
pages can still be resident in RAM. No successor job has run at Steam scale, so its runtime and peak
memory are unknown.

## Interface eligible for Stage 2

Stage 2 may bind only the historical `s1-v2-20260814` release:

- family and configuration: implicit ALS `als__k064__reg0p05__ao20__ownership_only`
- three seed-specific production refits, seeds 104729, 130363, and 155921, on 4,051,868 restored
  warm design edges
- the ordered 8,902-item production catalogue and 12,585 folded-in assessment users
- four frozen transformations: `global_shift_q90_scale`, `global_robust_softplus`,
  `within_user_midrank_percentile`, and `positive_part_user_standardization`
- production manifest `c8b76f330e382dc74a3b67361bc763cc4030f329d399e95e7926dd60b23e5ba1`
- Gate 2 manifest `3c0714a65cc9807d9a6a3be5688b1f932b0dc83af3e83609536e99a95353f87b`.

The three seeds are separate interfaces, not a score-averaged ensemble. Averaging scores across
seeds before or after a transformation would be a new policy that needs its own prospective
evaluation. Within-user transformations use the complete ordered catalogue before any pool columns
are selected. Identity BPR and identity plus genre BPR failed admission and are not eligible inputs.
The 128-user aggregate assessment peek must be carried into any Stage 2 access record.

A successor release can replace this interface only after its own selection, production refit,
fold-in, pseudo-utility, and evidence manifests are complete. Parameters, catalogue maps,
transformations, and admission evidence from different releases must never be mixed. No Stage 2
outcome may select a Stage 1 model, feature block, or transformation.

## Still open

These do not block the closeout:

- confirm redistribution terms for the tracked derived tables and whether the project and
  supervisor attribution may be public
- commit and tag the Stage 2 protocol before any further assessment access
- restore the private design inputs, then run the successor experiments above under a declared
  selection configuration.

Stage 2 remains a blueprint. It starts only after an explicit request, with the candidate-pool
registry and notebook 11 as its first dependency.
