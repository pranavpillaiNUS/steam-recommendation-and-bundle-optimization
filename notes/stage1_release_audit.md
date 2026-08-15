# Stage 1 post-freeze release audit

Audit date: 2026-08-14

## Publication note

This repository begins from the sanitized Stage 1 snapshot prepared on 2026-08-15. The privacy,
author-identity, tracked-file, and clean-history items recorded below were resolved when the clean
repository was created. Dataset redistribution permission and the software license remain open.

The `repository_baseline_commit` values in the frozen configurations refer to the private archival
research history. They are retained as internal provenance identifiers and are not publicly
resolvable commits or evidence of external preregistration.

I ran this audit after freezing the `s1-v2-20260814` evidence package. None of the findings changes
the reported ranking metrics or admission decision. Editing hash-recorded code, configurations, or
evidence would reopen the cycle, so I have recorded those fixes for the next cycle.

## Publication-history items resolved

- Notebook outputs in the private archival history contained one public Steam profile identifier,
  a username, review text, and local machine paths. The clean publication history contains only the
  sanitized files.
- The private archival commits used a machine-style author name and a personal email address. The
  clean publication history uses a professional display name and GitHub noreply address.
- The complete v2 source, tests, configurations, aggregate evidence, CI, model card, and provenance
  files are deliberately tracked in the clean repository.
- The public verifier checks the publishable evidence graph without requiring ignored raw or
  protected artifacts and without claiming computational reproduction.

## Remaining before public release

- Kaggle API metadata labels mirror version 1 as `Apache 2.0`, but the upstream page states no
  license and the mirror uploader's authority over every component has not been established.
  Confirm permission for tracked derived data and choose a software license with the project owner
  and supervisor before public release.
- Confirm that the project and supervisor attribution may be public.

## Scientific wording amendments

1. The Kaggle archive contains no user-bundle interaction file. `users_own_all` is this project's
   upper-bound ownership-compatibility proxy, not recovered purchases and not a reproduction of the
   source paper's 87,565 user-bundle interactions.
2. The frozen sample-size note calls a normal-approximation half-width "distribution-free." Only
   the bounded variance input is distribution-free. A Hoeffding 95 percent half-width at n=5,000 is
   about 0.0192, while the recorded worst-case normal-approximation half-width is below 0.014. The
   sample and results do not change.
3. The pseudo-cold popularity comparator uses support measured before the temporary cohort removal.
   It is an oracle-like pre-removal comparator, not a deployable new-item baseline. The diagnostic
   does not establish general cold-start performance.
4. The percentile bootstrap interval is conditional on this panel, evaluation sample, trained
   seeds, and protocol. It does not cover population shift or optimization randomness across a
   wider training-seed distribution.
5. The fallback BPR optimizer and frozen schedule are specific to this cycle. "Not admitted" is
   the supported result; a general claim that BPR or genre fails is not.

## Quantified legacy identity erratum

Notebook 02 historically chose display `user_id` before `steam_id` and kept the first duplicate.
The live Stage 1 source contract uses stable numeric `steam_id` and fieldwise maxima.

An audit of all 88,310 raw records found a bijection among active users and exactly the same
5,094,082 ownership edges. Only 10 user-item playtime records change: +217 lifetime minutes and +203
two-week minutes in total. Notebook 03 ownership metrics, notebook 04, notebook 05 regressions, and
all Stage 1 fits are unchanged. Stage 1 reads only `item_id` and `genres` from `game_features.csv`.
A corrected descriptive pipeline should use a new artifact version rather than overwrite the
hash-bound files.

## Engineering amendments for the next cycle

- Cached Gate 1, production, and Gate 2 runners currently validate their own manifest IDs but do
  not recheck every dependency and current runner hash before returning. The public wrapper is
  strict; a new scientific cycle should centralize strict dependency verification and add stale
  cache/corruption tests.
- The existing evidence assembler silently skips missing recorded paths. The public wrapper instead
  fails on every missing public path and allows absence only for explicitly classified raw or
  protected paths.
- Pseudo-utility diagnostics scored the first 128 assessment users and published aggregate score
  summaries after production but before the Stage 2 protocol was frozen. No bundle objective or
  policy outcome was accessed, but this is an aggregate assessment peek. Freeze Stage 2 before any
  further assessment access; future diagnostics should use design users only.
- Pseudo-utility score generation allocates a full 5,000 by 8,902 float64 result, about 356 MB,
  despite a bounded-block design statement. Replace it prospectively with exact blockwise/memmap
  scoring or correct the resource contract.
- The runtime evidence table labels ranking time as validation runtime. The validation ledger's
  total runtimes sum to about 8,554 seconds and remain below the 43,200-second budget, but future
  evidence should report fit, ranking, serialization, and total time separately and enforce all
  resource ceilings.
- `score_item_batch_size=4096` is configured but the evaluator scores the full 8,902-item catalogue
  for each user batch. Current blocks remain within 64 MiB, but the implementation and stated
  two-dimensional batching contract differ.
- The identity-BPR archive includes an empty `feature_factors` array not listed in its declared
  schema. A successor manifest should declare the zero-row tensor or omit it.
- The full `game_features.csv` byte hash is bound upstream, although Stage 1 consumes only
  `item_id,genres` and the notebook-produced table lacks its own raw-input build manifest. A new
  cycle should publish a small semantic genre input with explicit raw provenance.
- Fold-in should validate item bounds before returning an `insufficient_history` fallback, and the
  pseudo-utility diagnostic path should bind assessment user IDs and row order before slicing.
- Model archives need estimator-specific field, dtype, shape, and map-hash validation; the resource
  monitor should reject nonpositive polling intervals.
- Cycle routing remains fragmented between v1 defaults and v2 orchestration. A successor cycle
  should use one explicit cycle context and add a compact synthetic end-to-end test covering stale
  cache rejection, corruption, regeneration, selection, production, and evidence assembly.

## Release decision

The Stage 1 result, privacy review, clean history, tracked publication, and clean-clone verification
are ready to present. Public visibility still depends on data-redistribution confirmation,
supervisor approval, and selection of a software license. The correct status is "Stage 1 complete;
Stage 2 planned." The full project, economic identification problem, and clean-clone retraining
workflow are not complete.

## Release checklist

- [ ] Confirm that the repository and supervisor attribution may be public, and resolve the
  redistribution terms for every tracked derived data artifact.
- [ ] Select a software license that covers original code only; do not imply that it relicenses the
  third-party Steam data.
- [x] Publish from a sanitized history without the removed identifiers, review text, local paths,
  or unwanted author metadata.
- [x] Deliberately track the v2 source, tests, configurations, aggregate evidence, CI, model card,
  and provenance files without ignored raw or protected data.
- [x] Use a professional Git identity and pass the strict tests and public verifier from the exact
  publication candidate.
- [ ] Commit and tag the Stage 2 protocol before opening any further assessment or policy outcomes.
