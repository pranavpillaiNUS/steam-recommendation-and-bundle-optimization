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

## Before public release

- Notebook outputs contained one public Steam profile identifier, a username, review text, and
  local machine paths. The current files have been sanitized. Public release should start from a
  sanitized history (for example, a reviewed clean/squashed publication branch); rewriting the
  existing history is destructive and requires explicit owner approval.
- Existing commits use a machine-style author name and a personal email address. Set a professional
  display name and a GitHub verified/noreply address for future commits; changing old author metadata
  is a separate history rewrite and must be an explicit owner decision.
- Dataset redistribution terms are not clearly specified by the mirror. Confirm permission for
  tracked derived data and choose a software license with the project owner and supervisor before
  public release.
- The public Stage 1 files are currently untracked in the working tree and must be deliberately
  reviewed and committed; a push of the prior `main` commit would omit the evidence.
- The original public command required ignored protected artifacts. The new public verifier checks
  the publishable evidence graph without claiming computational reproduction.

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

The Stage 1 result is ready to present once the privacy, licensing, commit, and clean-clone checks
are complete. The correct status is "Stage 1 complete; Stage 2 planned." The full project,
economic identification problem, and clean-clone retraining workflow are not complete.

## Release checklist

1. Confirm with the project owner/supervisor that the repository and supervisor attribution may be
   public, and resolve the redistribution terms for every tracked derived data artifact.
2. Select a software license that covers original code only; do not imply that it relicenses the
   third-party Steam data.
3. Publish from a sanitized history so removed notebook identifiers, review text, local paths, and
   unwanted author metadata are not exposed in earlier commits.
4. Review and deliberately add the currently untracked v2 source, tests, configs, aggregate
   evidence, CI, model card, and provenance files. Do not bulk-add ignored raw/protected data.
5. Set a professional Git author identity, run the strict test suite and public verifier from the
   exact publication candidate, and require the CI workflow to pass before changing visibility.
6. Commit and tag the Stage 2 protocol before opening any further assessment or policy outcomes.
