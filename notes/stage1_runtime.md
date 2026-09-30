# Successor Stage 1 runtime

`src/stage1_runtime.py` provides the scoring and artifact boundary used by the
successor Stage 1 runner. It is additive: it does not rewrite the frozen v2
implementation or its evidence. It adds engineering controls. These controls
do not establish an improvement in held-out recommendation quality.

## Scoring and ranking

`FactorScoreSource(user_factors, item_factors, item_bias=None, ...)` scores
bounded user and item blocks. The source checks the requested shape and the
float64 score plus boolean candidate-mask allocation **before** scoring.
The byte requirement is `9 * block_users * block_items`. User and item block
limits apply even when the byte cap would allow a larger request.

Scoring uses float64 `numpy.einsum(..., optimize=False)` with a fixed factor
reduction. This keeps scores consistent when the user and item block sizes
change. Successor arithmetic can differ at rounding precision from the frozen
BLAS scorer, so a new runtime run is separate evidence. It does not replace an
old artifact or claim an identical frozen score hash.

`evaluate_factor_ranking(source, user_indices, target_indices,
excluded_by_user, ks=..., top_item_mask=..., candidate_mask_provider=...)` produces a `RankingResult` with
`metrics`, `aggregate`, and `resource_contract` attributes. The first pass
finds each top-K boundary and captures each target from its actual catalogue
block. The second pass uses the frozen `TargetTieCounter` and inclusion
probability primitives to calculate exact expected target metrics and
expected catalogue coverage. Ties use exact equality and uniform random
ordering, including ties that cross item blocks. Coverage assumes independent
tie ordering across users. No top-K tie is resolved by arbitrary item order.

The caller supplies exclusions in evaluation-user order. They must contain
training positives and the non-target holdout. Duplicate evaluation users,
invalid indices, non-boolean item masks, and excluded targets are rejected.
The runtime cannot infer whether an omitted history item was supposed to be
excluded. The runner still owns split integrity. An optional
`candidate_mask_provider(rows, start, stop)` returns a boolean block that is
combined with the exclusions. The runner uses it to apply the frozen opaque
other-holdout mask without receiving the masked coordinates.

Coverage and concentration keys use the actual maximum requested K:
`expected_catalogue_coverage_at_{K}` and
`expected_top_item_concentration_at_{K}`. The concentration key appears only
when a fixed item mask is supplied. Targets remain eligible even when fewer
than K items remain after exclusions.

## Exact pseudo-utility fitting and transformation

`fit_global_parameters_disk(source, user_indices, scratch_directory=...,
maximum_disk_bytes=...)` writes a disposable float64 score sample to a
memory-mapped file. It checks its exact disk requirement,
`8 * sample_users * catalogue_items`, and available free disk space before
creating that file. It hashes row-major scores before partitioning, then fits
the same minimum and linear quantiles as `pseudo_utility.fit_global_parameters`.
It returns `parameters`, `score_sample_sha256`, `score_sample_shape`, and a
resource contract. Temporary files close and are removed on success and
exceptions, including on Windows.

`iter_pseudo_utility_rows(source, user_indices, scenario_id, parameters=...,
maximum_row_workspace_bytes=...)` yields `(user_index, values)` for each user.
It applies the existing transformation to the **complete frozen eligible
catalogue**, including already-owned items. Candidate-pool selection must
occur after transformation: ranking or standardizing only a selected pool
would change the frozen within-user scenarios.

The row workspace gate uses a conservative 128 bytes per catalogue item for
the score row, transformation output, sorting and temporary indices. Consume
each yielded row immediately. Retaining every row or stacking generator
outputs defeats streaming and becomes the caller's allocation.

## Memory limits and their interpretation

The score/mask cap is an allocation contract, not a process RSS cap. Additional
memory includes resident model arrays, bounded factor conversions, per-row
ranking workspaces, O(catalogue items) coverage state, O(evaluation users)
metrics, and library workspaces. Full-catalogue transforms require O(catalogue
items) row workspace. Exact disk-backed quantiles avoid a dense NumPy heap
score matrix, but the operating system can retain mapped pages in RAM. Measure
actual process and child-process peaks during a production-size run and report
them separately from these caps.

`stage1_execution.run_supervised(command, cwd=..., log_path=...,
maximum_wall_seconds=..., maximum_rss_bytes=...)` applies these runtime guards
to an owned worker. It launches without a shell or a visible Windows console,
redirects stdout and stderr into a new private log, and samples summed parent
and descendant RSS. It terminates the owned tree on timeout or observed excess
RSS and reaps the worker. Observed descendants left behind at normal worker
exit are also cleaned up. Status is `complete`, `failed`, `timeout`, or
`memory_limit`. Invalid limits are rejected before launch, and an existing log
is preserved rather than overwritten.

Its returned resource contract explicitly records sampling limitations.
Between-sample peaks and short-lived descendants can be missed, shared pages
can count more than once in summed RSS, and termination takes additional time.
This process guard improves observability and stops detected overruns. It is
not an operating-system hard memory reservation or guarantee. Descendants
already observed remain tracked if their original parent exits.

## Parameter archives and fold-in validation

`save_bound_parameter_archive(path, arrays, family=..., user_ids=...,
item_ids=...)` atomically saves model parameters, a schema, and the ordered ID
maps. `load_bound_parameter_archive` requires the trusted expected file hash,
family, and both ordered maps. It checks ZIP expansion size before loading,
rejects duplicate/unexpected members, and validates exact parameter fields,
dimensions, float32/float64 factor dtypes, finite values, and model/map
alignment. Popularity parameters must be integer counts within the number of
users. Embedded map hashes alone are insufficient: the expected outer hash
and caller-provided ordered maps establish the binding.

This is a new schema. Legacy archives must continue through the frozen loader
and their existing dependency checks. A successor runner may export a new
bound archive only after establishing the legacy artifact's trusted maps. It
must not invent bindings from matching dimensions alone.

`validated_fold_in_triples` validates catalogue bounds before delegating to
the frozen deterministic sampler. It closes the legacy early-return case in
which an invalid history with at least as many distinct entries as catalogue
items could be reported as insufficient history. Valid histories retain the
same deterministic triple samples. Existing frozen fold-in code is unchanged.
Successor callers must explicitly use the validated entry point.

## Successor runner

`src/stage1_successor.py` is the opt-in development runner built on this
runtime. It never fits or scores assessment users, never opens the design
test, and never evaluates a bundle policy. Its default cycle is
`s1-dev-20260929` and its configuration is `configs/stage1_successor.json`.

| Command | What it does | Exit code |
| --- | --- | --- |
| `python -m src.stage1_successor preflight` | Runs the public verifier and checks the ten allowlisted design inputs against their manifest hashes | 0 when ready, 2 when private inputs are missing |
| `python -m src.stage1_successor smoke` | Fits, archives, reloads, and ranks all four families on a tiny synthetic dataset | 0 on success |
| `python -m src.stage1_successor validate --job NAME` | Runs one named configuration job on design validation under `run_supervised` | 0 on success, 1 on error |

Every command prints one JSON report. Errors go to standard error as JSON with
`status: error`, so a public clone gets a readable message rather than a
traceback. `validate` runs no grid: it needs an explicit job name, and each job
fits a single seed. The internal `_worker` command is what the supervisor
launches and is not meant to be called directly. Use `--cycle-id` to keep smoke
runs out of the default development cycle.

Outputs go under `outputs/modeling/protected/<cycle>/`, which Git ignores:

- `<job>/run.json`: the self-hashed record of the specification,
  dependencies, outputs, and summary. A completed run is never overwritten.
- `<job>/attempt.json`: written before fitting. A failed attempt blocks a
  silent retry under the same run name.
- `<job>/parameters.npz`, `validation_metrics.npz`, and
  `training_diagnostics.json`: the bound archive, per-user metrics, and fit
  trace.
- `supervision/<job>/<attempt>.log` and `.json`: worker output plus wall
  time, peak sampled RSS, and termination status.
- `access_ledger.jsonl`: one appended line per worker, written before any
  protected input is loaded for fitting or evaluation, naming every input
  path, hash, and access class. This is an intended load, not a claim that
  every file was successfully read. It distinguishes analytical access,
  design-test scoring, and opaque masking. Preflight integrity hashing happens
  earlier and may read any present private reference in the frozen graph,
  including assessment artifacts, without fitting or scoring those users.

The cache specification binds the Python version, operating system, key
package versions, and a hash of installed distribution names and versions. Each
run summary exports that same captured list, and a detected software environment
change before completion fails the run. It also records CPU and memory totals
and the BLAS libraries without local file paths. Run summaries report fit, serialization,
ranking, scenario, and total seconds separately.

## Verification

Run `python -m pytest -q tests/test_stage1_runtime.py`. The tests compare exact
ranking and coverage with dense reference calculations, exercise fractional
ties across blocks and block-size invariance, check refusal before oversized
allocation, compare disk quantiles and all four transforms against the frozen
definitions, verify cleanup on failure, reject malformed archives and swapped
ID maps, and reproduce the fold-in bounds defect as a regression case.

`python -m pytest -q tests/test_stage1_execution.py` exercises successful and
failed exits, stdout/stderr capture, literal argument passing, timeout and
allocation guards, descendant RSS, child cleanup, cleanup after a monitor
failure, preservation of prior logs, and validation before launch.

`python -m pytest -q tests/test_stage1_training.py` checks the minibatch BPR
gradients against the frozen equation oracle and finite differences, the
update counts, the triple stream, training-only negatives, and invalid inputs.

`python -m pytest -q tests/test_stage1_successor.py` covers cache
revalidation, corruption, immutability, path and cycle boundaries, the design
input allowlist, the synthetic end-to-end run, the readable preflight and
missing-input exits, supervision records, the access ledger, and environment
capture.
