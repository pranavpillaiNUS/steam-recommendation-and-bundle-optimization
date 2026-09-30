# Successor Stage 1 runtime

`src/stage1_runtime.py` provides the scoring and artifact boundary used by the
successor Stage 1 runner. It is additive: it does not rewrite the frozen v2
implementation or its evidence. It adds engineering controls; these controls
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
BLAS scorer, so a new runtime run is separate evidence; it does not replace an
old artifact or claim an identical frozen score hash.

`evaluate_factor_ranking(source, user_indices, target_indices,
excluded_by_user, ks=..., top_item_mask=...)` produces a `RankingResult` with
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
excluded; the runner still owns split integrity.

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
RSS and reaps the worker; observed descendants left behind at normal worker
exit are also cleaned up. Status is `complete`, `failed`, `timeout`, or
`memory_limit`. Invalid limits are rejected before launch, and an existing log
is preserved rather than overwritten.

Its returned resource contract explicitly records sampling limitations.
Between-sample peaks and short-lived descendants can be missed, shared pages
can count more than once in summed RSS, and termination takes additional time.
This process guard improves observability and stops detected overruns; it is
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
bound archive only after establishing the legacy artifact's trusted maps; it
must not invent bindings from matching dimensions alone.

`validated_fold_in_triples` validates catalogue bounds before delegating to
the frozen deterministic sampler. It closes the legacy early-return case in
which an invalid history with at least as many distinct entries as catalogue
items could be reported as insufficient history. Valid histories retain the
same deterministic triple samples. Existing frozen fold-in code is unchanged;
successor callers must explicitly use the validated entry point.

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
