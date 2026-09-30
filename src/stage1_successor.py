"""Opt-in Stage 1 development runner; never overwrites the frozen v2 evidence.

The real-data command uses only the existing design training/validation contract.
Its results are exploratory because v2's design test has already been inspected.
No command fits or scores assessment users or evaluates a bundle policy.
Preflight may integrity-hash existing protected files through the public verifier.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile
import time
import uuid
from typing import Any, Mapping

from src.stage1_public_verify import canonical_json_bytes, file_sha256, semantic_sha256

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FROZEN_CYCLE = "s1-v2-20260814"
DEFAULT_CYCLE = "s1-dev-20260929"
DEFAULT_CONFIG = "configs/stage1_successor.json"
SPLIT_INPUTS = (
    "design_training_ownership", "design_training_playtime_forever",
    "design_training_playtime_2weeks", "design_training_user_ids",
    "design_training_item_ids", "validation_targets", "evaluation_user_sample",
    "validation_other_holdout_mask",
)
FEATURE_INPUTS = ("genre", "item_ids")


def _json(path: Path) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON field: {key}")
            result[key] = value
        return result

    def invalid(value: str) -> None:
        raise ValueError(f"nonfinite JSON number: {value}")

    result = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                        parse_constant=invalid)
    if not isinstance(result, dict):
        raise ValueError("expected a JSON object")
    return result


def _inside(root: Path, relative: str) -> Path:
    parts = PurePosixPath(relative)
    if (not relative or parts.is_absolute() or ".." in parts.parts
            or "\\" in relative or ":" in relative):
        raise ValueError(f"invalid relative artifact path: {relative}")
    path = (root / relative).resolve()
    path.relative_to(root.resolve())
    return path


def _entry(root: Path, path: Path) -> dict[str, Any]:
    source = path.resolve()
    relative = source.relative_to(root.resolve()).as_posix()
    if not source.is_file():
        raise FileNotFoundError(f"required artifact is missing: {relative}")
    return {"path": relative, "sha256": file_sha256(source),
            "size_bytes": source.stat().st_size}


def _check_entry(root: Path, entry: Mapping[str, Any]) -> Path:
    path = _inside(root, str(entry["path"]))
    actual = _entry(root, path)
    if any(actual[key] != entry.get(key) for key in actual):
        raise ValueError(f"artifact size or hash changed: {entry['path']}")
    return path


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".stage1-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(value))
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


@dataclass(frozen=True)
class CycleContext:
    root: Path
    cycle_id: str

    def __post_init__(self) -> None:
        if (not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,79}", self.cycle_id)
                or self.cycle_id.startswith(("s1-v1-", "s1-v2-"))):
            raise ValueError("use a new, safe cycle ID; v1/v2 are immutable")
        object.__setattr__(self, "root", self.root.resolve())

    @property
    def directory(self) -> Path:
        return _inside(self.root, f"outputs/modeling/protected/{self.cycle_id}")


class VerifiedRunStore:
    """A cache is usable only when its specification AND all dependencies match."""

    def __init__(self, context: CycleContext, name: str) -> None:
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,119}", name):
            raise ValueError("invalid run name")
        self.context = context
        self.directory = context.directory / name
        self.path = self.directory / "run.json"

    def load(self, *, specification: Mapping[str, Any],
             dependencies: Mapping[str, Path]) -> dict[str, Any] | None:
        if not self.path.exists():
            return None
        value = _json(self.path)
        unsigned = dict(value)
        claimed = unsigned.pop("manifest_id", None)
        if claimed != semantic_sha256(unsigned):
            raise ValueError("cached run manifest hash changed")
        if value.get("cycle_id") != self.context.cycle_id:
            raise ValueError("cached run cycle mismatch")
        if value.get("status") != "complete" or value.get("schema_version") != 1:
            raise ValueError("cached run is not a completed supported record")
        if value.get("specification") != specification:
            raise ValueError("cached run specification changed; use a new run/cycle")
        current = {name: _entry(self.context.root, path)
                   for name, path in sorted(dependencies.items())}
        if value.get("dependencies") != current:
            raise ValueError("cached run dependencies changed; use a new run/cycle")
        if not value.get("outputs"):
            raise ValueError("cached run output inventory is empty")
        if set(value["outputs"]) != set(specification.get("required_outputs", value["outputs"])):
            raise ValueError("cached run output inventory changed")
        for entry in value["outputs"].values():
            path = _check_entry(self.context.root, entry)
            path.relative_to(self.directory.resolve())
        return value

    def save(self, *, specification: Mapping[str, Any], dependencies: Mapping[str, Path],
             outputs: Mapping[str, Path], summary: Mapping[str, Any]) -> dict[str, Any]:
        if self.path.exists():
            raise FileExistsError("completed runs are immutable; choose a new run/cycle")
        if not dependencies or not outputs:
            raise ValueError("dependencies and outputs are required")
        for path in outputs.values():
            path.resolve().relative_to(self.directory.resolve())
        value = {
            "schema_version": 1, "cycle_id": self.context.cycle_id,
            "status": "complete", "interpretation": "exploratory_stage1_development",
            "specification": dict(specification),
            "dependencies": {name: _entry(self.context.root, path)
                             for name, path in sorted(dependencies.items())},
            "outputs": {name: _entry(self.context.root, path)
                        for name, path in sorted(outputs.items())},
            "summary": dict(summary),
        }
        value["manifest_id"] = semantic_sha256(value)
        _atomic_json(self.path, value)
        return value


def design_input_inventory(root: Path) -> dict[str, Mapping[str, Any]]:
    """List an explicit allowlist of design inputs, never assessment artifacts."""
    cycle = root / "outputs" / "modeling" / "cycles" / FROZEN_CYCLE
    split = _json(cycle / "stage1_split_manifest.json")
    features = _json(cycle / "item_feature_manifest.json")
    if split["cycle_id"] != FROZEN_CYCLE or features["cycle_id"] != FROZEN_CYCLE:
        raise ValueError("source cycle mismatch")
    result = {name: split["artifacts"][name] for name in SPLIT_INPUTS}
    result.update({f"feature_{name}": features["artifacts"][name]
                   for name in FEATURE_INPUTS})
    for name, entry in result.items():
        path = _inside(root, entry["path"])
        path.relative_to((root / "outputs/modeling/protected" / FROZEN_CYCLE).resolve())
        access = entry.get("access", "")
        if "assessment" in access or "design_test" in access or "audit_only" in access:
            raise ValueError(f"forbidden input in development inventory: {name}")
    return result


def preflight(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    from src.stage1_public_verify import verify_public_stage1
    public = verify_public_stage1(root, cycle_id=FROZEN_CYCLE)
    missing: list[str] = []
    for entry in design_input_inventory(root).values():
        if not _inside(root, entry["path"]).is_file():
            missing.append(entry["path"])
        else:
            _check_entry(root, entry)
    return {
        "status": "ready_for_design_development" if not missing else "private_inputs_missing",
        "frozen_public_evidence": public,
        "required_design_inputs": len(design_input_inventory(root)),
        "missing_design_inputs": missing,
        "assessment_or_bundle_outcomes_used": False,
        "next_action": ("Run development jobs under a new cycle ID." if not missing else
                        "Restore the listed original hash-matching private files under this root. "
                        "The public verifier and synthetic smoke command work without them."),
    }


def _environment() -> dict[str, str]:
    return {"python": sys.version.split()[0], **{
        name: importlib.metadata.version(name)
        for name in ("numpy", "scipy", "pandas", "implicit", "threadpoolctl")}}


def _code_dependencies(root: Path) -> dict[str, Path]:
    # Include all source modules: a newly imported helper cannot evade cache invalidation.
    result = {f"source:{p.relative_to(root).as_posix()}": p
              for p in sorted((root / "src").glob("*.py"))}
    result["environment_spec"] = root / "requirements-frozen.txt"
    return result


class OpaqueValidationMask:
    """Consume the frozen masking API without exposing its coordinates to tuning.

    One boolean user-batch by catalogue mask is cached (separate from score
    memory). The size is checked before allocation; masks never become labels.
    """

    def __init__(self, root: Path, manifest: Path, user_ids: Any, item_ids: Any,
                 maximum_mask_bytes: int = 8 * 1024 * 1024) -> None:
        self.root, self.manifest = root, manifest
        self.user_ids, self.item_ids = user_ids, item_ids
        if isinstance(maximum_mask_bytes, bool) or not isinstance(maximum_mask_bytes, int) or maximum_mask_bytes <= 0:
            raise ValueError("maximum_mask_bytes must be a positive integer")
        self.maximum_mask_bytes = maximum_mask_bytes
        self._rows: Any = None
        self._mask: Any = None

    def __call__(self, rows: Any, start: int, stop: int) -> Any:
        import numpy as np
        from src.stage1_split_artifacts import mask_validation_other_holdouts
        if self._rows is None or not np.array_equal(rows, self._rows):
            if len(rows) * len(self.item_ids) > self.maximum_mask_bytes:
                raise MemoryError("opaque validation mask exceeds its separate byte budget")
            self._mask = mask_validation_other_holdouts(
                np.ones((len(rows), len(self.item_ids)), dtype=bool),
                self.user_ids[rows], self.item_ids,
                project_root=self.root, manifest_path=self.manifest)
            self._rows = rows.copy()
        return self._mask[:, start:stop]


def load_design_data(root: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    import numpy as np
    import scipy.sparse as sp
    from src.stage1_split_artifacts import (
        load_design_training_artifacts,
        load_evaluation_user_sample, load_validation_targets,
    )
    cycle = root / "outputs/modeling/cycles" / FROZEN_CYCLE
    manifest_path = cycle / "stage1_split_manifest.json"
    inventory = design_input_inventory(root)
    dependencies = {name: _check_entry(root, entry) for name, entry in inventory.items()}
    dependencies.update(split_manifest=manifest_path,
                        feature_manifest=cycle / "item_feature_manifest.json")
    training = load_design_training_artifacts(project_root=root, manifest_path=manifest_path)
    sample = load_evaluation_user_sample(project_root=root, manifest_path=manifest_path)
    targets = load_validation_targets(project_root=root, manifest_path=manifest_path)

    def positions(haystack: Any, needles: Any) -> Any:
        index = np.searchsorted(haystack, needles)
        if np.any(index >= len(haystack)) or not np.array_equal(haystack[index], needles):
            raise ValueError("input identifiers do not align")
        return index.astype(np.int64)

    rows = positions(training.user_ids, sample["user_ids"])
    target_order = np.argsort(targets["user_ids"])
    locations = positions(targets["user_ids"][target_order], sample["user_ids"])
    columns = positions(training.item_ids, targets["item_ids"][target_order][locations])
    exclusions = []
    for row, target in zip(rows, columns):
        owned = training.ownership.indices[training.ownership.indptr[row]:training.ownership.indptr[row + 1]]
        mask = owned.copy().astype(np.int64)
        if target in mask:
            raise ValueError("validation target is masked or present in training")
        exclusions.append(mask)
    genre_ids = np.load(dependencies["feature_item_ids"], allow_pickle=False)
    genres = sp.load_npz(dependencies["feature_genre"]).tocsr()
    genres = genres[positions(genre_ids, training.item_ids)]
    return {
        "ownership": training.ownership, "playtime": training.playtime_forever,
        "user_ids": training.user_ids, "item_ids": training.item_ids, "genres": genres,
        "evaluation_rows": rows, "target_columns": columns, "exclusions": exclusions,
        "other_holdout_mask": OpaqueValidationMask(root, manifest_path, training.user_ids, training.item_ids),
    }, dependencies


def _data_identity(data: Mapping[str, Any]) -> str:
    import hashlib
    import numpy as np
    from src.interactions import csr_semantic_sha256
    payload = {name: csr_semantic_sha256(data[name])
               for name in ("ownership", "playtime", "genres")}
    for name in ("user_ids", "item_ids", "evaluation_rows", "target_columns"):
        payload[name] = hashlib.sha256(np.asarray(data[name], dtype="<i8").tobytes()).hexdigest()
    digest = hashlib.sha256()
    for row in data["exclusions"]:
        values = np.asarray(row, dtype="<i8")
        digest.update(len(values).to_bytes(8, "little"))
        digest.update(values.tobytes())
    payload["exclusions"] = digest.hexdigest()
    return semantic_sha256(payload)


def _source_for(arrays: Mapping[str, Any], family: str, data: Mapping[str, Any],
                scoring: Mapping[str, Any]) -> Any:
    import numpy as np
    from src.stage1_runtime import FactorScoreSource
    if family == "popularity":
        users = np.ones((len(data["user_ids"]), 1), dtype=np.float32)
        items = arrays["item_counts"].reshape(-1, 1)
        bias = None
    else:
        users = arrays["user_factors"]
        bias = arrays.get("item_bias")
        if family == "implicit_als":
            items = arrays["item_factors"]
        else:
            items = np.asarray(arrays["identity_factors"], dtype=np.float64)
            if arrays["feature_factors"].shape[0]:
                items = items + data["genres"] @ np.asarray(arrays["feature_factors"], dtype=np.float64)
    return FactorScoreSource(users, items, bias, **scoring)


def execute_job(context: CycleContext, name: str, job: Mapping[str, Any],
                data: Mapping[str, Any], dependencies: Mapping[str, Path], *,
                scoring: Mapping[str, Any], scenarios: bool = False,
                maximum_saved_model_bytes: int = 268435456) -> dict[str, Any]:
    """Fit, serialize, rank, and bind one explicitly requested development job.

    Caller supplies a hash-bound design dataset. The CLI loader enforces its
    access allowlist. This routine never selects a model or opens a test set.
    """
    import numpy as np
    from threadpoolctl import threadpool_limits
    from src.stage1_backend import fit_implicit_als
    from src.stage1_training import fit_minibatch_bpr
    from src.stage1_runtime import (
        evaluate_factor_ranking, fit_global_parameters_disk,
        iter_pseudo_utility_rows, load_bound_parameter_archive,
        save_bound_parameter_archive,
    )
    family = str(job["family"])
    seed = job.get("training_seed", 104729)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("training_seed must be a nonnegative integer")
    if family not in {"popularity", "implicit_als", "feature_sum_bpr_identity", "feature_sum_bpr_identity_genre"}:
        raise ValueError("unsupported development family")
    if family.endswith("_genre") and (not data["genres"].shape[1] or not data["genres"].nnz):
        raise ValueError("the genre job requires a nonempty genre block; use the identity job otherwise")
    if (isinstance(maximum_saved_model_bytes, bool) or not isinstance(maximum_saved_model_bytes, int)
            or maximum_saved_model_bytes <= 0):
        raise ValueError("maximum_saved_model_bytes must be a positive integer")
    store = VerifiedRunStore(context, name)
    specification = {"job": dict(job), "scoring": dict(scoring), "scenarios": scenarios,
                     "data_identity": _data_identity(data), "environment": _environment(),
                     "metric_ks": [10, 20], "assessment_analytical_access": False,
                     "maximum_saved_model_bytes": maximum_saved_model_bytes,
                     "required_outputs": ["parameters", "validation_metrics", "training_diagnostics"]
                     + (["synthetic_scenarios"] if scenarios else [])}
    cached = store.load(specification=specification, dependencies=dependencies)
    if cached is not None:
        return {"cache_reused": True, **cached}
    store.directory.mkdir(parents=True, exist_ok=True)
    attempt = store.directory / "attempt.json"
    initial_dependencies = {key: _entry(context.root, path) for key, path in dependencies.items()}
    try:
        with attempt.open("xb") as handle:
            handle.write(canonical_json_bytes({"status": "running", "specification": specification,
                                               "dependencies": initial_dependencies}))
    except FileExistsError as exc:
        raise FileExistsError("an attempt already exists; concurrent or unrecorded retries are forbidden") from exc
    started = time.perf_counter()
    try:
        fit_started = time.perf_counter()
        parameters = dict(job.get("parameters", {}))
        with threadpool_limits(limits=1, user_api="blas"):
            if family == "popularity":
                arrays = {"item_counts": np.asarray(data["ownership"].sum(axis=0)).ravel().astype(np.int64)}
                diagnostics = {"backend": "design_training_ownership_count"}
            elif family == "implicit_als":
                fitted = fit_implicit_als(data["ownership"], data["playtime"],
                                          training_seed=seed, **parameters)
                arrays = {"user_factors": fitted.user_factors, "item_factors": fitted.item_factors}
                diagnostics = dict(fitted.diagnostics)
            else:
                fitted = fit_minibatch_bpr(data["ownership"], data["genres"],
                    cycle_id=context.cycle_id, training_seed=seed,
                    include_genre=family.endswith("_genre"), **parameters)
                arrays = {key: getattr(fitted.parameters, key) for key in
                          ("user_factors", "identity_factors", "feature_factors", "item_bias")}
                diagnostics = dict(fitted.diagnostics)
        fit_seconds = time.perf_counter() - fit_started
        serialization_started = time.perf_counter()
        archive_path = store.directory / "parameters.npz"
        archive = save_bound_parameter_archive(archive_path, arrays, family=family,
                          user_ids=data["user_ids"], item_ids=data["item_ids"])
        if archive["size_bytes"] > maximum_saved_model_bytes:
            raise MemoryError("saved model exceeds its registered byte budget")
        restored = load_bound_parameter_archive(archive_path, expected_sha256=archive["sha256"],
            expected_family=family, expected_user_ids=data["user_ids"], expected_item_ids=data["item_ids"])
        serialization_seconds = time.perf_counter() - serialization_started
        source = _source_for(restored, family, data, scoring)
        support = np.asarray(data["ownership"].sum(axis=0)).ravel().astype(np.int64)
        top = np.zeros(len(support), dtype=bool)
        top[np.lexsort((data["item_ids"], -support))[:max(1, int(np.ceil(len(support) * 0.01)))]] = True
        ranking_started = time.perf_counter()
        with threadpool_limits(limits=1, user_api="blas"):
            result = evaluate_factor_ranking(source, data["evaluation_rows"], data["target_columns"],
                data["exclusions"], top_item_mask=top,
                candidate_mask_provider=data.get("other_holdout_mask"))
        ranking_seconds = time.perf_counter() - ranking_started
        metrics_path = store.directory / "validation_metrics.npz"
        np.savez_compressed(metrics_path,
            user_ids=data["user_ids"][data["evaluation_rows"]],
            target_item_ids=data["item_ids"][data["target_columns"]], **result.metrics)
        diagnostics_path = store.directory / "training_diagnostics.json"
        _atomic_json(diagnostics_path, diagnostics)
        outputs = {"parameters": archive_path, "validation_metrics": metrics_path,
                   "training_diagnostics": diagnostics_path}
        scenario_seconds = 0.0
        if scenarios:
            # Synthetic integration only. Real development jobs do not silently freeze a new interface.
            scenario_started = time.perf_counter()
            fit = fit_global_parameters_disk(source, data["evaluation_rows"],
                scratch_directory=store.directory / "scratch", maximum_disk_bytes=512 * 1024 * 1024)
            summaries = {}
            for scenario in ("global_shift_q90_scale", "global_robust_softplus",
                             "within_user_midrank_percentile", "positive_part_user_standardization"):
                count = 0
                total = 0.0
                low, high = float("inf"), float("-inf")
                for _, row in iter_pseudo_utility_rows(source, data["evaluation_rows"], scenario,
                                                     parameters=fit["parameters"]):
                    count += row.size
                    total += float(row.sum())
                    low, high = min(low, float(row.min())), max(high, float(row.max()))
                summaries[scenario] = {"count": count, "mean": total / count, "min": low, "max": high}
            scenario_path = store.directory / "synthetic_scenarios.json"
            _atomic_json(scenario_path, {"fit": fit, "scenarios": summaries,
                                        "real_stage1_interface_frozen": False})
            outputs["synthetic_scenarios"] = scenario_path
            scenario_seconds = time.perf_counter() - scenario_started
        current_dependencies = {key: _entry(context.root, path) for key, path in dependencies.items()}
        if initial_dependencies != current_dependencies:
            raise ValueError("a dependency changed during execution; result was not completed")
        summary = {"family": family, "training_seed": seed, **result.aggregate,
                   "timing_seconds": {"fit": fit_seconds, "serialization": serialization_seconds,
                       "ranking": ranking_seconds, "scenario_diagnostics": scenario_seconds,
                       "total": time.perf_counter() - started},
                   "resource_contract": {**result.resource_contract,
                       "additional_opaque_mask_limit_bytes": getattr(data.get("other_holdout_mask"), "maximum_mask_bytes", 0)},
                   "assessment_or_bundle_outcomes_used": False}
        completed = store.save(specification=specification, dependencies=dependencies,
                               outputs=outputs, summary=summary)
        _atomic_json(attempt, {"status": "complete", "manifest_id": completed["manifest_id"]})
        return {"cache_reused": False, **completed}
    except Exception as exc:
        _atomic_json(attempt, {"status": "failed", "error_type": type(exc).__name__,
                              "error": str(exc), "specification": specification,
                              "dependencies": initial_dependencies,
                              "elapsed_seconds": time.perf_counter() - started})
        raise


def synthetic_design_data() -> dict[str, Any]:
    import numpy as np
    import scipy.sparse as sp
    rng = np.random.default_rng(20260929)
    users, items = 36, 24
    ownership = np.zeros((users, items), dtype=np.float32)
    targets = np.empty(users, dtype=np.int64)
    exclusions = []
    for user in range(users):
        choices = rng.permutation(np.arange((user % 3) * 8, (user % 3 + 1) * 8))
        ownership[user, choices[:4]] = 1
        targets[user] = choices[4]
        exclusions.append(np.sort(np.append(choices[:4], choices[5])))
    owned = sp.csr_matrix(ownership)
    return {"ownership": owned, "playtime": owned.copy(),
            "genres": sp.csr_matrix(np.eye(3, dtype=np.float32)[np.arange(items) // 8]),
            "user_ids": np.arange(users, dtype=np.int64) + 1,
            "item_ids": np.arange(items, dtype=np.int64) + 100,
            "evaluation_rows": np.arange(users, dtype=np.int64),
            "target_columns": targets, "exclusions": exclusions}


def run_smoke(root: Path = PROJECT_ROOT, cycle_id: str = DEFAULT_CYCLE) -> dict[str, Any]:
    context = CycleContext(root, cycle_id)
    data = synthetic_design_data()
    dependencies = _code_dependencies(root)
    scoring = {"user_block_size": 7, "item_block_size": 5, "maximum_score_block_bytes": 1024}
    jobs = {
        "popularity": {"family": "popularity"},
        "als": {"family": "implicit_als", "training_seed": 104729,
                "parameters": {"factors": 4, "regularization": 0.05, "alpha_o": 20.0,
                               "alpha_p": 0.0, "tau": 0.0, "iterations": 4, "num_threads": 1}},
        "bpr": {"family": "feature_sum_bpr_identity", "training_seed": 104729,
                "parameters": {"factors": 4, "regularization": 0.001, "learning_rate": 0.05,
                               "epochs": 3, "samples_per_epoch": 300, "batch_size": 32,
                               "diagnostic_sample_size": 128}},
    }
    jobs["bpr_genre"] = {**jobs["bpr"], "family": "feature_sum_bpr_identity_genre"}
    results = {}
    for name, job in jobs.items():
        result = execute_job(context, f"synthetic-{name}", job, data, dependencies,
                             scoring=scoring, scenarios=name == "als")
        results[name] = {"manifest_id": result["manifest_id"], "cache_reused": result["cache_reused"],
                         "summary": result["summary"]}
    return {"status": "ok", "cycle_id": cycle_id, "data": "synthetic_only",
            "assessment_access": False, "models": results}


def run_development_job(root: Path, cycle_id: str, config_path: str, job_name: str) -> dict[str, Any]:
    if root.resolve() != PROJECT_ROOT.resolve():
        raise ValueError("real development must run from the executing repository root so code provenance matches")
    context = CycleContext(root, cycle_id)
    config_file = _inside(root, config_path)
    config = _json(config_file)
    if type(config.get("schema_version")) is not int or config["schema_version"] != 1 or config.get("source_cycle") != FROZEN_CYCLE:
        raise ValueError("unsupported development schema or source_cycle")
    if config.get("status") != "exploratory_development" or config.get("assessment_access") is not False:
        raise ValueError("only design-side exploratory configurations are supported")
    if job_name not in config["jobs"]:
        raise ValueError(f"unknown job: {job_name}")
    report = preflight(root)
    if report["missing_design_inputs"]:
        raise FileNotFoundError(report["next_action"])
    data, dependencies = load_design_data(root)
    dependencies.update(_code_dependencies(root))
    dependencies["development_config"] = config_file
    result = execute_job(context, job_name, config["jobs"][job_name], data, dependencies,
                         scoring=config["scoring"],
                         maximum_saved_model_bytes=config["resource_budget"]["maximum_saved_model_bytes"])
    return {"status": "ok", "cycle_id": cycle_id, "job": job_name,
            "manifest_id": result["manifest_id"], "cache_reused": result["cache_reused"],
            "interpretation": "exploratory_validation_not_new_confirmation", "summary": result["summary"]}


def supervised_development_job(root: Path, cycle_id: str, config_path: str, job_name: str) -> dict[str, Any]:
    """Run the development worker with wall-time and sampled process-tree limits."""
    from src.stage1_execution import run_supervised
    if root.resolve() != PROJECT_ROOT.resolve():
        raise ValueError("real development must run from the executing repository root")
    context = CycleContext(root, cycle_id)
    # Validate the requested name and readiness before launching any expensive worker.
    VerifiedRunStore(context, job_name)
    config_file = _inside(root, config_path)
    config = _json(config_file)
    if job_name not in config.get("jobs", {}):
        raise ValueError(f"unknown job: {job_name}")
    report = preflight(root)
    if report["missing_design_inputs"]:
        raise FileNotFoundError(report["next_action"])
    budget = config["resource_budget"]
    config_hash = file_sha256(config_file)
    log_directory = context.directory / "supervision" / job_name
    log_directory.mkdir(parents=True, exist_ok=True)
    attempt_id = uuid.uuid4().hex
    log_path = log_directory / f"{attempt_id}.log"
    command = [sys.executable, "-m", "src.stage1_successor", "_worker",
               "--root", str(root), "--cycle-id", cycle_id,
               "--config", config_path, "--job", job_name]
    supervision = run_supervised(command, cwd=root, log_path=log_path,
        maximum_wall_seconds=budget["maximum_job_wall_seconds"],
        maximum_rss_bytes=budget["maximum_process_tree_rss_bytes"],
        poll_interval_seconds=budget["poll_interval_seconds"])
    supervision["configuration_sha256"] = config_hash
    supervision["log_path"] = log_path.relative_to(root).as_posix()
    _atomic_json(log_directory / f"{attempt_id}.json", supervision)
    if supervision["status"] != "complete":
        raise RuntimeError(f"development worker {supervision['status']}; inspect {supervision['log_path']}")
    if file_sha256(config_file) != config_hash:
        raise ValueError("development configuration changed during supervised execution")
    worker_report = None
    for line in reversed(log_path.read_text(encoding="utf-8", errors="replace").splitlines()):
        try:
            candidate = json.loads(line)
        except ValueError:
            continue
        if isinstance(candidate, dict) and candidate.get("status") == "ok":
            worker_report = candidate
            break
    if worker_report is None:
        raise RuntimeError("successful worker did not emit its structured result")
    return {**worker_report, "supervision": supervision}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "smoke", "validate", "_worker"))
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--cycle-id", default=DEFAULT_CYCLE)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--job", help="explicit job name from the development configuration")
    args = parser.parse_args(argv)
    try:
        if args.command == "preflight":
            report = preflight(args.root.resolve())
            code = 0 if report["status"] == "ready_for_design_development" else 2
        elif args.command == "smoke":
            report = run_smoke(args.root.resolve(), args.cycle_id)
            code = 0
        else:
            if not args.job:
                raise ValueError("validate requires an explicit --job; no grid runs implicitly")
            runner = run_development_job if args.command == "_worker" else supervised_development_job
            report = runner(args.root.resolve(), args.cycle_id, args.config, args.job)
            code = 0
        print(json.dumps(report, sort_keys=True, allow_nan=False))
        return code
    except Exception as exc:
        print(json.dumps({"status": "error", "error_type": type(exc).__name__, "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
