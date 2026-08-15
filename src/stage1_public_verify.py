"""Read-only verification for the public Stage 1 evidence package.

This module intentionally uses only the Python standard library.  It verifies
the committed evidence graph without importing the modeling stack and without
creating, rewriting, or normalizing any project artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CYCLE_ID = "s1-v2-20260814"

MANIFEST_FILES = {
    "source": "stage1_source_manifest.json",
    "protocol": "stage1_protocol_manifest.json",
    "interactions": "stage1_interaction_manifest.json",
    "splits": "stage1_split_manifest.json",
    "features": "item_feature_manifest.json",
    "estimator": "stage1_estimator_spec_manifest.json",
    "backend_spike": "stage1_backend_spike_manifest.json",
    "training": "stage1_training_manifest.json",
    "admission": "stage1_validation_admission_manifest.json",
    "gate1": "stage1_gate1_manifest.json",
    "production": "stage1_production_manifest.json",
    "pseudo_utility_gate2": "pseudo_utility_scenarios_manifest.json",
}

REQUIRED_OUTPUT_LABELS = frozenset(
    {
        "summary",
        "runtime_memory",
        "ranking_figure",
        "validation_leaderboard",
        "design_test_leaderboard",
        "design_test_segments",
        "pseudo_cold_results",
        "pseudo_utility_diagnostics",
        "mathematical_appendix",
        "seed_specific_contrasts",
    }
)

_ALLOWED_ABSENT_PREFIXES = (
    ("data", "raw"),
    ("outputs", "modeling", "protected"),
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_MISSING = object()


class Stage1VerificationError(ValueError):
    """Raised when the public evidence graph fails verification."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return the canonical representation used by the frozen Stage 1 IDs."""

    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise Stage1VerificationError(f"value is not canonical JSON: {exc}") from exc
    return (payload + "\n").encode("utf-8")


def semantic_sha256(value: Any) -> str:
    """Hash JSON semantics using the Stage 1 canonical representation."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path, *, chunk_bytes: int = 1024 * 1024) -> str:
    """Hash a file incrementally without modifying it."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, nested in pairs:
        if key in value:
            raise Stage1VerificationError(f"duplicate JSON object key: {key!r}")
        value[key] = nested
    return value


def _reject_nonfinite(token: str) -> None:
    raise Stage1VerificationError(f"non-finite JSON number is not allowed: {token}")


def _load_json(path: Path, *, description: str) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except Stage1VerificationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Stage1VerificationError(f"cannot load {description} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise Stage1VerificationError(f"{description} must contain a JSON object: {path}")
    return value


def _require_hash(value: Any, *, description: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise Stage1VerificationError(f"{description} must be a lowercase SHA-256 hex digest")
    return value


def _require_size(value: Any, *, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Stage1VerificationError(f"{description} must be a nonnegative integer")
    return value


def _require_mapping(value: Any, *, description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Stage1VerificationError(f"{description} must be an object")
    return value


def _require_list(value: Any, *, description: str) -> list[Any]:
    if not isinstance(value, list):
        raise Stage1VerificationError(f"{description} must be an array")
    return value


def _semantic_identity(
    value: Mapping[str, Any],
    *,
    description: str,
    candidates: Sequence[str] = ("manifest_id", "admission_id"),
) -> str:
    for field in candidates:
        if field not in value:
            continue
        claimed = _require_hash(value[field], description=f"{description}.{field}")
        unsigned = dict(value)
        unsigned.pop(field)
        expected = semantic_sha256(unsigned)
        if claimed != expected:
            raise Stage1VerificationError(
                f"{description} semantic hash mismatch for {field}: "
                f"expected {expected}, recorded {claimed}"
            )
        return claimed

    # The frozen protocol ID binds the semantic hashes of its two configs,
    # rather than the full provenance/environment wrapper.
    if "protocol_id" in value and "configs" in value:
        protocol_id = _require_hash(
            value["protocol_id"], description=f"{description}.protocol_id"
        )
        configs = _require_mapping(value["configs"], description=f"{description}.configs")
        preference = _require_mapping(
            configs.get("preference_models"),
            description=f"{description}.configs.preference_models",
        )
        ranking = _require_mapping(
            configs.get("ranking_evaluation"),
            description=f"{description}.configs.ranking_evaluation",
        )
        preference_hash = _require_hash(
            preference.get("semantic_sha256"),
            description=f"{description}.configs.preference_models.semantic_sha256",
        )
        ranking_hash = _require_hash(
            ranking.get("semantic_sha256"),
            description=f"{description}.configs.ranking_evaluation.semantic_sha256",
        )
        expected = semantic_sha256(
            {
                "model_config_sha256": preference_hash,
                "ranking_config_sha256": ranking_hash,
            }
        )
        if protocol_id != expected:
            raise Stage1VerificationError(
                f"{description} protocol semantic binding mismatch: "
                f"expected {expected}, recorded {protocol_id}"
            )
        return protocol_id

    joined = ", ".join(candidates)
    raise Stage1VerificationError(
        f"{description} has no recognized semantic identifier ({joined})"
    )


def _safe_relative_parts(raw_path: Any, *, description: str) -> tuple[str, ...]:
    if not isinstance(raw_path, str) or not raw_path:
        raise Stage1VerificationError(f"{description}.path must be a nonempty string")
    if "\x00" in raw_path:
        raise Stage1VerificationError(f"{description}.path contains a NUL byte")
    if "\\" in raw_path:
        raise Stage1VerificationError(
            f"{description}.path must use repository-relative POSIX separators"
        )
    windows_path = PureWindowsPath(raw_path)
    posix_path = PurePosixPath(raw_path)
    raw_parts = raw_path.split("/")
    if (
        posix_path.is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or any(part in {"", ".", ".."} for part in raw_parts)
    ):
        raise Stage1VerificationError(
            f"{description}.path is not a safe repository-relative path: {raw_path!r}"
        )
    return tuple(posix_path.parts)


def _is_allowed_absent(parts: tuple[str, ...]) -> bool:
    return any(parts[: len(prefix)] == prefix for prefix in _ALLOWED_ABSENT_PREFIXES)


class _Verifier:
    def __init__(self, root: Path) -> None:
        if not root.is_dir():
            raise Stage1VerificationError(f"project root is not a directory: {root}")
        self.root = root.resolve()
        self._digest_cache: dict[Path, tuple[str, int]] = {}
        self.public_references_verified = 0
        self.private_references_verified = 0
        self.allowed_absent_private_references = 0

    def resolve(self, raw_path: Any, *, description: str) -> tuple[Path, tuple[str, ...]]:
        parts = _safe_relative_parts(raw_path, description=description)
        try:
            resolved = (self.root.joinpath(*parts)).resolve(strict=False)
            resolved.relative_to(self.root)
        except (OSError, RuntimeError, ValueError) as exc:
            raise Stage1VerificationError(
                f"{description}.path escapes the project root: {raw_path!r}"
            ) from exc
        return resolved, parts

    def _digest(self, path: Path) -> tuple[str, int]:
        cached = self._digest_cache.get(path)
        if cached is not None:
            return cached
        before = path.stat()
        digest = file_sha256(path)
        after = path.stat()
        if (
            before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
        ):
            raise Stage1VerificationError(f"file changed while it was being verified: {path}")
        result = (digest, after.st_size)
        self._digest_cache[path] = result
        return result

    def verify_recorded_file(
        self,
        record: Any,
        *,
        description: str,
        require_public: bool = False,
        require_size: bool = True,
    ) -> Path | None:
        entry = _require_mapping(record, description=description)
        path, parts = self.resolve(entry.get("path"), description=description)
        expected_hash = _require_hash(
            entry.get("sha256"), description=f"{description}.sha256"
        )
        recorded_size = entry.get("size_bytes", _MISSING)
        expected_size = (
            None
            if recorded_size is _MISSING and not require_size
            else _require_size(
                recorded_size, description=f"{description}.size_bytes"
            )
        )
        private = _is_allowed_absent(parts)
        if require_public and private:
            raise Stage1VerificationError(
                f"{description} is a public evidence entry and cannot use a private path: "
                f"{entry['path']}"
            )
        if not path.is_file():
            if path.exists():
                raise Stage1VerificationError(
                    f"recorded artifact is not a regular file: {entry['path']}"
                )
            if private and not require_public:
                self.allowed_absent_private_references += 1
                return None
            raise Stage1VerificationError(
                f"required recorded artifact is missing: {entry['path']}"
            )
        actual_hash, actual_size = self._digest(path)
        if actual_hash != expected_hash:
            raise Stage1VerificationError(
                f"recorded artifact hash mismatch for {entry['path']}: "
                f"expected {expected_hash}, observed {actual_hash}"
            )
        if expected_size is not None and actual_size != expected_size:
            raise Stage1VerificationError(
                f"recorded artifact size mismatch for {entry['path']}: "
                f"expected {expected_size}, observed {actual_size}"
            )
        if private:
            self.private_references_verified += 1
        else:
            self.public_references_verified += 1
        return path

    def verify_nested_references(self, value: Any, *, description: str) -> None:
        if isinstance(value, Mapping):
            if "path" in value and "sha256" in value:
                self.verify_recorded_file(
                    value, description=description, require_size=False
                )
            for key, nested in value.items():
                self.verify_nested_references(
                    nested, description=f"{description}.{key}"
                )
        elif isinstance(value, list):
            for index, nested in enumerate(value):
                self.verify_nested_references(
                    nested, description=f"{description}[{index}]"
                )


def _expect_equal(actual: Any, expected: Any, *, description: str) -> None:
    if actual != expected:
        raise Stage1VerificationError(
            f"{description} mismatch: expected {expected!r}, observed {actual!r}"
        )


def _expect_cycle(value: Mapping[str, Any], cycle_id: str, *, description: str) -> None:
    _expect_equal(value.get("cycle_id"), cycle_id, description=f"{description}.cycle_id")


def _unique_records_by_id(
    records: Any, *, id_field: str, description: str
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for index, raw_record in enumerate(_require_list(records, description=description)):
        record = _require_mapping(raw_record, description=f"{description}[{index}]")
        identifier = _require_hash(
            record.get(id_field), description=f"{description}[{index}].{id_field}"
        )
        if identifier in result:
            raise Stage1VerificationError(
                f"duplicate {id_field} in {description}: {identifier}"
            )
        result[identifier] = record
    return result


def _verify_cross_manifest_bindings(
    manifests: Mapping[str, Mapping[str, Any]],
    evidence: Mapping[str, Any],
    validation_logs: Mapping[str, Mapping[str, Any]],
    production_logs: Mapping[str, Mapping[str, Any]],
) -> None:
    protocol = manifests["protocol"]
    interactions = manifests["interactions"]
    splits = manifests["splits"]
    features = manifests["features"]
    estimator = manifests["estimator"]
    backend = manifests["backend_spike"]
    training = manifests["training"]
    admission = manifests["admission"]
    gate1 = manifests["gate1"]
    production = manifests["production"]
    pseudo = manifests["pseudo_utility_gate2"]

    protocol_id = protocol["protocol_id"]
    for label in (
        "interactions",
        "splits",
        "features",
        "estimator",
        "backend_spike",
        "training",
        "admission",
    ):
        _expect_equal(
            manifests[label].get("protocol_id"),
            protocol_id,
            description=f"{label}.protocol_id",
        )

    interaction_set_id = interactions.get("interaction_set_id")
    _require_hash(interaction_set_id, description="interactions.interaction_set_id")
    for label in ("splits", "features", "estimator", "training"):
        _expect_equal(
            manifests[label].get("interaction_set_id"),
            interaction_set_id,
            description=f"{label}.interaction_set_id",
        )

    split_set_id = splits.get("split_set_id")
    _require_hash(split_set_id, description="splits.split_set_id")
    for label in ("features", "estimator", "backend_spike", "training"):
        _expect_equal(
            manifests[label].get("split_set_id"),
            split_set_id,
            description=f"{label}.split_set_id",
        )

    feature_set_id = features.get("feature_set_id")
    _require_hash(feature_set_id, description="features.feature_set_id")
    for label in ("estimator", "backend_spike", "training"):
        _expect_equal(
            manifests[label].get("feature_set_id"),
            feature_set_id,
            description=f"{label}.feature_set_id",
        )

    _expect_equal(
        training.get("estimator_specification_id"),
        estimator.get("specification_id"),
        description="training.estimator_specification_id",
    )
    _expect_equal(
        training.get("backend_spike_manifest_id"),
        backend.get("manifest_id"),
        description="training.backend_spike_manifest_id",
    )
    _expect_equal(
        admission.get("training_manifest_id"),
        training.get("manifest_id"),
        description="admission.training_manifest_id",
    )
    _expect_equal(
        admission.get("selection"),
        training.get("selection"),
        description="admission.selection",
    )
    _expect_equal(
        gate1.get("admission_id"),
        admission.get("admission_id"),
        description="gate1.admission_id",
    )
    _expect_equal(
        production.get("admission_id"),
        admission.get("admission_id"),
        description="production.admission_id",
    )
    _expect_equal(
        production.get("gate1_manifest_id"),
        gate1.get("manifest_id"),
        description="production.gate1_manifest_id",
    )
    _expect_equal(
        pseudo.get("production_manifest_id"),
        production.get("manifest_id"),
        description="pseudo_utility_gate2.production_manifest_id",
    )

    gate_inputs = _require_mapping(gate1.get("inputs"), description="gate1.inputs")
    _expect_equal(
        gate_inputs.get("training_manifest_id"),
        training.get("manifest_id"),
        description="gate1.inputs.training_manifest_id",
    )
    _expect_equal(
        gate_inputs.get("split_set_id"),
        split_set_id,
        description="gate1.inputs.split_set_id",
    )
    _expect_equal(
        gate_inputs.get("feature_set_id"),
        feature_set_id,
        description="gate1.inputs.feature_set_id",
    )

    production_inputs = _require_mapping(
        production.get("inputs"), description="production.inputs"
    )
    for field, expected in (
        ("gate1_manifest_id", gate1.get("manifest_id")),
        ("training_manifest_id", training.get("manifest_id")),
        ("interaction_set_id", interaction_set_id),
        ("split_set_id", split_set_id),
    ):
        _expect_equal(
            production_inputs.get(field),
            expected,
            description=f"production.inputs.{field}",
        )

    admitted = _require_list(
        admission.get("admitted_families"), description="admission.admitted_families"
    )
    if len(set(admitted)) != len(admitted) or not all(
        isinstance(family, str) and family for family in admitted
    ):
        raise Stage1VerificationError(
            "admission.admitted_families must contain unique nonempty strings"
        )
    for label, value in (
        ("gate1", gate1.get("admitted_families")),
        ("production", production.get("admitted_families")),
        ("evidence", evidence.get("admitted_families")),
    ):
        _expect_equal(value, admitted, description=f"{label}.admitted_families")

    _expect_equal(backend.get("status"), "pass", description="backend_spike.status")
    _expect_equal(gate1.get("status"), "pass", description="gate1.status")
    _expect_equal(production.get("status"), "complete", description="production.status")
    gate2 = _require_mapping(pseudo.get("gate2"), description="pseudo_utility_gate2.gate2")
    _expect_equal(gate2.get("status"), "pass", description="pseudo_utility_gate2.gate2.status")
    _expect_equal(evidence.get("status"), "complete", description="evidence.status")
    _expect_equal(evidence.get("gate1_status"), gate1.get("status"), description="evidence.gate1_status")
    _expect_equal(evidence.get("gate2_status"), gate2.get("status"), description="evidence.gate2_status")

    training_runs = _unique_records_by_id(
        training.get("runs"), id_field="run_id", description="training.runs"
    )
    if not training_runs:
        raise Stage1VerificationError("training.runs must not be empty")
    _expect_equal(
        set(validation_logs),
        set(training_runs),
        description="validation run-log ID inventory",
    )
    for run_id, training_run in training_runs.items():
        run_log = validation_logs[run_id]
        _expect_equal(run_log.get("protocol_id"), protocol_id, description=f"validation run {run_id}.protocol_id")
        _expect_equal(run_log.get("status"), "complete", description=f"validation run {run_id}.status")
        for field, expected in training_run.items():
            _expect_equal(
                run_log.get(field, _MISSING),
                expected,
                description=f"validation run {run_id}.{field}",
            )

    production_runs = _unique_records_by_id(
        production.get("runs"), id_field="run_id", description="production.runs"
    )
    if not production_runs:
        raise Stage1VerificationError("production.runs must not be empty")
    _expect_equal(
        set(production_logs),
        set(production_runs),
        description="production run-log ID inventory",
    )
    for run_id, production_run in production_runs.items():
        _expect_equal(
            production_logs[run_id],
            production_run,
            description=f"production run log {run_id}",
        )

    selection = _require_mapping(training.get("selection"), description="training.selection")
    expected_production_keys = {
        (run.get("family"), run.get("configuration_id"), run.get("training_seed"))
        for run in training_runs.values()
        if run.get("family") in admitted
        and run.get("configuration_id") == selection.get(run.get("family"))
    }
    observed_production_keys = {
        (run.get("family"), run.get("configuration_id"), run.get("training_seed"))
        for run in production_runs.values()
    }
    _expect_equal(
        observed_production_keys,
        expected_production_keys,
        description="selected production run set",
    )
    if any(run.get("family") not in admitted for run in production_runs.values()):
        raise Stage1VerificationError("production.runs contains a non-admitted family")

    pseudo_parameters = _require_list(
        pseudo.get("run_parameters"), description="pseudo_utility_gate2.run_parameters"
    )
    pseudo_run_ids: list[str] = []
    for index, parameter in enumerate(pseudo_parameters):
        entry = _require_mapping(
            parameter, description=f"pseudo_utility_gate2.run_parameters[{index}]"
        )
        pseudo_run_ids.append(
            _require_hash(
                entry.get("production_run_id"),
                description=(
                    f"pseudo_utility_gate2.run_parameters[{index}].production_run_id"
                ),
            )
        )
    if len(set(pseudo_run_ids)) != len(pseudo_run_ids):
        raise Stage1VerificationError("pseudo-utility production_run_id values are duplicated")
    _expect_equal(
        set(pseudo_run_ids),
        set(production_runs),
        description="pseudo-utility production run set",
    )


def verify_public_stage1(
    root: str | Path = PROJECT_ROOT,
    *,
    cycle_id: str = DEFAULT_CYCLE_ID,
) -> dict[str, Any]:
    """Verify the complete public Stage 1 evidence graph without mutation."""

    if not isinstance(cycle_id, str) or not cycle_id or "/" in cycle_id or "\\" in cycle_id:
        raise Stage1VerificationError("cycle_id must be a single nonempty path component")
    verifier = _Verifier(Path(root))
    cycle_relative = PurePosixPath("outputs", "modeling", "cycles", cycle_id)
    cycle_dir = verifier.root.joinpath(*cycle_relative.parts)
    evidence_path = cycle_dir / "stage1_evidence_manifest.json"
    if not evidence_path.is_file():
        raise Stage1VerificationError(
            f"Stage 1 evidence manifest is missing: {evidence_path}"
        )
    evidence = _load_json(evidence_path, description="Stage 1 evidence manifest")
    _expect_cycle(evidence, cycle_id, description="evidence")
    evidence_id = _semantic_identity(evidence, description="evidence")

    manifest_entries = _require_mapping(
        evidence.get("manifests"), description="evidence.manifests"
    )
    _expect_equal(
        set(manifest_entries),
        set(MANIFEST_FILES),
        description="evidence manifest label inventory",
    )
    manifests: dict[str, Mapping[str, Any]] = {}
    for label, filename in MANIFEST_FILES.items():
        entry = _require_mapping(
            manifest_entries[label], description=f"evidence.manifests.{label}"
        )
        expected_path = (cycle_relative / filename).as_posix()
        _expect_equal(
            entry.get("path"),
            expected_path,
            description=f"evidence.manifests.{label}.path",
        )
        path = verifier.verify_recorded_file(
            entry,
            description=f"evidence.manifests.{label}",
            require_public=True,
        )
        assert path is not None
        manifest = _load_json(path, description=f"{label} manifest")
        _expect_cycle(manifest, cycle_id, description=label)
        identity = _semantic_identity(manifest, description=label)
        _expect_equal(
            entry.get("semantic_id"),
            identity,
            description=f"evidence.manifests.{label}.semantic_id",
        )
        manifests[label] = manifest

    output_entries = _require_mapping(evidence.get("outputs"), description="evidence.outputs")
    missing_output_labels = REQUIRED_OUTPUT_LABELS - set(output_entries)
    if missing_output_labels:
        raise Stage1VerificationError(
            "evidence.outputs is missing required labels: "
            + ", ".join(sorted(missing_output_labels))
        )
    for label, entry in output_entries.items():
        verifier.verify_recorded_file(
            entry,
            description=f"evidence.outputs.{label}",
            require_public=True,
        )

    validation_entries = _require_list(
        evidence.get("validation_run_logs"),
        description="evidence.validation_run_logs",
    )
    validation_logs: dict[str, Mapping[str, Any]] = {}
    listed_validation_paths: set[str] = set()
    validation_prefix = (cycle_relative / "validation_runs").as_posix() + "/"
    for index, raw_entry in enumerate(validation_entries):
        description = f"evidence.validation_run_logs[{index}]"
        entry = _require_mapping(raw_entry, description=description)
        raw_path = entry.get("path")
        if not isinstance(raw_path, str) or not raw_path.startswith(validation_prefix):
            raise Stage1VerificationError(
                f"{description}.path must be inside {validation_prefix}"
            )
        if raw_path in listed_validation_paths:
            raise Stage1VerificationError(f"duplicate validation run-log path: {raw_path}")
        listed_validation_paths.add(raw_path)
        path = verifier.verify_recorded_file(
            entry, description=description, require_public=True
        )
        assert path is not None
        run_log = _load_json(path, description="validation run log")
        _expect_cycle(run_log, cycle_id, description=f"validation run log {raw_path}")
        run_id = _semantic_identity(
            run_log,
            description=f"validation run log {raw_path}",
            candidates=("run_id",),
        )
        _expect_equal(
            entry.get("run_id"), run_id, description=f"{description}.run_id"
        )
        if run_id in validation_logs:
            raise Stage1VerificationError(f"duplicate validation run_id: {run_id}")
        validation_logs[run_id] = run_log

    validation_dir = cycle_dir / "validation_runs"
    on_disk_validation_paths = {
        path.resolve().relative_to(verifier.root).as_posix()
        for path in validation_dir.glob("*.json")
        if path.is_file()
    }
    _expect_equal(
        listed_validation_paths,
        on_disk_validation_paths,
        description="validation run-log path inventory",
    )

    production_dir = cycle_dir / "production_runs"
    if not production_dir.is_dir():
        raise Stage1VerificationError(
            f"production run-log directory is missing: {production_dir}"
        )
    production_logs: dict[str, Mapping[str, Any]] = {}
    production_paths = sorted(path for path in production_dir.glob("*.json") if path.is_file())
    for path in production_paths:
        relative = path.resolve().relative_to(verifier.root).as_posix()
        run_log = _load_json(path, description="production run log")
        _expect_cycle(run_log, cycle_id, description=f"production run log {relative}")
        run_id = _semantic_identity(
            run_log,
            description=f"production run log {relative}",
            candidates=("run_id",),
        )
        if run_id in production_logs:
            raise Stage1VerificationError(f"duplicate production run_id: {run_id}")
        production_logs[run_id] = run_log

    _verify_cross_manifest_bindings(
        manifests, evidence, validation_logs, production_logs
    )

    for label, manifest in manifests.items():
        verifier.verify_nested_references(manifest, description=label)
    for run_id, run_log in validation_logs.items():
        verifier.verify_nested_references(
            run_log, description=f"validation_run_logs.{run_id}"
        )
    for run_id, run_log in production_logs.items():
        verifier.verify_nested_references(
            run_log, description=f"production_run_logs.{run_id}"
        )

    return {
        "status": "ok",
        "cycle_id": cycle_id,
        "evidence_manifest_id": evidence_id,
        "manifests_verified": len(manifests),
        "outputs_verified": len(output_entries),
        "validation_run_logs_verified": len(validation_logs),
        "production_run_logs_verified": len(production_logs),
        "public_references_verified": verifier.public_references_verified,
        "private_references_verified": verifier.private_references_verified,
        "allowed_absent_private_references": (
            verifier.allowed_absent_private_references
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify the public Stage 1 evidence graph without modifying it"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT,
        help="repository root (defaults to the parent of src/)",
    )
    parser.add_argument("--cycle-id", default=DEFAULT_CYCLE_ID)
    args = parser.parse_args(argv)
    try:
        report = verify_public_stage1(args.root, cycle_id=args.cycle_id)
    except Exception as exc:  # CLI boundary: always return machine-readable failure.
        print(
            json.dumps(
                {"status": "error", "error": str(exc)},
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
