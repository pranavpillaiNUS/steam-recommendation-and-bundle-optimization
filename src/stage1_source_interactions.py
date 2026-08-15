"""Build the canonical Stage 1 interaction source directly from raw records.

The original notebook preferred the display ``user_id`` over the stable numeric
``steam_id``.  That caused the S1-v1 canonical loader to discard more than half
of the interaction rows.  This module defines the prospective S1-v2 source
contract: ``steam_id`` is the account identifier and duplicate account-item
rows are collapsed by the fieldwise maximum nonnegative playtime.

The builder is outcome-free.  It reads only the raw ownership snapshot and
does not inspect any split, held-out coordinate, score, or bundle objective.
"""

from __future__ import annotations

import argparse
from array import array
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .stage1_protocol import canonical_json_bytes, file_sha256, semantic_sha256


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "australian_users_items.json"
DEFAULT_CYCLE_ID = "s1-v2-20260814"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "modeling"
    / "protected"
    / DEFAULT_CYCLE_ID
    / "source_interactions.csv"
)
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "outputs"
    / "modeling"
    / "cycles"
    / DEFAULT_CYCLE_ID
    / "stage1_source_manifest.json"
)
SCHEMA_VERSION = 1
OUTPUT_COLUMNS = (
    "user_id",
    "item_id",
    "playtime_forever",
    "playtime_2weeks",
)


def _canonical_int64(value: Any, *, label: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} cannot be boolean")
    text = str(value)
    if not text or (text != "0" and (text.startswith("0") or not text.isdigit())):
        raise ValueError(f"{label} must be an unpadded nonnegative decimal integer")
    if not text.isdigit():
        raise ValueError(f"{label} must be an unpadded nonnegative decimal integer")
    integer = int(text)
    if integer > np.iinfo(np.int64).max:
        raise ValueError(f"{label} exceeds signed int64")
    return integer


def _playtime(value: Any, *, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return 0.0 if number == 0.0 else number


def canonicalize_source_columns(
    user_ids: Sequence[int] | np.ndarray,
    item_ids: Sequence[int] | np.ndarray,
    playtime_forever: Sequence[float] | np.ndarray,
    playtime_2weeks: Sequence[float] | np.ndarray,
) -> dict[str, np.ndarray]:
    """Sort raw columns and collapse duplicate pairs by fieldwise maximum."""

    users = np.asarray(user_ids, dtype=np.int64)
    items = np.asarray(item_ids, dtype=np.int64)
    forever = np.asarray(playtime_forever, dtype=np.float64)
    recent = np.asarray(playtime_2weeks, dtype=np.float64)
    lengths = {users.size, items.size, forever.size, recent.size}
    if len(lengths) != 1:
        raise ValueError("source columns have inconsistent lengths")
    if users.ndim != 1 or items.ndim != 1 or forever.ndim != 1 or recent.ndim != 1:
        raise ValueError("source columns must be one-dimensional")
    if users.size == 0:
        raise ValueError("the source contains no interaction rows")
    if np.any(users < 0) or np.any(items < 0):
        raise ValueError("source identifiers must be nonnegative")
    if np.any(~np.isfinite(forever)) or np.any(forever < 0.0):
        raise ValueError("lifetime playtime must be finite and nonnegative")
    if np.any(~np.isfinite(recent)) or np.any(recent < 0.0):
        raise ValueError("recent playtime must be finite and nonnegative")

    order = np.lexsort((items, users))
    users = users[order]
    items = items[order]
    forever = forever[order]
    recent = recent[order]
    starts_mask = np.empty(users.size, dtype=bool)
    starts_mask[0] = True
    starts_mask[1:] = (users[1:] != users[:-1]) | (items[1:] != items[:-1])
    starts = np.flatnonzero(starts_mask)
    result = {
        "user_id": users[starts],
        "item_id": items[starts],
        "playtime_forever": np.maximum.reduceat(forever, starts),
        "playtime_2weeks": np.maximum.reduceat(recent, starts),
    }
    for values in result.values():
        values.setflags(write=False)
    return result


def records_to_columns(records: Iterable[Mapping[str, Any]]) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Convert parsed raw user records to canonical account-item columns."""

    users = array("q")
    items = array("q")
    forever = array("d")
    recent = array("d")
    record_count = 0
    zero_item_record_count = 0
    mapping_pairs: set[tuple[str, int]] = set()
    reverse_map: dict[int, set[str]] = {}

    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("each raw line must contain a mapping")
        record_count += 1
        steam_id = _canonical_int64(record.get("steam_id"), label="steam_id")
        user_id = str(record.get("user_id", ""))
        mapping_pairs.add((user_id, steam_id))
        reverse_map.setdefault(steam_id, set()).add(user_id)
        owned = record.get("items", [])
        if owned is None:
            owned = []
        if not isinstance(owned, list):
            raise ValueError("items must be a list")
        if not owned:
            zero_item_record_count += 1
        for item in owned:
            if not isinstance(item, Mapping):
                raise ValueError("each owned item must be a mapping")
            users.append(steam_id)
            items.append(_canonical_int64(item.get("item_id"), label="item_id"))
            forever.append(
                _playtime(item.get("playtime_forever", 0), label="playtime_forever")
            )
            recent.append(
                _playtime(item.get("playtime_2weeks", 0), label="playtime_2weeks")
            )

    columns = canonicalize_source_columns(
        np.frombuffer(users, dtype=np.int64),
        np.frombuffer(items, dtype=np.int64),
        np.frombuffer(forever, dtype=np.float64),
        np.frombuffer(recent, dtype=np.float64),
    )
    reverse_collisions = sum(len(names) > 1 for names in reverse_map.values())
    active_users = int(np.unique(columns["user_id"]).size)
    diagnostics = {
        "raw_record_count": record_count,
        "zero_item_record_count": zero_item_record_count,
        "raw_interaction_row_count": len(users),
        "canonical_edge_count": int(columns["user_id"].size),
        "duplicate_excess_rows": int(len(users) - columns["user_id"].size),
        "active_user_count": active_users,
        "item_count": int(np.unique(columns["item_id"]).size),
        "unique_user_id_steam_id_pairs": len(mapping_pairs),
        "steam_id_reverse_collision_count": reverse_collisions,
    }
    return columns, diagnostics


def _literal_records(path: Path) -> Iterable[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = ast.literal_eval(line)
            except (SyntaxError, ValueError) as exc:
                raise ValueError(f"could not parse raw line {line_number}") from exc
            yield value


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _write_csv(columns: Mapping[str, np.ndarray], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            for start in range(0, len(columns["user_id"]), 250_000):
                stop = min(start + 250_000, len(columns["user_id"]))
                frame = pd.DataFrame(
                    {name: np.asarray(columns[name])[start:stop] for name in OUTPUT_COLUMNS}
                )
                frame.to_csv(
                    handle,
                    index=False,
                    header=start == 0,
                    lineterminator="\n",
                    float_format="%.17g",
                )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_source_manifest(
    *,
    cycle_id: str,
    raw_path: Path,
    output_path: Path,
    diagnostics: Mapping[str, int],
    project_root: Path,
) -> dict[str, Any]:
    source_hash = file_sha256(raw_path)
    output_hash = file_sha256(output_path)
    code_hash = file_sha256(Path(__file__))
    contract = {
        "user_identifier": "steam_id",
        "item_identifier": "items[].item_id",
        "duplicate_rule": "fieldwise_max_nonnegative_playtime",
        "ordering": "ascending_numeric_steam_id_then_item_id",
        "output_columns": list(OUTPUT_COLUMNS),
        "protected_outcomes_accessed": False,
    }
    identity = {
        "cycle_id": cycle_id,
        "raw_sha256": source_hash,
        "output_sha256": output_hash,
        "contract": contract,
        "diagnostics": dict(diagnostics),
    }
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "cycle_id": cycle_id,
        "source_set_id": semantic_sha256(identity),
        "contract": contract,
        "diagnostics": dict(diagnostics),
        "inputs": {
            "raw": {
                "path": _relative(raw_path, project_root),
                "size_bytes": raw_path.stat().st_size,
                "sha256": source_hash,
            },
            "code": {
                "path": _relative(Path(__file__), project_root),
                "sha256": code_hash,
            },
        },
        "artifact": {
            "path": _relative(output_path, project_root),
            "size_bytes": output_path.stat().st_size,
            "sha256": output_hash,
        },
    }
    manifest["manifest_id"] = semantic_sha256(manifest)
    return manifest


def generate_source_interactions(
    *,
    raw_path: str | Path = DEFAULT_RAW_PATH,
    output_path: str | Path = DEFAULT_OUTPUT,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    cycle_id: str = DEFAULT_CYCLE_ID,
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Generate or exactly verify the no-clobber S1-v2 source artifact."""

    raw = Path(raw_path)
    output = Path(output_path)
    manifest_destination = Path(manifest_path)
    root = Path(project_root)
    if output.exists() or manifest_destination.exists():
        if not output.exists() or not manifest_destination.exists():
            raise FileExistsError("partial source publication already exists")
        saved = json.loads(manifest_destination.read_text(encoding="utf-8"))
        unsigned = dict(saved)
        claimed = unsigned.pop("manifest_id", None)
        if claimed != semantic_sha256(unsigned):
            raise ValueError("source manifest semantic hash mismatch")
        if saved.get("cycle_id") != cycle_id:
            raise FileExistsError("source publication belongs to another cycle")
        if file_sha256(output) != saved.get("artifact", {}).get("sha256"):
            raise ValueError("source interaction artifact hash mismatch")
        if file_sha256(raw) != saved.get("inputs", {}).get("raw", {}).get("sha256"):
            raise ValueError("raw source hash changed")
        return saved

    columns, diagnostics = records_to_columns(_literal_records(raw))
    _write_csv(columns, output)
    manifest = build_source_manifest(
        cycle_id=cycle_id,
        raw_path=raw,
        output_path=output,
        diagnostics=diagnostics,
        project_root=root,
    )
    manifest_destination.parent.mkdir(parents=True, exist_ok=True)
    manifest_destination.write_bytes(canonical_json_bytes(manifest) + b"\n")
    return manifest


def verify_source_interactions(
    *,
    raw_path: str | Path = DEFAULT_RAW_PATH,
    output_path: str | Path = DEFAULT_OUTPUT,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    cycle_id: str = DEFAULT_CYCLE_ID,
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Verify hashes and schema without rebuilding the multi-gigabyte table."""

    manifest = generate_source_interactions(
        raw_path=raw_path,
        output_path=output_path,
        manifest_path=manifest_path,
        cycle_id=cycle_id,
        project_root=project_root,
    )
    header = pd.read_csv(output_path, nrows=0).columns.tolist()
    if header != list(OUTPUT_COLUMNS):
        raise ValueError("source interaction CSV schema changed")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the steam_id Stage 1 source")
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cycle-id", default=DEFAULT_CYCLE_ID)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    function = verify_source_interactions if args.check_only else generate_source_interactions
    manifest = function(
        raw_path=args.raw,
        output_path=args.output,
        manifest_path=args.manifest,
        cycle_id=args.cycle_id,
    )
    print(
        json.dumps(
            {
                "cycle_id": manifest["cycle_id"],
                "source_set_id": manifest["source_set_id"],
                **manifest["diagnostics"],
                "wrote_artifact": not args.check_only,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
