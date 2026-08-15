"""Reproducible static audit of the observed Steam bundle catalogue.

The raw files used by this project have ``.json`` suffixes but contain one
Python-literal dictionary per line.  They are parsed with ``ast.literal_eval``
(never ``eval``), cross-referenced by exact application ID, and reduced to the
tracked bundle-mechanism audit schema.

This module deliberately supports only a *static catalogue cross-reference*.
It cannot identify Steam's historical mechanism, component exclusivity,
ownership-adjusted pricing, or indivisible-key status.  In particular, zero
affirmative SBR observations is not evidence that SBR never existed.

The pure transformation functions sort output by normalized bundle ID, so
permuting the input record order cannot change the generated CSV.  The original
frozen CSV predates this module and retained raw-file order; its rows reconcile
exactly by bundle ID even though its byte order is different.

Example (run from any directory)::

    python -m src.mechanism_audit

To compare with the frozen table without overwriting it::

    python -m src.mechanism_audit --check-only \
        --compare-to outputs/tables/bundle_mechanism_audit.csv
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import csv
import hashlib
import io
import json
from pathlib import Path
import platform
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE_PATH = PROJECT_ROOT / "data" / "raw" / "bundle_data.json"
DEFAULT_CATALOGUE_PATH = PROJECT_ROOT / "data" / "raw" / "steam_games.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "tables" / "bundle_mechanism_audit.csv"
DEFAULT_MANIFEST_PATH = (
    PROJECT_ROOT / "outputs" / "tables" / "bundle_mechanism_audit_manifest.json"
)

AUDIT_COLUMNS = (
    "bundle_id",
    "bundle_name",
    "n_items",
    "n_items_in_catalogue",
    "catalogue_coverage",
    "standalone_price_coverage",
    "n_distinct_publishers",
    "publisher_coherent",
    "developer_coherent",
    "n_missing_item_id",
    "mechanism_class",
    "ownership_adjusted",
    "indivisible",
    "confidence",
    "evidence",
)

SBA_LIKE = "B_SBA_like"
UNCLEAR = "E_unclear"
NOT_OBSERVABLE = "not_observable"


@dataclass(frozen=True)
class CatalogueMetadata:
    """Order-invariant publisher/developer evidence for one application ID.

    Duplicate catalogue records are combined rather than resolved by first/last
    occurrence.  This matters for order invariance and avoids silently choosing
    one conflicting metadata record.  The project snapshot's duplicate ID has
    identical publisher and developer labels, so combination reproduces the
    frozen audit exactly.
    """

    publishers: frozenset[str]
    developers: frozenset[str]


def parse_literal_records(
    lines: Iterable[str], *, source: str = "<memory>"
) -> list[dict[str, Any]]:
    """Parse one Python-literal mapping per nonblank line.

    The Kaggle snapshot is not strict JSON: its lines contain Python unicode
    prefixes and single-quoted dictionaries.  ``ast.literal_eval`` supports the
    format without permitting arbitrary code execution.
    """

    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = ast.literal_eval(line)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(
                f"could not parse {source} line {line_number} as a literal mapping"
            ) from exc
        if not isinstance(record, dict):
            raise ValueError(
                f"expected a mapping in {source} line {line_number}, "
                f"got {type(record).__name__}"
            )
        records.append(record)
    return records


def load_literal_records(path: str | Path) -> list[dict[str, Any]]:
    """Load the project's line-delimited Python-literal records from ``path``."""

    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return parse_literal_records(handle, source=str(path))


def _normalized_id(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _metadata_value(value: Any) -> str | None:
    """Return a raw nonblank metadata label without case/punctuation merging."""

    if value is None:
        return None
    text = str(value)
    return text if text.strip() else None


def _has_observed_value(record: Mapping[str, Any], key: str) -> bool:
    """Treat numeric zero as observed while rejecting absent/null/blank values."""

    if key not in record or record[key] is None:
        return False
    value = record[key]
    return not (isinstance(value, str) and not value.strip())


def build_catalogue_index(
    catalogue_records: Iterable[Mapping[str, Any]],
) -> dict[str, CatalogueMetadata]:
    """Build an order-invariant application-ID metadata index.

    Records without an ``id`` are outside the cross-reference universe.  For a
    repeated ID, all observed publisher/developer labels are retained.  The
    resulting distinct-label counts are therefore independent of source order.
    """

    publishers: dict[str, set[str]] = {}
    developers: dict[str, set[str]] = {}
    for record in catalogue_records:
        item_id = _normalized_id(record.get("id"))
        if item_id is None:
            continue
        publishers.setdefault(item_id, set())
        developers.setdefault(item_id, set())
        publisher = _metadata_value(record.get("publisher"))
        developer = _metadata_value(record.get("developer"))
        if publisher is not None:
            publishers[item_id].add(publisher)
        if developer is not None:
            developers[item_id].add(developer)

    return {
        item_id: CatalogueMetadata(
            publishers=frozenset(publishers[item_id]),
            developers=frozenset(developers[item_id]),
        )
        for item_id in publishers
    }


def _confidence(catalogue_coverage: float) -> str:
    if catalogue_coverage >= 0.8:
        return "high"
    if catalogue_coverage >= 0.5:
        return "medium"
    return "low"


def _evidence_text(
    *,
    n_items: int,
    n_items_in_catalogue: int,
    n_items_with_price: int,
) -> str:
    """Construct the frozen evidence text, with safe incomplete-data fallbacks."""

    if n_items == 0:
        return "bundle record contains no components"

    all_prices = n_items_with_price == n_items
    coverage = n_items_in_catalogue / n_items
    if n_items_in_catalogue == 0:
        if all_prices:
            return (
                "no components found in catalogue snapshot (likely catalogue "
                "incompleteness); standalone prices still shown"
            )
        return (
            "no components found in catalogue snapshot; standalone-price "
            f"evidence available for {n_items_with_price}/{n_items} components"
        )

    if all_prices and coverage >= 0.5:
        return (
            f"{n_items_in_catalogue}/{n_items} components individually listed in "
            "catalogue; all show a standalone price"
        )
    if all_prices:
        return (
            f"{n_items_in_catalogue}/{n_items} components individually listed; "
            "standalone prices shown for all"
        )
    return (
        f"{n_items_in_catalogue}/{n_items} components individually listed; "
        f"standalone prices shown for {n_items_with_price}/{n_items}"
    )


def audit_bundle(
    bundle: Mapping[str, Any],
    catalogue_index: Mapping[str, CatalogueMetadata],
) -> dict[str, Any]:
    """Audit one bundle against a prebuilt static catalogue index.

    ``B_SBA_like`` requires two pieces of static evidence: at least one exact
    catalogue match and a displayed standalone price for every component.
    Everything else is ``E_unclear``.  No input field identifies affirmative
    SBR exclusivity, ownership-adjusted pricing, or indivisible packages, so the
    function never manufactures those classifications.
    """

    bundle_id = _normalized_id(bundle.get("bundle_id"))
    if bundle_id is None:
        raise ValueError("bundle record is missing bundle_id")

    raw_items = bundle.get("items", [])
    if raw_items is None:
        raw_items = []
    if isinstance(raw_items, (str, bytes)) or not isinstance(raw_items, Sequence):
        raise ValueError(f"bundle {bundle_id} has a non-sequence items field")

    items: list[Mapping[str, Any]] = []
    for position, item in enumerate(raw_items):
        if not isinstance(item, Mapping):
            raise ValueError(
                f"bundle {bundle_id} item {position} is not a mapping"
            )
        items.append(item)

    n_items = len(items)
    n_missing_item_id = 0
    n_items_in_catalogue = 0
    n_items_with_price = 0
    publishers: set[str] = set()
    developers: set[str] = set()

    for item in items:
        item_id = _normalized_id(item.get("item_id"))
        if item_id is None:
            n_missing_item_id += 1
        elif item_id in catalogue_index:
            n_items_in_catalogue += 1
            metadata = catalogue_index[item_id]
            publishers.update(metadata.publishers)
            developers.update(metadata.developers)
        if _has_observed_value(item, "discounted_price"):
            n_items_with_price += 1

    if n_items:
        raw_catalogue_coverage = n_items_in_catalogue / n_items
        raw_price_coverage = n_items_with_price / n_items
    else:
        raw_catalogue_coverage = 0.0
        raw_price_coverage = 0.0

    all_prices = n_items > 0 and n_items_with_price == n_items
    mechanism_class = (
        SBA_LIKE if n_items_in_catalogue > 0 and all_prices else UNCLEAR
    )
    bundle_name = bundle.get("bundle_name")

    return {
        "bundle_id": bundle_id,
        "bundle_name": "" if bundle_name is None else str(bundle_name),
        "n_items": n_items,
        "n_items_in_catalogue": n_items_in_catalogue,
        "catalogue_coverage": round(raw_catalogue_coverage, 3),
        "standalone_price_coverage": round(raw_price_coverage, 3),
        "n_distinct_publishers": len(publishers),
        "publisher_coherent": len(publishers) == 1,
        "developer_coherent": len(developers) == 1,
        "n_missing_item_id": n_missing_item_id,
        "mechanism_class": mechanism_class,
        "ownership_adjusted": NOT_OBSERVABLE,
        "indivisible": NOT_OBSERVABLE,
        "confidence": _confidence(raw_catalogue_coverage),
        "evidence": _evidence_text(
            n_items=n_items,
            n_items_in_catalogue=n_items_in_catalogue,
            n_items_with_price=n_items_with_price,
        ),
    }


def _bundle_sort_key(row: Mapping[str, Any]) -> tuple[int, int, str]:
    bundle_id = str(row["bundle_id"])
    try:
        numeric_id = int(bundle_id)
    except ValueError:
        return (1, 0, bundle_id)
    return (0, numeric_id, bundle_id)


def build_audit_rows(
    bundle_records: Iterable[Mapping[str, Any]],
    catalogue_records: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Pure, order-invariant transformation from raw records to audit rows."""

    catalogue_index = build_catalogue_index(catalogue_records)
    rows = [audit_bundle(bundle, catalogue_index) for bundle in bundle_records]

    bundle_ids = [str(row["bundle_id"]) for row in rows]
    duplicate_ids = sorted(
        bundle_id for bundle_id, count in Counter(bundle_ids).items() if count > 1
    )
    if duplicate_ids:
        preview = ", ".join(duplicate_ids[:5])
        raise ValueError(f"duplicate bundle_id values are ambiguous: {preview}")

    return sorted(rows, key=_bundle_sort_key)


def serialize_audit_csv(rows: Iterable[Mapping[str, Any]]) -> bytes:
    """Serialize audit rows with the tracked schema and stable CRLF dialect."""

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=AUDIT_COLUMNS,
        extrasaction="raise",
        lineterminator="\r\n",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row[column] for column in AUDIT_COLUMNS})
    return buffer.getvalue().encode("utf-8")


def write_audit_csv(rows: Iterable[Mapping[str, Any]], path: str | Path) -> None:
    """Write a serialized audit CSV, creating only its parent directory."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(serialize_audit_csv(rows))


def read_audit_csv(path: str | Path) -> list[dict[str, str]]:
    """Read an audit CSV for exact cell-level comparison."""

    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != AUDIT_COLUMNS:
            raise ValueError(
                f"unexpected audit columns in {path}: {reader.fieldnames!r}"
            )
        return list(reader)


def _csv_cell(value: Any) -> str:
    return "" if value is None else str(value)


def compare_audit_rows(
    expected_rows: Iterable[Mapping[str, Any]],
    actual_rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare every CSV cell by bundle ID, intentionally ignoring row order."""

    def index(rows: Iterable[Mapping[str, Any]], label: str) -> dict[str, Mapping[str, Any]]:
        result: dict[str, Mapping[str, Any]] = {}
        for row in rows:
            bundle_id = _csv_cell(row.get("bundle_id"))
            if bundle_id in result:
                raise ValueError(f"duplicate bundle_id {bundle_id!r} in {label} rows")
            result[bundle_id] = row
        return result

    expected = index(expected_rows, "expected")
    actual = index(actual_rows, "actual")
    missing = sorted(set(expected) - set(actual), key=lambda value: (len(value), value))
    extra = sorted(set(actual) - set(expected), key=lambda value: (len(value), value))
    mismatches: dict[str, dict[str, list[str]]] = {}
    for bundle_id in sorted(set(expected) & set(actual), key=lambda value: (len(value), value)):
        field_mismatches: dict[str, list[str]] = {}
        for column in AUDIT_COLUMNS:
            expected_value = _csv_cell(expected[bundle_id].get(column))
            actual_value = _csv_cell(actual[bundle_id].get(column))
            if expected_value != actual_value:
                field_mismatches[column] = [expected_value, actual_value]
        if field_mismatches:
            mismatches[bundle_id] = field_mismatches

    return {
        "equal": not missing and not extra and not mismatches,
        "expected_row_count": len(expected),
        "actual_row_count": len(actual),
        "missing_bundle_ids": missing,
        "extra_bundle_ids": extra,
        "field_mismatches": mismatches,
    }


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def summarize_audit(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Return the auditable headline counts used in the mechanism memo."""

    rows = list(rows)
    mechanism_counts = Counter(str(row.get("mechanism_class", "")) for row in rows)
    confidence_counts = Counter(str(row.get("confidence", "")) for row in rows)
    affirmative_sbr = sum(
        "SBR" in str(row.get("mechanism_class", "")).upper() for row in rows
    )
    return {
        "row_count": len(rows),
        "mechanism_counts": dict(sorted(mechanism_counts.items())),
        "sba_like_count": mechanism_counts.get(SBA_LIKE, 0),
        "unclear_count": mechanism_counts.get(UNCLEAR, 0),
        "affirmative_sbr_evidence_count": affirmative_sbr,
        "full_standalone_price_coverage_count": sum(
            _as_float(row.get("standalone_price_coverage")) == 1.0 for row in rows
        ),
        "complete_catalogue_confirmation_count": sum(
            _as_float(row.get("catalogue_coverage")) == 1.0 for row in rows
        ),
        "high_confidence_count": confidence_counts.get("high", 0),
        "high_confidence_sba_like_count": sum(
            row.get("mechanism_class") == SBA_LIKE and row.get("confidence") == "high"
            for row in rows
        ),
        "single_publisher_count": sum(
            _as_bool(row.get("publisher_coherent")) for row in rows
        ),
    }


def _jsonable(value: Any) -> Any:
    """Convert literal-data values to a canonical JSON-compatible structure."""

    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        converted = [_jsonable(item) for item in value]
        return sorted(
            converted,
            key=lambda item: json.dumps(
                item, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ),
        )
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"cannot canonicalize value of type {type(value).__name__}")


def canonical_records_sha256(records: Iterable[Mapping[str, Any]]) -> str:
    """Hash a record multiset independently of top-level input row order."""

    serialized = [
        json.dumps(
            _jsonable(record),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        for record in records
    ]
    payload = ("\n".join(sorted(serialized)) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def bundle_id_order_sha256(rows: Iterable[Mapping[str, Any]]) -> str:
    """Hash the exact bundle-ID row sequence (unlike the order-free row check)."""

    bundle_ids = [_csv_cell(row.get("bundle_id")) for row in rows]
    payload = json.dumps(
        bundle_ids, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _display_path(path: str | Path) -> str:
    path = Path(path)
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def build_manifest(
    *,
    rows: Sequence[Mapping[str, Any]],
    bundle_records: Sequence[Mapping[str, Any]],
    catalogue_records: Sequence[Mapping[str, Any]],
    bundle_path: str | Path,
    catalogue_path: str | Path,
    output_path: str | Path,
    producing_command: str,
    code_sha256: str,
    generated_at_utc: str | None = None,
    existing_artifact_rows: Sequence[Mapping[str, Any]] | None = None,
    existing_artifact_payload: bytes | None = None,
) -> dict[str, Any]:
    """Build a deterministic provenance and identification-boundary manifest.

    ``generated_at_utc`` is null by default so identical semantic inputs produce
    byte-identical artifacts.  A freeze workflow may pass an ISO-8601 timestamp
    explicitly; the manifest records the policy either way.
    """

    canonical_payload = serialize_audit_csv(rows)
    if (existing_artifact_rows is None) != (existing_artifact_payload is None):
        raise ValueError(
            "existing_artifact_rows and existing_artifact_payload must be supplied together"
        )
    preserving_existing_artifact = existing_artifact_rows is not None
    if existing_artifact_rows is None:
        artifact_rows: Sequence[Mapping[str, Any]] = rows
        artifact_payload = canonical_payload
        artifact_row_order = "ascending normalized bundle_id; numeric IDs first"
    else:
        comparison = compare_audit_rows(rows, existing_artifact_rows)
        if not comparison["equal"]:
            raise ValueError(
                "existing artifact does not reconcile with canonical audit rows"
            )
        artifact_rows = existing_artifact_rows
        artifact_payload = existing_artifact_payload
        artifact_row_order = "preserved existing artifact order; CSV not rewritten"

    return {
        "schema_version": 2,
        "artifact": "bundle_mechanism_audit",
        "generated_at_utc": generated_at_utc,
        "timestamp_policy": (
            "explicit CLI value"
            if generated_at_utc is not None
            else "null for reproducible build; pass --generated-at-utc when freezing"
        ),
        "identification_boundary": {
            "evidence_scope": "static bundle/catalogue snapshot cross-reference only",
            "historical_mechanism_identified": False,
            "component_exclusivity_identified": False,
            "ownership_adjusted_pricing_observable": False,
            "indivisible_package_status_observable": False,
            "zero_affirmative_sbr_interpretation": (
                "absence of affirmative SBR evidence, not proven absence of SBR"
            ),
        },
        "inputs": {
            "bundle_records": {
                "path": _display_path(bundle_path),
                "record_count": len(bundle_records),
                "order_invariant_sha256": canonical_records_sha256(bundle_records),
            },
            "catalogue_records": {
                "path": _display_path(catalogue_path),
                "record_count": len(catalogue_records),
                "order_invariant_sha256": canonical_records_sha256(catalogue_records),
            },
        },
        "output": {
            "path": _display_path(output_path),
            "row_count": len(artifact_rows),
            "columns": list(AUDIT_COLUMNS),
            "row_order": artifact_row_order,
            "bundle_id_order_sha256": bundle_id_order_sha256(artifact_rows),
            "sha256": hashlib.sha256(artifact_payload).hexdigest(),
            "size_bytes": len(artifact_payload),
            "preserved_existing_artifact": preserving_existing_artifact,
            "verified_equal_by_bundle_id_and_cell": preserving_existing_artifact,
        },
        "canonical_generation": {
            "row_count": len(rows),
            "columns": list(AUDIT_COLUMNS),
            "row_order": "ascending normalized bundle_id; numeric IDs first",
            "bundle_id_order_sha256": bundle_id_order_sha256(rows),
            "sha256": hashlib.sha256(canonical_payload).hexdigest(),
            "size_bytes": len(canonical_payload),
        },
        "method": {
            "id_match": "exact normalized string application ID",
            "sba_like_rule": (
                "at least one catalogue-confirmed component and standalone-price "
                "evidence for every component"
            ),
            "confidence_rule": {
                "low": "catalogue coverage < 0.5",
                "medium": "0.5 <= catalogue coverage < 0.8",
                "high": "catalogue coverage >= 0.8",
            },
            "duplicate_catalogue_ids": (
                "union publisher/developer labels; never first/last-row resolution"
            ),
            "random_seeds": [],
        },
        "headline_counts": summarize_audit(rows),
        "runtime": {
            "python_version": platform.python_version(),
            "module_sha256": code_sha256,
        },
        "producing_command": producing_command,
    }


def write_manifest(manifest: Mapping[str, Any], path: str | Path) -> None:
    """Write a stable, human-readable JSON manifest."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.write_text(payload, encoding="utf-8", newline="\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate the static Steam bundle-mechanism audit and provenance manifest."
        )
    )
    parser.add_argument("--bundles", type=Path, default=DEFAULT_BUNDLE_PATH)
    parser.add_argument("--catalogue", type=Path, default=DEFAULT_CATALOGUE_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument(
        "--compare-to",
        type=Path,
        help="compare every cell by bundle ID before writing; return nonzero on mismatch",
    )
    write_mode = parser.add_mutually_exclusive_group()
    write_mode.add_argument(
        "--check-only",
        action="store_true",
        help="build and optionally compare in memory without writing CSV or manifest",
    )
    write_mode.add_argument(
        "--manifest-only",
        action="store_true",
        help=(
            "verify an existing --compare-to or --output CSV and write only its "
            "provenance manifest; never rewrite the CSV"
        ),
    )
    parser.add_argument(
        "--generated-at-utc",
        help="optional explicit ISO-8601 freeze timestamp; omitted for reproducible output",
    )
    return parser


def _command_for_manifest(args: argparse.Namespace) -> str:
    parts = [
        "python",
        "-m",
        "src.mechanism_audit",
        "--bundles",
        _display_path(args.bundles),
        "--catalogue",
        _display_path(args.catalogue),
        "--output",
        _display_path(args.output),
        "--manifest",
        _display_path(args.manifest),
    ]
    if args.generated_at_utc is not None:
        parts.extend(["--generated-at-utc", args.generated_at_utc])
    if args.compare_to is not None:
        parts.extend(["--compare-to", _display_path(args.compare_to)])
    if args.manifest_only:
        parts.append("--manifest-only")
    return " ".join(parts)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; returns zero only after generation/comparison succeeds."""

    args = _parser().parse_args(argv)

    existing_artifact_path = None
    if args.manifest_only:
        existing_artifact_path = args.compare_to or args.output
        if not existing_artifact_path.is_file():
            print(
                f"manifest-only artifact does not exist: {existing_artifact_path}",
                file=sys.stderr,
            )
            return 2
        if existing_artifact_path.resolve() == args.manifest.resolve():
            print(
                "--manifest must not point to the existing CSV in --manifest-only mode",
                file=sys.stderr,
            )
            return 2

    bundle_records = load_literal_records(args.bundles)
    catalogue_records = load_literal_records(args.catalogue)
    rows = build_audit_rows(bundle_records, catalogue_records)

    comparison = None
    existing_artifact_rows = None
    existing_artifact_payload = None
    if args.manifest_only:
        # Read and verify before creating the manifest directory or touching any
        # output.  A mismatch therefore leaves both the CSV and any pre-existing
        # manifest byte-for-byte unchanged.
        existing_artifact_payload = existing_artifact_path.read_bytes()
        existing_artifact_rows = read_audit_csv(existing_artifact_path)
        comparison = compare_audit_rows(existing_artifact_rows, rows)
    elif args.compare_to is not None:
        comparison = compare_audit_rows(read_audit_csv(args.compare_to), rows)

    if comparison is not None:
        if not comparison["equal"]:
            print(json.dumps(comparison, indent=2, sort_keys=True), file=sys.stderr)
            return 1

    if not args.check_only and not args.manifest_only:
        write_audit_csv(rows, args.output)

    if not args.check_only:
        code_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        manifest = build_manifest(
            rows=rows,
            bundle_records=bundle_records,
            catalogue_records=catalogue_records,
            bundle_path=args.bundles,
            catalogue_path=args.catalogue,
            output_path=existing_artifact_path or args.output,
            producing_command=_command_for_manifest(args),
            code_sha256=code_sha256,
            generated_at_utc=args.generated_at_utc,
            existing_artifact_rows=existing_artifact_rows,
            existing_artifact_payload=existing_artifact_payload,
        )
        write_manifest(manifest, args.manifest)

    result = summarize_audit(rows)
    if comparison is not None:
        result["comparison_equal"] = comparison["equal"]
    result["wrote_csv"] = not args.check_only and not args.manifest_only
    result["wrote_manifest"] = not args.check_only
    result["wrote_artifacts"] = result["wrote_csv"] or result["wrote_manifest"]
    result["manifest_only"] = args.manifest_only
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through ``main`` tests
    raise SystemExit(main())
