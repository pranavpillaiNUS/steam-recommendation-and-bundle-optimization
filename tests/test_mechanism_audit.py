"""Focused tests for the reproducible static bundle-mechanism audit."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import pytest


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import mechanism_audit as ma  # noqa: E402


def _item(item_id, price="$1.99"):
    return {"item_id": item_id, "discounted_price": price}


def _synthetic_records():
    catalogue = [
        {"id": "1", "publisher": "P", "developer": "D1"},
        {"id": "2", "publisher": "P", "developer": "D1"},
        {"id": "3", "publisher": "P", "developer": "D2"},
        {"id": "4", "publisher": "P", "developer": "D2"},
    ]
    bundles = [
        {
            "bundle_id": "20",
            "bundle_name": "High coverage",
            "items": [_item("1", 0), _item("2"), _item("3"), _item("4"), _item("x")],
        },
        {
            "bundle_id": "3",
            "bundle_name": "Low coverage",
            "items": [_item("1"), _item("x"), _item(None)],
        },
        {
            "bundle_id": "11",
            "bundle_name": "Outside catalogue",
            "items": [_item("x"), _item("y")],
        },
    ]
    return bundles, catalogue


def test_synthetic_classification_reproduces_frozen_rules():
    bundles, catalogue = _synthetic_records()
    rows = {row["bundle_id"]: row for row in ma.build_audit_rows(bundles, catalogue)}

    high = rows["20"]
    assert high["n_items"] == 5
    assert high["n_items_in_catalogue"] == 4
    assert high["catalogue_coverage"] == 0.8
    assert high["standalone_price_coverage"] == 1.0  # numeric zero is observed
    assert high["mechanism_class"] == ma.SBA_LIKE
    assert high["confidence"] == "high"
    assert high["publisher_coherent"] is True
    assert high["developer_coherent"] is False
    assert high["evidence"] == (
        "4/5 components individually listed in catalogue; all show a standalone price"
    )

    low = rows["3"]
    assert low["catalogue_coverage"] == 0.333
    assert low["n_missing_item_id"] == 1
    assert low["mechanism_class"] == ma.SBA_LIKE
    assert low["confidence"] == "low"
    assert low["evidence"] == (
        "1/3 components individually listed; standalone prices shown for all"
    )

    unclear = rows["11"]
    assert unclear["mechanism_class"] == ma.UNCLEAR
    assert unclear["ownership_adjusted"] == ma.NOT_OBSERVABLE
    assert unclear["indivisible"] == ma.NOT_OBSERVABLE
    assert unclear["evidence"] == (
        "no components found in catalogue snapshot (likely catalogue incompleteness); "
        "standalone prices still shown"
    )


def test_transform_and_serialization_are_order_invariant():
    bundles, catalogue = _synthetic_records()
    # Conflicting duplicate metadata are unioned, not selected by first/last row.
    catalogue.append({"id": "1", "publisher": "Q", "developer": "D1"})
    baseline = ma.build_audit_rows(bundles, catalogue)

    shuffled_bundles = copy.deepcopy(list(reversed(bundles)))
    for bundle in shuffled_bundles:
        bundle["items"] = list(reversed(bundle["items"]))
    reordered = ma.build_audit_rows(shuffled_bundles, list(reversed(catalogue)))

    assert baseline == reordered
    assert ma.serialize_audit_csv(baseline) == ma.serialize_audit_csv(reordered)
    assert [row["bundle_id"] for row in baseline] == ["3", "11", "20"]
    assert baseline[0]["n_distinct_publishers"] == 2


def test_literal_parser_is_safe_and_reports_bad_rows():
    parsed = ma.parse_literal_records(
        ["{'bundle_id': u'1', 'items': []}\n", "\n"], source="synthetic"
    )
    assert parsed == [{"bundle_id": "1", "items": []}]
    with pytest.raises(ValueError, match="synthetic line 1"):
        ma.parse_literal_records(["__import__('os').getcwd()\n"], source="synthetic")
    with pytest.raises(ValueError, match="expected a mapping"):
        ma.parse_literal_records(["[1, 2]\n"], source="synthetic")


def test_cli_writes_csv_manifest_and_can_check_without_overwrite(tmp_path, capsys):
    bundles, catalogue = _synthetic_records()
    bundle_path = tmp_path / "bundles.json"
    catalogue_path = tmp_path / "catalogue.json"
    output_path = tmp_path / "audit.csv"
    manifest_path = tmp_path / "audit_manifest.json"
    bundle_path.write_text("\n".join(map(repr, reversed(bundles))) + "\n", encoding="utf-8")
    catalogue_path.write_text("\n".join(map(repr, catalogue)) + "\n", encoding="utf-8")

    result = ma.main(
        [
            "--bundles",
            str(bundle_path),
            "--catalogue",
            str(catalogue_path),
            "--output",
            str(output_path),
            "--manifest",
            str(manifest_path),
            "--generated-at-utc",
            "2026-07-17T00:00:00Z",
        ]
    )
    assert result == 0
    with output_path.open(encoding="utf-8", newline="") as handle:
        output_rows = list(csv.DictReader(handle))
    assert [row["bundle_id"] for row in output_rows] == ["3", "11", "20"]

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["headline_counts"]["row_count"] == 3
    boundary = manifest["identification_boundary"]
    assert boundary["historical_mechanism_identified"] is False
    assert boundary["component_exclusivity_identified"] is False
    assert "not proven absence" in boundary["zero_affirmative_sbr_interpretation"]

    before_csv = output_path.read_bytes()
    before_manifest = manifest_path.read_bytes()
    check_result = ma.main(
        [
            "--bundles",
            str(bundle_path),
            "--catalogue",
            str(catalogue_path),
            "--compare-to",
            str(output_path),
            "--check-only",
        ]
    )
    assert check_result == 0
    assert output_path.read_bytes() == before_csv
    assert manifest_path.read_bytes() == before_manifest
    assert "comparison_equal" in capsys.readouterr().out


def test_manifest_only_hashes_preserved_order_and_is_atomic_on_mismatch(
    tmp_path, capsys
):
    bundles, catalogue = _synthetic_records()
    bundle_path = tmp_path / "bundles.json"
    catalogue_path = tmp_path / "catalogue.json"
    existing_path = tmp_path / "existing_audit.csv"
    manifest_path = tmp_path / "audit_manifest.json"
    bundle_path.write_text("\n".join(map(repr, bundles)) + "\n", encoding="utf-8")
    catalogue_path.write_text("\n".join(map(repr, catalogue)) + "\n", encoding="utf-8")

    index = ma.build_catalogue_index(catalogue)
    existing_rows = [ma.audit_bundle(bundle, index) for bundle in bundles]
    ma.write_audit_csv(existing_rows, existing_path)  # deliberately noncanonical order
    existing_before = existing_path.read_bytes()
    canonical_rows = ma.build_audit_rows(bundles, catalogue)
    canonical_payload = ma.serialize_audit_csv(canonical_rows)
    assert existing_before != canonical_payload

    result = ma.main(
        [
            "--bundles",
            str(bundle_path),
            "--catalogue",
            str(catalogue_path),
            "--output",
            str(existing_path),
            "--manifest",
            str(manifest_path),
            "--manifest-only",
        ]
    )
    assert result == 0
    assert existing_path.read_bytes() == existing_before

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output = manifest["output"]
    canonical = manifest["canonical_generation"]
    assert output["preserved_existing_artifact"] is True
    assert output["verified_equal_by_bundle_id_and_cell"] is True
    assert output["sha256"] == hashlib.sha256(existing_before).hexdigest()
    assert canonical["sha256"] == hashlib.sha256(canonical_payload).hexdigest()
    assert output["bundle_id_order_sha256"] != canonical["bundle_id_order_sha256"]
    assert "CSV not rewritten" in output["row_order"]

    # A cell mismatch returns nonzero before touching an existing manifest.
    mismatched_path = tmp_path / "mismatched_audit.csv"
    mismatched_rows = copy.deepcopy(existing_rows)
    mismatched_rows[0]["bundle_name"] = "tampered"
    ma.write_audit_csv(mismatched_rows, mismatched_path)
    mismatched_before = mismatched_path.read_bytes()
    sentinel = b"pre-existing manifest sentinel\n"
    manifest_path.write_bytes(sentinel)
    mismatch_result = ma.main(
        [
            "--bundles",
            str(bundle_path),
            "--catalogue",
            str(catalogue_path),
            "--compare-to",
            str(mismatched_path),
            "--manifest",
            str(manifest_path),
            "--manifest-only",
        ]
    )
    assert mismatch_result == 1
    assert mismatched_path.read_bytes() == mismatched_before
    assert manifest_path.read_bytes() == sentinel
    assert "field_mismatches" in capsys.readouterr().err


def test_manifest_builder_cannot_label_an_unverified_artifact_as_verified():
    bundles, catalogue = _synthetic_records()
    canonical_rows = ma.build_audit_rows(bundles, catalogue)
    tampered_rows = copy.deepcopy(canonical_rows)
    tampered_rows[0]["bundle_name"] = "tampered"

    with pytest.raises(ValueError, match="does not reconcile"):
        ma.build_manifest(
            rows=canonical_rows,
            bundle_records=bundles,
            catalogue_records=catalogue,
            bundle_path="bundles.json",
            catalogue_path="catalogue.json",
            output_path="audit.csv",
            producing_command="test",
            code_sha256="0" * 64,
            existing_artifact_rows=tampered_rows,
            existing_artifact_payload=ma.serialize_audit_csv(tampered_rows),
        )


def test_full_snapshot_reconciles_every_frozen_row_and_headline_count():
    project_root = Path(__file__).resolve().parents[1]
    bundle_path = project_root / "data" / "raw" / "bundle_data.json"
    catalogue_path = project_root / "data" / "raw" / "steam_games.json"
    frozen_path = project_root / "outputs" / "tables" / "bundle_mechanism_audit.csv"
    if not (bundle_path.exists() and catalogue_path.exists() and frozen_path.exists()):
        pytest.skip("ignored raw snapshot and/or frozen audit is unavailable")

    bundles = ma.load_literal_records(bundle_path)
    catalogue = ma.load_literal_records(catalogue_path)
    generated = ma.build_audit_rows(bundles, catalogue)
    frozen = ma.read_audit_csv(frozen_path)

    comparison = ma.compare_audit_rows(frozen, generated)
    assert comparison["equal"], json.dumps(comparison, indent=2, sort_keys=True)
    assert comparison["actual_row_count"] == 615

    headline = ma.summarize_audit(generated)
    assert headline["sba_like_count"] == 568
    assert headline["unclear_count"] == 47
    assert headline["affirmative_sbr_evidence_count"] == 0
    assert headline["full_standalone_price_coverage_count"] == 615
    assert headline["complete_catalogue_confirmation_count"] == 475
    assert headline["high_confidence_sba_like_count"] == 513

    # Reverse engineering check: applying the pure row function in legacy raw
    # order reproduces the tracked CSV byte-for-byte, including text and floats.
    index = ma.build_catalogue_index(catalogue)
    legacy_order_rows = [ma.audit_bundle(bundle, index) for bundle in bundles]
    assert ma.serialize_audit_csv(legacy_order_rows) == frozen_path.read_bytes()
