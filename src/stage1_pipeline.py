"""Idempotent cycle-scoped entry point for the complete live Stage 1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from src.stage1_backend_spike import run_backend_spike
from src.stage1_estimator_v2 import build_estimator_manifest
from src.stage1_evidence import assemble_evidence
from src.stage1_gate1 import run_gate1
from src.stage1_feature_artifacts import verify_feature_artifacts
from src.stage1_interaction_artifacts import verify_interaction_artifacts
from src.stage1_production import run_production
from src.stage1_pseudo_utility_freeze import freeze_pseudo_utilities
from src.stage1_source_interactions import PROJECT_ROOT, verify_source_interactions
from src.stage1_split_artifacts import verify_split_artifacts
from src.stage1_protocol import build_protocol_manifest, load_json
from src.stage1_validation import run_validation


def run_complete_stage1() -> dict[str, str]:
    cycle = "s1-v2-20260814"
    config_dir = PROJECT_ROOT / "configs" / "cycles" / cycle
    cycle_dir = PROJECT_ROOT / "outputs" / "modeling" / "cycles" / cycle
    protected = PROJECT_ROOT / "outputs" / "modeling" / "protected" / cycle
    source = verify_source_interactions()
    models = load_json(config_dir / "preference_models.json")
    ranking = load_json(config_dir / "ranking_evaluation.json")
    protocol = build_protocol_manifest(
        models,
        ranking,
        model_config_path=config_dir / "preference_models.json",
        ranking_config_path=config_dir / "ranking_evaluation.json",
    )
    saved_protocol = load_json(cycle_dir / "stage1_protocol_manifest.json")
    protocol.pop("environment_at_freeze", None)
    saved_protocol.pop("environment_at_freeze", None)
    if protocol != saved_protocol:
        raise ValueError("S1.0 protocol manifest changed")
    verify_interaction_artifacts(
        ranking_config_path=config_dir / "ranking_evaluation.json",
        protocol_manifest_path=cycle_dir / "stage1_protocol_manifest.json",
        output_dir=protected / "stage1_interactions",
        manifest_path=cycle_dir / "stage1_interaction_manifest.json",
    )
    verify_split_artifacts(
        ranking_config_path=config_dir / "ranking_evaluation.json",
        protocol_manifest_path=cycle_dir / "stage1_protocol_manifest.json",
        interaction_manifest_path=cycle_dir / "stage1_interaction_manifest.json",
        output_dir=protected / "stage1_splits",
        manifest_path=cycle_dir / "stage1_split_manifest.json",
    )
    verify_feature_artifacts(
        model_config_path=config_dir / "preference_models.json",
        ranking_config_path=config_dir / "ranking_evaluation.json",
        protocol_manifest_path=cycle_dir / "stage1_protocol_manifest.json",
        interaction_manifest_path=cycle_dir / "stage1_interaction_manifest.json",
        interaction_output_dir=protected / "stage1_interactions",
        split_manifest_path=cycle_dir / "stage1_split_manifest.json",
        output_dir=protected / "stage1_features",
        manifest_path=cycle_dir / "item_feature_manifest.json",
        coverage_path=cycle_dir / "stage1_feature_coverage.csv",
    )
    if build_estimator_manifest() != load_json(
        cycle_dir / "stage1_estimator_spec_manifest.json"
    ):
        raise ValueError("S1.4 estimator manifest changed")
    spike = run_backend_spike()
    admission = run_validation()
    gate1 = run_gate1()
    production = run_production()
    gate2 = freeze_pseudo_utilities()
    evidence = assemble_evidence()
    return {
        "source_set_id": source["source_set_id"],
        "backend_spike_manifest_id": spike["manifest_id"],
        "admission_id": admission["admission_id"],
        "gate1_manifest_id": gate1["manifest_id"],
        "production_manifest_id": production["manifest_id"],
        "gate2_manifest_id": gate2["manifest_id"],
        "evidence_manifest_id": evidence["manifest_id"],
        "status": "complete",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run or verify complete Stage 1")
    parser.parse_args(argv)
    print(json.dumps(run_complete_stage1(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
