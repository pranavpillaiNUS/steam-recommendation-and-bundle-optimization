"""Assemble the S1.12 evidence package and verify its dependency graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.stage1_protocol import canonical_json_bytes, file_sha256, load_json, semantic_sha256


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CYCLE_ID = "s1-v2-20260814"
CYCLE_DIR = PROJECT_ROOT / "outputs" / "modeling" / "cycles" / CYCLE_ID
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures" / CYCLE_ID
SUMMARY_PATH = CYCLE_DIR / "stage1_evidence_summary.md"
RUNTIME_TABLE = CYCLE_DIR / "stage1_runtime_memory.csv"
VALIDATION_RESOURCE_TRACE = CYCLE_DIR / "stage1_validation_resource_trace.csv"
SEED_CONTRAST_TABLE = CYCLE_DIR / "stage1_seed_specific_contrasts.csv"
FIGURE_PATH = FIGURE_DIR / "stage1_ranking_evidence.png"
OUTPUT_MANIFEST = CYCLE_DIR / "stage1_evidence_manifest.json"


MANIFESTS = {
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


def _verify_semantic_id(value: Mapping[str, Any], candidates: Sequence[str]) -> str:
    for field in candidates:
        if field in value:
            unsigned = dict(value)
            claimed = unsigned.pop(field)
            if claimed != semantic_sha256(unsigned):
                raise ValueError(f"manifest semantic hash mismatch: {field}")
            return str(claimed)
    # Protocol manifests deliberately identify the two frozen config hashes,
    # rather than hashing the full environment/provenance wrapper.
    if "protocol_id" in value and "configs" in value:
        expected = semantic_sha256(
            {
                "model_config_sha256": value["configs"]["preference_models"]["semantic_sha256"],
                "ranking_config_sha256": value["configs"]["ranking_evaluation"]["semantic_sha256"],
            }
        )
        if value["protocol_id"] != expected:
            raise ValueError("protocol semantic binding changed")
        return str(expected)
    raise ValueError("manifest has no recognized semantic identifier")


def _verify_recorded_paths(value: Any, root: Path) -> int:
    verified = 0
    if isinstance(value, Mapping):
        if {"path", "sha256"}.issubset(value) and isinstance(value["path"], str):
            path = root / value["path"]
            if path.is_file():
                if file_sha256(path) != value["sha256"]:
                    raise ValueError(f"recorded artifact hash mismatch: {path}")
                if "size_bytes" in value and path.stat().st_size != value["size_bytes"]:
                    raise ValueError(f"recorded artifact size mismatch: {path}")
                verified += 1
        for nested in value.values():
            verified += _verify_recorded_paths(nested, root)
    elif isinstance(value, list):
        for nested in value:
            verified += _verify_recorded_paths(nested, root)
    return verified


def _runtime_table(training: Mapping[str, Any], production: Mapping[str, Any]) -> pd.DataFrame:
    rows = [
        {
            "stage": "validation",
            "family": run["family"],
            "configuration_id": run["configuration_id"],
            "training_seed": run["training_seed"],
            "parameter_count": run["parameter_count"],
            "runtime_seconds": run["aggregate_validation_metrics"]["runtime_seconds"],
            "artifact_bytes": run["model_artifact"]["size_bytes"],
            "observed_process_peak_rss_bytes": "",
        }
        for run in training["runs"]
    ]
    rows.extend(
        {
            "stage": "production_and_fold_in",
            "family": run["family"],
            "configuration_id": run["configuration_id"],
            "training_seed": run["training_seed"],
            "parameter_count": "",
            "runtime_seconds": run["runtime_seconds"],
            "artifact_bytes": sum(entry["size_bytes"] for entry in run["artifacts"].values()),
            "observed_process_peak_rss_bytes": "",
        }
        for run in production["runs"]
    )
    if VALIDATION_RESOURCE_TRACE.exists():
        trace = pd.read_csv(VALIDATION_RESOURCE_TRACE)
        rows.append(
            {
                "stage": "validation_process_monitor",
                "family": "all_remaining_runs_after_monitor_start",
                "configuration_id": "",
                "training_seed": "",
                "parameter_count": "",
                "runtime_seconds": "",
                "artifact_bytes": "",
                "observed_process_peak_rss_bytes": int(trace["rss_bytes"].max()),
            }
        )
    return pd.DataFrame(rows)


def _figure(validation: pd.DataFrame, design_test: pd.DataFrame, pseudo_cold: pd.DataFrame) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    aggregate = validation.groupby(["family", "configuration_id"], as_index=False)["mean_ndcg_at_20"].mean()
    for family, rows in aggregate.groupby("family"):
        axes[0].scatter(np.arange(len(rows)), rows["mean_ndcg_at_20"], label=family, s=28)
    axes[0].set_title("Validation grid")
    axes[0].set_ylabel("Mean NDCG@20")
    axes[0].set_xlabel("Configuration within family")
    axes[0].legend(fontsize=7)
    test_aggregate = design_test.groupby("family", as_index=False)["mean_ndcg_at_20"].mean().sort_values("mean_ndcg_at_20")
    axes[1].barh(test_aggregate["family"], test_aggregate["mean_ndcg_at_20"])
    axes[1].set_title("One-time design test")
    axes[1].set_xlabel("Mean NDCG@20 across seeds")
    cold_aggregate = pseudo_cold.groupby("family", as_index=False)["mean_ndcg_at_20"].mean().sort_values("mean_ndcg_at_20")
    axes[2].barh(cold_aggregate["family"], cold_aggregate["mean_ndcg_at_20"])
    axes[2].set_title("Pseudo-cold cohort")
    axes[2].set_xlabel("Mean NDCG@20")
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _seed_contrasts(training: Mapping[str, Any], design_test: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    popularity_run = next(run for run in training["runs"] if run["family"] == "popularity")
    with np.load(PROJECT_ROOT / popularity_run["metric_artifact"]["path"], allow_pickle=False) as payload:
        popularity = {name: np.asarray(payload[name]) for name in ("ndcg_at_20", "recall_at_20")}
    selected = training["selection"]
    selected_runs = {
        family: sorted(
            (run for run in training["runs"] if run["family"] == family and run["configuration_id"] == configuration),
            key=lambda run: int(run["training_seed"]),
        )
        for family, configuration in selected.items()
    }
    validation_metrics: dict[str, list[dict[str, np.ndarray]]] = {}
    for family, family_runs in selected_runs.items():
        validation_metrics[family] = []
        for run in family_runs:
            with np.load(PROJECT_ROOT / run["metric_artifact"]["path"], allow_pickle=False) as payload:
                metric = {name: np.asarray(payload[name]) for name in ("ndcg_at_20", "recall_at_20")}
            validation_metrics[family].append(metric)
            rows.append({"split": "validation", "contrast": f"{family}_versus_popularity", "training_seed": run["training_seed"], "mean_ndcg_at_20_difference": float(np.mean(metric["ndcg_at_20"] - popularity["ndcg_at_20"])), "mean_recall_at_20_difference": float(np.mean(metric["recall_at_20"] - popularity["recall_at_20"]))})
    for identity, genre, run in zip(validation_metrics["feature_sum_bpr_identity"], validation_metrics["feature_sum_bpr_identity_genre"], selected_runs["feature_sum_bpr_identity_genre"]):
        rows.append({"split": "validation", "contrast": "genre_versus_identity", "training_seed": run["training_seed"], "mean_ndcg_at_20_difference": float(np.mean(genre["ndcg_at_20"] - identity["ndcg_at_20"])), "mean_recall_at_20_difference": float(np.mean(genre["recall_at_20"] - identity["recall_at_20"]))})
    pop_test = design_test.loc[design_test["family"] == "popularity"].iloc[0]
    for _, row in design_test.loc[design_test["family"] != "popularity"].iterrows():
        rows.append({"split": "design_test", "contrast": f"{row['family']}_versus_popularity", "training_seed": int(row["training_seed"]), "mean_ndcg_at_20_difference": float(row["mean_ndcg_at_20"] - pop_test["mean_ndcg_at_20"]), "mean_recall_at_20_difference": float(row["mean_recall_at_20"] - pop_test["mean_recall_at_20"])})
    identity_test = design_test.loc[design_test["family"] == "feature_sum_bpr_identity"].set_index("training_seed")
    genre_test = design_test.loc[design_test["family"] == "feature_sum_bpr_identity_genre"].set_index("training_seed")
    for seed in sorted(identity_test.index):
        rows.append({"split": "design_test", "contrast": "genre_versus_identity", "training_seed": int(seed), "mean_ndcg_at_20_difference": float(genre_test.loc[seed, "mean_ndcg_at_20"] - identity_test.loc[seed, "mean_ndcg_at_20"]), "mean_recall_at_20_difference": float(genre_test.loc[seed, "mean_recall_at_20"] - identity_test.loc[seed, "mean_recall_at_20"])})
    return pd.DataFrame(rows)


def assemble_evidence(*, output_path: str | Path = OUTPUT_MANIFEST) -> dict[str, Any]:
    destination = Path(output_path).resolve()
    manifests: dict[str, dict[str, Any]] = {}
    manifest_entries: dict[str, Any] = {}
    verified_artifacts = 0
    for label, filename in MANIFESTS.items():
        path = CYCLE_DIR / filename
        value = load_json(path)
        if value.get("cycle_id") != CYCLE_ID:
            raise ValueError(f"evidence dependency cycle mismatch: {label}")
        identity = _verify_semantic_id(
            value,
            ("manifest_id", "admission_id"),
        )
        verified_artifacts += _verify_recorded_paths(value, PROJECT_ROOT)
        manifests[label] = value
        manifest_entries[label] = {
            "path": path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": file_sha256(path),
            "semantic_id": identity,
        }
    if manifests["gate1"].get("status") != "pass":
        raise ValueError("Gate 1 did not pass")
    if manifests["pseudo_utility_gate2"].get("gate2", {}).get("status") != "pass":
        raise ValueError("Gate 2 did not pass")
    if manifests["production"].get("status") != "complete":
        raise ValueError("production refit/fold-in is incomplete")

    validation = pd.read_csv(CYCLE_DIR / "stage1_validation_leaderboard.csv")
    design_test = pd.read_csv(CYCLE_DIR / "stage1_design_test_leaderboard.csv")
    pseudo_cold = pd.read_csv(CYCLE_DIR / "stage1_pseudo_cold_results.csv")
    runtime = _runtime_table(manifests["training"], manifests["production"])
    runtime.to_csv(RUNTIME_TABLE, index=False, lineterminator="\n")
    seed_contrasts = _seed_contrasts(manifests["training"], design_test)
    seed_contrasts.to_csv(SEED_CONTRAST_TABLE, index=False, lineterminator="\n")
    _figure(validation, design_test, pseudo_cold)

    expected_run_ids = {run["run_id"] for run in manifests["training"]["runs"]}
    validation_log_entries: list[dict[str, Any]] = []
    observed_run_ids: set[str] = set()
    for path in sorted((CYCLE_DIR / "validation_runs").glob("*.json")):
        value = load_json(path)
        run_id = _verify_semantic_id(value, ("run_id",))
        observed_run_ids.add(run_id)
        validation_log_entries.append(
            {
                "path": path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
                "run_id": run_id,
            }
        )
    if observed_run_ids != expected_run_ids:
        raise ValueError("validation run-log inventory differs from training manifest")

    admitted = manifests["admission"]["admitted_families"]
    selected = manifests["training"]["selection"]
    test_means = design_test.groupby("family")["mean_ndcg_at_20"].mean().to_dict()
    lines = [
        "# Stage 1 evidence summary",
        "",
        f"Cycle: `{CYCLE_ID}`",
        "",
        "Stage 1 is complete under the prospective steam_id-based cycle. All ranking results are held-out ownership reconstruction metrics, not utility or monetary estimates.",
        "",
        "## Selected configurations",
        "",
    ]
    lines.extend(f"- `{family}`: `{configuration}`" for family, configuration in selected.items())
    lines.extend(["", "## Admission and design-test result", "", f"Admitted families: {', '.join(f'`{x}`' for x in admitted)}."])
    for family, value in sorted(test_means.items()):
        lines.append(f"- `{family}` mean design-test NDCG@20: {value:.6f}")
    lines.extend(
        [
            "",
            "The admission set was selected and hashed from validation before the design-test coordinates were opened. Design-test and pseudo-cold results did not replace the selected configurations.",
            "",
            "## Claims and nonclaims",
            "",
            "- Warm ranking evidence supports only predictive ownership reconstruction on this snapshot.",
            "- The genre comparison is a controlled predictive ablation, not a causal genre effect.",
            "- Pseudo-cold scores suppress collaborative identity and bias for cohort items, but do not prove general cold-start performance.",
            "- Pseudo-utility scenarios are deterministic nonnegative transformations. They do not identify willingness to pay, money, or interpersonal welfare.",
            "- No Stage 2 bundle objective or bundle outcome was used anywhere in Stage 1 selection or Gate 2 freezing.",
        ]
    )
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    outputs = {
        "summary": SUMMARY_PATH,
        "runtime_memory": RUNTIME_TABLE,
        "ranking_figure": FIGURE_PATH,
        "validation_leaderboard": CYCLE_DIR / "stage1_validation_leaderboard.csv",
        "design_test_leaderboard": CYCLE_DIR / "stage1_design_test_leaderboard.csv",
        "design_test_segments": CYCLE_DIR / "stage1_design_test_segments.csv",
        "pseudo_cold_results": CYCLE_DIR / "stage1_pseudo_cold_results.csv",
        "pseudo_utility_diagnostics": CYCLE_DIR / "stage1_pseudo_utility_diagnostics.csv",
        "mathematical_appendix": PROJECT_ROOT / "notes" / "stage1_v2_mathematical_appendix.md",
        "seed_specific_contrasts": SEED_CONTRAST_TABLE,
    }
    if VALIDATION_RESOURCE_TRACE.exists():
        outputs["validation_resource_trace"] = VALIDATION_RESOURCE_TRACE
    output_entries = {
        label: {
            "path": path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for label, path in outputs.items()
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "artifact": "stage1_complete_evidence_package",
        "cycle_id": CYCLE_ID,
        "status": "complete",
        "gate1_status": "pass",
        "gate2_status": "pass",
        "admitted_families": admitted,
        "manifests": manifest_entries,
        "validation_run_logs": validation_log_entries,
        "outputs": output_entries,
        "verified_recorded_artifact_references": verified_artifacts,
        "stop_rule": "recommender expansion stopped after required identity_genre_pseudo_cold_and_pseudo_utility_work",
        "next_permitted_stage": "Stage_2_candidate_pool_freeze_and_fixed_bundle_SBA_CP",
    }
    manifest["manifest_id"] = semantic_sha256(manifest)
    destination.write_bytes(canonical_json_bytes(manifest))
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Assemble and verify Stage 1 evidence")
    parser.parse_args(argv)
    result = assemble_evidence()
    print(json.dumps({"status": result["status"], "manifest_id": result["manifest_id"], "verified_references": result["verified_recorded_artifact_references"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
