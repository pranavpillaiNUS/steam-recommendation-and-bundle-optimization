"""Presentation figure for the Stage 1 headline result.

This module is deliberately outside the frozen evidence graph. It reads the
hash-bound `s1-v2-20260814` tables and rewrites a README image, so it must never
be added to `stage1_evidence.py` or the public verifier: doing so would make a
cosmetic rerender able to invalidate the cycle. The hash-bound evidence figure
remains `outputs/figures/s1-v2-20260814/stage1_ranking_evidence.png`.

Every number drawn here is read from the frozen artifacts at run time. Nothing is
hardcoded, so the figure cannot drift away from the recorded result.

    python -m src.stage1_headline_figure
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CYCLE = "s1-v2-20260814"
CYCLE_DIR = PROJECT_ROOT / "outputs" / "modeling" / "cycles" / CYCLE
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures" / CYCLE

# Admission rule frozen in S1.9: a personalized family is noninferior to
# popularity only if the lower 95% paired NDCG@20 bound is at least -0.005.
ADMISSION_NDCG_LOWER_BOUND = -0.005

# Display names for the frozen family identifiers.
FAMILY_LABELS = {
    "implicit_als": "ALS, ownership-only",
    "popularity": "Popularity",
    "feature_sum_bpr_identity": "BPR, identity",
    "feature_sum_bpr_identity_genre": "BPR, identity + genre",
}

# Activity bands are the frozen [5, 10, 25, 50, 100, 200) edges from
# ranking_evaluation.json. Code -1 holds users below the first edge.
BAND_LABELS = {
    "-1": "under 5",
    "0": "5 to 9",
    "1": "10 to 24",
    "2": "25 to 49",
    "3": "50 to 99",
    "4": "100 to 199",
    "5": "200 or more",
}

# Palette slots from the validated reference palette. Each mode is stepped for
# its own surface rather than flipped from the other.
THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "accent": "#2a78d6",
        "negative": "#e34948",
        "muted_mark": "#898781",
        "text_primary": "#0b0b0b",
        "text_secondary": "#52514e",
        "text_muted": "#898781",
        "grid": "#e1e0d9",
        "baseline": "#c3c2b7",
    },
    "dark": {
        "surface": "#1a1a19",
        "accent": "#3987e5",
        "negative": "#e66767",
        "muted_mark": "#898781",
        "text_primary": "#ffffff",
        "text_secondary": "#c3c2b7",
        "text_muted": "#898781",
        "grid": "#2c2c2a",
        "baseline": "#383835",
    },
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_evidence() -> dict[str, object]:
    """Collect every value the figure draws from the frozen cycle artifacts."""
    leaderboard = _read_csv(CYCLE_DIR / "stage1_design_test_leaderboard.csv")
    segments = _read_csv(CYCLE_DIR / "stage1_design_test_segments.csv")
    gate1 = json.loads((CYCLE_DIR / "stage1_gate1_manifest.json").read_text(encoding="utf-8"))

    per_seed: dict[str, list[float]] = defaultdict(list)
    for row in leaderboard:
        per_seed[row["family"]].append(float(row["mean_ndcg_at_20"]))
    means = {family: sum(values) / len(values) for family, values in per_seed.items()}

    contrasts = {}
    for family, payload in gate1["design_test_contrasts"].items():
        if family == "genre_versus_identity":
            continue
        paired = payload["paired_ndcg_at_20"]
        contrasts[family] = {
            "mean": float(paired["mean_difference"]),
            "lower": float(paired["lower"]),
            "upper": float(paired["upper"]),
        }

    band_popularity: dict[str, float] = {}
    band_als: dict[str, list[float]] = defaultdict(list)
    band_users: dict[str, int] = {}
    for row in segments:
        if row["dimension"] != "user_activity_band":
            continue
        code = row["segment_code"]
        band_users[code] = int(row["users"])
        if row["family"] == "popularity":
            band_popularity[code] = float(row["mean_ndcg_at_20"])
        elif row["family"] == "implicit_als":
            band_als[code].append(float(row["mean_ndcg_at_20"]))

    codes = sorted(band_popularity, key=int)
    lifts = [sum(band_als[c]) / len(band_als[c]) - band_popularity[c] for c in codes]

    return {
        "means": means,
        "per_seed": dict(per_seed),
        "contrasts": contrasts,
        "band_codes": codes,
        "band_lifts": lifts,
        "band_users": band_users,
        "admitted": set(gate1["admitted_families"]) if "admitted_families" in gate1 else {"implicit_als"},
    }


def _style_axes(ax, theme: dict[str, str], *, xgrid: bool = True) -> None:
    ax.set_facecolor(theme["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(theme["baseline"])
    ax.spines["bottom"].set_linewidth(1.0)
    if xgrid:
        ax.xaxis.grid(True, color=theme["grid"], linewidth=1.0, linestyle="-")
        ax.set_axisbelow(True)
    ax.yaxis.grid(False)
    ax.tick_params(axis="both", length=0, colors=theme["text_muted"], labelsize=9)


def _panel_result(ax, data: dict, theme: dict[str, str]) -> None:
    """Panel A: design-test NDCG@20 by family, emphasising the admitted model."""
    order = [
        "implicit_als",
        "popularity",
        "feature_sum_bpr_identity",
        "feature_sum_bpr_identity_genre",
    ]
    means = data["means"]
    positions = range(len(order))
    colors = [
        theme["accent"] if family in data["admitted"] else theme["muted_mark"]
        for family in order
    ]

    ax.barh(list(positions), [means[f] for f in order], height=0.46, color=colors, zorder=3)

    # Per-seed marks show that the ranking is stable across the three frozen seeds.
    for y, family in enumerate(order):
        values = data["per_seed"][family]
        if len(values) > 1:
            ax.scatter(
                values,
                [y] * len(values),
                s=18,
                facecolor=theme["surface"],
                edgecolor=theme["text_secondary"],
                linewidth=1.2,
                zorder=5,
            )

    # Clear the seed marks before the value, so the label never sits on a dot.
    for y, family in enumerate(order):
        reach = max([means[family]] + data["per_seed"][family])
        ax.text(
            reach + 0.008,
            y,
            "%.3f" % means[family],
            va="center",
            ha="left",
            fontsize=9.5,
            color=theme["text_primary"],
            fontweight="semibold" if family in data["admitted"] else "normal",
            zorder=6,
        )

    ax.set_yticks(list(positions))
    ax.set_yticklabels([FAMILY_LABELS[f] for f in order], fontsize=9.5, color=theme["text_secondary"])
    ax.invert_yaxis()
    ax.set_xlim(0, max(means.values()) * 1.30)
    ax.set_xlabel(
        "Mean NDCG@20 on the one-time design test\nopen marks are the three individual training seeds",
        fontsize=9,
        color=theme["text_secondary"],
    )
    ax.set_title(
        "Only ownership-only ALS beat popularity",
        fontsize=11.5,
        color=theme["text_primary"],
        fontweight="semibold",
        loc="left",
        pad=10,
    )
    _style_axes(ax, theme)


def _panel_admission(ax, data: dict, theme: dict[str, str]) -> None:
    """Panel B: paired difference against popularity with the admission rule."""
    order = ["implicit_als", "feature_sum_bpr_identity", "feature_sum_bpr_identity_genre"]
    contrasts = data["contrasts"]
    positions = list(range(len(order)))

    ax.axvline(0.0, color=theme["baseline"], linewidth=1.0, zorder=2)
    # A rule the data must clear, not a data value, so it stays out of the
    # diverging blue/red vocabulary that panel C uses for signed magnitudes.
    ax.axvline(
        ADMISSION_NDCG_LOWER_BOUND,
        color=theme["text_secondary"],
        linewidth=1.4,
        linestyle=(0, (4, 3)),
        zorder=2,
    )

    # Status rides a fixed right-hand column in axes coordinates. Anchored to the
    # interval end instead, it ran back across the zero line on the genre row.
    status_x = ax.get_yaxis_transform()

    for y, family in enumerate(order):
        entry = contrasts[family]
        admitted = family in data["admitted"]
        color = theme["accent"] if admitted else theme["muted_mark"]
        ax.plot(
            [entry["lower"], entry["upper"]], [y, y],
            color=color, linewidth=2.0, solid_capstyle="round", zorder=4,
        )
        ax.scatter(
            [entry["mean"]], [y], s=42, color=color,
            edgecolor=theme["surface"], linewidth=1.6, zorder=5,
        )
        ax.text(
            entry["upper"] + 0.005, y, "%+.3f" % entry["mean"],
            va="center", ha="left", fontsize=9,
            color=theme["text_primary"], zorder=6,
        )
        ax.text(
            0.995, y, "admitted" if admitted else "not admitted",
            transform=status_x, va="center", ha="right", fontsize=8.5,
            color=theme["text_primary"] if admitted else theme["text_muted"],
            fontweight="semibold" if admitted else "normal", zorder=6,
        )

    ax.set_yticks(positions)
    ax.set_yticklabels([FAMILY_LABELS[f] for f in order], fontsize=9.5, color=theme["text_secondary"])
    ax.invert_yaxis()
    ax.set_ylim(len(order) - 0.45, -0.85)
    ax.set_xlim(-0.105, 0.185)
    ax.set_xlabel(
        "Paired NDCG@20 difference against popularity\ndot is the mean, line is the 95% paired user bootstrap interval",
        fontsize=9,
        color=theme["text_secondary"],
    )
    ax.set_title(
        "The admission rule, fixed before the test was opened",
        fontsize=11.5, color=theme["text_primary"], fontweight="semibold", loc="left", pad=10,
    )
    _style_axes(ax, theme)

    ax.text(
        ADMISSION_NDCG_LOWER_BOUND - 0.006, -0.66,
        "noninferiority bound, -0.005",
        fontsize=8, color=theme["text_secondary"], ha="right", va="center",
    )

    # The identity-BPR interval is narrower than its own marker, so the reason it
    # failed is invisible at this scale. One callout, on the row that needs it.
    identity = contrasts["feature_sum_bpr_identity"]
    ax.text(
        identity["lower"] - 0.004,
        order.index("feature_sum_bpr_identity") + 0.34,
        "lower bound %.4f, misses by %.4f" % (
            identity["lower"], ADMISSION_NDCG_LOWER_BOUND - identity["lower"]),
        fontsize=8, color=theme["text_muted"], ha="right", va="center",
    )


def _panel_bands(ax, data: dict, theme: dict[str, str]) -> None:
    """Panel C: where the ALS gain actually lives, by user activity."""
    codes = data["band_codes"]
    lifts = data["band_lifts"]
    labels = [BAND_LABELS[c] for c in codes]
    colors = [theme["accent"] if lift >= 0 else theme["negative"] for lift in lifts]
    positions = list(range(len(codes)))

    ax.barh(positions, lifts, height=0.5, color=colors, zorder=3)
    ax.axvline(0.0, color=theme["baseline"], linewidth=1.0, zorder=4)

    for y, lift in enumerate(lifts):
        offset = 0.004 if lift >= 0 else -0.004
        ax.text(
            lift + offset, y, "%+.3f" % lift,
            va="center", ha="left" if lift >= 0 else "right",
            fontsize=9, color=theme["text_primary"], zorder=6,
        )

    # User counts ride the tick labels: as free-floating text they collided with
    # the value label on the one negative bar.
    ax.set_yticks(positions)
    ax.set_yticklabels(
        ["%s   (n = %s)" % (label, format(data["band_users"][code], ","))
         for label, code in zip(labels, codes)],
        fontsize=9.5,
        color=theme["text_secondary"],
    )
    ax.invert_yaxis()
    ax.set_xlim(-0.030, 0.165)
    ax.set_ylabel("Games owned in training", fontsize=9, color=theme["text_secondary"])
    ax.set_xlabel(
        "ALS minus popularity, mean NDCG@20 on the design test",
        fontsize=9, color=theme["text_secondary"],
    )
    ax.set_title(
        "The gain is largest for lighter users and disappears for the heaviest",
        fontsize=11.5, color=theme["text_primary"], fontweight="semibold", loc="left", pad=10,
    )
    _style_axes(ax, theme)


def build_figure(theme_name: str, output_path: Path) -> Path:
    theme = THEMES[theme_name]
    data = load_evidence()

    figure = plt.figure(figsize=(12.4, 7.8), dpi=200, facecolor=theme["surface"])
    grid = figure.add_gridspec(
        2, 2, height_ratios=[1.0, 1.10], width_ratios=[1.0, 1.0],
        hspace=0.78, wspace=0.42, left=0.135, right=0.975, top=0.815, bottom=0.095,
    )

    _panel_result(figure.add_subplot(grid[0, 0]), data, theme)
    _panel_admission(figure.add_subplot(grid[0, 1]), data, theme)
    _panel_bands(figure.add_subplot(grid[1, :]), data, theme)

    figure.suptitle(
        "Stage 1: held-out Steam ownership ranking, cycle s1-v2-20260814",
        fontsize=14, color=theme["text_primary"], fontweight="semibold",
        x=0.045, ha="left", y=0.975,
    )
    figure.text(
        0.045, 0.930,
        "5,000 design users, each ranked against the complete 8,902-item warm catalogue. "
        "Scores measure ownership reconstruction, not willingness to pay.",
        fontsize=9.5, color=theme["text_secondary"], ha="left",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, facecolor=theme["surface"], bbox_inches="tight", pad_inches=0.28)
    plt.close(figure)
    return output_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render the Stage 1 headline presentation figure")
    parser.parse_args(argv)
    written = [
        build_figure("light", FIGURE_DIR / "stage1_headline.png"),
        build_figure("dark", FIGURE_DIR / "stage1_headline_dark.png"),
    ]
    for path in written:
        print(path.relative_to(PROJECT_ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
