"""Deterministic Stage 1 user, interaction, and pseudo-cold splits.

The functions in this module implement the frozen rules in
``configs/ranking_evaluation.json``.  They operate only on identifiers,
activity/support counts, and pre-model metadata.  They do not inspect model
scores, validation outcomes, assessment outcomes, or Stage 2 objectives.

Every returned table is placed in a canonical numeric-ID order so its bytes
and content hash do not depend on the input row order.
"""

from __future__ import annotations

from fractions import Fraction
import html
import re
from typing import Any, Mapping, Sequence
import unicodedata

import numpy as np
import pandas as pd

from .interactions import canonical_numeric_ids
from .stage1_protocol import canonical_json_bytes, stable_hash_uint64


EXCLUDED_LOW_ACTIVITY = "excluded_low_activity"
EXCLUDED_NONWARM_ITEM = "excluded_nonwarm_item"
_WHITESPACE = re.compile(r"\s+")


def _canonical_ids(values: pd.Series, *, field: str) -> pd.Series:
    identifiers = canonical_numeric_ids(values.to_numpy(), label=field)
    return pd.Series(identifiers, index=values.index, name=field)


def _integer_values(values: pd.Series, *, field: str) -> pd.Series:
    """Validate and return finite, nonnegative integer values."""

    integers = canonical_numeric_ids(values.to_numpy(), label=field)
    return pd.Series(integers, index=values.index, name=field)


def _id_text(value: Any) -> str:
    return str(int(value))


def primary_genre_from_csv(value: Any) -> str | None:
    """Return the lexical first normalized genre from one CSV genre field.

    This is the outcome-independent S1.2 clarification of the previously
    underspecified ``primary_genre`` stratum. Genre labels are HTML-unescaped,
    Unicode-NFKC normalized, whitespace-collapsed, deduplicated, and sorted.
    Missing or blank fields return ``None``.
    """

    if value is None or pd.isna(value):
        return None
    if not isinstance(value, (str, np.str_)):
        raise ValueError("genre fields must be strings or missing")
    tokens: set[str] = set()
    for raw_token in str(value).split(","):
        token = html.unescape(raw_token)
        token = unicodedata.normalize("NFKC", token)
        token = _WHITESPACE.sub(" ", token).strip()
        if token:
            tokens.add(token)
    return min(tokens) if tokens else None


def _sorted_strata(values: Sequence[Any]) -> list[Any]:
    return sorted(values, key=canonical_json_bytes)


def _proportional_quotas(
    counts: Mapping[Any, int], target_size: int
) -> dict[Any, int]:
    """Allocate an exact target by Hamilton largest-remainder apportionment."""

    if target_size < 0:
        raise ValueError("target_size must be nonnegative")
    clean = {key: int(value) for key, value in counts.items()}
    if any(value < 0 for value in clean.values()):
        raise ValueError("stratum counts must be nonnegative")
    total = sum(clean.values())
    if target_size > total:
        raise ValueError(
            f"requested sample of {target_size} exceeds {total} available rows"
        )
    if total == 0:
        if target_size:
            raise ValueError("cannot allocate a nonzero sample from no rows")
        return {key: 0 for key in clean}

    quotas = {
        key: (target_size * count) // total for key, count in clean.items()
    }
    remainder_order = sorted(
        clean,
        key=lambda key: (
            -(target_size * clean[key] % total),
            canonical_json_bytes(key),
        ),
    )
    remaining = target_size - sum(quotas.values())
    for key in remainder_order[:remaining]:
        quotas[key] += 1

    if sum(quotas.values()) != target_size:
        raise AssertionError("proportional reconciliation failed")
    if any(quotas[key] > clean[key] for key in clean):
        raise AssertionError("a proportional quota exceeded its stratum")
    return quotas


def activity_band_outer_split(
    user_activity: pd.DataFrame,
    spec: Mapping[str, Any],
    *,
    user_col: str = "user_id",
    activity_col: str = "raw_ownership_count",
) -> pd.DataFrame:
    """Assign the exact activity-stratified outer 80/20 user split.

    Within each eligible activity band, the first
    ``floor(assessment_fraction * band_size)`` users in stable hash order are
    assigned to assessment.  All other eligible users enter design.  Users
    below the frozen minimum remain in the returned audit table with an
    explicit exclusion label.
    """

    required = {user_col, activity_col}
    missing = required.difference(user_activity.columns)
    if missing:
        raise ValueError(f"missing outer-split columns: {sorted(missing)}")

    frame = user_activity.loc[:, [user_col, activity_col]].copy()
    frame = frame.reset_index(drop=True)
    frame[user_col] = _canonical_ids(frame[user_col], field=user_col)
    frame[activity_col] = _integer_values(frame[activity_col], field=activity_col)
    if frame[user_col].duplicated().any():
        raise ValueError("outer split requires exactly one row per user")

    minimum = int(spec["minimum_raw_ownership_count"])
    bounds = [int(value) for value in spec["activity_bands"]]
    if not bounds or bounds[0] != minimum or any(
        left >= right for left, right in zip(bounds, bounds[1:])
    ):
        raise ValueError(
            "activity bands must be increasing lower bounds beginning at the minimum"
        )
    assessment_fraction = Fraction(str(spec["assessment_fraction"]))
    design_fraction = Fraction(str(spec["design_fraction"]))
    if assessment_fraction <= 0 or design_fraction <= 0:
        raise ValueError("outer split fractions must be positive")
    if assessment_fraction + design_fraction != 1:
        raise ValueError("outer split fractions must sum exactly to one")

    frame["activity_band"] = -1
    frame["split"] = EXCLUDED_LOW_ACTIVITY
    frame["assignment_hash"] = ""
    namespace = str(spec["namespace"])

    eligible_mask = frame[activity_col] >= minimum
    eligible = frame.loc[eligible_mask].copy()
    activities = eligible[activity_col].to_numpy(dtype=np.int64)
    eligible["activity_band"] = np.searchsorted(
        np.asarray(bounds, dtype=np.int64), activities, side="right"
    ) - 1

    for band in sorted(eligible["activity_band"].unique().tolist()):
        band_rows = eligible.loc[eligible["activity_band"] == band].copy()
        band_rows["_hash"] = band_rows[user_col].map(
            lambda user_id: stable_hash_uint64(
                namespace,
                int(band),
                _id_text(user_id),
            )
        )
        band_rows = band_rows.sort_values(
            ["_hash", user_col], kind="mergesort"
        )
        n_assessment = (
            len(band_rows)
            * assessment_fraction.numerator
            // assessment_fraction.denominator
        )
        assessment_ids = set(band_rows.iloc[:n_assessment][user_col])
        band_indices = band_rows.index
        frame.loc[band_indices, "activity_band"] = int(band)
        frame.loc[band_indices, "split"] = [
            "assessment" if user_id in assessment_ids else "design"
            for user_id in band_rows[user_col]
        ]
        frame.loc[band_indices, "assignment_hash"] = [
            f"{value:016x}" for value in band_rows["_hash"]
        ]

    frame["activity_band"] = frame["activity_band"].astype(np.int64)
    frame = frame.sort_values(user_col, kind="mergesort")
    return frame.reset_index(drop=True)


def capacity_aware_edge_split(
    design_edges: pd.DataFrame,
    warm_catalogue_spec: Mapping[str, Any],
    split_spec: Mapping[str, Any],
    *,
    user_col: str = "user_id",
    item_col: str = "item_id",
    outer_split_col: str = "outer_split",
) -> pd.DataFrame:
    """Assign distinct validation/test edges without exhausting item support.

    The caller supplies design-user ownership edges only.  Warm items are
    identified from their pre-holdout design-user support.  Users are visited
    in the frozen hash order.  For each user, the function transactionally
    searches for the requested test/validation pair: either both roles are
    committed or neither is, so no nominally evaluable user receives a
    partial holdout.  Item capacity is initial support minus the required
    post-holdout training support.
    """

    missing = {user_col, item_col, outer_split_col}.difference(
        design_edges.columns
    )
    if missing:
        raise ValueError(f"missing edge-split columns: {sorted(missing)}")
    is_design = design_edges[outer_split_col].eq("design").fillna(False)
    if not is_design.all():
        raise ValueError("nested split input must contain design users only")
    edges = design_edges.loc[:, [user_col, item_col]].copy()
    edges[user_col] = _canonical_ids(edges[user_col], field=user_col)
    edges[item_col] = _canonical_ids(edges[item_col], field=item_col)
    if edges.duplicated([user_col, item_col]).any():
        raise ValueError("nested split requires unique ownership edges")
    if edges.empty:
        return pd.DataFrame(
            {
                user_col: pd.Series(dtype=np.int64),
                item_col: pd.Series(dtype=np.int64),
                "role": pd.Series(dtype="string"),
                "is_warm_item": pd.Series(dtype=bool),
                "evaluable_user": pd.Series(dtype=bool),
                "user_split_status": pd.Series(dtype="string"),
                "item_support_before": pd.Series(dtype=np.int64),
                "item_training_support_after": pd.Series(dtype="Int64"),
            }
        )

    minimum_before = int(
        warm_catalogue_spec["minimum_design_user_support_before_holdout"]
    )
    minimum_after = int(
        warm_catalogue_spec["minimum_design_training_support_after_holdout"]
    )
    if minimum_before < minimum_after or minimum_after < 1:
        raise ValueError("invalid warm-catalogue support thresholds")

    support_before = edges.groupby(item_col, sort=False).size().to_dict()
    warm_items = {
        item_id
        for item_id, support in support_before.items()
        if support >= minimum_before
    }
    capacities = {
        item_id: support_before[item_id] - minimum_after
        for item_id in warm_items
    }

    required_edges = int(split_spec["required_warm_edges_per_evaluable_user"])
    role_order = list(split_spec["edge_role_order"])
    if role_order != ["test", "validation"]:
        raise ValueError("this protocol requires test then validation role order")
    if int(split_spec["test_positives_per_user"]) != 1 or int(
        split_spec["validation_positives_per_user"]
    ) != 1:
        raise ValueError("this protocol requires one test and one validation edge")
    if required_edges < len(role_order) + 1:
        raise ValueError("evaluable users must retain at least one training edge")

    warm_by_user: dict[str, list[str]] = {}
    for user_id, item_id in edges.itertuples(index=False, name=None):
        if item_id in warm_items:
            warm_by_user.setdefault(user_id, []).append(item_id)

    namespace = str(split_spec["namespace"])
    candidate_users = [
        user_id
        for user_id, items in warm_by_user.items()
        if len(items) >= required_edges
    ]
    candidate_users.sort(
        key=lambda user_id: (
            stable_hash_uint64(namespace, "user", _id_text(user_id)),
            user_id,
        )
    )

    assigned: dict[tuple[str, str], str] = {}
    evaluable_users: set[str] = set()
    first_role, second_role = role_order
    for user_id in candidate_users:
        items = warm_by_user[user_id]
        first_candidates = sorted(
            items,
            key=lambda item_id: (
                stable_hash_uint64(
                    namespace,
                    first_role,
                    _id_text(user_id),
                    _id_text(item_id),
                ),
                item_id,
            ),
        )
        chosen: tuple[str, str] | None = None
        for first_item in first_candidates:
            if capacities[first_item] <= 0:
                continue
            second_candidates = sorted(
                (item_id for item_id in items if item_id != first_item),
                key=lambda item_id: (
                    stable_hash_uint64(
                        namespace,
                        second_role,
                        _id_text(user_id),
                        _id_text(item_id),
                    ),
                    item_id,
                ),
            )
            for second_item in second_candidates:
                remaining = capacities[second_item]
                if second_item == first_item:
                    remaining -= 1
                if remaining > 0:
                    chosen = (first_item, second_item)
                    break
            if chosen is not None:
                break

        if chosen is None:
            continue
        first_item, second_item = chosen
        capacities[first_item] -= 1
        capacities[second_item] -= 1
        assigned[(user_id, first_item)] = first_role
        assigned[(user_id, second_item)] = second_role
        evaluable_users.add(user_id)

    result = edges.copy()
    result["is_warm_item"] = result[item_col].isin(warm_items)
    result["role"] = [
        (
            assigned.get((user_id, item_id), "training")
            if item_id in warm_items
            else EXCLUDED_NONWARM_ITEM
        )
        for user_id, item_id in zip(result[user_col], result[item_col])
    ]
    result["evaluable_user"] = result[user_col].isin(evaluable_users)
    candidate_user_set = set(candidate_users)
    status_by_user = {
        user_id: (
            "evaluable"
            if user_id in evaluable_users
            else (
                "capacity_exhausted"
                if user_id in candidate_user_set
                else "insufficient_warm_history"
            )
        )
        for user_id in result[user_col].unique()
    }
    result["user_split_status"] = result[user_col].map(status_by_user)
    result["item_support_before"] = result[item_col].map(support_before).astype(
        np.int64
    )

    warm_training = result.loc[
        result["is_warm_item"] & result["role"].eq("training")
    ]
    support_after = warm_training.groupby(item_col, sort=False).size().to_dict()
    result["item_training_support_after"] = result[item_col].map(
        lambda item_id: support_after.get(item_id, 0)
        if item_id in warm_items
        else pd.NA
    )
    result["item_training_support_after"] = result[
        "item_training_support_after"
    ].astype("Int64")

    if any(support_after.get(item_id, 0) < minimum_after for item_id in warm_items):
        raise AssertionError("edge assignment violated minimum item training support")
    if evaluable_users:
        evaluable_role_counts = (
            result.loc[result["evaluable_user"]]
            .groupby([user_col, "role"], sort=False)
            .size()
            .unstack(fill_value=0)
        )
        for role in ("test", "validation"):
            if role not in evaluable_role_counts:
                raise AssertionError(
                    "an evaluable user received a partial holdout"
                )
            if not evaluable_role_counts[role].eq(1).all():
                raise AssertionError(
                    "an evaluable user received a partial holdout"
                )
    retained_warm = result.loc[
        result["evaluable_user"]
        & result["is_warm_item"]
        & result["role"].eq("training")
    ].groupby(user_col, sort=False).size()
    if (retained_warm < required_edges - len(role_order)).any():
        raise AssertionError("an evaluable user retained too few warm training edges")

    result = result.sort_values(
        [user_col, item_col], kind="mergesort"
    )
    return result.reset_index(drop=True)


def proportional_evaluation_user_sample(
    users: pd.DataFrame,
    spec: Mapping[str, Any],
    *,
    user_col: str = "user_id",
    band_col: str = "activity_band",
    outer_split_col: str = "split",
    evaluable_col: str = "evaluable_user",
) -> pd.DataFrame:
    """Select the exact proportional evaluation sample within activity bands."""

    missing = {
        user_col,
        band_col,
        outer_split_col,
        evaluable_col,
    }.difference(users.columns)
    if missing:
        raise ValueError(f"missing evaluation-sample columns: {sorted(missing)}")
    is_design = users[outer_split_col].eq("design").fillna(False)
    if not is_design.all():
        raise ValueError("evaluation sampling is restricted to design users")
    is_evaluable = users[evaluable_col].eq(True).fillna(False)
    if not is_evaluable.all():
        raise ValueError("evaluation sampling is restricted to evaluable users")
    frame = users.copy()
    frame[user_col] = _canonical_ids(frame[user_col], field=user_col)
    if frame[user_col].duplicated().any():
        raise ValueError("evaluation sampling requires unique users")
    if frame[band_col].isna().any():
        raise ValueError("evaluation activity bands cannot be missing")

    counts = frame.groupby(band_col, sort=False).size().to_dict()
    sample_size = int(spec["sample_size"])
    quotas = _proportional_quotas(counts, sample_size)
    namespace = str(spec["namespace"])
    selected_parts: list[pd.DataFrame] = []
    for band in _sorted_strata(list(counts)):
        band_rows = frame.loc[frame[band_col].eq(band)].copy()
        band_rows["selection_hash"] = band_rows[user_col].map(
            lambda user_id: (
                f"{stable_hash_uint64(namespace, band, _id_text(user_id)):016x}"
            )
        )
        band_rows = band_rows.sort_values(
            ["selection_hash", user_col], kind="mergesort"
        ).iloc[: quotas[band]]
        band_rows["sample_stratum_quota"] = quotas[band]
        selected_parts.append(band_rows)

    result = pd.concat(selected_parts, ignore_index=True) if selected_parts else frame.iloc[:0]
    if len(result) != sample_size:
        raise AssertionError("evaluation sample did not meet its frozen exact size")
    result = result.sort_values(user_col, kind="mergesort")
    return result.reset_index(drop=True)


def select_pseudo_cold_items(
    items: pd.DataFrame,
    spec: Mapping[str, Any],
    *,
    item_col: str = "item_id",
    support_col: str = "design_training_support",
    primary_genre_col: str = "primary_genre",
) -> pd.DataFrame:
    """Select exact support-band samples stratified by frozen primary genre.

    ``primary_genre_col`` must already contain the pre-model, deterministically
    constructed primary-genre label.  This function deliberately does not
    infer a primary genre from an unordered token collection.
    """

    missing = {item_col, support_col, primary_genre_col}.difference(items.columns)
    if missing:
        raise ValueError(f"missing pseudo-cold columns: {sorted(missing)}")
    frame = items.loc[:, [item_col, support_col, primary_genre_col]].copy()
    frame[item_col] = _canonical_ids(frame[item_col], field=item_col)
    frame[support_col] = _integer_values(frame[support_col], field=support_col)
    if frame[item_col].duplicated().any():
        raise ValueError("pseudo-cold selection requires unique items")

    def normalize_genre(value: Any) -> str | None:
        if value is None or pd.isna(value):
            return None
        label = str(value).strip()
        return label if label else None

    frame[primary_genre_col] = frame[primary_genre_col].map(normalize_genre)
    requires_genre = bool(spec["candidate_requires_genre"])
    if requires_genre:
        frame = frame.loc[frame[primary_genre_col].notna()].copy()
    else:
        frame[primary_genre_col] = frame[primary_genre_col].fillna("__missing__")

    bands = [(int(lower), int(upper)) for lower, upper in spec["support_bands"]]
    if any(lower >= upper for lower, upper in bands):
        raise ValueError("each pseudo-cold support band must have lower < upper")
    sorted_bands = sorted(bands)
    if any(left[1] > right[0] for left, right in zip(sorted_bands, sorted_bands[1:])):
        raise ValueError("pseudo-cold support bands cannot overlap")
    if spec.get("upper_bound_is_exclusive") is not True:
        raise ValueError("the frozen pseudo-cold upper bound must be exclusive")

    items_per_band = int(spec["items_per_band"])
    namespace = str(spec["namespace"])
    selected_parts: list[pd.DataFrame] = []
    for band_index, (lower, upper) in enumerate(bands):
        candidates = frame.loc[
            frame[support_col].ge(lower) & frame[support_col].lt(upper)
        ].copy()
        if len(candidates) < items_per_band:
            raise ValueError(
                f"support band [{lower}, {upper}) has {len(candidates)} "
                f"eligible items, fewer than the frozen {items_per_band}"
            )
        genre_counts = candidates.groupby(primary_genre_col, sort=False).size().to_dict()
        quotas = _proportional_quotas(genre_counts, items_per_band)
        for genre in _sorted_strata(list(genre_counts)):
            genre_rows = candidates.loc[
                candidates[primary_genre_col].eq(genre)
            ].copy()

            def selection_hash(item_id: Any) -> str:
                value = stable_hash_uint64(
                    namespace,
                    band_index,
                    genre,
                    _id_text(item_id),
                )
                return f"{value:016x}"

            genre_rows["selection_hash"] = genre_rows[item_col].map(
                selection_hash
            )
            genre_rows = genre_rows.sort_values(
                ["selection_hash", item_col], kind="mergesort"
            ).iloc[: quotas[genre]]
            genre_rows["support_band_index"] = band_index
            genre_rows["support_band_lower"] = lower
            genre_rows["support_band_upper_exclusive"] = upper
            genre_rows["genre_stratum_quota"] = quotas[genre]
            selected_parts.append(genre_rows)

    result = pd.concat(selected_parts, ignore_index=True)
    expected_size = items_per_band * len(bands)
    if len(result) != expected_size:
        raise AssertionError("pseudo-cold cohort did not meet its frozen exact size")
    result = result.sort_values(item_col, kind="mergesort")
    return result.reset_index(drop=True)
