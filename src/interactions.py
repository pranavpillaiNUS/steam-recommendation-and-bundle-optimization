"""Canonical sparse interaction contract for Stage 1.

The live target is binary ownership. Playtime is carried only as an observed
confidence modifier.  All constructors operate on explicit edge arrays and
never materialize a dense user-by-item object.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp


REQUIRED_COLUMNS = (
    "user_id",
    "item_id",
    "playtime_forever",
    "playtime_2weeks",
)
_CANONICAL_DECIMAL_ID = re.compile(r"(?:0|[1-9][0-9]*)\Z")
_INT64_MAX_TEXT = str(np.iinfo(np.int64).max)


def _numeric_ids(values: Sequence[Any], *, label: str) -> np.ndarray:
    """Return exact nonnegative int64 IDs without truncating fractional values."""

    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{label} IDs must be one-dimensional")
    if array.dtype.kind == "b":
        raise ValueError(f"{label} IDs cannot be boolean")

    if array.dtype.kind in "iu":
        if array.dtype.kind == "u" and array.size and int(array.max()) > np.iinfo(
            np.int64
        ).max:
            raise ValueError(f"{label} IDs exceed the signed 64-bit range")
        numeric = array.astype(np.int64, copy=False)
    elif array.dtype.kind == "f":
        if not np.all(np.isfinite(array)) or np.any(array != np.floor(array)):
            raise ValueError(f"{label} IDs must be exact integers")
        if np.any(array < 0):
            raise ValueError(f"{label} IDs must be nonnegative")
        if np.any(array >= 2**63):
            raise ValueError(f"{label} IDs exceed the signed 64-bit range")
        numeric = array.astype(np.int64)
    else:
        parsed: list[int] = []
        for value in array.tolist():
            if isinstance(value, (bool, np.bool_)):
                raise ValueError(f"{label} IDs cannot be boolean")
            if isinstance(value, (int, np.integer)):
                integer = int(value)
            elif isinstance(value, (str, np.str_)) and _CANONICAL_DECIMAL_ID.fullmatch(
                str(value)
            ):
                integer = int(value)
            else:
                raise ValueError(
                    f"{label} IDs must be canonical nonnegative decimal integers"
                )
            if integer > np.iinfo(np.int64).max:
                raise ValueError(f"{label} IDs exceed the signed 64-bit range")
            parsed.append(integer)
        numeric = np.asarray(parsed, dtype=np.int64)

    if np.any(numeric < 0):
        raise ValueError(f"{label} IDs must be nonnegative")
    return numeric


def canonical_numeric_ids(values: Sequence[Any], *, label: str) -> np.ndarray:
    """Return a detached canonical int64 ID vector under the frozen contract."""

    return _numeric_ids(values, label=label).copy()


def _playtime(values: Sequence[Any], *, label: str) -> np.ndarray:
    array = np.array(values, dtype=np.float64, copy=True)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must be a finite one-dimensional vector")
    if np.any(array < 0):
        raise ValueError(f"{label} cannot be negative")
    array[array == 0.0] = 0.0
    return array


@dataclass(frozen=True)
class CanonicalEdges:
    """Unique user-item edges in ascending numeric ``(user_id, item_id)`` order."""

    user_id: np.ndarray
    item_id: np.ndarray
    playtime_forever: np.ndarray
    playtime_2weeks: np.ndarray
    input_row_count: int
    duplicate_excess_rows: int

    def __post_init__(self) -> None:
        users = _numeric_ids(self.user_id, label="user").copy()
        items = _numeric_ids(self.item_id, label="item").copy()
        forever = _playtime(self.playtime_forever, label="playtime_forever")
        recent = _playtime(self.playtime_2weeks, label="playtime_2weeks")
        for array in (users, items, forever, recent):
            array.setflags(write=False)
        object.__setattr__(self, "user_id", users)
        object.__setattr__(self, "item_id", items)
        object.__setattr__(self, "playtime_forever", forever)
        object.__setattr__(self, "playtime_2weeks", recent)
        lengths = {
            len(users),
            len(items),
            len(forever),
            len(recent),
        }
        if len(lengths) != 1:
            raise ValueError("canonical edge columns have inconsistent lengths")
        if int(self.input_row_count) != self.input_row_count or self.input_row_count < 0:
            raise ValueError("input row count must be a nonnegative integer")
        if (
            int(self.duplicate_excess_rows) != self.duplicate_excess_rows
            or self.duplicate_excess_rows < 0
        ):
            raise ValueError("duplicate count must be a nonnegative integer")
        if self.input_row_count < self.n_edges:
            raise ValueError("input row count cannot be smaller than the edge count")
        if self.duplicate_excess_rows != self.input_row_count - self.n_edges:
            raise ValueError("duplicate count does not reconcile with input rows")
        if self.n_edges:
            prior_is_smaller = (users[1:] > users[:-1]) | (
                (users[1:] == users[:-1]) & (items[1:] > items[:-1])
            )
            if not np.all(prior_is_smaller):
                raise ValueError("canonical edges must be unique and lexicographically sorted")

    @property
    def n_edges(self) -> int:
        return len(self.user_id)

    def subset(self, mask: Sequence[bool]) -> "CanonicalEdges":
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (self.n_edges,):
            raise ValueError("edge mask has the wrong shape")
        return CanonicalEdges(
            user_id=self.user_id[mask],
            item_id=self.item_id[mask],
            playtime_forever=self.playtime_forever[mask],
            playtime_2weeks=self.playtime_2weeks[mask],
            input_row_count=int(mask.sum()),
            duplicate_excess_rows=0,
        )


def canonicalize_edges(
    user_id: Sequence[Any],
    item_id: Sequence[Any],
    playtime_forever: Sequence[Any],
    playtime_2weeks: Sequence[Any] | None = None,
) -> CanonicalEdges:
    """Sort edges and collapse duplicates by maximum nonnegative playtime.

    Ownership is binary, so every duplicate group becomes one edge. Maximum
    playtime is chosen rather than a sum so repeated snapshot rows cannot
    manufacture additional engagement.
    """

    users = _numeric_ids(user_id, label="user")
    items = _numeric_ids(item_id, label="item")
    forever = _playtime(playtime_forever, label="playtime_forever")
    if playtime_2weeks is None:
        recent = np.zeros(len(users), dtype=np.float64)
    else:
        recent = _playtime(playtime_2weeks, label="playtime_2weeks")
    if not (len(users) == len(items) == len(forever) == len(recent)):
        raise ValueError("interaction columns have inconsistent lengths")

    order = np.lexsort((items, users))
    users = users[order]
    items = items[order]
    forever = forever[order]
    recent = recent[order]
    if len(users) == 0:
        return CanonicalEdges(
            users, items, forever, recent, input_row_count=0, duplicate_excess_rows=0
        )

    starts = np.r_[
        0,
        np.flatnonzero((users[1:] != users[:-1]) | (items[1:] != items[:-1])) + 1,
    ]
    return CanonicalEdges(
        user_id=users[starts],
        item_id=items[starts],
        playtime_forever=np.maximum.reduceat(forever, starts),
        playtime_2weeks=np.maximum.reduceat(recent, starts),
        input_row_count=len(order),
        duplicate_excess_rows=len(order) - len(starts),
    )


@dataclass(frozen=True)
class InteractionLoadAudit:
    """Aggregate exclusion and reconciliation counts for one source load."""

    source_rows: int
    eligible_rows_before_duplicate_collapse: int
    excluded_rows: int
    excluded_by_primary_reason: Mapping[str, int]
    raw_invalid_flags: Mapping[str, int]
    chunks_read: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_rows": int(self.source_rows),
            "eligible_rows_before_duplicate_collapse": int(
                self.eligible_rows_before_duplicate_collapse
            ),
            "excluded_rows": int(self.excluded_rows),
            "excluded_by_primary_reason": {
                key: int(value)
                for key, value in sorted(self.excluded_by_primary_reason.items())
            },
            "raw_invalid_flags": {
                key: int(value) for key, value in sorted(self.raw_invalid_flags.items())
            },
            "chunks_read": int(self.chunks_read),
        }


@dataclass(frozen=True)
class InteractionLoadResult:
    edges: CanonicalEdges
    audit: InteractionLoadAudit


def _id_validity(values: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Return missing and valid masks under the frozen decimal-ID contract."""

    text = values.astype("string")
    missing = text.isna() | text.eq("")
    canonical = text.str.fullmatch(r"(?:0|[1-9][0-9]*)", na=False)
    lengths = text.str.len().fillna(0)
    within_int64 = lengths.lt(len(_INT64_MAX_TEXT)) | (
        lengths.eq(len(_INT64_MAX_TEXT)) & text.le(_INT64_MAX_TEXT).fillna(False)
    )
    valid = (~missing) & canonical & within_int64
    return missing.astype(bool), valid.astype(bool)


def load_interaction_csv_audited(
    path: str | Path, *, chunksize: int = 500_000
) -> InteractionLoadResult:
    """Load, audit, filter, and canonicalize the frozen interaction CSV.

    Missing or noncanonical IDs are excluded and counted, as preregistered.
    Nonfinite or negative playtime is a hard failure even on a row whose ID is
    otherwise ineligible. The source is read in bounded chunks; only the four
    canonical edge arrays are retained before global duplicate collapse.
    """

    if isinstance(chunksize, (bool, np.bool_)):
        raise ValueError("chunksize must be a positive integer")
    chunk_rows = int(chunksize)
    if chunk_rows != chunksize or chunk_rows <= 0:
        raise ValueError("chunksize must be a positive integer")

    users: list[np.ndarray] = []
    items: list[np.ndarray] = []
    forever_values: list[np.ndarray] = []
    recent_values: list[np.ndarray] = []
    source_rows = 0
    excluded_by_reason = {
        "missing_user_id": 0,
        "invalid_user_id": 0,
        "missing_item_id": 0,
        "invalid_item_id": 0,
    }
    invalid_flags = {
        "missing_user_id": 0,
        "invalid_user_id": 0,
        "missing_item_id": 0,
        "invalid_item_id": 0,
    }
    chunks_read = 0

    reader = pd.read_csv(
        path,
        usecols=list(REQUIRED_COLUMNS),
        dtype={
            "user_id": "string",
            "item_id": "string",
            "playtime_forever": np.float64,
            "playtime_2weeks": np.float64,
        },
        chunksize=chunk_rows,
        keep_default_na=False,
    )
    for frame in reader:
        chunks_read += 1
        source_rows += len(frame)
        forever = frame["playtime_forever"].to_numpy(dtype=np.float64)
        recent = frame["playtime_2weeks"].to_numpy(dtype=np.float64)
        _playtime(forever, label="playtime_forever")
        _playtime(recent, label="playtime_2weeks")

        user_missing, user_valid = _id_validity(frame["user_id"])
        item_missing, item_valid = _id_validity(frame["item_id"])
        user_invalid = (~user_missing) & (~user_valid)
        item_invalid = (~item_missing) & (~item_valid)
        raw_masks = {
            "missing_user_id": user_missing,
            "invalid_user_id": user_invalid,
            "missing_item_id": item_missing,
            "invalid_item_id": item_invalid,
        }
        for reason, mask in raw_masks.items():
            invalid_flags[reason] += int(mask.sum())

        primary_masks = {
            "missing_user_id": user_missing,
            "invalid_user_id": (~user_missing) & user_invalid,
            "missing_item_id": user_valid & item_missing,
            "invalid_item_id": user_valid & (~item_missing) & item_invalid,
        }
        for reason, mask in primary_masks.items():
            excluded_by_reason[reason] += int(mask.sum())

        eligible = user_valid & item_valid
        if eligible.any():
            users.append(
                frame.loc[eligible, "user_id"].astype(np.int64).to_numpy(copy=True)
            )
            items.append(
                frame.loc[eligible, "item_id"].astype(np.int64).to_numpy(copy=True)
            )
            eligible_array = eligible.to_numpy(dtype=bool)
            forever_values.append(forever[eligible_array])
            recent_values.append(recent[eligible_array])

    empty_int = np.empty(0, dtype=np.int64)
    empty_float = np.empty(0, dtype=np.float64)
    all_users = np.concatenate(users) if users else empty_int
    all_items = np.concatenate(items) if items else empty_int
    all_forever = np.concatenate(forever_values) if forever_values else empty_float
    all_recent = np.concatenate(recent_values) if recent_values else empty_float
    edges = canonicalize_edges(all_users, all_items, all_forever, all_recent)
    excluded_rows = int(sum(excluded_by_reason.values()))
    if source_rows != edges.input_row_count + excluded_rows:
        raise AssertionError("source, eligible, and excluded row counts do not reconcile")
    audit = InteractionLoadAudit(
        source_rows=source_rows,
        eligible_rows_before_duplicate_collapse=edges.input_row_count,
        excluded_rows=excluded_rows,
        excluded_by_primary_reason=excluded_by_reason,
        raw_invalid_flags=invalid_flags,
        chunks_read=chunks_read,
    )
    return InteractionLoadResult(edges=edges, audit=audit)


def load_interaction_csv(path: str | Path) -> CanonicalEdges:
    """Load canonical edges while retaining the legacy edge-only return type."""

    return load_interaction_csv_audited(path).edges


@dataclass(frozen=True)
class SparseInteractionData:
    """Aligned sparse ownership and playtime structures."""

    ownership: sp.csr_matrix
    playtime_forever: sp.csr_matrix
    playtime_2weeks: sp.csr_matrix
    user_ids: np.ndarray
    item_ids: np.ndarray

    def __post_init__(self) -> None:
        users = _numeric_ids(self.user_ids, label="user").copy()
        items = _numeric_ids(self.item_ids, label="item").copy()
        if np.any(users[1:] <= users[:-1]) or np.any(items[1:] <= items[:-1]):
            raise ValueError("user_ids and item_ids must be unique ascending numeric IDs")
        users.setflags(write=False)
        items.setflags(write=False)
        object.__setattr__(self, "user_ids", users)
        object.__setattr__(self, "item_ids", items)

        expected = (len(users), len(items))
        for name, matrix in (
            ("ownership", self.ownership),
            ("playtime_forever", self.playtime_forever),
            ("playtime_2weeks", self.playtime_2weeks),
        ):
            if not sp.isspmatrix_csr(matrix) or matrix.shape != expected:
                raise ValueError(f"{name} violates the common CSR shape contract")
            if matrix.dtype != np.dtype(np.float32):
                raise ValueError(f"{name} must use the float32 storage contract")
            if not matrix.has_canonical_format:
                raise ValueError(f"{name} must use canonical sorted CSR storage")
            if not np.all(np.isfinite(matrix.data)):
                raise ValueError(f"{name} contains nonfinite data")
        if np.any(self.ownership.data != 1.0):
            raise ValueError("ownership must contain binary observed values")
        if np.any(self.playtime_forever.data < 0) or np.any(
            self.playtime_2weeks.data < 0
        ):
            raise ValueError("playtime matrices cannot contain negative values")
        if not (
            np.array_equal(self.ownership.indptr, self.playtime_forever.indptr)
            and np.array_equal(self.ownership.indices, self.playtime_forever.indices)
            and np.array_equal(self.ownership.indptr, self.playtime_2weeks.indptr)
            and np.array_equal(self.ownership.indices, self.playtime_2weeks.indices)
        ):
            raise ValueError("ownership and playtime sparsity patterns are misaligned")


def build_sparse_interactions(
    edges: CanonicalEdges,
    *,
    user_ids: Sequence[Any] | None = None,
    item_ids: Sequence[Any] | None = None,
) -> SparseInteractionData:
    """Build aligned CSR matrices in explicit ascending numeric ID order."""

    users = (
        np.unique(edges.user_id)
        if user_ids is None
        else _numeric_ids(user_ids, label="user")
    )
    items = (
        np.unique(edges.item_id)
        if item_ids is None
        else _numeric_ids(item_ids, label="item")
    )
    if np.any(users[1:] <= users[:-1]) or np.any(items[1:] <= items[:-1]):
        raise ValueError("explicit user_ids and item_ids must be unique ascending numeric IDs")

    rows = np.searchsorted(users, edges.user_id)
    cols = np.searchsorted(items, edges.item_id)
    valid = (
        (rows < len(users))
        & (cols < len(items))
        & (users[np.minimum(rows, len(users) - 1)] == edges.user_id)
        & (items[np.minimum(cols, len(items) - 1)] == edges.item_id)
    ) if len(users) and len(items) else np.zeros(edges.n_edges, dtype=bool)
    if not np.all(valid):
        raise ValueError("an edge ID is absent from the explicit ID contract")

    shape = (len(users), len(items))
    indices = (rows, cols)
    ownership = sp.csr_matrix(
        (np.ones(edges.n_edges, dtype=np.float32), indices), shape=shape
    )
    forever = sp.csr_matrix(
        (edges.playtime_forever.astype(np.float32), indices), shape=shape
    )
    recent = sp.csr_matrix(
        (edges.playtime_2weeks.astype(np.float32), indices), shape=shape
    )
    for matrix in (ownership, forever, recent):
        matrix.sort_indices()
    return SparseInteractionData(ownership, forever, recent, users, items)


def assert_exact_id_alignment(
    expected_ids: Sequence[Any],
    actual_ids: Sequence[Any],
    *,
    label: str,
) -> None:
    """Reject any difference between two ordered numeric ID contracts."""

    expected = _numeric_ids(expected_ids, label=f"expected {label}")
    actual = _numeric_ids(actual_ids, label=f"actual {label}")
    if expected.shape != actual.shape:
        raise ValueError(
            f"{label} ID-map length mismatch: expected {expected.size}, "
            f"got {actual.size}"
        )
    mismatch = np.flatnonzero(expected != actual)
    if mismatch.size:
        row = int(mismatch[0])
        raise ValueError(
            f"{label} ID-map order mismatch at row {row}: "
            f"expected {expected[row]}, got {actual[row]}"
        )


def assert_aligned_item_rows(
    expected_item_ids: Sequence[Any],
    actual_item_ids: Sequence[Any],
    matrix: sp.spmatrix,
) -> None:
    """Require a feature-like matrix to declare the exact model item row map."""

    if len(matrix.shape) != 2:
        raise ValueError("aligned item data must be a two-dimensional matrix")
    actual_count = len(actual_item_ids)
    if matrix.shape[0] != actual_count:
        raise ValueError("item-row matrix shape does not match its declared ID map")
    assert_exact_id_alignment(
        expected_item_ids,
        actual_item_ids,
        label="item",
    )


def observed_confidence(
    ownership: sp.csr_matrix,
    playtime: sp.csr_matrix,
    *,
    alpha_o: float,
    alpha_p: float,
    tau: float,
) -> sp.csr_matrix:
    """Build observed $c_{ui}>1$ values with implicit unobserved baseline one."""

    if not sp.isspmatrix_csr(ownership) or not sp.isspmatrix_csr(playtime):
        raise TypeError("ownership and playtime must be CSR")
    if ownership.shape != playtime.shape or not (
        np.array_equal(ownership.indptr, playtime.indptr)
        and np.array_equal(ownership.indices, playtime.indices)
    ):
        raise ValueError("ownership and playtime must have identical sparsity patterns")
    parameters = np.asarray([alpha_o, alpha_p, tau], dtype=np.float64)
    if not np.all(np.isfinite(parameters)):
        raise ValueError("confidence parameters must be finite")
    if alpha_o <= 0 or alpha_p < 0 or tau < 0:
        raise ValueError("confidence parameters violate the frozen nonnegative contract")
    if not np.all(np.isfinite(playtime.data)) or np.any(playtime.data < 0):
        raise ValueError("playtime confidence input must be finite and nonnegative")
    if np.any(ownership.data != 1.0):
        raise ValueError("ownership confidence input must be binary")
    with np.errstate(over="ignore", invalid="ignore"):
        raw_values = (
            1.0
            + float(alpha_o)
            + float(alpha_p) * np.minimum(np.log1p(playtime.data), float(tau))
        )
    if not np.all(np.isfinite(raw_values)) or np.any(
        raw_values > np.finfo(np.float32).max
    ):
        raise ValueError("confidence values must remain finite in float32 storage")
    values = raw_values.astype(np.float32)
    result = sp.csr_matrix(
        (values, ownership.indices.copy(), ownership.indptr.copy()),
        shape=ownership.shape,
    )
    if np.any(result.data <= 1.0):
        raise AssertionError("owned interactions must exceed unobserved confidence")
    return result


def remove_observed_pairs(
    data: SparseInteractionData,
    heldout_user_ids: Sequence[Any],
    heldout_item_ids: Sequence[Any],
) -> SparseInteractionData:
    """Remove observed pairs from ownership and both playtime structures.

    Every requested pair must be present exactly once. The input object is not
    mutated, and explicit zero-playtime entries remain aligned for all retained
    ownership edges.
    """

    users = _numeric_ids(heldout_user_ids, label="held-out user")
    items = _numeric_ids(heldout_item_ids, label="held-out item")
    if users.shape != items.shape:
        raise ValueError("held-out user and item arrays must have equal shape")
    pairs = np.column_stack((users, items))
    if pairs.size and np.unique(pairs, axis=0).shape[0] != pairs.shape[0]:
        raise ValueError("held-out pairs must be unique")

    rows = np.searchsorted(data.user_ids, users)
    cols = np.searchsorted(data.item_ids, items)
    if users.size:
        if np.any(rows >= len(data.user_ids)) or np.any(cols >= len(data.item_ids)):
            raise ValueError("a held-out ID is outside the sparse interaction contract")
        if not np.array_equal(data.user_ids[rows], users) or not np.array_equal(
            data.item_ids[cols], items
        ):
            raise ValueError("a held-out ID is outside the sparse interaction contract")

    remove_positions = np.empty(users.size, dtype=np.int64)
    for index, (row, col) in enumerate(zip(rows, cols)):
        start = int(data.ownership.indptr[row])
        stop = int(data.ownership.indptr[row + 1])
        offset = int(np.searchsorted(data.ownership.indices[start:stop], col))
        position = start + offset
        if position >= stop or data.ownership.indices[position] != col:
            raise ValueError("a requested held-out pair is not an observed ownership edge")
        remove_positions[index] = position

    keep = np.ones(data.ownership.nnz, dtype=bool)
    keep[remove_positions] = False
    removed_per_row = np.bincount(rows, minlength=data.ownership.shape[0])
    retained_per_row = np.diff(data.ownership.indptr) - removed_per_row
    new_indptr = np.empty(data.ownership.shape[0] + 1, dtype=data.ownership.indptr.dtype)
    new_indptr[0] = 0
    np.cumsum(retained_per_row, out=new_indptr[1:])

    def filtered(matrix: sp.csr_matrix) -> sp.csr_matrix:
        return sp.csr_matrix(
            (
                matrix.data[keep].copy(),
                matrix.indices[keep].copy(),
                new_indptr.copy(),
            ),
            shape=matrix.shape,
        )

    return SparseInteractionData(
        ownership=filtered(data.ownership),
        playtime_forever=filtered(data.playtime_forever),
        playtime_2weeks=filtered(data.playtime_2weeks),
        user_ids=data.user_ids.copy(),
        item_ids=data.item_ids.copy(),
    )


def sparse_storage_bytes(data: SparseInteractionData) -> int:
    """Return actual CSR and ID-array storage, excluding Python object overhead."""

    matrices = (data.ownership, data.playtime_forever, data.playtime_2weeks)
    return int(
        sum(
            matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
            for matrix in matrices
        )
        + data.user_ids.nbytes
        + data.item_ids.nbytes
    )


def array_sha256(array: np.ndarray) -> str:
    """Hash dtype, shape, and canonical C-order bytes."""

    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def id_map_sha256(values: Sequence[Any], *, label: str) -> str:
    """Hash ordered IDs as canonical unpadded UTF-8 decimals separated by LF."""

    identifiers = _numeric_ids(values, label=label)
    digest = hashlib.sha256()
    for identifier in identifiers:
        digest.update(str(int(identifier)).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def edge_sha256(edges: CanonicalEdges) -> str:
    """Hash the canonical unique edge table including both playtime fields."""

    digest = hashlib.sha256()
    for array in (
        edges.user_id,
        edges.item_id,
        edges.playtime_forever,
        edges.playtime_2weeks,
    ):
        digest.update(bytes.fromhex(array_sha256(array)))
    return digest.hexdigest()


def csr_semantic_sha256(matrix: sp.spmatrix) -> str:
    """Hash canonical CSR semantics independently of platform index width."""

    canonical = matrix.tocsr().astype(np.float32, copy=True)
    canonical.sum_duplicates()
    canonical.sort_indices()
    canonical.data[canonical.data == 0.0] = 0.0
    digest = hashlib.sha256()
    digest.update(np.asarray(canonical.shape, dtype="<i8").tobytes())
    digest.update(canonical.indptr.astype("<i8", copy=False).tobytes())
    digest.update(canonical.indices.astype("<i8", copy=False).tobytes())
    digest.update(canonical.data.astype("<f4", copy=False).tobytes())
    return digest.hexdigest()


def save_sparse_interactions(
    data: SparseInteractionData,
    directory: str | Path,
    *,
    prefix: str,
) -> dict[str, str]:
    """Save matrices and fixed-dtype IDs; return their file hashes."""

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "ownership": directory / f"{prefix}_ownership.npz",
        "playtime_forever": directory / f"{prefix}_playtime_forever.npz",
        "playtime_2weeks": directory / f"{prefix}_playtime_2weeks.npz",
        "user_ids": directory / f"{prefix}_user_ids.npy",
        "item_ids": directory / f"{prefix}_item_ids.npy",
    }
    sp.save_npz(paths["ownership"], data.ownership, compressed=True)
    sp.save_npz(paths["playtime_forever"], data.playtime_forever, compressed=True)
    sp.save_npz(paths["playtime_2weeks"], data.playtime_2weeks, compressed=True)
    np.save(paths["user_ids"], data.user_ids, allow_pickle=False)
    np.save(paths["item_ids"], data.item_ids, allow_pickle=False)
    from src.stage1_protocol import file_sha256

    return {name: file_sha256(path) for name, path in paths.items()}


def load_sparse_interactions(
    directory: str | Path,
    *,
    prefix: str,
    expected_file_hashes: Mapping[str, str] | None = None,
) -> SparseInteractionData:
    """Load saved sparse structures, optionally verifying physical file hashes."""

    directory = Path(directory)
    paths = {
        "ownership": directory / f"{prefix}_ownership.npz",
        "playtime_forever": directory / f"{prefix}_playtime_forever.npz",
        "playtime_2weeks": directory / f"{prefix}_playtime_2weeks.npz",
        "user_ids": directory / f"{prefix}_user_ids.npy",
        "item_ids": directory / f"{prefix}_item_ids.npy",
    }
    if expected_file_hashes is not None:
        from src.stage1_protocol import file_sha256

        for name, path in paths.items():
            expected = expected_file_hashes.get(name)
            if expected is None or file_sha256(path) != expected:
                raise ValueError(f"saved interaction artifact hash mismatch: {name}")

    return SparseInteractionData(
        ownership=sp.load_npz(paths["ownership"]).tocsr(),
        playtime_forever=sp.load_npz(paths["playtime_forever"]).tocsr(),
        playtime_2weeks=sp.load_npz(paths["playtime_2weeks"]).tocsr(),
        user_ids=np.load(paths["user_ids"], allow_pickle=False),
        item_ids=np.load(paths["item_ids"], allow_pickle=False),
    )
