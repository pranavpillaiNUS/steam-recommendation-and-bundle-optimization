"""Additive Stage 1 scoring runtime; the frozen v2 implementation is untouched.

Scores and candidate masks are bounded in BOTH axes before allocation. Model
arrays, O(n_items) coverage, metrics, factor slices, and NumPy/BLAS workspaces
are additional memory. Exact global quantiles use disposable disk storage;
memory mapping does not impose a hard limit on operating-system resident RAM.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable, Iterator, Mapping, Sequence
import zipfile

import numpy as np

from src.pseudo_utility import transform
from src.ranking import (
    DEFAULT_KS,
    DEFAULT_MAX_SCORE_BLOCK_BYTES,
    TargetTieCounter,
    TopKBoundaryAccumulator,
    inclusion_probabilities_at_boundary,
)


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _indices(values: Any, name: str, *, size: int | None = None) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1 or (raw.size and raw.dtype.kind not in "iu"):
        raise ValueError(f"{name} must be a one-dimensional integer array")
    if raw.size and (np.any(raw < 0) or np.any(raw > np.iinfo(np.int64).max)):
        raise ValueError(f"{name} must contain nonnegative int64 values")
    result = raw.astype(np.int64, copy=False)
    if size is not None and np.any(result >= size):
        raise IndexError(f"{name} contains an index outside its axis")
    return result


def _finite_real_array(values: Any, name: str, ndim: int) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != ndim or array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a real numeric array of dimension {ndim}")
    # Validate in row chunks without materializing a full bool factor matrix.
    for start in range(0, array.shape[0], 1024):
        if not np.all(np.isfinite(array[start : start + 1024])):
            raise ValueError(f"{name} must be finite")
    return array


class FactorScoreSource:
    """Reusable float64 factor scorer with an explicit score/mask byte cap.

    Caller-owned parameters must not be mutated during an operation. Two
    identical passes use identical row/column blocks, including the block
    containing the target: a separately calculated dot product is never used
    to decide exact score ties. Unoptimized einsum makes the float64 reduction
    independent of the user and item blocking. This successor arithmetic may
    differ at rounding precision from the frozen BLAS implementation.
    """

    def __init__(
        self,
        user_factors: np.ndarray,
        item_factors: np.ndarray,
        item_bias: np.ndarray | None = None,
        *,
        maximum_score_block_bytes: int = DEFAULT_MAX_SCORE_BLOCK_BYTES,
        user_block_size: int = 128,
        item_block_size: int = 4096,
    ) -> None:
        self.user_factors = _finite_real_array(user_factors, "user_factors", 2)
        self.item_factors = _finite_real_array(item_factors, "item_factors", 2)
        if min(*self.user_factors.shape, *self.item_factors.shape) < 1:
            raise ValueError("factor axes must be nonempty")
        if self.user_factors.shape[1] != self.item_factors.shape[1]:
            raise ValueError("user and item factors must have the same factor count")
        self.n_users, self.n_factors = self.user_factors.shape
        self.n_items = self.item_factors.shape[0]
        self.item_bias = None if item_bias is None else _finite_real_array(item_bias, "item_bias", 1)
        if self.item_bias is not None and self.item_bias.shape != (self.n_items,):
            raise ValueError("item_bias must have one value per item")
        self.maximum_score_block_bytes = _integer(maximum_score_block_bytes, "maximum_score_block_bytes", minimum=9)
        self.user_block_size = _integer(user_block_size, "user_block_size", minimum=1)
        self.item_block_size = _integer(item_block_size, "item_block_size", minimum=1)

    def user_batches(self, user_indices: Sequence[int] | np.ndarray) -> Iterator[tuple[int, np.ndarray]]:
        rows = _indices(user_indices, "user_indices", size=self.n_users)
        batch_size = min(self.user_block_size, self.maximum_score_block_bytes // 9)
        for start in range(0, rows.size, batch_size):
            yield start, rows[start : start + batch_size]

    def item_blocks(self, user_indices: Sequence[int] | np.ndarray) -> Iterator[tuple[int, int, np.ndarray]]:
        rows = _indices(user_indices, "user_indices", size=self.n_users)
        if not rows.size or rows.size > self.user_block_size:
            raise ValueError("item_blocks requires a nonempty bounded user batch")
        width = min(self.item_block_size, self.maximum_score_block_bytes // (9 * rows.size))
        if width < 1:
            raise MemoryError("one score column and mask exceed the byte budget")
        for start in range(0, self.n_items, width):
            stop = min(start + width, self.n_items)
            yield start, stop, self.score(rows, start, stop)

    def score(self, user_indices: Sequence[int] | np.ndarray, start: int, stop: int) -> np.ndarray:
        rows = _indices(user_indices, "user_indices", size=self.n_users)
        start = _integer(start, "start")
        stop = _integer(stop, "stop", minimum=1)
        if start >= stop or stop > self.n_items or not rows.size:
            raise ValueError("invalid score block coordinates")
        if rows.size > self.user_block_size or stop - start > self.item_block_size:
            raise MemoryError("score block exceeds the configured axis bounds")
        if rows.size * (stop - start) * 9 > self.maximum_score_block_bytes:
            raise MemoryError("score block and candidate mask exceed the byte budget")
        users = np.asarray(self.user_factors[rows], dtype=np.float64)
        items = np.asarray(self.item_factors[start:stop], dtype=np.float64)
        with np.errstate(over="raise", invalid="raise"):
            values = np.einsum("uf,if->ui", users, items, optimize=False)
            if self.item_bias is not None:
                values += self.item_bias[start:stop]
        if not np.all(np.isfinite(values)):
            raise FloatingPointError("factor scores became nonfinite")
        return values

    @property
    def resource_contract(self) -> dict[str, Any]:
        return {
            "maximum_score_and_mask_bytes": self.maximum_score_block_bytes,
            "user_block_size": self.user_block_size,
            "item_block_size": self.item_block_size,
            "score_dtype": "float64",
            "score_reduction": "numpy_einsum_optimize_false",
            "additional_memory": "resident model arrays; bounded factor conversions; row workspaces; O(n_items) coverage; O(n_users) metrics; BLAS workspace",
            "full_user_catalogue_score_matrix_in_ram": False,
            "hard_process_rss_limit": False,
        }


@dataclass(frozen=True)
class RankingResult:
    metrics: Mapping[str, np.ndarray]
    aggregate: Mapping[str, float | int]
    resource_contract: Mapping[str, Any]


def evaluate_factor_ranking(
    source: FactorScoreSource,
    user_indices: Sequence[int] | np.ndarray,
    target_indices: Sequence[int] | np.ndarray,
    excluded_by_user: Sequence[Sequence[int] | np.ndarray],
    *,
    ks: Sequence[int] = DEFAULT_KS,
    top_item_mask: np.ndarray | None = None,
    candidate_mask_provider: Callable[[np.ndarray, int, int], np.ndarray] | None = None,
) -> RankingResult:
    """Exact full-catalogue target metrics and expected top-K item coverage.

    Exclusions must include training positives and the other held-out target.
    This function validates the target remains eligible; split membership is
    the caller's responsibility. Coverage assumes independent random tie
    ordering across users, as in the frozen ranking primitives.
    """
    users = _indices(user_indices, "user_indices", size=source.n_users)
    targets = _indices(target_indices, "target_indices", size=source.n_items)
    checked_ks = tuple(_integer(k, "k", minimum=1) for k in ks)
    if not users.size or targets.shape != users.shape or len(excluded_by_user) != users.size:
        raise ValueError("provide one target and exclusion vector per nonempty user sample")
    if not checked_ks or len(set(checked_ks)) != len(checked_ks):
        raise ValueError("ks must be nonempty and contain no duplicates")
    if np.unique(users).size != users.size:
        raise ValueError("evaluation user_indices must be unique")
    excluded = [np.unique(_indices(x, "exclusions", size=source.n_items)) for x in excluded_by_user]
    for target, omissions in zip(targets, excluded):
        if np.any(omissions == target):
            raise ValueError("a held-out target was included in the exclusions")
    if top_item_mask is not None:
        top_item_mask = np.asarray(top_item_mask)
        if top_item_mask.dtype.kind != "b" or top_item_mask.shape != (source.n_items,):
            raise ValueError("top_item_mask must be a boolean catalogue vector")
    metrics = {name: np.empty(users.size, dtype=dtype) for name, dtype in (
        ("strictly_above", np.int64), ("tied_block_size", np.int64), ("expected_rank", np.float64)
    )}
    for k in checked_ks:
        metrics[f"recall_at_{k}"] = np.empty(users.size, dtype=np.float64)
        metrics[f"ndcg_at_{k}"] = np.empty(users.size, dtype=np.float64)
    not_included = np.ones(source.n_items, dtype=np.float64)
    top_exposure = total_exposure = 0.0

    def masks_for(offset: int, count: int, start: int, stop: int) -> np.ndarray:
        masks = np.ones((count, stop - start), dtype=bool)
        if candidate_mask_provider is not None:
            supplied = np.asarray(candidate_mask_provider(users[offset:offset + count], start, stop))
            if supplied.dtype != np.dtype(bool) or supplied.shape != masks.shape:
                raise ValueError("opaque candidate mask provider returned an invalid block")
            masks &= supplied
        for local in range(count):
            omit = excluded[offset + local]
            left, right = np.searchsorted(omit, [start, stop])
            masks[local, omit[left:right] - start] = False
        return masks

    for offset, rows in source.user_batches(users):
        boundaries = [TopKBoundaryAccumulator(max(checked_ks)) for _ in rows]
        target_scores = np.full(rows.size, np.nan)
        local_targets = targets[offset : offset + rows.size]
        for start, stop, scores in source.item_blocks(rows):
            masks = masks_for(offset, rows.size, start, stop)
            for local in range(rows.size):
                boundaries[local].update(scores[local], masks[local])
                if start <= local_targets[local] < stop:
                    if not masks[local, local_targets[local] - start]:
                        raise ValueError("a held-out target was excluded by the mask provider")
                    target_scores[local] = scores[local, local_targets[local] - start]
            del masks, scores
        counters = [TargetTieCounter(score) for score in target_scores]
        final_boundaries = [accumulator.boundary() for accumulator in boundaries]
        for start, stop, scores in source.item_blocks(rows):
            masks = masks_for(offset, rows.size, start, stop)
            for local in range(rows.size):
                target = int(local_targets[local])
                counters[local].update(scores[local], masks[local], target_offset=target - start if start <= target < stop else None)
                probabilities = inclusion_probabilities_at_boundary(scores[local], final_boundaries[local], masks[local])
                not_included[start:stop] *= 1.0 - probabilities
                total_exposure += float(probabilities.sum())
                if top_item_mask is not None:
                    top_exposure += float(probabilities[top_item_mask[start:stop]].sum())
            del masks, scores
        for local, counter in enumerate(counters):
            for name, value in counter.metrics(checked_ks).as_flat_dict().items():
                metrics[name][offset + local] = value
    aggregate: dict[str, float | int] = {f"mean_{name}": float(values.mean()) for name, values in metrics.items()}
    aggregate.update({
        "evaluation_users": int(users.size),
        f"expected_catalogue_coverage_at_{max(checked_ks)}": float(np.mean(1.0 - not_included)),
    })
    if top_item_mask is not None:
        aggregate[f"expected_top_item_concentration_at_{max(checked_ks)}"] = top_exposure / total_exposure
    return RankingResult(metrics, aggregate, {**source.resource_contract, "score_passes": 2})


def fit_global_parameters_disk(
    source: FactorScoreSource,
    user_indices: Sequence[int] | np.ndarray,
    *,
    scratch_directory: str | Path,
    maximum_disk_bytes: int,
) -> dict[str, Any]:
    """Fit exact linear quantiles without a dense heap score matrix.

    A temporary flattened float64 score sample occupies 8*n_users*n_items
    disk bytes. It is hashed in row-major order BEFORE in-place partitioning.
    Scratch is removed on success or failure, including on Windows. The
    returned fit includes parameters, the pre-partition score hash and the
    resource contract. OS mapped-page residency is not bounded by NumPy heap.
    """
    users = _indices(user_indices, "user_indices", size=source.n_users)
    if not users.size or np.unique(users).size != users.size:
        raise ValueError("parameter-fit user_indices must be nonempty and unique")
    needed = int(users.size) * source.n_items * 8
    limit = _integer(maximum_disk_bytes, "maximum_disk_bytes", minimum=1)
    if needed > limit:
        raise MemoryError("exact global score sample exceeds maximum_disk_bytes")
    scratch = Path(scratch_directory)
    scratch.mkdir(parents=True, exist_ok=True)
    if needed > shutil.disk_usage(scratch).free:
        raise OSError("insufficient free disk space for exact global score sample")
    fd, name = tempfile.mkstemp(prefix="stage1-score-", suffix=".f64", dir=scratch)
    os.close(fd)
    mapped: np.memmap | None = None
    try:
        mapped = np.memmap(name, mode="w+", dtype="<f8", shape=(needed // 8,))
        for offset, rows in source.user_batches(users):
            for start, stop, scores in source.item_blocks(rows):
                for local in range(rows.size):
                    row_start = (offset + local) * source.n_items
                    mapped[row_start + start : row_start + stop] = scores[local]
                del scores
        mapped.flush()
        digest = hashlib.sha256()
        hash_cells = max(1, source.maximum_score_block_bytes // 8)
        minimum = float("inf")
        for start in range(0, mapped.size, hash_cells):
            block = mapped[start : start + hash_cells]
            digest.update(memoryview(block).cast("B"))
            minimum = min(minimum, float(block.min()))
            del block
        quantiles = np.quantile(mapped, [0.05, 0.25, 0.5, 0.75, 0.95], method="linear", overwrite_input=True)
        parameters = dict(zip(("global_q05", "global_q25", "global_median", "global_q75", "global_q95"), map(float, quantiles)))
        parameters.update({"global_min": minimum, "sample_count": int(mapped.size)})
        return {
            "parameters": parameters,
            "score_sample_sha256": digest.hexdigest(),
            "score_sample_shape": [int(users.size), source.n_items],
            "resource_contract": {**source.resource_contract, "temporary_disk_bytes": needed, "quantiles": "exact_linear_in_place_memmap", "temporary_scores_retained": False},
        }
    finally:
        if mapped is not None:
            mapped._mmap.close()
        Path(name).unlink(missing_ok=True)


def iter_pseudo_utility_rows(
    source: FactorScoreSource,
    user_indices: Sequence[int] | np.ndarray,
    scenario_id: str,
    *,
    parameters: Mapping[str, Any] | None = None,
    maximum_row_workspace_bytes: int = DEFAULT_MAX_SCORE_BLOCK_BYTES,
) -> Iterator[tuple[int, np.ndarray]]:
    """Apply each frozen transformation with one full catalogue row in RAM.

    The conservative 128 bytes/item allowance covers row transformation
    arrays, sorting/inverse indices, and the score row; score block and factor
    conversion memory are additional. Consumers should process each yielded
    row immediately, rather than retaining all rows. Within-user transforms
    require the complete eligible catalogue, including already-owned items.
    """
    limit = _integer(maximum_row_workspace_bytes, "maximum_row_workspace_bytes", minimum=1)
    if source.n_items * 128 > limit:
        raise MemoryError("one pseudo-utility row workspace exceeds its byte budget")
    for user in _indices(user_indices, "user_indices", size=source.n_users):
        row = np.empty(source.n_items, dtype=np.float64)
        for start, stop, scores in source.item_blocks(np.asarray([user], dtype=np.int64)):
            row[start:stop] = scores[0]
            del scores
        yield int(user), transform(scenario_id, row, parameters)


def map_ids_sha256(values: Sequence[int] | np.ndarray) -> str:
    """Hash a unique ordered ID map using canonical little-endian int64."""
    ids = _indices(values, "IDs")
    if not ids.size or np.unique(ids).size != ids.size:
        raise ValueError("ID maps must be nonempty and unique")
    return hashlib.sha256(np.ascontiguousarray(ids, dtype="<i8").tobytes()).hexdigest()


def _validate_model_arrays(arrays: Mapping[str, Any], *, family: str, n_users: int, n_items: int) -> dict[str, np.ndarray]:
    if family == "implicit_als":
        required = {"user_factors", "item_factors"}
    elif family in {"feature_sum_bpr_identity", "feature_sum_bpr_identity_genre"}:
        required = {"user_factors", "identity_factors", "feature_factors", "item_bias"}
    elif family == "popularity":
        required = {"item_counts"}
    else:
        raise ValueError("unsupported model family")
    if set(arrays) != required:
        raise ValueError("parameter field names do not match the model family schema")
    result = {name: _finite_real_array(values, name, 1 if name in {"item_bias", "item_counts"} else 2) for name, values in arrays.items()}
    if family == "popularity":
        counts = result["item_counts"]
        if counts.shape != (n_items,) or counts.dtype.kind not in "iu" or np.any(counts < 0) or np.any(counts > n_users):
            raise ValueError("item_counts must be integer ownership counts bounded by n_users")
        return result
    factors = result["user_factors"].shape[1]
    if factors < 1 or result["user_factors"].shape[0] != n_users:
        raise ValueError("user factors do not match the user map or have no factors")
    item_name = "item_factors" if family == "implicit_als" else "identity_factors"
    if result[item_name].shape != (n_items, factors):
        raise ValueError("item factor shape does not match the item map and factor count")
    if any(value.dtype.kind != "f" or value.dtype.itemsize not in (4, 8) for value in result.values()):
        raise ValueError("factor parameters must use float32 or float64")
    if family != "implicit_als":
        if result["item_bias"].shape != (n_items,) or result["feature_factors"].shape[1] != factors:
            raise ValueError("BPR bias or feature-factor shape is invalid")
        if family == "feature_sum_bpr_identity" and result["feature_factors"].shape[0] != 0:
            raise ValueError("identity-only BPR must have no feature factors")
        if family == "feature_sum_bpr_identity_genre" and result["feature_factors"].shape[0] < 1:
            raise ValueError("genre BPR requires feature factors")
    return result


def save_bound_parameter_archive(
    path: str | Path,
    arrays: Mapping[str, np.ndarray],
    *,
    family: str,
    user_ids: Sequence[int] | np.ndarray,
    item_ids: Sequence[int] | np.ndarray,
) -> dict[str, Any]:
    """Atomically save strict numeric parameters together with ordered ID maps."""
    users, items = _indices(user_ids, "user_ids"), _indices(item_ids, "item_ids")
    user_hash, item_hash = map_ids_sha256(users), map_ids_sha256(items)
    checked = _validate_model_arrays(arrays, family=family, n_users=users.size, n_items=items.size)
    metadata = {"schema_version": 1, "family": family, "user_ids_sha256": user_hash, "item_ids_sha256": item_hash}
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, metadata=np.asarray(json.dumps(metadata, sort_keys=True)), user_ids=users, item_ids=items, **checked)
        os.replace(name, destination)
    finally:
        Path(name).unlink(missing_ok=True)
    return {**metadata, "path": destination.as_posix(), "sha256": _file_sha256(destination), "size_bytes": destination.stat().st_size}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_bound_parameter_archive(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_family: str,
    expected_user_ids: Sequence[int] | np.ndarray,
    expected_item_ids: Sequence[int] | np.ndarray,
    maximum_uncompressed_bytes: int = 2_147_483_648,
) -> dict[str, np.ndarray]:
    """Reject wrong maps, fields, dimensions, dtype, nonfinites, and file hashes.

    The trusted outer hash binds the embedded schema and map hashes to the
    artifact. The ZIP uncompressed-size gate precedes loading all parameters.
    Returned arrays exclude metadata/maps; expected ordered maps are mandatory.
    """
    source = Path(path)
    if not isinstance(expected_sha256, str) or _file_sha256(source) != expected_sha256:
        raise ValueError("parameter archive hash mismatch")
    limit = _integer(maximum_uncompressed_bytes, "maximum_uncompressed_bytes", minimum=1)
    with zipfile.ZipFile(source) as archive:
        members = archive.infolist()
        if len({member.filename for member in members}) != len(members):
            raise ValueError("parameter archive has duplicate fields")
        if sum(member.file_size for member in members) > limit:
            raise MemoryError("uncompressed parameter archive exceeds the byte budget")
        if any(not member.filename.endswith(".npy") or "/" in member.filename or "\\" in member.filename for member in members):
            raise ValueError("unexpected parameter archive member")
    with np.load(source, allow_pickle=False) as payload:
        if not {"metadata", "user_ids", "item_ids"}.issubset(payload.files):
            raise ValueError("parameter archive lacks its schema or ID maps")
        metadata_array = payload["metadata"]
        if metadata_array.shape != () or metadata_array.dtype.kind != "U":
            raise ValueError("parameter archive metadata must be a Unicode JSON scalar")
        metadata = json.loads(str(metadata_array.item()))
        if not isinstance(metadata, dict) or type(metadata.get("schema_version")) is not int:
            raise ValueError("parameter archive metadata has an invalid schema version")
        expected_metadata = {
            "schema_version": 1,
            "family": expected_family,
            "user_ids_sha256": map_ids_sha256(expected_user_ids),
            "item_ids_sha256": map_ids_sha256(expected_item_ids),
        }
        if metadata != expected_metadata:
            raise ValueError("parameter archive schema, family or ID-map binding mismatch")
        users, items = _indices(payload["user_ids"], "user_ids"), _indices(payload["item_ids"], "item_ids")
        if not np.array_equal(users, expected_user_ids) or not np.array_equal(items, expected_item_ids):
            raise ValueError("parameter archive ordered ID maps differ from expected maps")
        arrays = {name: payload[name] for name in payload.files if name not in {"metadata", "user_ids", "item_ids"}}
    return _validate_model_arrays(arrays, family=expected_family, n_users=users.size, n_items=items.size)


def validated_fold_in_triples(
    positive_items: Sequence[int] | np.ndarray,
    *,
    n_items: int,
    cycle_id: str,
    user_id: int,
) -> np.ndarray:
    """Validate all positive bounds before the frozen sampler's early returns."""
    from src.stage1_backend import construct_fold_in_triples

    count = _integer(n_items, "n_items", minimum=1)
    positives = _indices(positive_items, "positive_items", size=count)
    user = _integer(user_id, "user_id")
    if not isinstance(cycle_id, str) or not cycle_id:
        raise ValueError("cycle_id must be a nonempty string")
    return construct_fold_in_triples(positives, n_items=count, cycle_id=cycle_id, user_id=user)
