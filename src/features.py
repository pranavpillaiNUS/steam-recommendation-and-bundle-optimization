"""Deterministic sparse item features for the frozen Stage 1 contract.

The caller supplies the canonical item order.  This module validates that the
IDs are unique, unpadded nonnegative decimal Steam application IDs in ascending
numeric order and never derives a competing order from catalogue row order.

The required feature hierarchy is deliberately small:

* an identity CSR block with one unit feature per item; and
* a genre CSR block with equal weights over each item's observed genres.

Items without an observed genre have an all-zero genre row.  Identity and
genre are separate blocks so downstream code can add the genre block without
renormalizing or otherwise changing the identity coefficient.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import html
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence
import unicodedata

import numpy as np
import scipy.sparse as sp

from src.interactions import array_sha256, canonical_numeric_ids, id_map_sha256


SCHEMA_VERSION = 1
ITEM_ORDER_POLICY = "ascending_numeric_item_id"
ITEM_ID_ENCODING = "utf8_decimal_without_padding"
ITEM_ID_DTYPE = "int64"
GENRE_WEIGHTING = "l1_normalized_multi_hot"
MISSING_GENRE_POLICY = "zero_content_row"
TOKEN_NORMALIZATION = "html_unescape_then_unicode_nfkc_then_whitespace_collapse"

ITEM_IDS_FILENAME = "item_ids.npy"
IDENTITY_FILENAME = "item_features_identity.npz"
GENRE_FILENAME = "item_features_genre.npz"
IDENTITY_NAMES_FILENAME = "item_feature_names_identity.npy"
GENRE_NAMES_FILENAME = "item_feature_names_genre.npy"
MANIFEST_FILENAME = "item_feature_manifest.json"

_WHITESPACE = re.compile(r"\s+")


class FeatureAlignmentError(ValueError):
    """Raised when feature rows and the frozen model item map disagree."""


@dataclass(frozen=True)
class ItemFeatureArtifacts:
    """One aligned item map and its two atomic sparse feature blocks."""

    item_ids: np.ndarray
    identity: sp.csr_matrix
    genre: sp.csr_matrix
    identity_feature_names: np.ndarray
    genre_feature_names: np.ndarray


@dataclass(frozen=True)
class ItemFeatureView:
    """One model-facing row projection under the controlled genre toggle."""

    item_ids: np.ndarray
    matrix: sp.csr_matrix
    feature_names: np.ndarray
    include_genre: bool


def _fixed_unicode(values: Sequence[str]) -> np.ndarray:
    width = max((len(value) for value in values), default=1)
    return np.asarray(values, dtype=f"<U{width}")


def _canonical_item_id(value: Any) -> int:
    return int(canonical_numeric_ids([value], label="item")[0])


def canonical_item_ids(
    item_ids: Sequence[Any] | np.ndarray, *, require_sorted: bool = True
) -> np.ndarray:
    """Validate item IDs and return the frozen int64 representation.

    Input order is preserved.  Under the frozen contract it must already be
    strictly ascending numerically; sorting inside a feature builder would
    make accidental disagreement with the interaction matrix too easy.
    """

    if isinstance(item_ids, (str, bytes)):
        raise ValueError("item_ids must be a sequence, not a scalar string")
    values = canonical_numeric_ids(item_ids, label="item")
    if np.unique(values).size != values.size:
        raise ValueError("item IDs must be unique")
    if require_sorted and values.size > 1 and not np.all(
        values[1:] > values[:-1]
    ):
        raise ValueError("item IDs must be in strictly ascending numeric order")
    return values


def assert_exact_item_alignment(
    expected_item_ids: Sequence[Any] | np.ndarray,
    actual_item_ids: Sequence[Any] | np.ndarray,
) -> None:
    """Reject any item-row mismatch, including a pure permutation."""

    expected = canonical_item_ids(expected_item_ids, require_sorted=False)
    actual = canonical_item_ids(actual_item_ids, require_sorted=False)
    if expected.size != actual.size:
        raise FeatureAlignmentError(
            f"item-map length mismatch: expected {expected.size}, got {actual.size}"
        )
    mismatch = np.flatnonzero(expected != actual)
    if mismatch.size:
        row = int(mismatch[0])
        raise FeatureAlignmentError(
            "item-map order mismatch at row "
            f"{row}: expected {int(expected[row])}, got {int(actual[row])}"
        )


def normalize_feature_token(value: Any) -> str:
    """Conservatively normalize one label without semantic alias merging."""

    if not isinstance(value, (str, np.str_)):
        raise ValueError(f"feature tokens must be strings: {value!r}")
    token = html.unescape(str(value))
    token = unicodedata.normalize("NFKC", token)
    token = _WHITESPACE.sub(" ", token).strip()
    if not token:
        raise ValueError("feature tokens cannot be blank")
    return token


def normalize_genre_tokens(tokens: Iterable[Any] | None) -> tuple[str, ...]:
    """Return unique normalized genre labels in deterministic lexical order."""

    if tokens is None:
        return ()
    if isinstance(tokens, (str, np.str_)):
        tokens = (tokens,)
    normalized = {normalize_feature_token(value) for value in tokens}
    return tuple(sorted(normalized))


def aggregate_genre_records(
    records: Mapping[Any, Iterable[Any] | None]
    | Iterable[tuple[Any, Iterable[Any] | None]],
) -> dict[int, tuple[str, ...]]:
    """Union repeated item-genre records independently of record order."""

    iterator = records.items() if isinstance(records, Mapping) else records
    aggregated: dict[int, set[str]] = {}
    for raw_item_id, raw_tokens in iterator:
        item_id = _canonical_item_id(raw_item_id)
        aggregated.setdefault(item_id, set()).update(
            normalize_genre_tokens(raw_tokens)
        )
    return {
        item_id: tuple(sorted(aggregated[item_id]))
        for item_id in sorted(aggregated)
    }


def _canonical_vocabulary(
    vocabulary: Iterable[Any], *, require_sorted: bool
) -> tuple[str, ...]:
    values = tuple(normalize_feature_token(value) for value in vocabulary)
    if len(set(values)) != len(values):
        raise ValueError("genre vocabulary contains duplicate normalized labels")
    if require_sorted and values != tuple(sorted(values)):
        raise ValueError("genre vocabulary must be in deterministic lexical order")
    return values


def _canonical_csr(matrix: sp.spmatrix, *, dtype: np.dtype = np.float32) -> sp.csr_matrix:
    result = matrix.tocsr().astype(dtype, copy=False)
    result.sum_duplicates()
    result.eliminate_zeros()
    result.sort_indices()
    return result


def build_item_features(
    item_ids: Sequence[Any] | np.ndarray,
    genre_records: Mapping[Any, Iterable[Any] | None]
    | Iterable[tuple[Any, Iterable[Any] | None]],
    *,
    genre_vocabulary: Iterable[Any] | None = None,
) -> ItemFeatureArtifacts:
    """Build aligned identity and genre CSR blocks under the frozen contract.

    ``genre_records`` may omit items with no observed genre, but it may not
    contain an item outside the supplied universe.  When a frozen vocabulary
    is supplied it must already be sorted and every observed token must occur
    in it; labels are never silently pruned.
    """

    canonical_ids = canonical_item_ids(item_ids, require_sorted=True)
    genres_by_item = aggregate_genre_records(genre_records)
    item_set = set(canonical_ids.tolist())
    extra_items = sorted(set(genres_by_item) - item_set)
    if extra_items:
        preview = ", ".join(str(value) for value in extra_items[:3])
        raise FeatureAlignmentError(
            f"genre records contain {len(extra_items)} unknown item IDs: {preview}"
        )

    observed_labels = {
        token for tokens in genres_by_item.values() for token in tokens
    }
    if genre_vocabulary is None:
        vocabulary = tuple(sorted(observed_labels))
    else:
        vocabulary = _canonical_vocabulary(
            genre_vocabulary, require_sorted=True
        )
        unknown_labels = sorted(observed_labels - set(vocabulary))
        if unknown_labels:
            raise ValueError(
                "observed genre labels are absent from the frozen vocabulary: "
                + ", ".join(unknown_labels[:3])
            )

    genre_index = {token: column for column, token in enumerate(vocabulary)}
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for row, item_id in enumerate(canonical_ids.tolist()):
        tokens = genres_by_item.get(item_id, ())
        if not tokens:
            continue
        weight = 1.0 / len(tokens)
        rows.extend([row] * len(tokens))
        columns.extend(genre_index[token] for token in tokens)
        values.extend([weight] * len(tokens))

    n_items = canonical_ids.size
    identity = sp.identity(n_items, format="csr", dtype=np.float32)
    genre = sp.coo_matrix(
        (
            np.asarray(values, dtype=np.float32),
            (np.asarray(rows, dtype=np.int64), np.asarray(columns, dtype=np.int64)),
        ),
        shape=(n_items, len(vocabulary)),
        dtype=np.float32,
    )
    artifacts = ItemFeatureArtifacts(
        item_ids=canonical_ids,
        identity=_canonical_csr(identity),
        genre=_canonical_csr(genre),
        identity_feature_names=_fixed_unicode(
            [f"item::{int(item_id)}" for item_id in canonical_ids]
        ),
        genre_feature_names=_fixed_unicode(
            [f"genre::{token}" for token in vocabulary]
        ),
    )
    validate_feature_artifacts(artifacts, expected_item_ids=canonical_ids)
    return artifacts


def _validate_csr(name: str, matrix: sp.spmatrix) -> sp.csr_matrix:
    if not sp.isspmatrix_csr(matrix):
        raise ValueError(f"{name} must be a CSR matrix")
    if matrix.dtype != np.dtype(np.float32):
        raise ValueError(f"{name} must use float32 data")
    if not matrix.has_canonical_format or not matrix.has_sorted_indices:
        raise ValueError(f"{name} must have canonical sorted CSR indices")
    if not np.isfinite(matrix.data).all():
        raise ValueError(f"{name} contains nonfinite values")
    return matrix


def validate_feature_artifacts(
    artifacts: ItemFeatureArtifacts,
    *,
    expected_item_ids: Sequence[Any] | np.ndarray | None = None,
) -> None:
    """Validate shapes, sparse invariants, weighting, names, and alignment."""

    raw_item_ids = np.asarray(artifacts.item_ids)
    if raw_item_ids.ndim != 1 or raw_item_ids.dtype != np.dtype(np.int64):
        raise ValueError("stored item IDs must be a one-dimensional int64 array")
    item_ids = canonical_item_ids(raw_item_ids, require_sorted=True)
    if expected_item_ids is not None:
        assert_exact_item_alignment(expected_item_ids, item_ids)

    identity = _validate_csr("identity", artifacts.identity)
    genre = _validate_csr("genre", artifacts.genre)
    n_items = item_ids.size
    if identity.shape != (n_items, n_items):
        raise ValueError("identity block shape does not match the item map")
    if genre.shape[0] != n_items:
        raise ValueError("genre block row count does not match the item map")

    expected_indices = np.arange(n_items, dtype=identity.indices.dtype)
    expected_indptr = np.arange(n_items + 1, dtype=identity.indptr.dtype)
    if (
        identity.nnz != n_items
        or not np.array_equal(identity.indices, expected_indices)
        or not np.array_equal(identity.indptr, expected_indptr)
        or not np.array_equal(identity.data, np.ones(n_items, dtype=np.float32))
    ):
        raise ValueError("identity block is not exact sparse identity")

    row_mass = np.asarray(genre.sum(axis=1)).ravel()
    if np.any(genre.data <= 0.0):
        raise ValueError("genre weights must be strictly positive where observed")
    if not np.all(np.isclose(row_mass, 0.0) | np.isclose(row_mass, 1.0)):
        raise ValueError("genre rows must have L1 mass zero or one")
    row_counts = np.diff(genre.indptr)
    nonzero_rows = row_counts > 0
    expected_weights = np.repeat(
        1.0 / row_counts[nonzero_rows],
        row_counts[nonzero_rows],
    ).astype(np.float32)
    if not np.allclose(
        genre.data,
        expected_weights,
        rtol=1e-6,
        atol=1e-7,
    ):
        raise ValueError("genre rows must weight every label exactly equally")

    identity_names = np.asarray(artifacts.identity_feature_names)
    genre_names = np.asarray(artifacts.genre_feature_names)
    if identity_names.dtype.kind != "U" or genre_names.dtype.kind != "U":
        raise ValueError("feature-name arrays must use non-object Unicode dtype")
    if identity_names.shape != (n_items,):
        raise ValueError("identity feature-name count does not match identity columns")
    if genre_names.shape != (genre.shape[1],):
        raise ValueError("genre feature-name count does not match genre columns")
    expected_identity_names = _fixed_unicode(
        [f"item::{int(item_id)}" for item_id in item_ids]
    )
    if not np.array_equal(identity_names, expected_identity_names):
        raise ValueError("identity feature names do not match the item map")
    if len(set(genre_names.tolist())) != genre_names.size:
        raise ValueError("genre feature names must be unique")
    if any(not name.startswith("genre::") for name in genre_names.tolist()):
        raise ValueError("genre feature names must use the genre:: namespace")
    raw_genre_names = tuple(name.removeprefix("genre::") for name in genre_names)
    if raw_genre_names != tuple(sorted(raw_genre_names)):
        raise ValueError("genre feature names must be in lexical order")


def model_feature_view(
    artifacts: ItemFeatureArtifacts,
    *,
    include_genre: bool,
    requested_item_ids: Sequence[Any] | np.ndarray | None = None,
) -> ItemFeatureView:
    """Return aligned model rows with genre as the sole feature toggle.

    Identity columns always remain first, have unit weight, and are never
    renormalized. When ``requested_item_ids`` is supplied, only rows are
    projected; the frozen full-catalogue feature columns remain unchanged.
    """

    if not isinstance(include_genre, (bool, np.bool_)):
        raise ValueError("include_genre must be boolean")
    validate_feature_artifacts(
        artifacts,
        expected_item_ids=artifacts.item_ids,
    )
    if requested_item_ids is None:
        selected_ids = artifacts.item_ids.copy()
        positions = np.arange(selected_ids.size, dtype=np.int64)
    else:
        selected_ids = canonical_item_ids(
            requested_item_ids,
            require_sorted=True,
        )
        positions = np.searchsorted(artifacts.item_ids, selected_ids)
        if (
            np.any(positions >= artifacts.item_ids.size)
            or not np.array_equal(
                artifacts.item_ids[positions],
                selected_ids,
            )
        ):
            raise FeatureAlignmentError(
                "requested item rows are absent from the feature item map"
            )

    identity_rows = artifacts.identity[positions, :].tocsr()
    if include_genre:
        matrix = sp.hstack(
            (identity_rows, artifacts.genre[positions, :]),
            format="csr",
            dtype=np.float32,
        )
        names = np.concatenate(
            (
                artifacts.identity_feature_names,
                artifacts.genre_feature_names,
            )
        )
    else:
        matrix = identity_rows
        names = artifacts.identity_feature_names.copy()
    matrix = _canonical_csr(matrix)
    return ItemFeatureView(
        item_ids=selected_ids,
        matrix=matrix,
        feature_names=names,
        include_genre=bool(include_genre),
    )


def canonical_json_bytes(value: Any) -> bytes:
    """Return the canonical JSON representation used for semantic hashes."""

    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def semantic_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path, *, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def string_array_sha256(values: Sequence[Any] | np.ndarray) -> str:
    strings = [str(value) for value in np.asarray(values).tolist()]
    return semantic_sha256(strings)


def csr_semantic_sha256(matrix: sp.spmatrix) -> str:
    """Hash CSR semantics independently of platform index width."""

    canonical = _canonical_csr(matrix, dtype=np.float32)
    digest = hashlib.sha256()
    digest.update(
        canonical_json_bytes(
            {
                "format": "csr",
                "shape": list(canonical.shape),
                "data_dtype": "float32-little-endian",
                "index_dtype": "int64-little-endian",
            }
        )
    )
    digest.update(canonical.indptr.astype("<i8", copy=False).tobytes(order="C"))
    digest.update(canonical.indices.astype("<i8", copy=False).tobytes(order="C"))
    digest.update(canonical.data.astype("<f4", copy=False).tobytes(order="C"))
    return digest.hexdigest()


def _feature_semantics(artifacts: ItemFeatureArtifacts) -> dict[str, Any]:
    row_mass = np.asarray(artifacts.genre.sum(axis=1)).ravel()
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": {
            "item_order": ITEM_ORDER_POLICY,
            "item_id_encoding": ITEM_ID_ENCODING,
            "item_id_dtype": ITEM_ID_DTYPE,
            "token_normalization": TOKEN_NORMALIZATION,
            "genre_weighting": GENRE_WEIGHTING,
            "missing_genre_policy": MISSING_GENRE_POLICY,
            "identity_weight": 1.0,
            "genre_block_weight": 1.0,
            "combined_row_normalization": False,
        },
        "item_map": {
            "count": int(artifacts.item_ids.size),
            "array_sha256": array_sha256(artifacts.item_ids),
            "semantic_sha256": id_map_sha256(
                artifacts.item_ids,
                label="item",
            ),
        },
        "blocks": {
            "identity": {
                "shape": list(artifacts.identity.shape),
                "nnz": int(artifacts.identity.nnz),
                "dtype": str(artifacts.identity.dtype),
                "semantic_sha256": csr_semantic_sha256(artifacts.identity),
                "feature_names_sha256": string_array_sha256(
                    artifacts.identity_feature_names
                ),
            },
            "genre": {
                "shape": list(artifacts.genre.shape),
                "nnz": int(artifacts.genre.nnz),
                "dtype": str(artifacts.genre.dtype),
                "semantic_sha256": csr_semantic_sha256(artifacts.genre),
                "feature_names_sha256": string_array_sha256(
                    artifacts.genre_feature_names
                ),
                "covered_items": int(np.count_nonzero(row_mass)),
                "zero_content_items": int(np.count_nonzero(row_mass == 0.0)),
            },
        },
    }


def build_feature_manifest(
    artifacts: ItemFeatureArtifacts,
    *,
    input_files: Mapping[str, str | Path] | None = None,
    path_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build a deterministic content manifest before physical file metadata."""

    validate_feature_artifacts(artifacts, expected_item_ids=artifacts.item_ids)
    semantics = _feature_semantics(artifacts)
    root = Path(path_root).resolve() if path_root is not None else None
    inputs: dict[str, dict[str, Any]] = {}
    for label, raw_path in sorted((input_files or {}).items()):
        path = Path(raw_path).resolve()
        if root is None:
            recorded_path = path.as_posix()
        else:
            try:
                recorded_path = path.relative_to(root).as_posix()
            except ValueError as exc:
                raise ValueError(
                    f"feature input is outside path_root: {label}"
                ) from exc
        inputs[str(label)] = {
            "path": recorded_path,
            "size_bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
    content_identity = {
        "feature_semantics": semantics,
        "input_sha256": {
            label: entry["sha256"] for label, entry in inputs.items()
        },
    }
    return {
        **semantics,
        "inputs": inputs,
        "feature_set_id": semantic_sha256(content_identity),
    }


def save_feature_artifacts(
    artifacts: ItemFeatureArtifacts,
    output_dir: str | Path,
    *,
    input_files: Mapping[str, str | Path] | None = None,
    path_root: str | Path | None = None,
) -> dict[str, Any]:
    """Save validated artifacts and a manifest with semantic and file hashes."""

    validate_feature_artifacts(artifacts, expected_item_ids=artifacts.item_ids)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "item_ids": output / ITEM_IDS_FILENAME,
        "identity": output / IDENTITY_FILENAME,
        "genre": output / GENRE_FILENAME,
        "identity_feature_names": output / IDENTITY_NAMES_FILENAME,
        "genre_feature_names": output / GENRE_NAMES_FILENAME,
    }
    np.save(paths["item_ids"], artifacts.item_ids, allow_pickle=False)
    sp.save_npz(paths["identity"], artifacts.identity, compressed=True)
    sp.save_npz(paths["genre"], artifacts.genre, compressed=True)
    np.save(
        paths["identity_feature_names"],
        artifacts.identity_feature_names,
        allow_pickle=False,
    )
    np.save(
        paths["genre_feature_names"],
        artifacts.genre_feature_names,
        allow_pickle=False,
    )

    manifest = build_feature_manifest(
        artifacts,
        input_files=input_files,
        path_root=path_root,
    )
    manifest["artifacts"] = {
        label: {
            "path": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for label, path in sorted(paths.items())
    }
    manifest["manifest_id"] = semantic_sha256(manifest)
    manifest_path = output / MANIFEST_FILENAME
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def load_feature_artifacts(
    output_dir: str | Path,
    *,
    expected_item_ids: Sequence[Any] | np.ndarray | None = None,
) -> ItemFeatureArtifacts:
    """Load artifacts, verify every recorded hash, and enforce row alignment."""

    output = Path(output_dir)
    manifest_path = output / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported item-feature manifest schema")
    manifest_id = manifest.get("manifest_id")
    unsigned_manifest = dict(manifest)
    unsigned_manifest.pop("manifest_id", None)
    if manifest_id != semantic_sha256(unsigned_manifest):
        raise ValueError("item-feature manifest semantic hash mismatch")

    expected_names = {
        "item_ids": ITEM_IDS_FILENAME,
        "identity": IDENTITY_FILENAME,
        "genre": GENRE_FILENAME,
        "identity_feature_names": IDENTITY_NAMES_FILENAME,
        "genre_feature_names": GENRE_NAMES_FILENAME,
    }
    if set(manifest.get("artifacts", {})) != set(expected_names):
        raise ValueError("item-feature artifact inventory changed")
    for label, filename in expected_names.items():
        entry = manifest.get("artifacts", {}).get(label)
        if not isinstance(entry, dict) or entry.get("path") != filename:
            raise ValueError(f"missing or unexpected artifact entry: {label}")
        path = output / filename
        if path.stat().st_size != entry.get("size_bytes"):
            raise ValueError(f"artifact size mismatch: {filename}")
        if file_sha256(path) != entry.get("sha256"):
            raise ValueError(f"artifact hash mismatch: {filename}")

    artifacts = ItemFeatureArtifacts(
        item_ids=np.load(output / ITEM_IDS_FILENAME, allow_pickle=False),
        identity=sp.load_npz(output / IDENTITY_FILENAME),
        genre=sp.load_npz(output / GENRE_FILENAME),
        identity_feature_names=np.load(
            output / IDENTITY_NAMES_FILENAME, allow_pickle=False
        ),
        genre_feature_names=np.load(
            output / GENRE_NAMES_FILENAME, allow_pickle=False
        ),
    )
    validate_feature_artifacts(artifacts, expected_item_ids=expected_item_ids)
    semantics = _feature_semantics(artifacts)
    for key in ("contract", "item_map", "blocks"):
        if manifest.get(key) != semantics[key]:
            raise ValueError(f"artifact semantics disagree with manifest field: {key}")
    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict) or any(
        not isinstance(entry, dict) or not isinstance(entry.get("sha256"), str)
        for entry in inputs.values()
    ):
        raise ValueError("item-feature manifest input inventory is invalid")
    expected_feature_set_id = semantic_sha256(
        {
            "feature_semantics": semantics,
            "input_sha256": {
                label: entry["sha256"]
                for label, entry in sorted(inputs.items())
            },
        }
    )
    if manifest.get("feature_set_id") != expected_feature_set_id:
        raise ValueError("item-feature set semantic hash mismatch")
    return artifacts
