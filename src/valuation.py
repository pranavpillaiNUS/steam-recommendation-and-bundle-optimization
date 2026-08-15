"""Stage 1 preference engine: latent-factor scores from Steam ownership.

This legacy foundation fits SVD / NMF (and optionally Hu-Koren-Volinsky
implicit ALS) to the user-by-game ownership matrix and returns latent factors
and *preference scores*. They are ownership-derived ranking scores, not
utilities, willingness to pay, purchase probabilities, or monetary values.
The live Stage 2 interface will apply separately declared pseudo-utility
scenarios; it will not economically calibrate these scores.

This file predates the frozen Stage 1 protocol. Its SVD/NMF paths are retained
for provenance and do not satisfy the required popularity/ALS/identity-only/
identity-plus-genre ladder by themselves.

Matrix conventions (kept distinct so nothing is mislabeled "HKV"):
- ``preference_matrix``: binary CSR, p_ui = 1 if user u owns game i.
- ``weighted_interaction_matrix``: exploratory, r_ui = log(1 + playtime) at
  observed entries (the log tames the heavy playtime tail).
- ``observed_confidence``: c_ui = 1 + alpha * r_ui, stored ONLY at observed
  entries. The baseline confidence of one on every unobserved pair is implicit
  and is never materialized (that would densify the matrix).
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp


@dataclass
class InteractionData:
    """Sparse ownership/playtime matrices plus id<->index maps."""
    preference_matrix: sp.csr_matrix          # binary p_ui
    weighted_interaction_matrix: sp.csr_matrix  # r_ui = log(1 + playtime)
    observed_confidence: sp.csr_matrix        # c_ui = 1 + alpha * r_ui (observed only)
    user_ids: np.ndarray                      # row index -> user_id
    item_ids: np.ndarray                      # col index -> item_id
    user_index: dict                          # user_id -> row index
    item_index: dict                          # item_id -> col index
    alpha: float


def build_user_item_matrices(user_items_df, alpha=40.0,
                             user_col="user_id", 
                             item_col="item_id",
                             playtime_col="playtime_forever"):
    users = user_items_df[user_col].to_numpy()
    items = user_items_df[item_col].to_numpy()
    playtime = user_items_df[playtime_col].to_numpy(dtype=float)

    user_ids, user_inv = np.unique(users, return_inverse=True)
    item_ids, item_inv = np.unique(items, return_inverse=True)
    n_users, n_items = user_ids.size, item_ids.size
    shape = (n_users, n_items)
    r = np.log1p(np.clip(playtime, 0.0, None))

    preference = sp.csr_matrix((np.ones(user_inv.size, dtype=float), (user_inv, item_inv)), shape=shape)
    preference.data = np.minimum(preference.data, 1.0)

    weighted = sp.csr_matrix((r, (user_inv, item_inv)), shape=shape)
    confidence = sp.csr_matrix((1.0 + alpha * r, (user_inv, item_inv)), shape=shape)

    return InteractionData(
        preference_matrix=preference,
        weighted_interaction_matrix=weighted,
        observed_confidence=confidence,
        user_ids=user_ids,
        item_ids=item_ids,
        user_index={uid: i for i, uid in enumerate(user_ids)},
        item_index={iid: j for j, iid in enumerate(item_ids)},
        alpha=float(alpha),
    )


# Latent-factor models on the binary ownership matrix
def fit_svd(M, k, random_state=0):
    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=k, random_state=random_state)
    P = svd.fit_transform(M)
    Q = svd.components_.T
    return P, Q


def fit_nmf(M, k, random_state=0, max_iter=300):
    from sklearn.decomposition import NMF
    nmf = NMF(n_components=k, 
              init="nndsvda", 
              random_state=random_state,
              max_iter=max_iter)
    P = nmf.fit_transform(M)
    Q = nmf.components_.T
    return P, Q


def fit_als(user_confidence, factors=64, regularization=0.01, iterations=15, random_state=0):
    try:
        from implicit.als import AlternatingLeastSquares
    except ImportError as exc:
        raise ImportError(
            "fit_als needs the optional 'implicit' package (pip install implicit)"
        ) from exc

    model = AlternatingLeastSquares(
        factors=factors, 
        regularization=regularization, 
        iterations=iterations,
        random_state=random_state,
    )
    model.fit(sp.csr_matrix(user_confidence), show_progress=False)
    return np.asarray(model.user_factors), np.asarray(model.item_factors)


# Scoring API: explicitly bounded, no full-matrix default path
def score_pairs(P, Q, user_indices, item_indices):
    user_indices = np.asarray(user_indices)
    item_indices = np.asarray(item_indices)
    if user_indices.shape != item_indices.shape:
        raise ValueError("score_pairs needs equal-length user and item indices")
    return np.einsum("ij,ij->i", P[user_indices], Q[item_indices])


def score_user_items(P, Q, user_indices, item_indices):
    user_indices = np.asarray(user_indices)
    item_indices = np.asarray(item_indices)
    return P[user_indices] @ Q[item_indices].T


def iter_score_batches(P, Q, user_indices, item_indices, batch_size=512):
    user_indices = np.asarray(user_indices)
    item_indices = np.asarray(item_indices)
    Qsub = Q[item_indices]
    for start in range(0, user_indices.size, batch_size):
        batch = user_indices[start:start + batch_size]
        yield batch, P[batch] @ Qsub.T
