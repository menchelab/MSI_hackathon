"""
Simplex embedding + random projection visualisation of diffusion profiles.

Each MSI node is placed in the (K-1)-simplex whose vertices are the K input
profiles, via barycentric coordinates:

    w_k(v) = P_k(v) / sum_j P_j(v)

A node near vertex k is highly specific to profile k; a node near the
centroid visits all profiles equally (non-specific). The centroid is
subtracted so non-specific nodes cluster at the origin.

The simplex lives in K-1 dimensions. For K > target_dim, random orthogonal
projections are used to produce 2-D or 3-D views. Multiple projections
reveal the robustness of any visible clustering.

Node size encodes relevance (from node_relevance_and_specificity).

Usage
-----
    from visualize import simplex_visualize
    fig = simplex_visualize(profile_dict, relevance, n_projections=4)
    fig.savefig("simplex.png", dpi=150, bbox_inches="tight")
"""

import numpy as np
import matplotlib.pyplot as plt


# ── helpers ───────────────────────────────────────────────────────────────────

def _barycentric(profile_dict):
    """
    w_k(v) = P_k(v) / sum_j P_j(v)
    Returns W (n_nodes, K), labels list.
    """
    labels = list(profile_dict.keys())
    P = np.stack([profile_dict[l] for l in labels], axis=1)   # (N, K)
    totals = P.sum(axis=1, keepdims=True)
    W = P / np.maximum(totals, np.finfo(float).tiny)
    return W, labels


def _random_ortho(K, d, rng):
    """(K, d) matrix with orthonormal columns via QR decomposition.
    d is silently capped at K (simplex has at most K-1 free dimensions)."""
    d = min(d, K)
    Q, _ = np.linalg.qr(rng.standard_normal((K, d)))
    return Q[:, :d]


def _size_array(rel, lo=5, hi=200):
    """Map relevance scores to marker areas in [lo, hi] pt²."""
    upper = np.percentile(rel[rel > 0], 99) if (rel > 0).any() else 1.0
    clipped = np.clip(rel, 0, upper)
    norm = clipped / (upper + np.finfo(float).tiny)
    return lo + (hi - lo) * norm


# ── main function ─────────────────────────────────────────────────────────────

def simplex_visualize(
    profile_dict,
    relevance,
    n_projections=4,
    target_dim=2,
    min_relevance_percentile=75,
    seed=None,
    figsize=None,
    save_path=None,
):
    """
    Random-projection views of the profile simplex, sized by relevance.

    Parameters
    ----------
    profile_dict : dict {label: np.ndarray (n_nodes,)}
    relevance    : pd.Series — output A of node_relevance_and_specificity
    n_projections: how many independent random projections to draw
    target_dim   : 2 (default) or 3
    min_relevance_percentile : drop nodes below this relevance percentile
    seed         : int for reproducible projections (None = random every run)
    figsize      : (w, h) in inches; auto-sized if None
    save_path    : optional file path to save the figure

    Returns
    -------
    matplotlib Figure
    """
    assert target_dim in (2, 3), "target_dim must be 2 or 3"

    # ── embed ────────────────────────────────────────────────────────────────
    W, labels = _barycentric(profile_dict)
    K = len(labels)

    rel = relevance.values
    threshold = np.percentile(rel, min_relevance_percentile)
    mask = rel >= threshold

    W_filt = W[mask]
    rel_filt = rel[mask]

    # Centre: subtract uniform point (1/K, …, 1/K) so non-specific → origin
    W_c = W_filt - 1.0 / K

    # ── colour & size ────────────────────────────────────────────────────────
    dominant = np.argmax(W_filt, axis=1)            # index of "owning" profile
    palette = plt.cm.tab10.colors                    # 10 distinct colours
    node_colors = [palette[d % 10] for d in dominant]
    sizes = _size_array(rel_filt)

    # ── layout ───────────────────────────────────────────────────────────────
    ncols = min(n_projections, 4)
    nrows = (n_projections + ncols - 1) // ncols
    rng = np.random.default_rng(seed)

    if target_dim == 2:
        if figsize is None:
            figsize = (5 * ncols, 4.5 * nrows + 1.0)
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

        for i, ax in enumerate(axes.flat):
            if i >= n_projections:
                ax.set_visible(False)
                continue
            proj = W_c @ _random_ortho(K, 2, rng)
            ax.scatter(proj[:, 0], proj[:, 1],
                       c=node_colors, s=sizes, alpha=0.55, linewidths=0)
            ax.set_title(f"Projection {i + 1}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

    else:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        effective_dim = min(3, K)  # simplex is at most (K-1)-dimensional
        if effective_dim < 3:
            print(f"Note: K={K} profiles → simplex is {K-1}-D; "
                  f"projecting to {effective_dim}-D instead of 3-D.")
        if figsize is None:
            figsize = (5 * ncols, 5 * nrows + 1.0)
        fig = plt.figure(figsize=figsize)

        for i in range(n_projections):
            ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
            proj = W_c @ _random_ortho(K, effective_dim, rng)
            # Pad to 3 columns if needed so the 3-D axes get real coordinates
            if proj.shape[1] < 3:
                proj = np.hstack([proj, np.zeros((proj.shape[0], 3 - proj.shape[1]))])
            ax.scatter(proj[:, 0], proj[:, 1], proj[:, 2],
                       c=node_colors, s=sizes, alpha=0.5, linewidths=0)
            ax.set_title(f"Projection {i + 1}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

    # ── legend ───────────────────────────────────────────────────────────────
    profile_handles = [
        plt.scatter([], [], c=[palette[k % 10]], s=50, linewidths=0,
                    label=str(labels[k]))
        for k in range(K)
    ]

    upper = np.percentile(rel_filt[rel_filt > 0], 99)
    size_handles = [
        plt.scatter([], [], c=["#777777"], linewidths=0, alpha=0.7,
                    s=_size_array(np.array([v]), hi=200)[0],
                    label=f"rel p{p}")
        for v, p in zip(
            np.percentile(rel_filt[rel_filt > 0], [25, 75, 99]),
            [25, 75, 99],
        )
    ]

    fig.legend(
        handles=profile_handles + size_handles,
        loc="lower center",
        ncol=K + 3,
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, 0),
        columnspacing=1.0,
    )
    fig.suptitle(
        f"Simplex projection  ·  top {100 - min_relevance_percentile}% nodes "
        f"by relevance  (n = {mask.sum():,})",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.96])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")

    return fig
