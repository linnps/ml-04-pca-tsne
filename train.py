"""
Project the synthetic high-D Swiss roll into 2-D with PCA and t-SNE
(at three perplexities), then render the dashboard figures.

Palette (shared across the portfolio):
    background  : white
    grid / axes : light gray (#E5E5E5)
    primary     : muted blue (#3B6EA8)
    accent      : muted red  (#C04040)
    neutral     : medium gray (#7A7A7A)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness

from generate_data import DataConfig, generate

# ---------------------------------------------------------------- style ----
COLOR_BG = "#FFFFFF"
COLOR_GRID = "#E5E5E5"
COLOR_TEXT = "#333333"
COLOR_BLUE = "#3B6EA8"
COLOR_RED = "#C04040"
COLOR_GRAY = "#7A7A7A"
COLOR_LIGHT_GRAY = "#CCCCCC"

# Position-along-roll colormap: red → gray → blue (matches portfolio palette).
CMAP_T = LinearSegmentedColormap.from_list(
    "t_colormap", [COLOR_RED, "#D88080", "#BFBFBF", "#9EB7D6", COLOR_BLUE], N=256,
)

mpl.rcParams.update({
    "figure.facecolor": COLOR_BG,
    "axes.facecolor": COLOR_BG,
    "axes.edgecolor": COLOR_LIGHT_GRAY,
    "axes.labelcolor": COLOR_TEXT,
    "axes.titlecolor": COLOR_TEXT,
    "axes.titleweight": "bold",
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": COLOR_TEXT,
    "ytick.color": COLOR_TEXT,
    "grid.color": COLOR_GRID,
    "grid.linewidth": 0.6,
    "axes.grid": True,
    "legend.frameon": False,
    "font.family": "sans-serif",
    "font.size": 11,
})


# ---------------------------------------------------------------- figures --
def fig_swiss_3d(swiss_3d: np.ndarray, t: np.ndarray, out_path: Path) -> None:
    fig = plt.figure(figsize=(7, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(swiss_3d[:, 0], swiss_3d[:, 1], swiss_3d[:, 2],
               c=t, cmap=CMAP_T, s=12, alpha=0.85,
               edgecolor="white", linewidth=0.2)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title("Original 2-D manifold embedded in 3-D space\n"
                 "(color = position along the roll — the latent we want to recover)")
    ax.grid(False)
    ax.xaxis.pane.set_edgecolor(COLOR_LIGHT_GRAY)
    ax.yaxis.pane.set_edgecolor(COLOR_LIGHT_GRAY)
    ax.zaxis.pane.set_edgecolor(COLOR_LIGHT_GRAY)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_projections(
    X_pca: np.ndarray,
    X_tsne_dict: dict[int, np.ndarray],
    t: np.ndarray,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(15, 4), constrained_layout=True)

    panels = [("PCA (linear)", X_pca, "PCA 1", "PCA 2")]
    for perp in sorted(X_tsne_dict):
        panels.append((f"t-SNE (perplexity={perp})", X_tsne_dict[perp],
                       "t-SNE 1", "t-SNE 2"))

    for ax, (name, Z, xlab, ylab) in zip(axes, panels):
        sc = ax.scatter(Z[:, 0], Z[:, 1], c=t, cmap=CMAP_T,
                        s=10, alpha=0.85, edgecolor="white", linewidth=0.2)
        ax.set_title(name)
        ax.set_xlabel(xlab); ax.set_ylabel(ylab)

    cbar = fig.colorbar(sc, ax=axes, shrink=0.85, fraction=0.025, pad=0.02)
    cbar.set_label("position along the roll  →", color=COLOR_TEXT)
    fig.suptitle("PCA cannot unroll. t-SNE can — at the right perplexity.",
                 fontsize=14, fontweight="bold", color=COLOR_TEXT, y=1.05)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_pca_scree(explained_ratio: np.ndarray, out_path: Path) -> None:
    cum = np.cumsum(explained_ratio)
    n = len(explained_ratio)
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    ax.bar(range(1, n + 1), explained_ratio,
           color=COLOR_BLUE, edgecolor=COLOR_LIGHT_GRAY, linewidth=0.6,
           label="per-component")
    ax2 = ax.twinx()
    ax2.plot(range(1, n + 1), cum, color=COLOR_RED, marker="o",
             linewidth=1.6, label="cumulative")
    ax2.set_ylim(0, 1.05)
    ax2.grid(False)

    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance (per-component)")
    ax2.set_ylabel("Cumulative explained variance", color=COLOR_RED)
    ax.set_title("PCA scree plot — informative dims dominate, noise dims are flat")
    ax.set_xticks(range(1, n + 1))

    # Legend combining both axes.
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center right")

    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_trustworthiness(
    X_high: np.ndarray,
    X_pca: np.ndarray,
    X_tsne_dict: dict[int, np.ndarray],
    out_path: Path,
) -> None:
    """
    Trustworthiness ∈ [0, 1] measures whether neighborhood structure in
    the high-D space is preserved in the projection. 1.0 means perfect
    local fidelity; 0.5 is random.
    """
    ks = [5, 10, 30, 60]

    methods = [("PCA", X_pca, COLOR_BLUE)]
    for perp in sorted(X_tsne_dict):
        methods.append((f"t-SNE (perp={perp})", X_tsne_dict[perp], None))

    palette_extra = [COLOR_GRAY, COLOR_RED, "#5A8FCC"]
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)

    for i, (name, Z, color) in enumerate(methods):
        c = color if color is not None else palette_extra[i - 1]
        scores = [trustworthiness(X_high, Z, n_neighbors=k) for k in ks]
        ax.plot(ks, scores, marker="o", linewidth=1.8, color=c, label=name)

    ax.set_xlabel("Neighborhood size k")
    ax.set_ylabel("Trustworthiness")
    ax.set_ylim(0.5, 1.02)
    ax.set_title("How well does each projection preserve local neighborhoods?")
    ax.legend(loc="lower left")
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------- main ----
def main() -> None:
    cfg = DataConfig()
    X, swiss_3d, t = generate(cfg)

    pca = PCA(n_components=2).fit(X)
    X_pca = pca.transform(X)
    explained_full = PCA().fit(X).explained_variance_ratio_

    perplexities = [5, 30, 80]
    X_tsne_dict = {}
    for perp in perplexities:
        tsne = TSNE(n_components=2, perplexity=perp, init="pca",
                    learning_rate="auto", random_state=42)
        X_tsne_dict[perp] = tsne.fit_transform(X)

    # Trustworthiness summary @ k=10.
    tw_pca_10 = float(trustworthiness(X, X_pca, n_neighbors=10))
    tw_tsne_10 = {p: float(trustworthiness(X, Z, n_neighbors=10))
                  for p, Z in X_tsne_dict.items()}

    print(f"\nDataset: n={len(X)} points, dim={X.shape[1]} "
          f"(3 informative + {cfg.n_noise_dims} noise)")
    print(f"\nPCA explained variance per component:")
    for i, v in enumerate(explained_full[:5], 1):
        print(f"  PC{i}: {v:.3f}")

    print(f"\nTrustworthiness @ k=10 (higher = better local fidelity):")
    print(f"  PCA            {tw_pca_10:.3f}")
    for p, s in tw_tsne_10.items():
        print(f"  t-SNE perp={p:>3}  {s:.3f}")

    Path("results").mkdir(exist_ok=True)
    summary = {
        "config": cfg.__dict__,
        "pca_explained_variance_ratio": explained_full.tolist(),
        "trustworthiness_at_k10": {
            "PCA": tw_pca_10,
            **{f"tSNE_perp_{p}": s for p, s in tw_tsne_10.items()},
        },
    }
    with open("results/metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    assets = Path("assets"); assets.mkdir(exist_ok=True)
    fig_swiss_3d(swiss_3d, t, assets / "01_swiss_3d.png")
    fig_projections(X_pca, X_tsne_dict, t, assets / "02_projections.png")
    fig_pca_scree(explained_full, assets / "03_scree.png")
    fig_trustworthiness(X, X_pca, X_tsne_dict, assets / "04_trustworthiness.png")

    print(f"\nFigures saved to: {assets.resolve()}")


if __name__ == "__main__":
    main()
