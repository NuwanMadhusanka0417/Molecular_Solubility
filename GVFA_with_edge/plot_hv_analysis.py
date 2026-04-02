"""
Publication-quality hypervector distribution analysis from exported .npz batches.
"""
from __future__ import annotations

import glob
import math
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Configuration: training hypervector export
# - Set to a directory: loads every train_batch_*.npz inside it (full training set).
# - Or a single .npz file path (one batch only).
# - Or a glob pattern string, e.g. "my_hv_dump/train_batch_*.npz"
# ---------------------------------------------------------------------------
NPZ_SOURCE = "my_hv_dump/5000"

OUTPUT_DIR = Path("figures")
DPI = 300
RNG_SEED = 42
EPS = 1e-10


def _cosine_rows(H_a: np.ndarray, H_b: np.ndarray) -> np.ndarray:
    """Per-row cosine similarity between matching rows of H_a and H_b. Shape (N,)."""
    dotp = (H_a * H_b).sum(axis=1)
    na = np.linalg.norm(H_a, axis=1)
    nb = np.linalg.norm(H_b, axis=1)
    return dotp / (na * nb + EPS)


def _cosine_pair(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + EPS))


def discover_layer_indices(keys: list[str]) -> list[int]:
    found = set()
    for k in keys:
        m = re.match(r"layer_(\d+)_pre_bin", k)
        if m:
            found.add(int(m.group(1)))
    return sorted(found)


def sample_unique_pairs(n_nodes: int, n_pairs: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    max_pairs = n_nodes * (n_nodes - 1) // 2
    if max_pairs < 1:
        return []
    target = min(n_pairs, max_pairs)
    pairs: set[tuple[int, int]] = set()
    guard = 0
    max_iter = target * 500
    while len(pairs) < target and guard < max_iter:
        i, j = int(rng.integers(0, n_nodes)), int(rng.integers(0, n_nodes))
        guard += 1
        if i == j:
            continue
        if i > j:
            i, j = j, i
        pairs.add((i, j))
    if len(pairs) < target:
        raise RuntimeError(f"Could not sample {target} unique pairs (n_nodes={n_nodes})")
    return list(pairs)


def resolve_npz_files(source: str | Path) -> list[Path]:
    """Return sorted list of .npz paths: one file, a directory of train_batch_*.npz, or a glob."""
    p = Path(source)
    if p.is_file() and p.suffix.lower() == ".npz":
        return [p.resolve()]
    if p.is_dir():
        files = sorted(p.glob("train_batch_*.npz"))
        if not files:
            files = sorted(p.glob("*.npz"))
        return files
    matches = sorted(Path(x).resolve() for x in glob.glob(str(source)))
    return matches


def merge_training_npz_batches(files: list[Path]) -> dict[str, Any]:
    """
    Concatenate all batch files along the node axis (and graphs for y).
    Node-aligned arrays (N, D) are stacked; y is (total_graphs,).
    batch_node_graph_id is offset so graph indices are unique across batches.
    """
    if not files:
        raise FileNotFoundError("No .npz files to load.")

    accum: dict[str, list[np.ndarray]] = {}
    y_parts: list[np.ndarray] = []
    gid_parts: list[np.ndarray] = []
    graph_offset = 0

    for fp in files:
        d = np.load(fp, allow_pickle=True)
        y = np.asarray(d["y"], dtype=np.float64).ravel()
        y_parts.append(y)
        b = len(y)

        if "batch_node_graph_id" in d:
            gid = np.asarray(d["batch_node_graph_id"], dtype=np.int64) + graph_offset
            gid_parts.append(gid)

        graph_offset += b

        for k in d.files:
            if k in ("y", "start_idx"):
                continue
            if k == "batch_node_graph_id":
                continue
            arr = np.asarray(d[k])
            accum.setdefault(k, []).append(arr)

    out: dict[str, Any] = {k: np.concatenate(parts, axis=0) for k, parts in accum.items()}
    out["y"] = np.concatenate(y_parts, axis=0)
    if gid_parts:
        out["batch_node_graph_id"] = np.concatenate(gid_parts, axis=0)
    return out


def subplot_axes(n_plots: int, figsize_per: tuple[float, float] = (4.0, 3.2)):
    """Return (fig, axes ndarray) with enough subplots for n_plots panels."""
    if n_plots <= 0:
        raise ValueError("n_plots must be positive")
    ncols = int(math.ceil(math.sqrt(n_plots)))
    nrows = int(math.ceil(n_plots / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_per[0] * ncols, figsize_per[1] * nrows))
    if n_plots == 1:
        axes = np.array([axes])
    else:
        axes = np.atleast_1d(axes).ravel()
    return fig, axes


def main() -> None:
    for style in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot"):
        try:
            plt.style.use(style)
            break
        except OSError:
            continue
    plt.rcParams.update(
        {
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "font.size": 10,
            "figure.dpi": DPI,
        }
    )

    files = resolve_npz_files(NPZ_SOURCE)
    if not files:
        raise FileNotFoundError(f"No .npz files found for NPZ_SOURCE={NPZ_SOURCE!r}")

    print(f"Loading {len(files)} NPZ file(s) from {NPZ_SOURCE!r} …")
    data = merge_training_npz_batches(files)
    keys = list(data.keys())
    layer_ids = discover_layer_indices(keys)
    if not layer_ids:
        raise ValueError(f"No layer_*_pre_bin keys found in merged data (source={NPZ_SOURCE!r})")

    n_nodes_total = int(np.asarray(data[f"layer_{layer_ids[0]}_post_bin"]).shape[0])
    n_graphs_total = len(np.asarray(data["y"]).ravel())
    print(
        f"  Merged: {n_nodes_total} nodes, {n_graphs_total} graphs (targets y). "
        f"Drawing figures on full dataset."
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(RNG_SEED)

    n_layers = len(layer_ids)

    # --- Figure 1: Pre-binarization distributions ---
    fig1, axes1 = subplot_axes(n_layers, (4.2, 3.2))
    for idx, lid in enumerate(layer_ids):
        ax = axes1[idx]
        c = f"C{idx % 10}"
        arr = np.asarray(data[f"layer_{lid}_pre_bin"]).ravel()
        p1, p99 = np.percentile(arr, [1, 99])
        arr_clip = np.clip(arr, p1, p99)
        ax.hist(
            arr_clip,
            bins=80,
            color=c,
            edgecolor="white",
            alpha=0.85,
        )
        mu, sd = float(np.mean(arr)), float(np.std(arr))
        kurt = float(stats.kurtosis(arr, fisher=True))
        ax.set_title(f"Layer {lid}")
        ax.set_xlabel("Pre-binarization value")
        ax.set_ylabel("Count")
        ax.text(
            0.02,
            0.98,
            f"mean={mu:.4g}\nstd={sd:.4g}\nkurtosis={kurt:.4g}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
        )
    for j in range(n_layers, len(axes1)):
        axes1[j].set_visible(False)
    fig1.suptitle("Pre-binarization value distributions", y=1.02, fontsize=13)
    fig1.tight_layout()
    for ext in ("png", "pdf"):
        p = OUTPUT_DIR / f"fig1_pre_bin_distributions.{ext}"
        fig1.savefig(p, dpi=DPI if ext == "png" else None, bbox_inches="tight")
        print(f"Saved {p}")
    plt.close(fig1)

    # --- Figure 2: Per-node HV mean distribution (post-binarization) ---
    n_nodes = None
    colors_fig2 = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    all_node_means: list[np.ndarray] = []
    for lid in layer_ids:
        post = np.asarray(data[f"layer_{lid}_post_bin"], dtype=np.float64)
        if n_nodes is None:
            n_nodes = post.shape[0]
        all_node_means.append(post.mean(axis=1))

    if all_node_means:
        global_min = min(float(m.min()) for m in all_node_means)
        global_max = max(float(m.max()) for m in all_node_means)
        span = global_max - global_min
        pad = 0.1 * span if span > 0 else 1e-6
        x_range = (global_min - pad, global_max + pad)
    else:
        x_range = (-1.0, 1.0)

    if n_layers == 4:
        fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8))
        axes2 = np.atleast_1d(axes2).ravel()
    else:
        fig2, axes2 = subplot_axes(n_layers, (4.2, 3.2))

    fig2.suptitle(
        "Per-node hypervector mean distribution (post-binarization)",
        fontsize=14,
        y=1.02,
    )
    for idx, lid in enumerate(layer_ids):
        ax = axes2[idx]
        nm = all_node_means[idx]
        c = colors_fig2[idx % len(colors_fig2)]
        ax.hist(
            nm,
            bins=40,
            range=x_range,
            color=c,
            alpha=0.85,
            edgecolor="white",
        )
        ax.axvline(0.0, color="red", linestyle="--", linewidth=1.0)
        ax.set_title(f"Layer {lid}", fontsize=13)
        ax.set_xlabel("Mean of HV (0 = balanced)", fontsize=12)
        ax.set_ylabel("Number of nodes", fontsize=12)
        m, s = float(np.mean(nm)), float(np.std(nm))
        ax.text(
            0.05,
            0.92,
            f"mean={m:.4f}\nstd={s:.4f}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
    for j in range(n_layers, len(axes2)):
        axes2[j].set_visible(False)
    fig2.tight_layout()
    for ext in ("png", "pdf"):
        p = OUTPUT_DIR / f"fig2_post_bin_node_means.{ext}"
        fig2.savefig(p, dpi=DPI if ext == "png" else None, bbox_inches="tight")
        print(f"Saved {p}")
    plt.close(fig2)

    # --- Figure 3: Inter-layer cosine heatmap ---
    mat = np.zeros((n_layers, n_layers), dtype=np.float64)
    for i, li in enumerate(layer_ids):
        Hi = np.asarray(data[f"layer_{li}_post_bin"], dtype=np.float64)
        for j, lj in enumerate(layer_ids):
            Hj = np.asarray(data[f"layer_{lj}_post_bin"], dtype=np.float64)
            mat[i, j] = float(_cosine_rows(Hi, Hj).mean())

    fig3, ax3 = plt.subplots(figsize=(6.5, 5.5))
    im = ax3.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ax3.set_xticks(range(n_layers))
    ax3.set_yticks(range(n_layers))
    labels = [f"Layer {lid}" for lid in layer_ids]
    ax3.set_xticklabels(labels)
    ax3.set_yticklabels(labels)
    for i in range(n_layers):
        for j in range(n_layers):
            ax3.text(j, i, f"{mat[i, j]:.4f}", ha="center", va="center", color="black", fontsize=9)
    plt.colorbar(im, ax=ax3, label="Mean cosine similarity")
    ax3.set_title("Inter-layer mean cosine similarity (post-bin)")
    fig3.tight_layout()
    for ext in ("png", "pdf"):
        p = OUTPUT_DIR / f"fig3_inter_layer_cosine.{ext}"
        fig3.savefig(p, dpi=DPI if ext == "png" else None, bbox_inches="tight")
        print(f"Saved {p}")
    plt.close(fig3)

    # --- Figure 4: Hamming distance consecutive layers ---
    n_pairs_h = len(layer_ids) - 1
    if n_pairs_h < 1:
        print("Skipping fig4: need at least 2 layers for consecutive Hamming distances.")
    else:
        fig4, axes4 = plt.subplots(1, n_pairs_h, figsize=(4 * n_pairs_h, 3.8))
        if n_pairs_h == 1:
            axes4 = [axes4]
        for k in range(n_pairs_h):
            a = np.asarray(data[f"layer_{layer_ids[k]}_post_bin"])
            b = np.asarray(data[f"layer_{layer_ids[k + 1]}_post_bin"])
            ham = (a != b).mean(axis=1)
            ax = axes4[k]
            ax.hist(
                ham,
                bins=40,
                range=(0, 1),
                color="purple",
                edgecolor="white",
                alpha=0.85,
            )
            ax.axvline(0.5, color="red", linestyle="--", linewidth=1.2)
            ax.set_title(f"L{layer_ids[k]} → L{layer_ids[k + 1]}")
            ax.set_xlabel("Hamming distance (fraction)")
            ax.set_ylabel("Count")
            hm, hs = float(np.mean(ham)), float(np.std(ham))
            ax.text(
                0.02,
                0.98,
                f"mean={hm:.4f}\nstd={hs:.4f}",
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
            )
        fig4.suptitle("Per-node Hamming distance between consecutive layers", y=1.05, fontsize=13)
        fig4.tight_layout()
        for ext in ("png", "pdf"):
            p = OUTPUT_DIR / f"fig4_hamming_distance.{ext}"
            fig4.savefig(p, dpi=DPI if ext == "png" else None, bbox_inches="tight")
            print(f"Saved {p}")
        plt.close(fig4)

    # --- Figure 5: Within-layer pairwise cosine (500 random pairs) ---
    N = n_nodes if n_nodes is not None else np.asarray(data[f"layer_{layer_ids[0]}_post_bin"]).shape[0]
    pair_list = sample_unique_pairs(N, 500, rng)

    fig5, axes5 = subplot_axes(n_layers, (4.2, 3.2))
    for idx, lid in enumerate(layer_ids):
        ax = axes5[idx]
        H = np.asarray(data[f"layer_{lid}_post_bin"], dtype=np.float64)
        if pair_list:
            sims = np.asarray(
                [_cosine_pair(H[i], H[j]) for i, j in pair_list],
                dtype=np.float64,
            )
            ax.hist(
                sims,
                bins=40,
                range=(-1, 1),
                color="teal",
                edgecolor="white",
                alpha=0.85,
            )
            ax.text(
                0.02,
                0.98,
                f"mean={np.mean(sims):.4f}\nstd={np.std(sims):.4f}",
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
            )
        else:
            ax.text(0.5, 0.5, "Not enough nodes for pairs", ha="center", va="center")
        ax.axvline(0.0, color="red", linestyle="--", linewidth=1.2)
        ax.set_title(f"Layer {lid}")
        ax.set_xlabel("Cosine similarity")
        ax.set_ylabel("Count")
    for j in range(n_layers, len(axes5)):
        axes5[j].set_visible(False)
    fig5.suptitle("Within-layer cosine similarity (500 random node pairs)", y=1.02, fontsize=13)
    fig5.tight_layout()
    for ext in ("png", "pdf"):
        p = OUTPUT_DIR / f"fig5_within_layer_similarity.{ext}"
        fig5.savefig(p, dpi=DPI if ext == "png" else None, bbox_inches="tight")
        print(f"Saved {p}")
    plt.close(fig5)

    # --- Figure 6: Sigma-pi distributions ---
    s0 = np.asarray(data["sigma_pi_order_0"]).ravel()
    s1 = np.asarray(data["sigma_pi_order_1"]).ravel()
    sc = np.asarray(data["sigma_pi_combined"]).ravel()
    p1_g = min(np.percentile(s0, 1), np.percentile(s1, 1), np.percentile(sc, 1))
    p99_g = max(np.percentile(s0, 99), np.percentile(s1, 99), np.percentile(sc, 99))
    x_range = (p1_g, p99_g)

    fig6, axes6 = plt.subplots(1, 3, figsize=(12.5, 3.8))
    titles = [
        "Σπ Order 0 (= F1 tap)",
        "Σπ Order 1",
        "Σπ Combined",
    ]
    arrays = [s0, s1, sc]
    for ax, title, arr in zip(axes6, titles, arrays):
        arr_clip = np.clip(arr, x_range[0], x_range[1])
        ax.hist(
            arr_clip,
            bins=80,
            range=x_range,
            color="darkslateblue",
            edgecolor="white",
            alpha=0.85,
        )
        ax.set_title(title)
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
        ax.set_xlim(x_range)
        ax.text(
            0.02,
            0.98,
            f"mean={np.mean(arr):.6f}\nstd={np.std(arr):.6f}\nmin={np.min(arr):.6f}\nmax={np.max(arr):.6f}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
        )
    fig6.suptitle("Sigma-pi stage value distributions (1–99% clip, shared x-range)", y=1.05, fontsize=13)
    fig6.tight_layout()
    for ext in ("png", "pdf"):
        p = OUTPUT_DIR / f"fig6_sigma_pi_distributions.{ext}"
        fig6.savefig(p, dpi=DPI if ext == "png" else None, bbox_inches="tight")
        print(f"Saved {p}")
    plt.close(fig6)

    # --- Figure 7: Sigma-pi pairwise cosine per node ---
    S0 = np.asarray(data["sigma_pi_order_0"], dtype=np.float64)
    S1 = np.asarray(data["sigma_pi_order_1"], dtype=np.float64)
    Sc = np.asarray(data["sigma_pi_combined"], dtype=np.float64)
    cos_01 = _cosine_rows(S0, S1)
    cos_0c = _cosine_rows(S0, Sc)
    cos_1c = _cosine_rows(S1, Sc)

    fig7, axes7 = plt.subplots(1, 3, figsize=(12.5, 3.8))
    plots = [
        (cos_01, "Order 0 vs Order 1"),
        (cos_0c, "Order 0 vs Combined"),
        (cos_1c, "Order 1 vs Combined"),
    ]
    for ax, (vec, title) in zip(axes7, plots):
        ax.hist(
            vec,
            bins=40,
            range=(-1, 1),
            color="coral",
            edgecolor="white",
            alpha=0.85,
        )
        ax.axvline(0.0, color="red", linestyle="--", linewidth=1.2)
        ax.set_title(title)
        ax.set_xlabel("Cosine similarity")
        ax.set_ylabel("Count")
        ax.text(
            0.02,
            0.98,
            f"mean={np.mean(vec):.4f}\nstd={np.std(vec):.4f}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
        )
    fig7.suptitle("Per-node sigma-pi cosine similarity", y=1.05, fontsize=13)
    fig7.tight_layout()
    for ext in ("png", "pdf"):
        p = OUTPUT_DIR / f"fig7_sigma_pi_similarity.{ext}"
        fig7.savefig(p, dpi=DPI if ext == "png" else None, bbox_inches="tight")
        print(f"Saved {p}")
    plt.close(fig7)

    # --- Figure 8: Target y ---
    y = np.asarray(data["y"], dtype=np.float64).ravel()
    fig8_bins = min(100, max(30, max(10, len(y) // 80)))
    fig8, ax8 = plt.subplots(figsize=(6, 4))
    ax8.hist(y, bins=fig8_bins, color="gray", edgecolor="white", alpha=0.85)
    ax8.set_xlabel("Target value (y)")
    ax8.set_ylabel("Count")
    ax8.set_title("Target (y) distribution")
    ax8.text(
        0.02,
        0.98,
        f"mean={np.mean(y):.4f}\nstd={np.std(y):.4f}\nmin={np.min(y):.4f}\nmax={np.max(y):.4f}",
        transform=ax8.transAxes,
        fontsize=10,
        verticalalignment="top",
    )
    fig8.tight_layout()
    for ext in ("png", "pdf"):
        p = OUTPUT_DIR / f"fig8_y_distribution.{ext}"
        fig8.savefig(p, dpi=DPI if ext == "png" else None, bbox_inches="tight")
        print(f"Saved {p}")
    plt.close(fig8)

    print("Done.")


if __name__ == "__main__":
    main()
