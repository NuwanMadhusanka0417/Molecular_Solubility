"""
src/tsne_viz.py
===============
t-SNE visualisation of GVFA hypervector representations at each pipeline stage.
Called from GVFA_edge_main.py when --tsne is passed.

Stages visualised (9 panels, A-I):
  A  Layer 0     VSA projection + L2 norm (input to GNN, before message passing)
  B  Layer 1     after GNN message-passing step 1 + bind + L2 norm
  C  Layer 2     after GNN message-passing step 2
  D  Layer 3     after GNN message-passing step 3
  E  Layer 4     after GNN message-passing step 4
  F  All layers  mean of layers 0-4, then mean-pool atoms to molecule
  G  Tap buffer  hop-decay weighted sum of all layers, mean-pooled to molecule
  H  Sigma-Pi    polynomial HV expansion, mean-pooled to molecule
  I  Graph pool  multi-stat pool output (mean | max | mean-sq), 1 vector/molecule

For stages A-H: node HVs [N_total, D] are mean-pooled per graph → [G, D].
For stage  I:   model output is already [G, 3D] — no pooling needed.

Colour + marker (logS class):
  Dark blue circle #001f3f   logS < -3    (low; includes logS < -4)
  Red triangle     #b71c1c   -3 <= logS < -1 (moderate)
  Green square     #1b5e20   logS >= -1   (high)
"""

import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE


# ---------------------------------------------------------------------------
# Colour / marker constants (one shape per solubility class)
# ---------------------------------------------------------------------------

_COLORS = ["#001f3f", "#b71c1c", "#1b5e20"]
_MARKERS = ["o", "^", "s"]  # circle, triangle, square
_LABELS = [
    "logS < -3   (low; incl. logS < -4)",
    "-3 ≤ logS < -1   (moderate)",
    "logS ≥ -1   (high)",
]


def _assign_class(log_s: np.ndarray) -> np.ndarray:
    # 0 dark blue, 1 red, 2 green — contiguous bins on logS
    return np.where(log_s >= -1, 2, np.where(log_s >= -3, 1, 0))


# ---------------------------------------------------------------------------
# Pooling helper
# ---------------------------------------------------------------------------

def _mean_pool(node_mat: torch.Tensor, start_idx: list) -> np.ndarray:
    """
    Mean-pool node-level tensor [N_total, D] → graph-level [G, D].
    start_idx: list of cumulative node offsets, length G+1 (from aux['start_idx']).
    """
    rows = []
    for i in range(len(start_idx) - 1):
        lo, hi = start_idx[i], start_idx[i + 1]
        rows.append(node_mat[lo:hi].mean(dim=0))
    return torch.stack(rows).numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# t-SNE wrapper
# ---------------------------------------------------------------------------

def _run_tsne(X: np.ndarray, perplexity: int, seed: int) -> np.ndarray:
    X = X.astype(np.float32)
    n = X.shape[0]
    if n < 2:
        return np.zeros((n, 2), dtype=np.float32)
    perp = min(perplexity, max(5, n // 5))
    perp = min(perp, n - 1)
    perp = max(perp, 1)
    kw = dict(
        n_components=2,
        perplexity=perp,
        random_state=seed,
        init="pca",
    )
    try:
        tsne = TSNE(**kw, max_iter=1000)
    except TypeError:
        tsne = TSNE(**kw, n_iter=1000)
    return tsne.fit_transform(X)


# ---------------------------------------------------------------------------
# Stage capture
# ---------------------------------------------------------------------------

def _capture_stages(graphs, model, device) -> tuple:
    """
    Run one forward pass with capture_aux=True and return all 9 stage matrices.

    Returns
    -------
    stages : list of (title: str, X: np.ndarray [G, D_i])
    log_s  : np.ndarray [G]
    """
    log_s = np.array(
        [float(torch.as_tensor(g.label).reshape(-1)[0].item()) for g in graphs],
        dtype=np.float32,
    )

    # Reconstruct X_concat (layer 0) — same ops as GraphCNN.forward()
    # Must be done before model() call (forward does not modify g.node_features)
    with torch.no_grad():
        X_cat = torch.cat([g.node_features for g in graphs], dim=0).to(device)
        X_concat = F.normalize(X_cat, p=2, dim=1, eps=1e-8).cpu()  # [N_total, D]

    # Full forward pass
    model.eval()
    with torch.no_grad():
        graph_out = model(graphs, capture_aux=True)   # [1, G, 3D]

    aux = model._aux
    if aux is None:
        raise RuntimeError(
            "model._aux is None after forward(capture_aux=True). "
            "Make sure the model was built with use_reservoir=True."
        )

    start_idx   = aux["start_idx"]        # list[int], length G+1
    post_bins   = aux["layer_post_bin"]   # list of [N_total, D], one per GNN layer
    n_gnn       = len(post_bins)          # 4 (num_layers - 1)

    stages = []

    # Panel A: layer 0 (VSA projection + L2 norm, before any message passing)
    stages.append((
        "A  |  Layer 0\nVSA proj + L2 norm\n(input to GNN)",
        _mean_pool(X_concat, start_idx),
    ))

    # Panels B-E: GNN layers 1-4
    panel_letters = "BCDE"
    for li, post_h in enumerate(post_bins):
        stages.append((
            f"{panel_letters[li]}  |  Layer {li + 1}\nMsg-passing step {li + 1} + L2 norm",
            _mean_pool(post_h.cpu(), start_idx),
        ))

    # Panel F: all layers combined (mean across layers 0-4, then mean-pool)
    all_layers = [X_concat] + [p.cpu() for p in post_bins]
    combined   = torch.stack(all_layers, dim=0).mean(dim=0)   # [N_total, D]
    stages.append((
        f"F  |  All {n_gnn + 1} layers combined\nmean across layers → mean-pool",
        _mean_pool(combined, start_idx),
    ))

    # Panel G: after tap buffer
    if aux["F1_tap"] is None:
        raise RuntimeError("F1_tap is None — build model with use_reservoir=True.")
    stages.append((
        "G  |  After tap buffer\nhop-decay weighted sum → mean-pool",
        _mean_pool(aux["F1_tap"].cpu(), start_idx),
    ))

    # Panel H: after Sigma-Pi expansion
    if aux["sigma_pi_combined"] is None:
        raise RuntimeError("sigma_pi_combined is None.")
    stages.append((
        "H  |  After Sigma-Pi expansion\npolynomial HV binding → mean-pool",
        _mean_pool(aux["sigma_pi_combined"].cpu(), start_idx),
    ))

    # Panel I: after graph pooling (already [G, 3D])
    if graph_out.dim() == 3:
        graph_out = graph_out.squeeze(0)
    stages.append((
        "I  |  After graph pooling\nmulti-stat: mean | max | mean-sq\n(1 vector/molecule → Ridge)",
        graph_out.cpu().numpy().astype(np.float32),
    ))

    return stages, log_s


# ---------------------------------------------------------------------------
# Figure drawing
# ---------------------------------------------------------------------------

def _draw_figure(stages, log_s, out_path, perplexity, seed, dpi=150):
    n     = len(stages)
    ncols = 3
    nrows = math.ceil(n / ncols)

    classes = _assign_class(log_s)
    print("\n  [t-SNE] Solubility class breakdown:")
    for ci, lbl in enumerate(_LABELS):
        print(f"    {lbl}: {(classes == ci).sum()}")

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 4.6, nrows * 4.2),
        facecolor="white",
    )
    axes = np.array(axes).reshape(-1)

    for idx, (title, X) in enumerate(stages):
        ax = axes[idx]
        print(f"  [t-SNE] Panel {idx+1}/{n}: {title.split(chr(10))[0]}  shape={X.shape}")
        coords = _run_tsne(X, perplexity=perplexity, seed=seed)
        for ci, col in enumerate(_COLORS):
            mask = classes == ci
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                marker=_MARKERS[ci],
                c=col,
                s=36,
                alpha=0.85,
                linewidths=0.25,
                edgecolors="none",
            )
        ax.set_title(title, fontsize=8.0, fontweight="bold",
                     pad=5, linespacing=1.45)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
        ax.text(0.03, 0.97, chr(65 + idx),
                transform=ax.transAxes, fontsize=10,
                fontweight="bold", va="top", ha="left", color="#333333")

    for ax in axes[n:]:
        ax.set_visible(False)

    legend_handles = [
        Line2D(
            [0],
            [0],
            linestyle="None",
            marker=_MARKERS[i],
            markersize=9,
            markerfacecolor=_COLORS[i],
            markeredgecolor=_COLORS[i],
            label=lbl,
        )
        for i, lbl in enumerate(_LABELS)
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        fontsize=9.5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.005),
    )
    fig.suptitle(
        "t-SNE of GVFA Hypervector Representations at Each Pipeline Stage",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  [t-SNE] Saved → {out_path}")


# ---------------------------------------------------------------------------
# Public entry point — called from GVFA_edge_main.py
# ---------------------------------------------------------------------------

def draw_tsne_pipeline(
    model,
    train_HVs,
    test_HVs,
    device,
    split: str = "test",
    max_mols: int = 500,
    out_dir: str = "tsne_plots",
    tsne_perplexity: int = 30,
    seed: int = 42,
    dim: int = 0,
    sigma_tag: str = "",
):
    """
    Capture HV representations at all 9 GVFA pipeline stages and save
    a 3x3 t-SNE panel figure to out_dir.

    Parameters
    ----------
    model        : GraphCNN — already built, edge stats set, use_reservoir=True
    train_HVs    : list[S2VGraph] — after VSA_conversion (projected node features)
    test_HVs     : list[S2VGraph] — after VSA_conversion with train stats
    device       : torch.device
    split        : 'train', 'test', or 'both'
    max_mols     : downsample to this many molecules before t-SNE (0 = all)
    out_dir      : directory for output PNGs
    tsne_perplexity : t-SNE perplexity
    seed         : RNG seed for downsampling and t-SNE
    dim          : VSA dim (used only in output filename)
    sigma_tag    : sigma_pi tag string (used only in output filename)
    """
    os.makedirs(out_dir, exist_ok=True)

    split_map = {
        "train": [(train_HVs, "train")],
        "test":  [(test_HVs,  "test")],
        "both":  [(train_HVs, "train"), (test_HVs, "test")],
    }
    if split not in split_map:
        raise ValueError(f"split must be 'train', 'test', or 'both', got {split!r}")

    for graphs, split_name in split_map[split]:

        # Optional random downsample for t-SNE speed
        if max_mols > 0 and len(graphs) > max_mols:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(graphs), max_mols, replace=False)
            idx.sort()
            graphs = [graphs[i] for i in idx]
            print(f"  [t-SNE] Downsampled {split_name} → {max_mols} molecules")

        print(f"\n  [t-SNE] Capturing pipeline stages: split={split_name}, "
              f"n={len(graphs)}, dim={dim}, sigma={sigma_tag}")

        stages, log_s = _capture_stages(graphs, model, device)

        fname = f"tsne_{split_name}_dim{dim}_{sigma_tag}.png"
        out_path = os.path.join(out_dir, fname)

        _draw_figure(
            stages, log_s, out_path,
            perplexity=tsne_perplexity,
            seed=seed,
        )
