"""
Feature-wise standardization for node and edge features before VSA projection.

Three-tier strategy (matches bipolar HV space [-1,+1]):

  Binary {0,1}       →  x * 2 - 1          →  {-1, +1}
  Bounded continuous →  min-max → [-1,+1]   (fit on train only — no leakage)
  Unbounded float    →  tanh(x / σ)         →  (-1, +1)  soft-clips outliers

Node feature layout (18 columns, from expand_atomic_features):
  col  0      atomic_num          bounded int  → min-max
  col  1      degree              bounded int  → min-max
  col  2      valence_electrons   bounded int  → min-max
  cols 3-5    hybridization (3)   binary       → ×2-1
  col  6      aromaticity         binary       → ×2-1
  col  7      formal_charge       unbounded    → tanh
  cols 8-9    hbond_flags (2)     binary       → ×2-1
  cols 10-11  chirality (2)       binary       → ×2-1
  col  12     num_attached_h      bounded int  → min-max
  col  13     gasteiger_charge    unbounded    → tanh
  col  14     crippen_logp        unbounded    → tanh
  col  15     tpsa_contrib        bounded ≥0   → min-max
  col  16     is_in_aromatic_ring binary       → ×2-1
  col  17     smallest_ring_size  bounded int  → min-max

Edge feature layout (5 columns, from bond_node_features_geognn):
  col  0      bond_type           bounded int  → min-max
  col  1      is_conjugated       binary       → ×2-1
  col  2      in_ring             binary       → ×2-1
  col  3      bond_length         unbounded ≥0 → tanh
  col  4      stereo              bounded int  → min-max
"""

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Feature column assignments
# ---------------------------------------------------------------------------

_NODE_BINARY_COLS   = [3, 4, 5, 6, 8, 9, 10, 11, 16]
_NODE_BOUNDED_COLS  = [0, 1, 2, 12, 15, 17]
_NODE_UNBOUNDED_COLS = [7, 13, 14]

_EDGE_BINARY_COLS   = [1, 2]
_EDGE_BOUNDED_COLS  = [0, 4]
_EDGE_UNBOUNDED_COLS = [3]


# ---------------------------------------------------------------------------
# Scaler
# ---------------------------------------------------------------------------

class MolecularFeatureScaler:
    """
    Fit statistics on training graphs; apply the same transformation to any
    split (train or test) without leakage.

    Usage::

        scaler = MolecularFeatureScaler()
        scaler.fit(train_graphs)          # learn stats from training set
        scaler.transform(train_graphs)    # normalise in-place
        scaler.transform(test_graphs)     # same stats applied to test

    Or combined::

        scaler = MolecularFeatureScaler()
        scaler.fit_transform(train_graphs)
        scaler.transform(test_graphs)
    """

    def __init__(self):
        self._node_min   = None   # np.ndarray [F_node]
        self._node_max   = None
        self._node_sigma = None   # std for tanh columns
        self._edge_min   = None   # np.ndarray [F_edge]
        self._edge_max   = None
        self._edge_sigma = None
        self._fitted = False

    # ------------------------------------------------------------------
    def fit(self, graph_list):
        """Collect per-column statistics from all graphs in graph_list."""
        node_arrays, edge_arrays = [], []

        for g in graph_list:
            if g.node_features is not None and g.node_features.numel() > 0:
                node_arrays.append(g.node_features.detach().cpu().float().numpy())
            if g.edge_attr is not None and g.edge_attr.numel() > 0:
                edge_arrays.append(g.edge_attr.detach().cpu().float().numpy())

        if node_arrays:
            X = np.concatenate(node_arrays, axis=0)      # [total_atoms, F_node]
            self._node_min   = X.min(axis=0)
            self._node_max   = X.max(axis=0)
            self._node_sigma = X.std(axis=0)

        if edge_arrays:
            E = np.concatenate(edge_arrays, axis=0)      # [total_bonds, F_edge]
            self._edge_min   = E.min(axis=0)
            self._edge_max   = E.max(axis=0)
            self._edge_sigma = E.std(axis=0)

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    def transform(self, graph_list):
        """Normalise node and edge features in-place. Returns graph_list."""
        if not self._fitted:
            raise RuntimeError("Call fit() on the training set before transform().")

        for g in graph_list:
            if g.node_features is not None and g.node_features.numel() > 0:
                g.node_features = _apply_normalization(
                    g.node_features,
                    self._node_min, self._node_max, self._node_sigma,
                    _NODE_BINARY_COLS, _NODE_BOUNDED_COLS, _NODE_UNBOUNDED_COLS,
                )
            if g.edge_attr is not None and g.edge_attr.numel() > 0:
                g.edge_attr = _apply_normalization(
                    g.edge_attr,
                    self._edge_min, self._edge_max, self._edge_sigma,
                    _EDGE_BINARY_COLS, _EDGE_BOUNDED_COLS, _EDGE_UNBOUNDED_COLS,
                )
        return graph_list

    # ------------------------------------------------------------------
    def fit_transform(self, graph_list):
        """Convenience: fit then transform the same graph list."""
        self.fit(graph_list)
        return self.transform(graph_list)

    # ------------------------------------------------------------------
    def summary(self):
        """Print fitted statistics for inspection."""
        if not self._fitted:
            print("Scaler not fitted yet.")
            return
        print("=== Node feature stats ===")
        _print_stats("node", self._node_min, self._node_max, self._node_sigma,
                     _NODE_BINARY_COLS, _NODE_BOUNDED_COLS, _NODE_UNBOUNDED_COLS)
        print("=== Edge feature stats ===")
        _print_stats("edge", self._edge_min, self._edge_max, self._edge_sigma,
                     _EDGE_BINARY_COLS, _EDGE_BOUNDED_COLS, _EDGE_UNBOUNDED_COLS)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _apply_normalization(X, x_min, x_max, x_sigma,
                         binary_cols, bounded_cols, unbounded_cols):
    """
    X          : torch.FloatTensor  [N, F]
    x_min/max  : np.ndarray [F]  — training-set min/max
    x_sigma    : np.ndarray [F]  — training-set std
    Returns a new torch.FloatTensor [N, F] with values in [-1, +1].
    """
    arr = X.detach().cpu().float().numpy().copy()  # work on a fresh copy

    # --- Binary {0,1} → {-1, +1} ---
    for c in binary_cols:
        arr[:, c] = arr[:, c] * 2.0 - 1.0

    # --- Bounded continuous → min-max → [-1, +1] ---
    for c in bounded_cols:
        lo, hi = float(x_min[c]), float(x_max[c])
        if hi > lo:
            arr[:, c] = 2.0 * (arr[:, c] - lo) / (hi - lo) - 1.0
        else:
            arr[:, c] = 0.0   # constant feature → neutral midpoint

    # --- Unbounded float → tanh(x / σ) → (-1, +1) ---
    for c in unbounded_cols:
        sigma = float(x_sigma[c])
        if sigma < 1e-6:
            sigma = 1.0       # near-constant fallback
        arr[:, c] = np.tanh(arr[:, c] / sigma)

    return torch.from_numpy(arr)


def _print_stats(tag, mn, mx, sigma, binary_cols, bounded_cols, unbounded_cols):
    if mn is None:
        print(f"  No {tag} data collected.")
        return
    F = len(mn)
    for c in range(F):
        if c in binary_cols:
            kind = "binary  "
        elif c in bounded_cols:
            kind = "bounded "
        else:
            kind = "unbound "
        print(f"  col {c:2d} [{kind}]  min={mn[c]:+.4f}  max={mx[c]:+.4f}  σ={sigma[c]:.4f}")
