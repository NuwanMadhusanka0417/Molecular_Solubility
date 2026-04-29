
import torch
import math
import torch.nn.functional as F

# Node feature layout [18 cols] — pre-projection maps each column into ~[-1, +1]
BINARY_COLS = [3, 4, 5, 6, 8, 9, 10, 11, 16]  # {0,1} → {-1,+1}
MINMAX_COLS = [0, 1, 2, 12, 15, 17]              # min-max on train → [-1,+1]
TANH_COLS = {7: 1.0, 13: 0.3, 14: 0.5}          # formal_charge, gasteiger, crippen


def hv_bind(a, b):
    """
    Hypervector binding for bipolar HVs: elementwise multiplication.
    a, b: [D]
    """
    return a * b

def vsa_message_passing(node_H, edge_H, edge_index, alpha=1.0):
    """
    One GNN-style message passing step in VSA space.

    node_H:   [N, D]  node hypervectors
    edge_H:   [E, D]  edge/bond hypervectors
    edge_index: [2, E]  (u, v) indices for each edge
    alpha: message strength
    """
    N, D = node_H.shape
    E = edge_H.shape[0]

    messages = torch.zeros_like(node_H)

    for e in range(E):
        u = int(edge_index[0, e])
        v = int(edge_index[1, e])

        b  = edge_H[e]
        hu = node_H[u]
        hv = node_H[v]

        msg_u = hv_bind(b, hv)
        msg_v = hv_bind(b, hu)

        messages[u] += msg_u
        messages[v] += msg_v

    updated = node_H + alpha * messages
    updated = F.normalize(updated, p=2, dim=1)

    return updated

def _random_projection_matrix(in_dim, out_dim, orthogonal=False, seed=0):
    """
    Build shared random projection W: (in_dim, out_dim).
    - orthogonal=True: QR-based orthonormal columns → better preserves norms (info-preserving).
    - orthogonal=False: standard Gaussian / sqrt(in_dim) (JL-style).
    """
    g = torch.Generator().manual_seed(seed)

    if orthogonal:
        if out_dim <= in_dim:
            Q, _ = torch.linalg.qr(torch.randn(in_dim, out_dim, generator=g))
            W = Q[:, :out_dim]
        else:
            Q, _ = torch.linalg.qr(torch.randn(out_dim, in_dim, generator=g))
            W = Q[:, :in_dim].T
    else:
        W = torch.randn(in_dim, out_dim, generator=g)
        W = W / math.sqrt(in_dim)
    return W


def _apply_node_bounded_features(X, train_stats):
    """X [N, 18] float; maps columns to ~[-1,+1] using train_stats from fit."""
    X = X.clone().float()
    bc = train_stats["BINARY_COLS"]
    mm = train_stats["MINMAX_COLS"]
    X[:, bc] = X[:, bc] * 2.0 - 1.0
    X[:, mm] = (
        (X[:, mm] - train_stats["col_min"]) / train_stats["col_range"]
    ) * 2.0 - 1.0
    for col, scale in train_stats["TANH_COLS"].items():
        X[:, col] = torch.tanh(X[:, col] / scale)
    return X


def project_with_vsa(g_list, new_dim, projection_type="orthogonal", seed=0, train_stats=None):
    """
    Map raw node features to ~[-1,+1] per column (train stats for min-max), then random-project.

    train_stats: if None, fit from g_list (training). Else use this dict (test / no leakage).

    Returns: (g_list, train_stats)
    """
    torch.manual_seed(seed)
    F_node = g_list[0].node_features.shape[1]
    EXPECTED_NODE_FEAT_DIM = 18
    assert F_node == EXPECTED_NODE_FEAT_DIM or train_stats is not None, (
        f"Node feature dim is {F_node}, expected {EXPECTED_NODE_FEAT_DIM}. "
        f"Update BINARY_COLS / MINMAX_COLS / TANH_COLS if features changed."
    )
    use_orthogonal = projection_type == "orthogonal"
    W_node = _random_projection_matrix(F_node, new_dim, orthogonal=use_orthogonal, seed=seed)

    print("VSA_conversion: node feature dim =", F_node, "new_dim =", new_dim,
          "projection =", projection_type)

    if train_stats is None:
        all_X = torch.cat([g.node_features for g in g_list], dim=0)
        col_min = all_X[:, MINMAX_COLS].min(dim=0).values
        col_range = (all_X[:, MINMAX_COLS].max(dim=0).values - col_min).clamp(min=1e-6)
        train_stats = {
            "BINARY_COLS": BINARY_COLS,
            "MINMAX_COLS": MINMAX_COLS,
            "TANH_COLS": dict(TANH_COLS),
            "col_min": col_min,
            "col_range": col_range,
        }
        print("  [Bounded features] Fit min-max on train (cols %s)." % MINMAX_COLS)
        print(f"    col_min:   {col_min.tolist()}")
        print(f"    col_range: {col_range.tolist()}")
    else:
        print("  [Bounded features] Applying pre-computed train stats (test data).")

    for g in g_list:
        X = _apply_node_bounded_features(g.node_features, train_stats)
        g.node_features = torch.matmul(X, W_node)

    print("g list item shape after VSA:", g_list[0].node_features.shape)
    return g_list, train_stats


def VSA_conversion(g_list, new_dim=None, projection_type="orthogonal", seed=0, train_stats=None):
    """
    Build neighbors & edge_mat for GraphCNN. If new_dim is set, project node features to HVs.

    train_stats: if None on first call, fit bounded transforms from g_list; pass the returned
        dict into the test call so test uses training min-max (no leakage).

    Returns: (g_list, train_stats) — train_stats is None when new_dim is None.
    """
    for g in g_list:
        g.neighbors = [[] for _ in range(len(g.g))]
        for i, j in g.g.edges():
            g.neighbors[i].append(j)
            g.neighbors[j].append(i)
        degree_list = [len(g.neighbors[i]) for i in range(len(g.g))]
        g.max_neighbor = max(degree_list) if degree_list else 0

        if hasattr(g, "edge_index") and g.edge_index is not None and g.edge_index.numel() > 0:
            g.edge_mat = g.edge_index.clone()
        else:
            edges = [list(pair) for pair in g.g.edges()]
            edges.extend([[j, i] for i, j in edges])
            if edges:
                g.edge_mat = torch.LongTensor(edges).transpose(0, 1)
            else:
                g.edge_mat = torch.zeros((2, 0), dtype=torch.long)

    if not new_dim:
        return g_list, None

    g_list, train_stats = project_with_vsa(
        g_list, new_dim, projection_type=projection_type, seed=seed, train_stats=train_stats,
    )
    return g_list, train_stats
