
import torch
import numpy as np
import math
import torch.nn.functional as F
# import torch
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

    # accumulate messages for each node
    messages = torch.zeros_like(node_H)  # [N, D]

    for e in range(E):
        u = int(edge_index[0, e])
        v = int(edge_index[1, e])

        b  = edge_H[e]    # bond HV
        hu = node_H[u]
        hv = node_H[v]

        # message from v -> u and u -> v
        msg_u = hv_bind(b, hv)
        msg_v = hv_bind(b, hu)

        messages[u] += msg_u
        messages[v] += msg_v

    # update node HVs (like h' = h + m)
    updated = node_H + alpha * messages

    # binarize back to {-1,+1} for stability
    # updated = torch.sign(updated)
    # updated[updated == 0] = 1.0
    updated = F.normalize(updated, p=2, dim=1)

    return updated

def _random_projection_matrix(in_dim, out_dim, orthogonal=False, seed=0):
    """
    Build shared random projection W: (in_dim, out_dim).
    - orthogonal=True: QR-based orthonormal columns → better preserves norms (info-preserving).
    - orthogonal=False: standard Gaussian / sqrt(in_dim) (JL-style).
    """
    g = torch.Generator().manual_seed(seed)
    # W = torch.randn(in_dim, out_dim, generator=g)
    # if orthogonal and out_dim <= in_dim:
    #     # Orthonormal columns: preserves ||x|| when out_dim >= in_dim; minimizes distortion when out_dim < in_dim
    #     Q, _ = torch.linalg.qr(W)
    #     W = Q[:, :out_dim]
    # else:
    #     W = W / math.sqrt(in_dim)

    if orthogonal:
        if out_dim <= in_dim:
            # Orthonormal columns: W^T W = I
            Q, _ = torch.linalg.qr(torch.randn(in_dim, out_dim, generator=g))
            W = Q[:, :out_dim]
        else:
            # Orthonormal rows: W W^T = I
            Q, _ = torch.linalg.qr(torch.randn(out_dim, in_dim, generator=g))
            W = Q[:, :in_dim].T  # shape (in_dim, out_dim)
    else:
        W = torch.randn(in_dim, out_dim, generator=g)
        W = W / math.sqrt(in_dim)
    return W


def project_with_vsa(g_list, new_dim, projection_type="orthogonal", seed=0,
                     feature_mean=None, feature_std=None):
    """
    Project node features to hypervectors, with optional pre-projection standardization.

    If feature_mean and feature_std are None, computes them from the current g_list
    (training mode). Otherwise applies the provided stats (test mode).

    Returns: (g_list, feature_mean, feature_std)
        feature_mean, feature_std: torch.Tensor [F_node] — always returned so callers
        can pass train stats to test projection.
    """
    torch.manual_seed(seed)
    F_node = g_list[0].node_features.shape[1]
    EXPECTED_NODE_FEAT_DIM = 18  # update if expand_atomic_features in create_graphs.py changes
    assert F_node == EXPECTED_NODE_FEAT_DIM or feature_mean is not None, (
        f"Node feature dim is {F_node}, expected {EXPECTED_NODE_FEAT_DIM}. "
        f"Update CONTINUOUS_COLS in project_with_vsa if features were added."
    )
    use_orthogonal = projection_type == "orthogonal"
    W_node = _random_projection_matrix(F_node, new_dim, orthogonal=use_orthogonal, seed=seed)

    print("VSA_conversion: node feature dim =", F_node, "new_dim =", new_dim,
          "projection =", projection_type)

    # --- Selective standardization: continuous columns only ---
    # Binary/one-hot columns already live in {0,1} — standardizing them is unnecessary
    # and harmful when nearly-constant (std ≈ 0 → clamp explosion).
    # Layout (18 cols from expand_atomic_features in create_graphs.py):
    #   0 atomic_number       7 formal_charge       12 num_attached_h
    #   13 gasteiger  14 crippen_logp  15 tpsa_contrib  17 smallest_ring_size
    CONTINUOUS_COLS = [0, 1, 2, 7, 12, 13, 14, 15, 17]

    if feature_mean is None:
        all_X = torch.cat([g.node_features for g in g_list], dim=0)
        feature_mean = torch.zeros(F_node, dtype=all_X.dtype)
        feature_std = torch.ones(F_node, dtype=all_X.dtype)
        cont = all_X[:, CONTINUOUS_COLS]
        feature_mean[CONTINUOUS_COLS] = cont.mean(dim=0)
        feature_std[CONTINUOUS_COLS] = cont.std(dim=0).clamp(min=0.1)

        print("  [Standardization] Selective (continuous cols only).")
        print(f"    Mean  (cont): {feature_mean[CONTINUOUS_COLS].tolist()}")
        print(f"    Std   (cont): {feature_std[CONTINUOUS_COLS].tolist()}")
    else:
        print("  [Standardization] Applying pre-computed train stats to test data.")

    for g in g_list:
        X = (g.node_features - feature_mean) / feature_std
        g.node_features = torch.matmul(X, W_node)

    print("g list item shape after VSA:", g_list[0].node_features.shape)
    return g_list, feature_mean, feature_std


def VSA_conversion(g_list, new_dim=None, projection_type="orthogonal", seed=0,
                   feature_mean=None, feature_std=None):
    """
    Build neighbors & edge_mat for GraphCNN. If new_dim is set, project node
    features to HVs with optional pre-projection standardization.

    feature_mean, feature_std: if None, computed from g_list (training).
        Pass training stats here for test data to avoid data leakage.

    Returns: (g_list, feature_mean, feature_std)
        feature_mean/std are None if new_dim is None (no projection done).
    """
    # Build neighbors and edge_mat; use edge_index when available (aligns with edge_attr)
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
        return g_list, None, None

    g_list, feature_mean, feature_std = project_with_vsa(
        g_list, new_dim, projection_type=projection_type, seed=seed,
        feature_mean=feature_mean, feature_std=feature_std,
    )
    return g_list, feature_mean, feature_std
