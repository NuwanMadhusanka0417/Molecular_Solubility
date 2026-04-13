
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
    # g = torch.Generator().manual_seed(seed)
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


def project_with_vsa(g_list, new_dim, projection_type="orthogonal", seed=0):
    """
    Project node features to hypervectors only.

    Edge conditioning is done solely in GraphCNN (single projection + FFT binding)
    so edge_attr is not projected or used here.

    projection_type: "orthogonal" (default) = orthonormal random projection, better
        preserves norms and distances; "gaussian" = standard JL-style random projection.
    seed: controls the random projection matrix (same seed => same W across train/test).
    """
    torch.manual_seed(seed)
    F_node = g_list[0].node_features.shape[1]
    use_orthogonal = projection_type == "orthogonal"
    W_node = _random_projection_matrix(F_node, new_dim, orthogonal=use_orthogonal, seed=seed)

    print("VSA_conversion: node feature dim =", F_node, "new_dim =", new_dim,
          "projection =", projection_type)

    for g in g_list:
        X = g.node_features  # [N, F_node]
        node_H = torch.matmul(X, W_node)  # [N, D]
        g.node_features = node_H

    print("g list item shape after VSA:", g_list[0].node_features.shape)
    return g_list

def VSA_conversion(g_list, new_dim=None, projection_type="orthogonal", seed=0):
    """
    Build neighbors & edge_mat for GraphCNN. If new_dim is set, project node
    features to HVs only (edge conditioning is done solely in GraphCNN).

    projection_type: "orthogonal" (info-preserving) or "gaussian".
    seed: passed to node projection when new_dim is set.
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

    # If no projection requested, just return graphs as-is
    if not new_dim:
        return g_list

    g_list = project_with_vsa(g_list, new_dim, projection_type=projection_type, seed=seed)
    return g_list
'''
def project_node_features(g_list, original_feature_dim, new_dim):
    # Set a random seed for reproducibility
    torch.manual_seed(0)
    # Generate a random projection matrix
    # R = np.random.randn(original_feature_dim, new_dim) / np.sqrt(new_dim)
    # Initialize a random weight matrix for projection
    W = torch.randn(original_feature_dim, new_dim) / np.sqrt(new_dim)
    # print("W : ", W.shape)
    # Project node features for each graph

    print("g list item shape before : ", g_list[0].node_features.shape)
    for g in g_list:
        # Assuming g.node_features is a torch.Tensor
        if g.node_features is not None:
            # print(g.node_features)
            g.node_features  = torch.matmul(g.node_features, W)
            # print("new g.node_features : ",g.node_features.shape)
    print("g list item shape after : ", g_list[0].node_features.shape)
    return g_list

def VSA_conversion(g_list, new_dim=None):
    # Add labels and edge_mat
    for g in g_list:
        g.neighbors = [[] for _ in range(len(g.g))]

        # Build neighbors list
        for i, j in g.g.edges():
            g.neighbors[i].append(j)
            g.neighbors[j].append(i)

        # Compute max degree
        degree_list = [len(g.neighbors[i]) for i in range(len(g.g))]
        g.max_neighbor = max(degree_list)

        # Create edge matrix
        edges = [list(pair) for pair in g.g.edges()]
        edges.extend([[j, i] for i, j in edges])
        g.edge_mat = torch.LongTensor(edges).transpose(0, 1)

    #Extracting unique tag labels
    # tagset = set([])
    # for g in g_list:
    #     tagset = tagset.union(set(g.node_tags))

    # tagset = list(tagset)
    # tag2index = {tagset[i]:i for i in range(len(tagset))}


    ########## This part make one hit encoding of each node as they contain different atoms
    # for g in g_list:
    #     g.node_features = torch.zeros(len(g.node_tags), len(tagset))
    #     g.node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1
            # hypervector[range(len(g.node_tags)), [tag2index[tag] for tag in node_tags if tag in tag2index]] = 1

    original_feature_dim = len(g_list[0].node_features[0])# len(tagset)
    # print(len(tagset))
    print("VSA_conversion",len(g_list[0].node_features[0]))


    if new_dim:
        g_list = project_node_features(g_list, original_feature_dim, new_dim)
    return g_list
    '''