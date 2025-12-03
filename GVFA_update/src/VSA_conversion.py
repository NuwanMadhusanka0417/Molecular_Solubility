
import torch
import numpy as np
'''
def project_node_features(g_list, original_feature_dim, new_dim):
    # Set a random seed for reproducibility
    torch.manual_seed(0)
    # Generate a random projection matrix
    # R = np.random.randn(original_feature_dim, new_dim) / np.sqrt(new_dim)
    # Initialize a random weight matrix for projection
    W = torch.randn(original_feature_dim, new_dim) / np.sqrt(new_dim)
    print("W : ", W.shape)
    # Project node features for each graph
    for g in g_list:
        # Assuming g.node_features is a torch.Tensor
        if g.node_features is not None:
            # print(g.node_features)
            g.node_features  = torch.matmul(g.node_features, W)
            # print(g.node_features.shape)

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

    original_feature_dim = len(g_list[0].node_features[0])# len(tagset)
    # print(len(tagset))
    print("VSA_conversion",len(g_list[0].node_features[0]))


    if new_dim:
        g_list = project_node_features(g_list, original_feature_dim, new_dim)
    return g_list'''

def make_random_W(in_dim, out_dim, seed=0, device="cpu"):
    g = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(in_dim, out_dim, generator=g, device=device) / (out_dim ** 0.5)

def VSA_conversion(g_list, new_dim=None, seed=0):
    # do nothing to neighbors/edges here
    W = None
    if new_dim is not None:
        dx = g_list[0].node_features.size(1)
        W = make_random_W(dx, new_dim, seed=seed, device=g_list[0].node_features.device)
        for g in g_list:
            # keep original features; store projection separately
            # g.node_hv = g.node_features @ W     # [N, new_dim]
            g.node_hv  = torch.matmul(g.node_features, W)
    return g_list, W
