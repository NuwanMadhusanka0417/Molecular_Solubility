
import torch
import numpy as np

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