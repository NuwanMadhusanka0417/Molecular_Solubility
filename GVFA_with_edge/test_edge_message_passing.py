"""
Minimal test for edge-conditioned message passing.
Run with: python test_edge_message_passing.py  (or your env's python)
"""
from src.create_graphs import create_graph_list
from src.load_data import load_data
from src.VSA_conversion import VSA_conversion, get_feature_stats
from models.graphcnnVSA_Binding_FULL import GraphCNN
import torch

def main():
    train_data, test_data = load_data()
    n_train, n_test = 5, 3
    train_graphs = create_graph_list([train_data[i] for i in range(n_train)])
    test_graphs = create_graph_list([test_data[i] for i in range(n_test)])
    train_feat_mean, train_feat_std = get_feature_stats(train_graphs)
    train_HVs = VSA_conversion(
        train_graphs, 100, feat_mean=train_feat_mean, feat_std=train_feat_std,
    )
    test_HVs = VSA_conversion(
        test_graphs, 100, feat_mean=train_feat_mean, feat_std=train_feat_std,
    )

    model = GraphCNN(100, 3, 1, "sum", "sum", torch.device("cpu"), 10, edge_feat_dim=5)
    out = model(train_HVs)
    print("Output shape:", out.shape)
    print("Expected: (num_layers, batch_size, dim) = (3, 5, 100)")
    print("OK")

if __name__ == "__main__":
    main()
