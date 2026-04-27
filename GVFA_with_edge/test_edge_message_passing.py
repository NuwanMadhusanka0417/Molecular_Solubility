"""
Minimal test for edge-conditioned message passing.
Run with: python test_edge_message_passing.py  (or your env's python)
"""
from src.create_graphs import create_graph_list
from src.load_data import load_data
from src.VSA_conversion import VSA_conversion
from models.graphcnnVSA_Binding_FULL import GraphCNN
import torch

def main():
    train_data, test_data = load_data()
    n_train, n_test = 5, 3
    train_graphs = create_graph_list([train_data[i] for i in range(n_train)])
    test_graphs = create_graph_list([test_data[i] for i in range(n_test)])
    train_HVs, m, s = VSA_conversion(train_graphs, 100)
    test_HVs, _, _ = VSA_conversion(test_graphs, 100, feature_mean=m, feature_std=s)

    model = GraphCNN(100, 3, 1, "sum", "sum", torch.device("cpu"), 10, edge_feat_dim=5)
    _ea = [g.edge_attr for g in train_HVs if g.edge_attr is not None and g.edge_attr.numel() > 0]
    if _ea:
        all_e = torch.cat([e.to(torch.float32) for e in _ea], dim=0)
        model.set_edge_stats(all_e.mean(dim=0), all_e.std(dim=0).clamp(min=1e-6))
    out = model(train_HVs)
    print("Output shape:", out.shape)
    print("Expected: (num_layers, batch_size, dim) = (3, 5, 100)")
    print("OK")

if __name__ == "__main__":
    main()
