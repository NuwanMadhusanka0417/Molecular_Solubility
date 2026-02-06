"""
Compare equation variants: 10 (baseline), 12 (adaptive rotation), 13 (edge strength),
14 (directional), 15 (full). Reports MAE, RMSE, R2 for each.

Recommended order to try: 10 -> 12 -> 13 -> 15. Equation 12 is zero-cost; 13 uses
edge features; 15 combines all improvements.
"""

from src.create_graphs import create_graph_list
from src.load_data import load_data
from src.VSA_conversion import VSA_conversion
from src.embeddings import getEmbedding
from models.graphcnnVSA_Binding_FULL import GraphCNN
import torch
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def run_eval(equation, dim=5000):
    train_data, test_data = load_data()
    train_graphs = create_graph_list(train_data)
    test_graphs = create_graph_list(test_data)
    ts_graph = test_graphs.copy()
    tr_graph = train_graphs.copy()

    test_HVs = VSA_conversion(ts_graph, dim, projection_type="orthogonal")
    train_HVs = VSA_conversion(tr_graph, dim, projection_type="orthogonal")

    model = GraphCNN(
        test_HVs[0].node_features.shape[1], num_layers=5, delta=1,
        graph_pooling_type="sum", neighbor_pooling_type="sum", device=torch.device("cpu"),
        equation=equation, edge_feat_dim=5, edge_projection_type="orthogonal",
    )

    train_emb, train_labels = getEmbedding(
        model, torch.device("cpu"), train_HVs, use_size_aware=True
    )
    test_emb, test_labels = getEmbedding(
        model, torch.device("cpu"), test_HVs, use_size_aware=True
    )

    train_emb = train_emb.squeeze(0)
    test_emb = test_emb.squeeze(0)

    xgb = XGBRegressor(
        n_estimators=2000, learning_rate=0.03, max_depth=7,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, reg_alpha=0.0,
        random_state=42, n_jobs=4, tree_method="hist",
    )
    xgb.fit(train_emb, train_labels, eval_set=[(test_emb, test_labels)], verbose=False)

    pred = xgb.predict(test_emb)
    mae = mean_absolute_error(test_labels, pred)
    rmse = mean_squared_error(test_labels, pred) ** 0.5
    r2 = r2_score(test_labels, pred)
    return mae, rmse, r2


def main():
    dim = 5000
    equations = [
        (10, "Baseline (fixed shift, bind+residual)"),
        (12, "Adaptive rotation (shift=1+layer)"),
        (13, "Edge strength modulation"),
        (14, "Directional binding"),
        (15, "Full (directional + strength + attention)"),
    ]
    print("=" * 70)
    print("Equation comparison (HV dim={})".format(dim))
    print("=" * 70)

    baseline_mae = None
    for eq, desc in equations:
        print("\nEquation {}: {}".format(eq, desc))
        mae, rmse, r2 = run_eval(eq, dim=dim)
        if baseline_mae is None:
            baseline_mae = mae
        imp = (baseline_mae - mae) / baseline_mae * 100 if baseline_mae else 0
        print("  MAE={:.4f}  RMSE={:.4f}  R2={:.4f}  vs eq10: {:+.1f}%".format(mae, rmse, r2, imp))

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
