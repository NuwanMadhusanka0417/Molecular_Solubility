"""
Compare Baseline vs Resonator Consensus: measure impact of resonator refinement.

Runs the pipeline with use_resonator=False (baseline) and use_resonator=True,
then reports MAE, RMSE, R2 and relative improvement.

Expected: 5-15% improvement in MAE/RMSE with resonator consensus.
"""

from src.create_graphs import create_graph_list
from src.load_data import load_data
from src.VSA_conversion import VSA_conversion
from src.embeddings import getEmbedding
from models.graphcnnVSA_Binding_FULL import GraphCNN
import torch
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def run_eval(use_resonator, dim=5000):
    """Run full pipeline and return MAE, RMSE, R2."""
    train_data, test_data = load_data()
    train_graphs = create_graph_list(train_data)
    test_graphs = create_graph_list(test_data)
    ts_graph = test_graphs.copy()
    tr_graph = train_graphs.copy()

    train_HVs, node_feat_mean, node_feat_std = VSA_conversion(
        tr_graph, dim, projection_type="orthogonal", seed=42,
    )
    test_HVs, _, _ = VSA_conversion(
        ts_graph, dim, projection_type="orthogonal", seed=42,
        feature_mean=node_feat_mean, feature_std=node_feat_std,
    )

    model = GraphCNN(
        test_HVs[0].node_features.shape[1], num_layers=5, delta=1,
        graph_pooling_type="sum", neighbor_pooling_type="sum", device=torch.device("cpu"),
        equation=10, edge_feat_dim=5, edge_projection_type="orthogonal",
        use_resonator=use_resonator, resonator_iters=7, resonator_beta=0.75,
        rng_seed=42,
    )
    _edge_attrs = []
    for g in train_HVs:
        if g.edge_attr is not None and g.edge_attr.numel() > 0:
            _edge_attrs.append(g.edge_attr.to(torch.float32))
    if _edge_attrs:
        all_edge_feats = torch.cat(_edge_attrs, dim=0)
        model.set_edge_stats(all_edge_feats.mean(dim=0), all_edge_feats.std(dim=0).clamp(min=1e-6))

    train_emb, train_labels = getEmbedding(
        model, torch.device("cpu"), train_HVs, use_size_aware=True, hop_alpha=0.8
    )
    test_emb, test_labels = getEmbedding(
        model, torch.device("cpu"), test_HVs, use_size_aware=True, hop_alpha=0.8
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
    dim = 5000  # single dim for faster comparison
    print("=" * 60)
    print("Comparing Baseline vs Resonator Consensus")
    print(f"HV dim = {dim}, hop_alpha = 0.8")
    print("=" * 60)

    print("\n[1/2] Running BASELINE (use_resonator=False)...")
    mae_base, rmse_base, r2_base = run_eval(use_resonator=False, dim=dim)
    print(f"  MAE  = {mae_base:.4f}")
    print(f"  RMSE = {rmse_base:.4f}")
    print(f"  R2   = {r2_base:.4f}")

    print("\n[2/2] Running RESONATOR (use_resonator=True)...")
    mae_res, rmse_res, r2_res = run_eval(use_resonator=True, dim=dim)
    print(f"  MAE  = {mae_res:.4f}")
    print(f"  RMSE = {rmse_res:.4f}")
    print(f"  R2   = {r2_res:.4f}")

    print("\n" + "=" * 60)
    print("IMPROVEMENT (Resonator vs Baseline)")
    print("=" * 60)
    mae_imp = (mae_base - mae_res) / mae_base * 100
    rmse_imp = (rmse_base - rmse_res) / rmse_base * 100
    r2_delta = r2_res - r2_base
    print(f"  MAE:  {mae_imp:+.1f}% {'better' if mae_imp > 0 else 'worse'}")
    print(f"  RMSE: {rmse_imp:+.1f}% {'better' if rmse_imp > 0 else 'worse'}")
    print(f"  R2:   {r2_delta:+.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
