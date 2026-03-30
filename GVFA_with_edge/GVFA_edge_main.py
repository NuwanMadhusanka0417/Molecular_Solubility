"""
GVFA for molecular solubility prediction.

Models:
  gvfa_ridge:  GVFA encoder -> embeddings -> Ridge/XGBoost (no training).
  attn_gvfa:   GVFA encoder (frozen) + learned attention readout + MLP regressor.

Train: solubility_1.csv.  Test: testset_novel.csv.
"""
import argparse
import os

from src.create_graphs import create_graph_list
from src.load_data import load_data
from src.VSA_conversion import VSA_conversion
from src.embeddings import getEmbedding
from models.graphcnnVSA_Binding_FULL import GraphCNN
from models.attn_gvfa_readout import AttnGVFARegressor

import torch
import torch.nn as nn
import numpy as np
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_labels(graphs, device=None):
    """Extract labels [N] from list of S2VGraph."""
    labels = torch.tensor(
        [float(torch.as_tensor(g.label).item()) for g in graphs],
        dtype=torch.float32,
    )
    if device is not None:
        labels = labels.to(device)
    return labels


def compute_metrics(y_true, y_pred):
    """RMSE, MAE, std of residuals, R² (COD), Pearson R² — all on same arrays in original logS units."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    err = y_true - y_pred
    rmse = np.sqrt(np.mean(err ** 2))
    std_err = float(np.std(err, ddof=0))
    mae = np.mean(np.abs(err))
    sse = np.sum(err ** 2)
    sst = np.sum((y_true - np.mean(y_true)) ** 2)
    r2_cod = 1.0 - (sse / sst) if sst > 0 else 0.0
    pr, _ = pearsonr(y_true, y_pred) if len(y_true) >= 2 else (0.0, 1.0)
    return {
        "rmse": rmse,
        "std_err": std_err,
        "mae": mae,
        "r2_cod": r2_cod,
        "pearson_r2": pr ** 2,
        "pearson_r": pr,
    }


def print_split_stats(name, graphs):
    labels = np.array([float(torch.as_tensor(g.label).item()) for g in graphs])
    print(f"  {name}: n={len(labels)}, logS range=[{labels.min():.2f}, {labels.max():.2f}], "
          f"mean={labels.mean():.2f}, std={labels.std():.2f}")


@torch.no_grad()
def export_hypervector_analysis(encoder, graphs, device, out_dir, split_name, batch_size):
    """
    Save one compressed .npz per batch with node-level hypervectors and solubility labels.
    encoder: GraphCNN with use_reservoir=True (same forward as training; capture_aux only adds storage).
    """
    os.makedirs(out_dir, exist_ok=True)
    encoder.eval()
    batch_ix = 0
    for start in range(0, len(graphs), batch_size):
        bg = graphs[start : start + batch_size]
        y = np.array([float(torch.as_tensor(g.label).item()) for g in bg], dtype=np.float32)
        _ = encoder(bg, return_node_rep=True, capture_aux=True)
        aux = getattr(encoder, "_aux", None)
        if aux is None:
            continue
        payload = {"y": y, "start_idx": np.array(aux["start_idx"], dtype=np.int64)}
        if aux.get("batch_node_graph_id") is not None:
            payload["batch_node_graph_id"] = aux["batch_node_graph_id"].numpy()
        for li, t in enumerate(aux["layer_pre_bin"]):
            payload[f"layer_{li}_pre_bin"] = t.cpu().numpy().astype(np.float32)
        for li, t in enumerate(aux["layer_post_bin"]):
            payload[f"layer_{li}_post_bin"] = t.cpu().numpy().astype(np.float32)
        if aux.get("F1_tap") is not None:
            payload["F1_tap"] = aux["F1_tap"].cpu().numpy().astype(np.float32)
        if aux.get("sigma_pi_terms") is not None:
            for order, ten in aux["sigma_pi_terms"].items():
                payload[f"sigma_pi_order_{order}"] = ten.cpu().numpy().astype(np.float32)
        if aux.get("sigma_pi_combined") is not None:
            payload["sigma_pi_combined"] = aux["sigma_pi_combined"].cpu().numpy().astype(np.float32)
        path = os.path.join(out_dir, f"{split_name}_batch_{batch_ix:05d}.npz")
        np.savez_compressed(path, **payload)
        batch_ix += 1


@torch.no_grad()
def predict_batched(model, graphs, batch_size, device):
    """Run model on graphs in batches, return (y_pred [N], y_true [N]) numpy in original logS."""
    model.eval()
    preds, truths = [], []
    for start in range(0, len(graphs), batch_size):
        bg = graphs[start : start + batch_size]
        yhat = model(bg)
        preds.append(yhat.cpu().numpy().ravel())
        truths.append(np.array([float(torch.as_tensor(g.label).item()) for g in bg]))
    return np.concatenate(preds), np.concatenate(truths)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='GVFA for molecular solubility')
    p.add_argument('--model', type=str, default='gvfa_ridge', choices=['gvfa_ridge', 'attn_gvfa'])
    p.add_argument('--dataset', type=str, default='solubility_novel',
                   choices=['old', 'solubility_novel', 'new'],
                   help='solubility_novel: train solubility_1.csv, test testset_novel.csv')
    p.add_argument('--dim', type=int, default=1000, help='VSA dimension')
    p.add_argument('--dims', type=str, default='1000, 2000, 5000, 7000, 9000, 10000, 15000',
                   help='Comma-separated dims for gvfa_ridge loop')
    p.add_argument('--epochs', type=int, default=200, help='Max epochs for attn_gvfa')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=5e-4, help='Initial learning rate')
    p.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    p.add_argument('--dropout', type=float, default=0.15)
    p.add_argument('--save_dir', type=str, default='checkpoints')
    p.add_argument('--use_ridge', action='store_true', default=True)
    p.add_argument('--no_ridge', action='store_false', dest='use_ridge')
    p.add_argument('--attn_heads', type=int, default=1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument(
        '--export_analysis_dir', type=str, default=None,
        help='If set, save per-batch .npz files: GVFA layer pre/post binarization HV, '
             'sigma-pi per order + combined, F1 tap buffer, y (logS), and node graph ids.',
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# GVFA + Ridge/XGBoost
# ---------------------------------------------------------------------------

def run_gvfa_ridge(args, train_data, test_data, device):
    dims = [int(x) for x in args.dims.split(',')]
    for dim in dims:
        train_graphs = create_graph_list(train_data)
        test_graphs = create_graph_list(test_data)
        test_HVs = VSA_conversion(test_graphs.copy(), dim, projection_type="orthogonal")
        train_HVs = VSA_conversion(train_graphs.copy(), dim, projection_type="orthogonal")

        model_eq1 = GraphCNN(
            test_HVs[0].node_features.shape[1], 5, 1, 'sum', 'sum', device, 10,
            edge_feat_dim=5, edge_projection_type="orthogonal",
            use_reservoir=True, hop_decay=0.85, sigma_pi_orders=[0, 1],
        )
        train_emb, train_labels = getEmbedding(model_eq1, device, train_HVs, use_size_aware=True, hop_alpha=1.0)
        test_emb, test_labels = getEmbedding(model_eq1, device, test_HVs, use_size_aware=True, hop_alpha=1.0)
        train_emb = train_emb.squeeze(0)
        test_emb = test_emb.squeeze(0)

        if args.use_ridge:
            reg = RidgeCV(alphas=np.logspace(-4, 2, 50), cv=5, scoring='neg_mean_squared_error')
            reg.fit(train_emb, train_labels)
            pred = reg.predict(test_emb)
        else:
            from xgboost import XGBRegressor
            reg = XGBRegressor(
                n_estimators=2000, learning_rate=0.03, max_depth=7,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, reg_alpha=0.0,
                random_state=42, n_jobs=4, tree_method="hist",
            )
            reg.fit(train_emb, train_labels, eval_set=[(test_emb, test_labels)], verbose=False)
            pred = reg.predict(test_emb)

        m = compute_metrics(test_labels, pred)
        print(f"Dim={dim}  RMSE={m['rmse']:.4f}  STD_err={m['std_err']:.4f}  MAE={m['mae']:.4f}  "
              f"R2_COD={m['r2_cod']:.4f}  Pearson_R2={m['pearson_r2']:.4f}")
        if args.export_analysis_dir:
            sub = os.path.join(args.export_analysis_dir, f"ridge_dim_{dim}")
            export_hypervector_analysis(
                model_eq1, train_graphs, device, sub, "train", args.batch_size,
            )
            export_hypervector_analysis(
                model_eq1, test_graphs, device, sub, "test", args.batch_size,
            )
            print(f"  Saved HV analysis under {sub}")


# ---------------------------------------------------------------------------
# GVFA + Attention readout (trained)
# ---------------------------------------------------------------------------

def run_attn_gvfa(args, train_data, test_data, device):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- Build graphs and project to VSA space ----
    train_graphs = create_graph_list(train_data)
    test_graphs = create_graph_list(test_data)
    train_HVs = VSA_conversion(train_graphs, args.dim, projection_type="orthogonal")
    test_HVs = VSA_conversion(test_graphs, args.dim, projection_type="orthogonal")

    # ---- Train / validation split (90 / 10 from training set) ----
    n = len(train_HVs)
    indices = np.arange(n)
    tr_idx, val_idx = train_test_split(indices, test_size=0.1, random_state=args.seed, shuffle=True)
    tr_HVs = [train_HVs[i] for i in tr_idx]
    val_HVs = [train_HVs[i] for i in val_idx]

    print("\n--- Dataset statistics ---")
    print_split_stats("Train", tr_HVs)
    print_split_stats("Val", val_HVs)
    print_split_stats("Test (independent)", test_HVs)

    # ---- Target standardization (fit on train split only) ----
    tr_labels = np.array([float(torch.as_tensor(g.label).item()) for g in tr_HVs])
    y_mean, y_std = float(tr_labels.mean()), float(tr_labels.std())
    if y_std < 1e-8:
        y_std = 1.0
    print(f"Target standardization: mean={y_mean:.4f}, std={y_std:.4f}")

    # ---- Encoder (frozen) ----
    D = train_HVs[0].node_features.shape[1]
    encoder = GraphCNN(
        D, 5, 1, 'sum', 'sum', device, 10,
        edge_feat_dim=5, edge_projection_type="orthogonal",
        use_reservoir=True, hop_decay=0.85, sigma_pi_orders=[0, 1],
    )
    encoder.to(device)

    # ---- Model: attention readout + deeper MLP regressor ----
    model = AttnGVFARegressor(
        encoder, D,
        readout_hidden=128,
        regressor_hidden=64,
        dropout=args.dropout,
        use_layernorm=True,
        num_heads=args.attn_heads,
        regressor_hidden_dims=[256, 64],
    )
    model.to(device)

    trainable_params = sum(p.numel() for p in model.get_trainable_parameters())
    print(f"Trainable parameters: {trainable_params}")

    # ---- Optimizer, scheduler, loss ----
    optimizer = torch.optim.AdamW(model.get_trainable_parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=7, min_lr=1e-6,
    )
    criterion = nn.MSELoss()

    os.makedirs(args.save_dir, exist_ok=True)
    best_val_rmse = float('inf')
    best_state = None
    wait = 0
    best_path = os.path.join(args.save_dir, f"attn_gvfa_dim{args.dim}_best.pt")

    # ---- Training loop ----
    for epoch in range(args.epochs):
        model.train()
        perm = np.random.permutation(len(tr_HVs))
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, len(tr_HVs), args.batch_size):
            idx = perm[start : start + args.batch_size]
            bg = [tr_HVs[i] for i in idx]
            # Standardized targets
            raw_y = torch.tensor(
                [float(torch.as_tensor(tr_HVs[i].label).item()) for i in idx],
                dtype=torch.float32, device=device,
            ).unsqueeze(1)
            z_y = (raw_y - y_mean) / y_std

            optimizer.zero_grad()
            yhat = model(bg)
            loss = criterion(yhat, z_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.get_trainable_parameters(), max_norm=5.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        # ---- Validation (un-standardize predictions for metrics in original logS) ----
        val_pred_z, val_true = predict_batched(model, val_HVs, args.batch_size, device)
        val_pred = val_pred_z * y_std + y_mean
        val_m = compute_metrics(val_true, val_pred)
        scheduler.step(val_m["rmse"])

        if val_m["rmse"] < best_val_rmse:
            best_val_rmse = val_m["rmse"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch+1:3d}  loss={epoch_loss/n_batches:.4f}  "
                  f"val_RMSE={val_m['rmse']:.4f}  val_STD_err={val_m['std_err']:.4f}  "
                  f"val_R2={val_m['r2_cod']:.4f}  lr={lr_now:.2e}")

        if wait >= args.patience:
            print(f"Early stopping at epoch {epoch+1} (patience={args.patience})")
            break

    # ---- Restore best and save ----
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save({"model_state_dict": model.state_dict(), "y_mean": y_mean, "y_std": y_std}, best_path)
    print(f"Best val RMSE: {best_val_rmse:.4f}  (saved to {best_path})")

    # ---- Final evaluation on independent test set ----
    test_pred_z, test_true = predict_batched(model, test_HVs, args.batch_size, device)
    test_pred = test_pred_z * y_std + y_mean
    m = compute_metrics(test_true, test_pred)

    print("\n=== Independent Test Results ===")
    print(f"  RMSE:       {m['rmse']:.4f}")
    print(f"  STD (err):  {m['std_err']:.4f}")
    print(f"  MAE:        {m['mae']:.4f}")
    print(f"  R2 (COD):   {m['r2_cod']:.4f}")
    print(f"  Pearson R2: {m['pearson_r2']:.4f}")
    print(f"  y_test:  min={test_true.min():.2f}  max={test_true.max():.2f}  "
          f"mean={test_true.mean():.2f}  std={test_true.std():.2f}")
    print(f"  y_pred:  min={test_pred.min():.2f}  max={test_pred.max():.2f}  "
          f"mean={test_pred.mean():.2f}  std={test_pred.std():.2f}")

    if args.export_analysis_dir:
        base = os.path.join(args.export_analysis_dir, f"attn_gvfa_dim_{args.dim}")
        export_hypervector_analysis(model.encoder, tr_HVs, device, base, "train", args.batch_size)
        export_hypervector_analysis(model.encoder, val_HVs, device, base, "val", args.batch_size)
        export_hypervector_analysis(model.encoder, test_HVs, device, base, "test", args.batch_size)
        print(f"\nSaved HV / sigma-pi analysis under {base}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Running for dataset: ", args.dataset)
    train_data, test_data = load_data(dataset=args.dataset)

    if args.model == 'attn_gvfa':
        run_attn_gvfa(args, train_data, test_data, device)
    else:
        run_gvfa_ridge(args, train_data, test_data, device)


if __name__ == '__main__':
    main()
