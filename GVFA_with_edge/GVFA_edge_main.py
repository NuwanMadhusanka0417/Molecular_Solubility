"""
GVFA for molecular solubility prediction.

Models:
  gvfa_ridge:  GVFA encoder -> embeddings -> Ridge/XGBoost (no training).
  attn_gvfa:   GVFA encoder (frozen) + learned attention readout + MLP regressor.

Train: solubility_1.csv.  Test: testset_novel.csv.
"""
import argparse
import json
import os

from src.create_graphs import create_graph_list
from src.hv_analysis_save import save_experiment_layout
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
    """RMSE, MAE, R² (COD), Pearson R² — all on same arrays in original logS units."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    sse = np.sum((y_true - y_pred) ** 2)
    sst = np.sum((y_true - np.mean(y_true)) ** 2)
    r2_cod = 1.0 - (sse / sst) if sst > 0 else 0.0
    pr, _ = pearsonr(y_true, y_pred) if len(y_true) >= 2 else (0.0, 1.0)
    return {"rmse": rmse, "mae": mae, "r2_cod": r2_cod, "pearson_r2": pr ** 2, "pearson_r": pr}


def print_split_stats(name, graphs):
    labels = np.array([float(torch.as_tensor(g.label).item()) for g in graphs])
    print(f"  {name}: n={len(labels)}, logS range=[{labels.min():.2f}, {labels.max():.2f}], "
          f"mean={labels.mean():.2f}, std={labels.std():.2f}")


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
    p.add_argument('--dims', type=str, default='1000, 2000, 5000, 10000',
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
    p.add_argument('--sigma_pi_orders', type=str, default='0,1',
                   help='Comma-separated sigma-pi orders, e.g. "0" for linear, "0,1" for 2nd-order')
    p.add_argument('--sigma_pi_mode', type=str, default='bundle', choices=['concat', 'bundle'],
                   help='How to combine sigma-pi terms: concat or bundle')
    p.add_argument(
        '--gvfa_binding_mode',
        type=str,
        default='circular_conv',
        choices=['circular_conv', 'hadamard'],
        help='GVFA layer binding (next_layer_eps): circular convolution (FFT, default) or element-wise multiply',
    )
    p.add_argument(
        '--other_binding_mode',
        type=str,
        default='circular_conv',
        choices=['circular_conv', 'hadamard'],
        help='Binding elsewhere: edges, sigma-pi, multi_stat_pool F∘F, resonator, legacy reservoir',
    )
    p.add_argument('--sweep_sigma_pi', action='store_true', default=False,
                   help='Sweep over multiple sigma-pi configs: linear, 2nd, 3rd, 4th order')
    p.add_argument('--full_sweep', action='store_true', default=False,
                   help='Run all sigma-pi experiments: individual orders + concat combos + bundle combos')
    p.add_argument('--save_analysis_data', action='store_true', default=False,
                   help='Save HV tensors at each pipeline stage (train + test) for offline analysis')
    p.add_argument('--analysis_save_dir', type=str, default='hv_analysis_exports',
                   help='Root directory for analysis exports (used when --save_analysis_data)')
    return p.parse_args()


# ---------------------------------------------------------------------------
# GVFA + Ridge/XGBoost
# ---------------------------------------------------------------------------

def _parse_sigma_pi_configs(args):
    """Return list of (orders, mode, label) tuples to run."""
    if args.full_sweep:
        configs = []
        for t in range(4):
            configs.append(([t], "concat", f"order-{t}-only"))
        for orders in [[0, 1], [0, 1, 2], [0, 1, 2, 3]]:
            label = f"orders-{','.join(map(str, orders))}-concat"
            configs.append((orders, "concat", label))
        for orders in [[0, 1], [0, 1, 2], [0, 1, 2, 3]]:
            label = f"orders-{','.join(map(str, orders))}-bundle"
            configs.append((orders, "bundle", label))
        return configs

    orders = [int(x) for x in args.sigma_pi_orders.split(',')]
    # Must match argparse default for --sigma_pi_mode ('bundle'), not 'concat'
    mode = getattr(args, 'sigma_pi_mode', 'bundle')
    if args.sweep_sigma_pi:
        configs = []
        for o in [[0], [0, 1], [0, 1, 2], [0, 1, 2, 3]]:
            label = f"orders-{','.join(map(str, o))}-{mode}"
            configs.append((o, mode, label))
        return configs

    label = f"orders-{','.join(map(str, orders))}-{mode}"
    return [(orders, mode, label)]


def _run_single_ridge(args, train_data, test_data, device, dim, sp_orders, sp_mode):
    """Run a single (dim, orders, mode) config. Returns dict with 'train' and 'test' metrics."""
    train_graphs = create_graph_list(train_data)
    test_graphs = create_graph_list(test_data)
    test_HVs = VSA_conversion(test_graphs.copy(), dim, projection_type="orthogonal", projection_seed=args.seed)
    train_HVs = VSA_conversion(train_graphs.copy(), dim, projection_type="orthogonal", projection_seed=args.seed)

    model_eq1 = GraphCNN(
        test_HVs[0].node_features.shape[1], 5, 1, 'sum', 'sum', device, 10,
        edge_feat_dim=5, edge_projection_type="orthogonal",
        use_reservoir=True, hop_decay=0.85,
        sigma_pi_orders=sp_orders, sigma_pi_mode=sp_mode,
        gvfa_binding_mode=args.gvfa_binding_mode,
        other_binding_mode=args.other_binding_mode,
    )
    train_emb, train_labels = getEmbedding(model_eq1, device, train_HVs, use_size_aware=True, hop_alpha=1.0)
    test_emb, test_labels = getEmbedding(model_eq1, device, test_HVs, use_size_aware=True, hop_alpha=1.0)
    train_emb = train_emb.squeeze(0)
    test_emb = test_emb.squeeze(0)

    if args.use_ridge:
        reg = RidgeCV(alphas=np.logspace(-4, 2, 50), cv=5, scoring='neg_mean_squared_error')
        reg.fit(train_emb, train_labels)
        pred_test = reg.predict(test_emb)
        pred_train = reg.predict(train_emb)
        reg_info = {"type": "RidgeCV", "alphas": "logspace(-4,2,50)", "cv": 5}
    else:
        from xgboost import XGBRegressor
        reg = XGBRegressor(
            n_estimators=2000, learning_rate=0.03, max_depth=7,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, reg_alpha=0.0,
            random_state=42, n_jobs=4, tree_method="hist",
        )
        reg.fit(train_emb, train_labels, eval_set=[(test_emb, test_labels)], verbose=False)
        pred_test = reg.predict(test_emb)
        pred_train = reg.predict(train_emb)
        reg_info = {"type": "XGBRegressor"}

    train_labels_np = train_labels.cpu().numpy() if torch.is_tensor(train_labels) else np.asarray(train_labels)
    test_labels_np = test_labels.cpu().numpy() if torch.is_tensor(test_labels) else np.asarray(test_labels)
    m_train = compute_metrics(train_labels_np, pred_train)
    m_test = compute_metrics(test_labels_np, pred_test)

    if getattr(args, "save_analysis_data", False):
        bs = getattr(args, "batch_size", 64)
        cfg = {
            "dataset": args.dataset,
            "sigma_pi_orders": sp_orders,
            "sigma_pi_mode": sp_mode,
            "gvfa_binding_mode": args.gvfa_binding_mode,
            "other_binding_mode": args.other_binding_mode,
            "equation": 10,
            "delta": 1,
            "num_layers": 5,
            "use_ridge": args.use_ridge,
        }
        exp_path = save_experiment_layout(
            args.analysis_save_dir,
            dim,
            args.seed,
            train_HVs,
            test_HVs,
            model_eq1,
            device,
            bs,
            cfg,
            train_labels_np,
            test_labels_np,
            m_train,
            m_test,
            reg_info=reg_info,
        )
        print(f"Saved HV experiment layout to {exp_path}")

    return {"train": m_train, "test": m_test}


def run_gvfa_ridge(args, train_data, test_data, device):
    dims = [int(x) for x in args.dims.split(',')]
    configs = _parse_sigma_pi_configs(args)
    all_results = []

    for sp_orders, sp_mode, sp_label in configs:
        print(f"\n{'='*70}")
        print(f"  Sigma-Pi config: orders={sp_orders}  mode={sp_mode}  ({sp_label})")
        print(f"{'='*70}")

        for dim in dims:
            m = _run_single_ridge(args, train_data, test_data, device, dim, sp_orders, sp_mode)
            tr, te = m["train"], m["test"]
            row = {
                "label": sp_label, "dim": dim, "orders": str(sp_orders), "mode": sp_mode,
                "rmse_train": tr["rmse"], "rmse_test": te["rmse"],
                "mae_train": tr["mae"], "mae_test": te["mae"],
                "r2_cod_train": tr["r2_cod"], "r2_cod_test": te["r2_cod"],
                "pearson_r2_train": tr["pearson_r2"], "pearson_r2_test": te["pearson_r2"],
            }
            all_results.append(row)
            print(f"  Dim={dim:>6d}  train_RMSE={tr['rmse']:.4f}  test_RMSE={te['rmse']:.4f}  "
                  f"train_R2={tr['r2_cod']:.4f}  test_R2={te['r2_cod']:.4f}")

    _print_summary_table(all_results)

    if getattr(args, "save_analysis_data", False) and all_results:
        os.makedirs(args.analysis_save_dir, exist_ok=True)
        summ_path = os.path.join(args.analysis_save_dir, f"summary_metrics_seed{args.seed}.json")
        def _json_safe(o):
            if isinstance(o, dict):
                return {k: _json_safe(v) for k, v in o.items()}
            if isinstance(o, list):
                return [_json_safe(v) for v in o]
            if isinstance(o, (np.floating, np.integer)):
                return o.item()
            return o

        with open(summ_path, "w", encoding="utf-8") as f:
            json.dump(
                _json_safe({"seed": args.seed, "dataset": args.dataset, "runs": all_results}),
                f,
                indent=2,
            )
        print(f"\nSaved aggregated train/test metrics to {summ_path}")


def _print_summary_table(all_results):
    """Print a formatted summary table of all experiment results."""
    print(f"\n\n{'#'*80}")
    print(f"{'SUMMARY OF ALL RESULTS':^80}")
    print(f"{'#'*80}")

    header = (f"{'Config':<28s} {'Dim':>5s}  {'trRMSE':>7s}  {'teRMSE':>7s}  "
              f"{'trR2':>7s}  {'teR2':>7s}  {'trP2':>7s}  {'teP2':>7s}")
    print(f"\n{header}")
    print("-" * len(header))
    for r in all_results:
        print(f"{r['label']:<28s} {r['dim']:>5d}  {r['rmse_train']:>7.4f}  {r['rmse_test']:>7.4f}  "
              f"{r['r2_cod_train']:>7.4f}  {r['r2_cod_test']:>7.4f}  "
              f"{r['pearson_r2_train']:>7.4f}  {r['pearson_r2_test']:>7.4f}")

    dims = sorted(set(r['dim'] for r in all_results))
    labels = list(dict.fromkeys(r['label'] for r in all_results))

    for dim in dims:
        print(f"\n--- Dim = {dim} ---")
        print(f"  {'Config':<33s} {'trRMSE':>7s} {'teRMSE':>7s} {'trR2':>7s} {'teR2':>7s}")
        print(f"  {'-'*68}")
        for label in labels:
            matches = [r for r in all_results if r['dim'] == dim and r['label'] == label]
            for r in matches:
                print(f"  {r['label']:<33s} {r['rmse_train']:>7.4f} {r['rmse_test']:>7.4f} "
                      f"{r['r2_cod_train']:>7.4f} {r['r2_cod_test']:>7.4f}")


# ---------------------------------------------------------------------------
# GVFA + Attention readout (trained)
# ---------------------------------------------------------------------------

def run_attn_gvfa(args, train_data, test_data, device):
    configs = _parse_sigma_pi_configs(args)

    for sp_orders, sp_mode, sp_label in configs:
        print(f"\n{'='*70}")
        print(f"  Sigma-Pi config: orders={sp_orders}  mode={sp_mode}  ({sp_label})")
        print(f"{'='*70}")
        _run_attn_gvfa_single(args, train_data, test_data, device, sp_orders, sp_mode, sp_label)


def _run_attn_gvfa_single(args, train_data, test_data, device, sp_orders, sp_mode, sp_label):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_graphs = create_graph_list(train_data)
    test_graphs = create_graph_list(test_data)
    train_HVs = VSA_conversion(train_graphs, args.dim, projection_type="orthogonal", projection_seed=args.seed)
    test_HVs = VSA_conversion(test_graphs, args.dim, projection_type="orthogonal", projection_seed=args.seed)

    n = len(train_HVs)
    indices = np.arange(n)
    tr_idx, val_idx = train_test_split(indices, test_size=0.1, random_state=args.seed, shuffle=True)
    tr_HVs = [train_HVs[i] for i in tr_idx]
    val_HVs = [train_HVs[i] for i in val_idx]

    print("\n--- Dataset statistics ---")
    print_split_stats("Train", tr_HVs)
    print_split_stats("Val", val_HVs)
    print_split_stats("Test (independent)", test_HVs)

    tr_labels = np.array([float(torch.as_tensor(g.label).item()) for g in tr_HVs])
    y_mean, y_std = float(tr_labels.mean()), float(tr_labels.std())
    if y_std < 1e-8:
        y_std = 1.0
    print(f"Target standardization: mean={y_mean:.4f}, std={y_std:.4f}")

    D = train_HVs[0].node_features.shape[1]
    encoder = GraphCNN(
        D, 5, 1, 'sum', 'sum', device, 10,
        edge_feat_dim=5, edge_projection_type="orthogonal",
        use_reservoir=True, hop_decay=0.85,
        sigma_pi_orders=sp_orders, sigma_pi_mode=sp_mode,
        gvfa_binding_mode=args.gvfa_binding_mode,
        other_binding_mode=args.other_binding_mode,
    )
    encoder.to(device)
    D_out = encoder.output_node_dim
    print(f"Encoder output node dim: {D_out} (T={len(sp_orders)} x D={D}, mode={sp_mode})")

    # ---- Model: attention readout + deeper MLP regressor ----
    model = AttnGVFARegressor(
        encoder, D_out,
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
    best_path = os.path.join(args.save_dir, f"attn_gvfa_dim{args.dim}_{sp_label}_best.pt")

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
                  f"val_RMSE={val_m['rmse']:.4f}  val_R2={val_m['r2_cod']:.4f}  lr={lr_now:.2e}")

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

    print(f"\n=== Independent Test Results [{sp_label}] ===")
    print(f"  RMSE:       {m['rmse']:.4f}")
    print(f"  MAE:        {m['mae']:.4f}")
    print(f"  R2 (COD):   {m['r2_cod']:.4f}")
    print(f"  Pearson R2: {m['pearson_r2']:.4f}")
    print(f"  y_test:  min={test_true.min():.2f}  max={test_true.max():.2f}  "
          f"mean={test_true.mean():.2f}  std={test_true.std():.2f}")
    print(f"  y_pred:  min={test_pred.min():.2f}  max={test_pred.max():.2f}  "
          f"mean={test_pred.mean():.2f}  std={test_pred.std():.2f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Running for dataset: ", args.dataset)
    train_data, test_data = load_data(dataset=args.dataset)

    if args.model == 'attn_gvfa':
        run_attn_gvfa(args, train_data, test_data, device)
    else:
        run_gvfa_ridge(args, train_data, test_data, device)


if __name__ == '__main__':
    main()
