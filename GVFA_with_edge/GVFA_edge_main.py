"""
GVFA for molecular solubility prediction.

Encoder uses FHRR (complex phasors, element-wise bind); graph embeddings are real with
real/imag interleaved (2x hypervector dimension per complex block).

GVFA encoder -> embeddings -> Ridge/XGBoost (no training).

Train: solubility_1.csv.  Test: testset_novel.csv.
"""
import argparse
import csv
import os
import random
from datetime import datetime

from src.create_graphs import create_graph_list
from src.load_data import load_data
from src.VSA_conversion import VSA_conversion
from src.embeddings import getEmbedding
from models.graphcnnVSA_Binding_FULL import GraphCNN

import torch
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tensor_to_np_analysis(t):
    """FHRR tensors may be complex; preserve complex64 in .npz exports."""
    x = t.detach().cpu().numpy()
    if x.dtype.kind == "c":
        return x.astype(np.complex64)
    return x.astype(np.float32)


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
            payload[f"layer_{li}_pre_bin"] = _tensor_to_np_analysis(t)
        for li, t in enumerate(aux["layer_post_bin"]):
            payload[f"layer_{li}_post_bin"] = _tensor_to_np_analysis(t)
        if aux.get("F1_tap") is not None:
            payload["F1_tap"] = _tensor_to_np_analysis(aux["F1_tap"])
        if aux.get("sigma_pi_terms") is not None:
            for order, ten in aux["sigma_pi_terms"].items():
                payload[f"sigma_pi_order_{order}"] = _tensor_to_np_analysis(ten)
        if aux.get("sigma_pi_combined") is not None:
            payload["sigma_pi_combined"] = _tensor_to_np_analysis(aux["sigma_pi_combined"])
        path = os.path.join(out_dir, f"{split_name}_batch_{batch_ix:05d}.npz")
        np.savez_compressed(path, **payload)
        batch_ix += 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='GVFA for molecular solubility')
    p.add_argument('--dataset', type=str, default='solubility_novel',
                   choices=['old', 'solubility_novel', 'new'],
                   help='solubility_novel: train solubility_1.csv, test testset_novel.csv')
    p.add_argument('--dim', type=int, default=1000, help='FHRR hypervector dimension D (embeddings use 2*D real/imag per block)')
    p.add_argument('--dims', type=str, default='1000, 2000, 5000, 10000',
                   help='Comma-separated dims for gvfa_ridge loop')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--use_ridge', action='store_true', default=True)
    p.add_argument('--no_ridge', action='store_false', dest='use_ridge')
    p.add_argument('--seed', type=int, default=42, help='Single RNG seed (PyTorch, NumPy, Python, VSA/edge init, XGB; train/test split when dataset=old)')
    p.add_argument('--seeds', type=str, default=None,
                   help='Comma-separated seeds for multi-seed runs, e.g. "42,43,44". '
                        'Overrides --seed. Results are printed per seed, then mean +/- std summary.')
    p.add_argument(
        '--sigma_pi',
        type=str,
        default='all',
        help='Sigma-Pi order sets per --dims value. Presets: "all" = [0],[1],[2],[0,1],[0,1,2]; '
             '"legacy" = [0,1] only. Or one set as comma-separated orders, e.g. "0,1,2".',
    )
    p.add_argument(
        '--export_analysis_dir', type=str, default=None,
        help='If set, save per-batch .npz files: GVFA layer pre/post FHRR torus HV, '
             'sigma-pi per order + combined, F1 tap buffer, y (logS), and node graph ids.',
    )
    p.add_argument('--save_csv', action='store_true', default=False,
                   help='Save all results (per seed/dim/sigma_pi) and summary stats to a '
                        'timestamped CSV file in the results/ directory.')
    return p.parse_args()


# ---------------------------------------------------------------------------
# GVFA + Ridge/XGBoost
# ---------------------------------------------------------------------------

def _parse_sigma_pi_arg(s: str):
    """Return list of (sigma_pi_orders, tag) for the ridge loop."""
    key = s.strip().lower()
    if key == 'all':
        return [
            ([0], 'o0'),
            ([1], 'o1'),
            ([2], 'o2'),
            ([0, 1], 'o0_o1'),
            ([0, 1, 2], 'o0_o1_o2'),
        ]
    if key == 'legacy':
        return [([0, 1], 'o0_o1')]
    orders = [int(x.strip()) for x in s.split(',') if x.strip() != '']
    if not orders:
        raise ValueError(f'Invalid --sigma_pi: {s!r}')
    tag = 'o' + '_o'.join(str(t) for t in orders)
    return [(orders, tag)]


def run_gvfa_ridge(args, train_data, test_data, device, seed):
    """Run the GVFA + Ridge/XGBoost pipeline for a single seed.

    Returns a list of result dicts: [{dim, sigma_tag, seed, **metrics}].
    """
    dims = [int(x) for x in args.dims.split(',')]
    sigma_configs = _parse_sigma_pi_arg(args.sigma_pi)
    results = []

    for dim in dims:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        train_graphs = create_graph_list(train_data)
        test_graphs = create_graph_list(test_data)
        test_HVs = VSA_conversion(
            test_graphs.copy(), dim, projection_type="orthogonal", seed=seed,
        )
        train_HVs = VSA_conversion(
            train_graphs.copy(), dim, projection_type="orthogonal", seed=seed,
        )

        for sigma_pi_orders, sigma_tag in sigma_configs:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            model_eq1 = GraphCNN(
                test_HVs[0].node_features.shape[1], 5, 1, 'sum', 'sum', device, 10,
                edge_feat_dim=5, edge_projection_type="orthogonal",
                use_reservoir=True, hop_decay=0.85, sigma_pi_orders=sigma_pi_orders,
                rng_seed=seed,
            )
            train_emb, train_labels = getEmbedding(
                model_eq1, device, train_HVs, use_size_aware=True, hop_alpha=1.0,
            )
            test_emb, test_labels = getEmbedding(
                model_eq1, device, test_HVs, use_size_aware=True, hop_alpha=1.0,
            )
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
                    random_state=seed, n_jobs=4, tree_method="hist",
                )
                reg.fit(train_emb, train_labels, eval_set=[(test_emb, test_labels)], verbose=False)
                pred = reg.predict(test_emb)

            m = compute_metrics(test_labels, pred)
            orders_str = ','.join(str(x) for x in sigma_pi_orders)
            print(
                f"Seed={seed}  Dim={dim}  sigma_pi=[{orders_str}]  ({sigma_tag})  "
                f"RMSE={m['rmse']:.4f}  STD_err={m['std_err']:.4f}  MAE={m['mae']:.4f}  "
                f"R2_COD={m['r2_cod']:.4f}  Pearson_R2={m['pearson_r2']:.4f}",
            )
            results.append({"dim": dim, "sigma_tag": sigma_tag, "seed": seed, **m})

            if args.export_analysis_dir:
                sub = os.path.join(
                    args.export_analysis_dir, f"ridge_dim_{dim}_sigma_{sigma_tag}_seed_{seed}",
                )
                export_hypervector_analysis(
                    model_eq1, train_graphs, device, sub, "train", args.batch_size,
                )
                export_hypervector_analysis(
                    model_eq1, test_graphs, device, sub, "test", args.batch_size,
                )
                print(f"  Saved HV analysis under {sub}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _print_multi_seed_summary(all_results):
    """Print mean +/- std across seeds for each (dim, sigma_tag) configuration."""
    from collections import defaultdict
    groups = defaultdict(list)
    for r in all_results:
        groups[(r["dim"], r["sigma_tag"])].append(r)

    metric_keys = ["rmse", "std_err", "mae", "r2_cod", "pearson_r2"]
    print("\n" + "=" * 90)
    print("MULTI-SEED SUMMARY  (mean +/- std)")
    print("=" * 90)
    for (dim, sigma_tag), runs in sorted(groups.items()):
        seeds_used = [r["seed"] for r in runs]
        parts = [f"Dim={dim}  ({sigma_tag})  seeds={seeds_used}"]
        for k in metric_keys:
            vals = np.array([r[k] for r in runs])
            parts.append(f"{k}={vals.mean():.4f}+/-{vals.std():.4f}")
        print("  ".join(parts))
    print("=" * 90)


def _save_results_csv(all_results, args):
    """Write per-run rows and (if multi-seed) summary rows to a timestamped CSV."""
    from collections import defaultdict

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"gvfa_results_{args.dataset}_{ts}.csv"
    filepath = os.path.join(results_dir, filename)

    metric_keys = ["rmse", "std_err", "mae", "r2_cod", "pearson_r2", "pearson_r"]
    header = ["type", "seed", "dim", "sigma_tag"] + metric_keys

    rows = []
    for r in all_results:
        rows.append(["per_seed", r["seed"], r["dim"], r["sigma_tag"]]
                     + [f"{r[k]:.6f}" for k in metric_keys])

    seeds = sorted(set(r["seed"] for r in all_results))
    if len(seeds) > 1:
        groups = defaultdict(list)
        for r in all_results:
            groups[(r["dim"], r["sigma_tag"])].append(r)
        for (dim, sigma_tag), runs in sorted(groups.items()):
            means = {k: np.mean([r[k] for r in runs]) for k in metric_keys}
            stds  = {k: np.std([r[k] for r in runs])  for k in metric_keys}
            rows.append(["mean", "", dim, sigma_tag]
                         + [f"{means[k]:.6f}" for k in metric_keys])
            rows.append(["std", "", dim, sigma_tag]
                         + [f"{stds[k]:.6f}" for k in metric_keys])

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"\nResults saved to: {filepath}")


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Running for dataset: ", args.dataset)

    if args.seeds is not None:
        seeds = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]
    else:
        seeds = [args.seed]

    all_results = []
    for seed in seeds:
        print(f"\n{'─' * 60}")
        print(f"  Seed = {seed}")
        print(f"{'─' * 60}")
        train_data, test_data = load_data(dataset=args.dataset, seed=seed)
        results = run_gvfa_ridge(args, train_data, test_data, device, seed=seed)
        all_results.extend(results)

    if len(seeds) > 1:
        _print_multi_seed_summary(all_results)

    if args.save_csv:
        _save_results_csv(all_results, args)


if __name__ == '__main__':
    main()
