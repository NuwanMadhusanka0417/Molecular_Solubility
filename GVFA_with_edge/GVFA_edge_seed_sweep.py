"""
GVFA seed sweep with train / validation / test protocol.

Phase 1 – Sweep:
    Split training CSV into train (85%) and validation (15%).
    For every (seed, dim, sigma_pi) combo, build GVFA embeddings,
    fit Ridge on train split, evaluate on validation split.

Phase 2 – Final evaluation:
    Take the best config (lowest validation RMSE).
    Retrain Ridge on the FULL training data (train+val) with that config.
    Evaluate once on the held-out test set and report.

Usage:
    python GVFA_edge_seed_sweep.py --seeds 0-49 --dims 1000,2000,5000,10000
    python GVFA_edge_seed_sweep.py --seeds 0,1,42,123 --dims 2000,5000
"""
import argparse
import csv
import os
import pickle
import random
import time

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split

from src.create_graphs import create_graph_list
from src.load_data import load_data, ZINCLikeCSV
from src.VSA_conversion import _random_projection_matrix
from src.embeddings import getEmbedding
from models.graphcnnVSA_Binding_FULL import GraphCNN


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred):
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
        "rmse": rmse, "std_err": std_err, "mae": mae,
        "r2_cod": r2_cod, "pearson_r2": pr ** 2, "pearson_r": pr,
    }


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_seeds(s: str):
    """Parse '0-49' or '0,1,42,123' into a list of ints."""
    s = s.strip()
    if '-' in s and ',' not in s:
        lo, hi = s.split('-')
        return list(range(int(lo), int(hi) + 1))
    return [int(x.strip()) for x in s.split(',') if x.strip()]


def parse_sigma_pi(s: str):
    key = s.strip().lower()
    if key == 'all':
        return [
            ([0], 'o0'), ([1], 'o1'), ([2], 'o2'),
            ([0, 1], 'o0_o1'), ([0, 1, 2], 'o0_o1_o2'),
        ]
    if key == 'legacy':
        return [([0, 1], 'o0_o1')]
    orders = [int(x.strip()) for x in s.split(',') if x.strip()]
    tag = 'o' + '_o'.join(str(t) for t in orders)
    return [(orders, tag)]


def precompute_graphs(data, cache_path=None):
    """
    Build graph list + neighbor/edge_mat ONCE.  This is the expensive step
    (RDKit 3D conformers, atom features).  The result is seed/dim independent.
    Optionally load from / save to a pickle cache on disk.
    """
    if cache_path and os.path.exists(cache_path):
        print(f"  Loading cached graphs from {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    graphs = create_graph_list(data)

    for g in graphs:
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

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(graphs, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  Saved cached graphs to {cache_path}")

    return graphs


def build_embeddings_fast(graphs, original_features, dim, seed,
                          sigma_pi_orders, device):
    """
    Project cached graphs and extract embeddings.  Skips create_graph_list
    and neighbor-building entirely — only does the cheap projection + encoder.

    graphs:            pre-computed graph list (from precompute_graphs)
    original_features: list of [N_i, F_node] tensors (saved once before loop)
    """
    set_all_seeds(seed)

    W_node = _random_projection_matrix(
        original_features[0].shape[1], dim, orthogonal=True, seed=seed,
    )
    for g, orig_feat in zip(graphs, original_features):
        g.node_features = torch.matmul(orig_feat, W_node)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    encoder = GraphCNN(
        dim, 5, 1, 'sum', 'sum', device, 10,
        edge_feat_dim=5, edge_projection_type="orthogonal",
        use_reservoir=True, hop_decay=0.85, sigma_pi_orders=sigma_pi_orders,
        rng_seed=seed,
    )
    emb, labels = getEmbedding(
        encoder, device, graphs, use_size_aware=True, hop_alpha=1.0,
    )
    return emb.squeeze(0), labels


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='GVFA seed sweep with train/val/test')
    p.add_argument('--dataset', type=str, default='solubility_novel',
                   choices=['old', 'solubility_novel', 'new'])
    p.add_argument('--seeds', type=str, default='0-49',
                   help='Seed range "0-49" or comma-separated "0,1,42,123"')
    p.add_argument('--dims', type=str, default='1000,2000,5000,10000',
                   help='Comma-separated VSA dimensions')
    p.add_argument('--sigma_pi', type=str, default='all',
                   help='Sigma-Pi config: "all", "legacy", or "0,1,2"')
    p.add_argument('--val_size', type=float, default=0.15,
                   help='Fraction of training data for validation (default 0.15)')
    p.add_argument('--val_seed', type=int, default=42,
                   help='Seed for the train/val split (kept fixed across all experiments)')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--save_dir', type=str, default='seed_sweep_results',
                   help='Directory for output CSVs')
    p.add_argument('--top_k', type=int, default=5,
                   help='Show top-K configs at the end')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)

    seeds = parse_seeds(args.seeds)
    dims = [int(x) for x in args.dims.split(',')]
    sigma_configs = parse_sigma_pi(args.sigma_pi)

    print(f"Seeds: {len(seeds)} values  ({seeds[0]}..{seeds[-1]})")
    print(f"Dims:  {dims}")
    print(f"Sigma-Pi configs: {[tag for _, tag in sigma_configs]}")
    total_combos = len(seeds) * len(dims) * len(sigma_configs)
    print(f"Total combinations: {total_combos}")
    print()

    # ------------------------------------------------------------------
    # Load data & create the fixed train / val split
    # ------------------------------------------------------------------
    if args.dataset == 'solubility_novel':
        train_path = "final_data/solubility_1.csv"
        test_path = "final_data/testset_novel.csv"
        train_df = pd.read_csv(train_path).dropna(subset=["SMILES", "logS"])
        test_df = pd.read_csv(test_path).dropna(subset=["SMILES", "logS"])

        train_sub_df, val_df = train_test_split(
            train_df, test_size=args.val_size,
            random_state=args.val_seed, shuffle=True,
        )

        train_sub_data = ZINCLikeCSV(train_sub_df, smiles_col="SMILES", target_col="logS")
        val_data = ZINCLikeCSV(val_df, smiles_col="SMILES", target_col="logS")
        full_train_data = ZINCLikeCSV(train_df, smiles_col="SMILES", target_col="logS")
        test_data = ZINCLikeCSV(test_df, smiles_col="SMILES", target_col="logS")
    else:
        full_train_data, test_data = load_data(dataset=args.dataset, seed=args.val_seed)
        n = len(full_train_data)
        n_val = int(n * args.val_size)
        n_train = n - n_val
        gen = torch.Generator().manual_seed(args.val_seed)
        train_sub_data, val_data = torch.utils.data.random_split(
            full_train_data, [n_train, n_val], generator=gen,
        )

    print(f"Data sizes — train: {len(train_sub_data)}, val: {len(val_data)}, "
          f"full_train: {len(full_train_data)}, test: {len(test_data)}")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Pre-compute graph lists ONCE (the expensive RDKit step)
    # ------------------------------------------------------------------
    cache_dir = os.path.join(args.save_dir, 'graph_cache')
    os.makedirs(cache_dir, exist_ok=True)

    t_cache = time.time()
    print("Pre-computing graph lists (RDKit 3D + atom features)...")

    print(f"  train ({len(train_sub_data)} molecules)...")
    train_graphs = precompute_graphs(
        train_sub_data,
        os.path.join(cache_dir, 'train_sub_graphs.pkl'),
    )
    train_orig_feats = [g.node_features.clone() for g in train_graphs]

    print(f"  val ({len(val_data)} molecules)...")
    val_graphs = precompute_graphs(
        val_data,
        os.path.join(cache_dir, 'val_graphs.pkl'),
    )
    val_orig_feats = [g.node_features.clone() for g in val_graphs]

    print(f"  full_train ({len(full_train_data)} molecules)...")
    full_train_graphs = precompute_graphs(
        full_train_data,
        os.path.join(cache_dir, 'full_train_graphs.pkl'),
    )
    full_train_orig_feats = [g.node_features.clone() for g in full_train_graphs]

    print(f"  test ({len(test_data)} molecules)...")
    test_graphs = precompute_graphs(
        test_data,
        os.path.join(cache_dir, 'test_graphs.pkl'),
    )
    test_orig_feats = [g.node_features.clone() for g in test_graphs]

    print(f"Graph pre-computation done in {time.time() - t_cache:.1f}s  "
          f"(cached to {cache_dir}/)")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Phase 1: Sweep on validation
    # ------------------------------------------------------------------
    sweep_csv = os.path.join(args.save_dir, 'phase1_validation_sweep.csv')
    sweep_fields = [
        'seed', 'dim', 'sigma_pi', 'sigma_tag',
        'val_RMSE', 'val_MAE', 'val_R2_COD', 'val_Pearson_R2', 'val_Pearson_R',
        'ridge_alpha',
    ]
    with open(sweep_csv, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=sweep_fields).writeheader()

    results = []
    done = 0

    for seed in seeds:
        for dim in dims:
            for sigma_pi_orders, sigma_tag in sigma_configs:
                t0 = time.time()

                tr_emb, tr_labels = build_embeddings_fast(
                    train_graphs, train_orig_feats, dim, seed,
                    sigma_pi_orders, device,
                )
                va_emb, va_labels = build_embeddings_fast(
                    val_graphs, val_orig_feats, dim, seed,
                    sigma_pi_orders, device,
                )

                reg = RidgeCV(
                    alphas=np.logspace(-4, 2, 50), cv=5,
                    scoring='neg_mean_squared_error',
                )
                reg.fit(tr_emb.numpy() if torch.is_tensor(tr_emb) else tr_emb,
                        tr_labels.numpy() if torch.is_tensor(tr_labels) else tr_labels)

                va_pred = reg.predict(
                    va_emb.numpy() if torch.is_tensor(va_emb) else va_emb)
                m = compute_metrics(
                    va_labels.numpy() if torch.is_tensor(va_labels) else va_labels,
                    va_pred)

                elapsed = time.time() - t0
                done += 1
                orders_str = ','.join(str(x) for x in sigma_pi_orders)

                row = {
                    'seed': seed, 'dim': dim,
                    'sigma_pi': f'[{orders_str}]', 'sigma_tag': sigma_tag,
                    'val_RMSE': m['rmse'], 'val_MAE': m['mae'],
                    'val_R2_COD': m['r2_cod'], 'val_Pearson_R2': m['pearson_r2'],
                    'val_Pearson_R': m['pearson_r'],
                    'ridge_alpha': float(reg.alpha_),
                }
                results.append(row)

                with open(sweep_csv, 'a', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=sweep_fields)
                    w.writerow({k: f'{v:.6f}' if isinstance(v, float) else v
                                for k, v in row.items()})

                print(
                    f"[{done}/{total_combos}] seed={seed} dim={dim} "
                    f"sigma=[{orders_str}] | "
                    f"val_RMSE={m['rmse']:.4f}  val_R2={m['r2_cod']:.4f}  "
                    f"({elapsed:.1f}s)"
                )

    # ------------------------------------------------------------------
    # Rank results
    # ------------------------------------------------------------------
    results.sort(key=lambda r: r['val_RMSE'])
    print("\n" + "=" * 80)
    print(f"TOP {args.top_k} CONFIGS BY VALIDATION RMSE")
    print("=" * 80)
    for i, r in enumerate(results[:args.top_k]):
        print(
            f"  #{i+1}  seed={r['seed']}  dim={r['dim']}  "
            f"sigma={r['sigma_pi']}  val_RMSE={r['val_RMSE']:.4f}  "
            f"val_R2={r['val_R2_COD']:.4f}  alpha={r['ridge_alpha']:.4f}"
        )

    # ------------------------------------------------------------------
    # Phase 2: Retrain best config on FULL training data, evaluate on test
    # ------------------------------------------------------------------
    best = results[0]
    best_seed = best['seed']
    best_dim = best['dim']
    best_sigma_tag = best['sigma_tag']
    best_sigma_orders = [
        cfg for cfg in sigma_configs if cfg[1] == best_sigma_tag
    ][0][0]

    print("\n" + "=" * 80)
    print("PHASE 2: FINAL EVALUATION")
    print(f"Best config: seed={best_seed}, dim={best_dim}, "
          f"sigma_pi={best['sigma_pi']}")
    print(f"Retraining on full training data ({len(full_train_data)} molecules)...")
    print("=" * 80)

    full_emb, full_labels = build_embeddings_fast(
        full_train_graphs, full_train_orig_feats, best_dim, best_seed,
        best_sigma_orders, device,
    )
    test_emb, test_labels = build_embeddings_fast(
        test_graphs, test_orig_feats, best_dim, best_seed,
        best_sigma_orders, device,
    )

    reg_final = RidgeCV(
        alphas=np.logspace(-4, 2, 50), cv=5,
        scoring='neg_mean_squared_error',
    )
    reg_final.fit(
        full_emb.numpy() if torch.is_tensor(full_emb) else full_emb,
        full_labels.numpy() if torch.is_tensor(full_labels) else full_labels,
    )
    test_pred = reg_final.predict(
        test_emb.numpy() if torch.is_tensor(test_emb) else test_emb)

    m_test = compute_metrics(
        test_labels.numpy() if torch.is_tensor(test_labels) else test_labels,
        test_pred)

    print(f"\n  TEST RESULTS (seed={best_seed}, dim={best_dim}, "
          f"sigma={best['sigma_pi']}):")
    print(f"    RMSE      = {m_test['rmse']:.4f}")
    print(f"    MAE       = {m_test['mae']:.4f}")
    print(f"    STD_err   = {m_test['std_err']:.4f}")
    print(f"    R2 (COD)  = {m_test['r2_cod']:.4f}")
    print(f"    Pearson R = {m_test['pearson_r']:.4f}")
    print(f"    Pearson R²= {m_test['pearson_r2']:.4f}")
    print(f"    Ridge α   = {reg_final.alpha_:.6f}")

    # Save final results
    final_csv = os.path.join(args.save_dir, 'phase2_final_test_result.csv')
    with open(final_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['metric', 'value'])
        w.writerow(['best_seed', best_seed])
        w.writerow(['best_dim', best_dim])
        w.writerow(['best_sigma_pi', best['sigma_pi']])
        w.writerow(['ridge_alpha', f"{reg_final.alpha_:.6f}"])
        w.writerow(['val_RMSE', f"{best['val_RMSE']:.6f}"])
        for k, v in m_test.items():
            w.writerow([f'test_{k}', f'{v:.6f}'])

    # Save per-molecule predictions
    pred_csv = os.path.join(args.save_dir, 'phase2_test_predictions.csv')
    y_true = np.asarray(
        test_labels.numpy() if torch.is_tensor(test_labels) else test_labels
    ).ravel()
    y_pred = np.asarray(test_pred).ravel()
    with open(pred_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['y_true', 'y_pred', 'error'])
        for yt, yp in zip(y_true, y_pred):
            w.writerow([f'{yt:.6f}', f'{yp:.6f}', f'{yt - yp:.6f}'])

    # Also evaluate top-K configs on test (for comparison in paper)
    topk_csv = os.path.join(args.save_dir, 'phase2_topk_test_results.csv')
    topk_fields = [
        'rank', 'seed', 'dim', 'sigma_pi', 'val_RMSE',
        'test_RMSE', 'test_MAE', 'test_R2_COD', 'test_Pearson_R2',
    ]
    with open(topk_csv, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=topk_fields).writeheader()

    print(f"\nTop-{args.top_k} configs evaluated on test set:")
    for i, r in enumerate(results[:args.top_k]):
        r_seed = r['seed']
        r_dim = r['dim']
        r_sigma_tag = r['sigma_tag']
        r_sigma_orders = [
            cfg for cfg in sigma_configs if cfg[1] == r_sigma_tag
        ][0][0]

        fe, fl = build_embeddings_fast(
            full_train_graphs, full_train_orig_feats, r_dim, r_seed,
            r_sigma_orders, device,
        )
        te, tl = build_embeddings_fast(
            test_graphs, test_orig_feats, r_dim, r_seed,
            r_sigma_orders, device,
        )
        reg_k = RidgeCV(
            alphas=np.logspace(-4, 2, 50), cv=5,
            scoring='neg_mean_squared_error',
        )
        reg_k.fit(fe.numpy() if torch.is_tensor(fe) else fe,
                  fl.numpy() if torch.is_tensor(fl) else fl)
        tp = reg_k.predict(te.numpy() if torch.is_tensor(te) else te)
        mt = compute_metrics(
            tl.numpy() if torch.is_tensor(tl) else tl, tp)

        print(
            f"  #{i+1} seed={r_seed} dim={r_dim} sigma={r['sigma_pi']} | "
            f"val_RMSE={r['val_RMSE']:.4f} → test_RMSE={mt['rmse']:.4f} "
            f"test_R2={mt['r2_cod']:.4f}"
        )

        with open(topk_csv, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=topk_fields).writerow({
                'rank': i + 1, 'seed': r_seed, 'dim': r_dim,
                'sigma_pi': r['sigma_pi'], 'val_RMSE': f"{r['val_RMSE']:.6f}",
                'test_RMSE': f"{mt['rmse']:.6f}", 'test_MAE': f"{mt['mae']:.6f}",
                'test_R2_COD': f"{mt['r2_cod']:.6f}",
                'test_Pearson_R2': f"{mt['pearson_r2']:.6f}",
            })

    print(f"\nAll results saved to: {args.save_dir}/")
    print(f"  - phase1_validation_sweep.csv   ({len(results)} rows)")
    print(f"  - phase2_final_test_result.csv")
    print(f"  - phase2_test_predictions.csv")
    print(f"  - phase2_topk_test_results.csv")


if __name__ == '__main__':
    main()
