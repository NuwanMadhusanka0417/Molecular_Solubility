"""
GVFA k-fold cross-validation on the full training set.

Same pipeline as GVFA_edge_main.py (GVFA encoder -> Ridge/XGBoost), but evaluates on
k held-out folds of solubility_1.csv (or dataset train split) instead of testset_novel.csv.

Prints per-fold metrics and mean ± std across folds (same metric keys as main).

After CV, optionally trains on the full training CSV and evaluates on a held-out test CSV
(same print format as GVFA_edge_main: Dim=... RMSE=... STD_err=...). No import from main.
"""
import argparse
import csv
import os
import random

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

from src.create_graphs import create_graph_list
from src.load_data import ZINCLikeCSV
from src.VSA_conversion import VSA_conversion
from src.embeddings import getEmbedding
from models.graphcnnVSA_Binding_FULL import GraphCNN, EDGE_MINMAX_COLS


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
        "rmse": rmse,
        "std_err": std_err,
        "mae": mae,
        "r2_cod": r2_cod,
        "pearson_r2": pr ** 2,
        "pearson_r": pr,
    }


def _parse_seeds(s: str):
    s = s.strip()
    if '-' in s and ',' not in s:
        lo, hi = s.split('-')
        return list(range(int(lo), int(hi) + 1))
    return [int(x.strip()) for x in s.split(',') if x.strip()]


def _parse_sigma_pi_arg(s: str):
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


def _dims_list(args):
    return [int(x.strip()) for x in args.dims.replace(' ', '').split(',') if x.strip()]


def run_gvfa_ridge_train_test(args, train_data, test_data, device):
    """
    Train regression on full training set; evaluate on test_data.
    Loops all --dims and --sigma_pi with args.seed (multiple seeds from outer loop).
    """
    seed = args.seed
    dims = _dims_list(args)
    sigma_configs = _parse_sigma_pi_arg(args.sigma_pi)

    save = args.save_results is not None
    summary_path = None
    summary_fields = [
        'dim', 'sigma_pi', 'sigma_tag', 'seed',
        'RMSE', 'STD_err', 'MAE', 'R2_COD', 'Pearson_R2', 'Pearson_R',
    ]
    if save:
        os.makedirs(args.save_results, exist_ok=True)
        summary_path = os.path.join(args.save_results, 'results_summary.csv')
        with open(summary_path, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=summary_fields).writeheader()

    for dim in dims:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        train_graphs = create_graph_list(train_data)
        test_graphs = create_graph_list(test_data)
        train_HVs, node_train_stats = VSA_conversion(
            train_graphs.copy(), dim, projection_type="orthogonal", seed=seed,
        )
        test_HVs, _ = VSA_conversion(
            test_graphs.copy(), dim, projection_type="orthogonal", seed=seed,
            train_stats=node_train_stats,
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
            _eref = next((g.edge_attr for g in train_HVs if g.edge_attr is not None and g.edge_attr.numel() > 0), None)
            if _eref is not None:
                assert _eref.shape[1] == 5, (
                    f"edge_attr should be raw [E,5] bond features but got shape {_eref.shape}. "
                    f"VSA_conversion must not modify edge_attr."
                )
            _edge_attrs = []
            for g in train_HVs:
                if g.edge_attr is not None and g.edge_attr.numel() > 0:
                    _edge_attrs.append(g.edge_attr.to(torch.float32))
            if _edge_attrs:
                all_edge_feats = torch.cat(_edge_attrs, dim=0)
                edge_col_min = all_edge_feats[:, EDGE_MINMAX_COLS].min(dim=0).values
                edge_col_range = (
                    all_edge_feats[:, EDGE_MINMAX_COLS].max(dim=0).values - edge_col_min
                ).clamp(min=1e-6)
                model_eq1.set_edge_stats(edge_col_min, edge_col_range)
                print(f"  [Edge features] Bounded [-1,1] fit on {all_edge_feats.shape[0]} "
                      f"training edges. bond_length min/range: "
                      f"{edge_col_min[1]:.4f} / {edge_col_range[1]:.4f}")
            else:
                print("  [Edge features] No edge attrs found; skipping.")
            train_emb, train_labels = getEmbedding(
                model_eq1, device, train_HVs, use_size_aware=True, hop_alpha=1.0,
            )
            test_emb, test_labels = getEmbedding(
                model_eq1, device, test_HVs, use_size_aware=True, hop_alpha=1.0,
            )
            train_emb = train_emb.squeeze(0)
            test_emb = test_emb.squeeze(0)

            if args.use_ridge:
                reg = RidgeCV(
                    alphas=np.logspace(-4, 2, 50), cv=5,
                    scoring='neg_mean_squared_error',
                )
                reg.fit(train_emb, train_labels)
                pred = reg.predict(test_emb)
            else:
                from xgboost import XGBRegressor
                reg = XGBRegressor(
                    n_estimators=2000, learning_rate=0.03, max_depth=7,
                    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, reg_alpha=0.0,
                    random_state=seed, n_jobs=4, tree_method="hist",
                )
                reg.fit(
                    train_emb, train_labels,
                    eval_set=[(test_emb, test_labels)], verbose=False,
                )
                pred = reg.predict(test_emb)

            m = compute_metrics(test_labels, pred)
            orders_str = ','.join(str(x) for x in sigma_pi_orders)
            print(
                f"Dim={dim}  sigma_pi=[{orders_str}]  ({sigma_tag})  "
                f"RMSE={m['rmse']:.4f}  STD_err={m['std_err']:.4f}  MAE={m['mae']:.4f}  "
                f"R2_COD={m['r2_cod']:.4f}  Pearson_R2={m['pearson_r2']:.4f}",
            )

            if save and summary_path:
                y_true = np.asarray(test_labels).ravel()
                y_pred = np.asarray(pred).ravel()
                pred_path = os.path.join(
                    args.save_results, f'predictions_dim{dim}_{sigma_tag}.csv',
                )
                with open(pred_path, 'w', newline='') as f:
                    w = csv.writer(f)
                    w.writerow(['y_true', 'y_pred', 'error'])
                    for yt, yp in zip(y_true, y_pred):
                        w.writerow([f'{yt:.6f}', f'{yp:.6f}', f'{yt - yp:.6f}'])

                with open(summary_path, 'a', newline='') as f:
                    csv.DictWriter(f, fieldnames=summary_fields).writerow({
                        'dim': dim,
                        'sigma_pi': f'[{orders_str}]',
                        'sigma_tag': sigma_tag,
                        'seed': seed,
                        'RMSE': f'{m["rmse"]:.6f}',
                        'STD_err': f'{m["std_err"]:.6f}',
                        'MAE': f'{m["mae"]:.6f}',
                        'R2_COD': f'{m["r2_cod"]:.6f}',
                        'Pearson_R2': f'{m["pearson_r2"]:.6f}',
                        'Pearson_R': f'{m["pearson_r"]:.6f}',
                    })
                print(f"  Saved predictions to {pred_path}")


def run_gvfa_ridge_one_split(args, train_data, eval_data, device, fold_label=""):
    """
    Train on train_data, evaluate on eval_data (val fold).
    Same math as GVFA_edge_main.run_gvfa_ridge for one dim/sigma block.
    Returns list of {dim, sigma_tag, sigma_pi, **compute_metrics}.
    """
    seed = args.seed
    dims = _dims_list(args)
    sigma_configs = _parse_sigma_pi_arg(args.sigma_pi)

    save = args.save_results is not None
    summary_path = None
    summary_fields = [
        'fold', 'dim', 'sigma_pi', 'sigma_tag', 'seed',
        'RMSE', 'STD_err', 'MAE', 'R2_COD', 'Pearson_R2', 'Pearson_R',
    ]
    if save:
        os.makedirs(args.save_results, exist_ok=True)
        summary_path = os.path.join(args.save_results, 'results_summary.csv')
        if not os.path.exists(summary_path):
            with open(summary_path, 'w', newline='') as f:
                csv.DictWriter(f, fieldnames=summary_fields).writeheader()

    results_out = []

    for dim in dims:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        train_graphs = create_graph_list(train_data)
        eval_graphs = create_graph_list(eval_data)
        train_HVs, node_train_stats = VSA_conversion(
            train_graphs.copy(), dim, projection_type="orthogonal", seed=seed,
        )
        eval_HVs, _ = VSA_conversion(
            eval_graphs.copy(), dim, projection_type="orthogonal", seed=seed,
            train_stats=node_train_stats,
        )

        for sigma_pi_orders, sigma_tag in sigma_configs:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            model_eq1 = GraphCNN(
                eval_HVs[0].node_features.shape[1], 5, 1, 'sum', 'sum', device, 10,
                edge_feat_dim=5, edge_projection_type="orthogonal",
                use_reservoir=True, hop_decay=0.85, sigma_pi_orders=sigma_pi_orders,
                rng_seed=seed,
            )
            _eref = next((g.edge_attr for g in train_HVs if g.edge_attr is not None and g.edge_attr.numel() > 0), None)
            if _eref is not None:
                assert _eref.shape[1] == 5, (
                    f"edge_attr should be raw [E,5] bond features but got shape {_eref.shape}. "
                    f"VSA_conversion must not modify edge_attr."
                )
            _edge_attrs = []
            for g in train_HVs:
                if g.edge_attr is not None and g.edge_attr.numel() > 0:
                    _edge_attrs.append(g.edge_attr.to(torch.float32))
            if _edge_attrs:
                all_edge_feats = torch.cat(_edge_attrs, dim=0)
                edge_col_min = all_edge_feats[:, EDGE_MINMAX_COLS].min(dim=0).values
                edge_col_range = (
                    all_edge_feats[:, EDGE_MINMAX_COLS].max(dim=0).values - edge_col_min
                ).clamp(min=1e-6)
                model_eq1.set_edge_stats(edge_col_min, edge_col_range)
                print(f"  [Edge features] Bounded [-1,1] fit on {all_edge_feats.shape[0]} "
                      f"training edges. bond_length min/range: "
                      f"{edge_col_min[1]:.4f} / {edge_col_range[1]:.4f}")
            else:
                print("  [Edge features] No edge attrs found; skipping.")
            train_emb, train_labels = getEmbedding(
                model_eq1, device, train_HVs, use_size_aware=True, hop_alpha=1.0,
            )
            eval_emb, eval_labels = getEmbedding(
                model_eq1, device, eval_HVs, use_size_aware=True, hop_alpha=1.0,
            )
            train_emb = train_emb.squeeze(0)
            eval_emb = eval_emb.squeeze(0)

            if args.use_ridge:
                reg = RidgeCV(
                    alphas=np.logspace(-4, 2, 50), cv=5,
                    scoring='neg_mean_squared_error',
                )
                reg.fit(train_emb, train_labels)
                pred = reg.predict(eval_emb)
            else:
                from xgboost import XGBRegressor
                reg = XGBRegressor(
                    n_estimators=2000, learning_rate=0.03, max_depth=7,
                    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, reg_alpha=0.0,
                    random_state=seed, n_jobs=4, tree_method="hist",
                )
                reg.fit(
                    train_emb, train_labels,
                    eval_set=[(eval_emb, eval_labels)], verbose=False,
                )
                pred = reg.predict(eval_emb)

            m = compute_metrics(eval_labels, pred)
            orders_str = ','.join(str(x) for x in sigma_pi_orders)

            prefix = f"[{fold_label}] " if fold_label else ""
            print(
                f"{prefix}Dim={dim}  sigma_pi=[{orders_str}]  ({sigma_tag})  "
                f"RMSE={m['rmse']:.4f}  STD_err={m['std_err']:.4f}  MAE={m['mae']:.4f}  "
                f"R2_COD={m['r2_cod']:.4f}  Pearson_R2={m['pearson_r2']:.4f}",
            )

            row_agg = {
                'dim': dim, 'sigma_tag': sigma_tag, 'sigma_pi': f'[{orders_str}]',
                **m,
            }
            results_out.append(row_agg)

            if save and summary_path:
                with open(summary_path, 'a', newline='') as f:
                    csv.DictWriter(f, fieldnames=summary_fields).writerow({
                        'fold': fold_label,
                        'dim': dim,
                        'sigma_pi': f'[{orders_str}]',
                        'sigma_tag': sigma_tag,
                        'seed': seed,
                        'RMSE': f'{m["rmse"]:.6f}',
                        'STD_err': f'{m["std_err"]:.6f}',
                        'MAE': f'{m["mae"]:.6f}',
                        'R2_COD': f'{m["r2_cod"]:.6f}',
                        'Pearson_R2': f'{m["pearson_r2"]:.6f}',
                        'Pearson_R': f'{m["pearson_r"]:.6f}',
                    })

            if save:
                pred_path = os.path.join(
                    args.save_results,
                    f'predictions_fold{fold_label}_dim{dim}_{sigma_tag}.csv',
                )
                y_true = np.asarray(eval_labels).ravel()
                y_pred = np.asarray(pred).ravel()
                with open(pred_path, 'w', newline='') as f:
                    w = csv.writer(f)
                    w.writerow(['y_true', 'y_pred', 'error'])
                    for yt, yp in zip(y_true, y_pred):
                        w.writerow([f'{yt:.6f}', f'{yp:.6f}', f'{yt - yp:.6f}'])
                print(f"  Saved predictions to {pred_path}")

    return results_out


def parse_args():
    p = argparse.ArgumentParser(description='GVFA k-fold CV on full training set')
    p.add_argument(
        '--train_csv', type=str, default='final_data/solubility_1.csv',
        help='Training CSV with SMILES and logS',
    )
    p.add_argument('--k', '--k_folds', type=int, default=5, dest='k_folds',
                   help='Number of folds (default 5)')
    p.add_argument('--cv_seed', type=int, default=42,
                   help='Seed for KFold shuffling')
    p.add_argument('--dim', type=int, default=None,
                   help='Ignored; use --dims')
    p.add_argument('--dims', type=str, default='1000, 2000, 5000, 10000',
                   help='Comma-separated VSA dimensions')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--use_ridge', action='store_true', default=True)
    p.add_argument('--no_ridge', action='store_false', dest='use_ridge')
    p.add_argument('--seed', type=int, default=42, help='RNG seed for GVFA/Ridge')
    p.add_argument('--seeds', type=str, default=None,
                   help='Multiple seeds; overrides --seed')
    p.add_argument(
        '--sigma_pi', type=str, default='all',
        help='Same as GVFA_edge_main: all, legacy, or custom',
    )
    p.add_argument(
        '--save_results', type=str, default=None,
        help='Directory for per-fold CSV summary and predictions',
    )
    p.add_argument(
        '--test_csv', type=str, default='final_data/testset_novel.csv',
        help='Test CSV (SMILES, logS) for evaluation after k-fold (full-train model).',
    )
    p.add_argument(
        '--no_test', action='store_true',
        help='Skip test-set evaluation after cross-validation.',
    )
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    seeds = _parse_seeds(args.seeds) if args.seeds else [args.seed]

    train_df = pd.read_csv(args.train_csv).dropna(subset=['SMILES', 'logS'])
    n = len(train_df)
    print(f"K-fold CV: {args.k_folds} folds on full training set ({n} molecules)")
    print(f"CSV: {args.train_csv}")
    print(f"Shuffle seed (KFold): {args.cv_seed}")
    print(f"Seeds (GVFA): {seeds}")

    test_data = None
    if not args.no_test:
        if not os.path.isfile(args.test_csv):
            raise FileNotFoundError(
                f"Test CSV not found: {args.test_csv!r} "
                f"(use --no_test to skip, or set --test_csv)",
            )
        test_df = pd.read_csv(args.test_csv).dropna(subset=['SMILES', 'logS'])
        test_data = ZINCLikeCSV(test_df, smiles_col='SMILES', target_col='logS')
        print(f"Test evaluation: {args.test_csv} ({len(test_df)} molecules)")
    else:
        print("Test evaluation: skipped (--no_test)")

    kf = KFold(
        n_splits=args.k_folds, shuffle=True, random_state=args.cv_seed,
    )

    agg_path = None
    if args.save_results:
        os.makedirs(args.save_results, exist_ok=True)
        agg_path = os.path.join(args.save_results, 'kfold_aggregate.csv')
        with open(agg_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow([
                'seed', 'dim', 'sigma_tag', 'fold',
                'RMSE', 'MAE', 'R2_COD', 'Pearson_R2', 'Pearson_R',
            ])

    for seed in seeds:
        args.seed = seed
        if args.save_results:
            seed_dir = os.path.join(args.save_results, f"seed_{seed}")
            os.makedirs(seed_dir, exist_ok=True)
            args._save_dir_seed = seed_dir
        else:
            args._save_dir_seed = None

        print("\n" + "=" * 80)
        print(f"SEED = {seed}")
        print("=" * 80)

        fold_metrics = {}  # (dim, sigma_tag) -> list of metric dicts

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(np.arange(n))):
            tr_df = train_df.iloc[train_idx].reset_index(drop=True)
            va_df = train_df.iloc[val_idx].reset_index(drop=True)
            train_data = ZINCLikeCSV(tr_df, smiles_col='SMILES', target_col='logS')
            val_data = ZINCLikeCSV(va_df, smiles_col='SMILES', target_col='logS')

            fold_tag = f"{fold_idx + 1}"
            print(f"\n--- Fold {fold_idx + 1}/{args.k_folds}  "
                  f"(train={len(tr_df)}, val={len(va_df)}) ---")

            save_fold = args._save_dir_seed
            if save_fold:
                fold_save = os.path.join(save_fold, f"fold_{fold_tag}")
            else:
                fold_save = None

            orig_save = args.save_results
            if fold_save:
                args.save_results = fold_save

            fold_rows = run_gvfa_ridge_one_split(
                args, train_data, val_data, device, fold_label=fold_tag,
            )

            args.save_results = orig_save

            for row in fold_rows:
                key = (row['dim'], row['sigma_tag'])
                m = {k: row[k] for k in ('rmse', 'std_err', 'mae', 'r2_cod', 'pearson_r2', 'pearson_r')}
                fold_metrics.setdefault(key, []).append(m)
                if agg_path:
                    with open(agg_path, 'a', newline='') as f:
                        csv.writer(f).writerow([
                            seed, row['dim'], row['sigma_tag'], fold_idx + 1,
                            f"{m['rmse']:.6f}", f"{m['mae']:.6f}",
                            f"{m['r2_cod']:.6f}", f"{m['pearson_r2']:.6f}",
                            f"{m['pearson_r']:.6f}",
                        ])

        print("\n" + "-" * 80)
        print(f"Mean ± std across {args.k_folds} folds (SEED={seed})")
        print("-" * 80)
        for (dim, sigma_tag), mlist in sorted(fold_metrics.items()):
            rmses = [x['rmse'] for x in mlist]
            maes = [x['mae'] for x in mlist]
            r2s = [x['r2_cod'] for x in mlist]
            pr2s = [x['pearson_r2'] for x in mlist]
            print(
                f"  dim={dim}  {sigma_tag}:  "
                f"RMSE={np.mean(rmses):.4f}±{np.std(rmses):.4f}  "
                f"MAE={np.mean(maes):.4f}±{np.std(maes):.4f}  "
                f"R2_COD={np.mean(r2s):.4f}±{np.std(r2s):.4f}  "
                f"Pearson_R2={np.mean(pr2s):.4f}±{np.std(pr2s):.4f}",
            )

        if test_data is not None:
            full_train_data = ZINCLikeCSV(
                train_df, smiles_col='SMILES', target_col='logS',
            )
            print("\n" + "=" * 80)
            print(f"SEED = {seed}")
            print("=" * 80)
            print(
                f"Test set: {args.test_csv}  |  "
                f"train on full {args.train_csv} ({n} molecules)",
            )

            orig_save_results = args.save_results
            if args._save_dir_seed:
                test_save = os.path.join(args._save_dir_seed, 'test_eval')
                os.makedirs(test_save, exist_ok=True)
                args.save_results = test_save
            run_gvfa_ridge_train_test(args, full_train_data, test_data, device)
            args.save_results = orig_save_results


if __name__ == '__main__':
    main()
