"""
GVFA for molecular solubility prediction.

GVFA encoder -> embeddings -> RidgeCV, RBF Kernel Ridge, or XGBoost (no encoder training).

Train: solubility_1.csv.  Test: testset_novel.csv.
"""
import argparse
import copy
import csv
import os
import random

from src.create_graphs import create_graph_list
from src.load_data import load_data
from src.VSA_conversion import VSA_conversion
from src.embeddings import getEmbedding
from src.tsne_viz import draw_tsne_pipeline
from models.graphcnnVSA_Binding_FULL import GraphCNN, EDGE_MINMAX_COLS

import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _mean_pool_layers(node_mat: torch.Tensor, start_idx: list) -> np.ndarray:
    """Mean-pool node-level [N_total, D] tensor to graph-level [G, D] array."""
    rows = []
    for i in range(len(start_idx) - 1):
        lo, hi = start_idx[i], start_idx[i + 1]
        rows.append(node_mat[lo:hi].mean(dim=0))
    return torch.stack(rows).numpy().astype(np.float32)


@torch.no_grad()
def get_gvfa_layer_embedding(model, graphs, device):
    """
    Build a [G, 5*D] embedding by concatenating mean-pooled node HVs from all
    5 GNN layers (layer 0 = VSA projection + L2-norm, layers 1-4 = message passing).

    Uses capture_aux=True — does not affect model weights.
    Layer 0 is reconstructed from g.node_features BEFORE the forward pass
    (the forward does not modify node_features in-place).

    Returns
    -------
    emb    : np.ndarray [G, 5*D]
    labels : np.ndarray [G]
    """
    labels = np.array(
        [float(torch.as_tensor(g.label).item()) for g in graphs],
        dtype=np.float32,
    )

    # Reconstruct layer 0 (same ops as GraphCNN.forward) before the forward pass
    X_cat = torch.cat([g.node_features for g in graphs], dim=0).to(device)
    X_concat = F.normalize(X_cat, p=2, dim=1, eps=1e-8).cpu()   # [N_total, D]

    model.eval()
    _ = model(graphs, capture_aux=True)

    aux = model._aux
    if aux is None:
        raise RuntimeError(
            "model._aux is None. Make sure the model was built with use_reservoir=True."
        )

    start_idx = aux["start_idx"]          # list[int], length G+1
    post_bins = aux["layer_post_bin"]     # list of 4 tensors [N_total, D]

    all_layers = [X_concat] + [p.cpu() for p in post_bins]   # 5 tensors
    pooled     = [_mean_pool_layers(lyr, start_idx) for lyr in all_layers]
    emb        = np.concatenate(pooled, axis=1)               # [G, 5*D]

    return emb, labels


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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='GVFA for molecular solubility')
    p.add_argument('--dataset', type=str, default='solubility_novel',
                   choices=['old', 'solubility_novel', 'new'],
                   help='solubility_novel: train solubility_1.csv, test testset_novel.csv')
    p.add_argument('--dim', type=int, default=1000, help='VSA dimension (used when --dims is omitted)')
    p.add_argument('--dims', type=str, default=None,
                   help='Comma-separated dims for gvfa_ridge loop (overrides --dim). '
                        'Example: "1000,2000,5000". Default: use --dim only.')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--use_ridge', action='store_true', default=True)
    p.add_argument('--no_ridge', action='store_false', dest='use_ridge')
    p.add_argument(
        '--ridge_type',
        type=str,
        default='ridgecv',
        choices=['ridgecv', 'rbf_ridge'],
        help='When --use_ridge: linear RidgeCV (alphas via CV) or RBF KernelRidge (alpha+gamma via GridSearchCV).',
    )
    p.add_argument('--seed', type=int, default=42, help='Single RNG seed (use --seeds for multiple)')
    p.add_argument('--seeds', type=str, default=None,
                   help='Multiple seeds: range "0-49" or comma-separated "0,1,42,123". Overrides --seed.')
    p.add_argument(
        '--sigma_pi',
        type=str,
        default='all',
        help='Sigma-Pi order sets per --dims value. Presets: "all" = [0],[1],[2],[0,1],[0,1,2]; '
             '"legacy" = [0,1] only. Or one set as comma-separated orders, e.g. "0,1,2".',
    )
    p.add_argument(
        '--save_results', type=str, default=None,
        help='If set, save CSV files (summary + per-molecule predictions) to this directory.',
    )
    p.add_argument(
        '--export_analysis_dir', type=str, default=None,
        help='If set, save per-batch .npz files: GVFA layer pre/post L2-normalization HV, '
             'sigma-pi per order + combined, F1 tap buffer, y (logS), and node graph ids.',
    )
    p.add_argument(
        '--gvfa_ridge',
        action='store_true',
        default=False,
        help='If set, run a second Ridge regression using only the concatenated '
             'GNN layer outputs (layers 0-4, before tap buffer / Sigma-Pi / graph '
             'pooling). Results are printed and saved alongside the main results. '
             'The original regression is always run regardless of this flag.',
    )
    p.add_argument(
        '--tsne',
        action='store_true',
        default=False,
        help='If set, draw t-SNE plots of HV representations at each pipeline stage '
             'after the model is built. Plots are saved to --tsne_out_dir.',
    )
    p.add_argument(
        '--tsne_split',
        type=str,
        default='test',
        choices=['train', 'test', 'both'],
        help='Which split to visualise with t-SNE (default: test).',
    )
    p.add_argument(
        '--tsne_max_mols',
        type=int,
        default=500,
        help='Max molecules to use for t-SNE. 0 = all (slow). Default: 500.',
    )
    p.add_argument(
        '--tsne_out_dir',
        type=str,
        default='tsne_plots',
        help='Directory to save t-SNE PNG files (default: tsne_plots).',
    )
    p.add_argument(
        '--tsne_perplexity',
        type=int,
        default=30,
        help='t-SNE perplexity (default: 30).',
    )
    p.add_argument(
        '--binding',
        type=str,
        default='circular',
        choices=['circular', 'hadamard'],
        help=(
            'VSA binding operator used in all bind() calls. '
            '"circular" = FFT circular convolution (default, HRR-style). '
            '"hadamard" = elementwise multiplication (MAP-style, preserves geometry better).'
        ),
    )
    p.add_argument(
        '--delta',
        type=int,
        default=1,
        choices=[0, 1, 2],
        help=(
            'Residual mixing mode in next_layer_eps. '
            '0 = h + pooled_nb (no bind, least scrambling). '
            '1 = h + bind(h, pooled_nb) (default). '
            '2 = h + bind(h, pooled_nb) + pooled_nb.'
        ),
    )
    p.add_argument(
        '--equation',
        type=int,
        default=10,
        choices=[10, 11, 12],
        help=(
            'Message-passing equation variant. '
            '10 = rotate h before neighbour-pool, no final rotate (default). '
            '11 = no pre-rotate, rotate output. '
            '12 = rotate h before pool AND rotate output.'
        ),
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# GVFA + Ridge/XGBoost
# ---------------------------------------------------------------------------

def _parse_seeds(s: str):
    """Parse '0-49' or '0,1,42,123' into a list of ints."""
    s = s.strip()
    if '-' in s and ',' not in s:
        lo, hi = s.split('-')
        return list(range(int(lo), int(hi) + 1))
    return [int(x.strip()) for x in s.split(',') if x.strip()]


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


def run_gvfa_ridge(args, train_data, test_data, device,
                   train_graphs_base=None, test_graphs_base=None):
    seed = args.seed
    if args.dims is not None:
        dims = [int(x.strip()) for x in args.dims.split(',') if x.strip()]
    else:
        dims = [args.dim]
    sigma_configs = _parse_sigma_pi_arg(args.sigma_pi)

    save = args.save_results is not None
    if save:
        os.makedirs(args.save_results, exist_ok=True)
        summary_path = os.path.join(args.save_results, 'results_summary.csv')
        summary_fields = [
            'dim', 'sigma_pi', 'sigma_tag', 'seed', 'regressor',
            'RMSE', 'STD_err', 'MAE', 'R2_COD', 'Pearson_R2', 'Pearson_R',
        ]
        with open(summary_path, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=summary_fields).writeheader()

    for dim in dims:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if train_graphs_base is not None and test_graphs_base is not None:
            train_graphs = copy.deepcopy(train_graphs_base)
            test_graphs = copy.deepcopy(test_graphs_base)
        else:
            train_graphs = create_graph_list(train_data)
            test_graphs = create_graph_list(test_data)

        train_HVs, node_train_stats = VSA_conversion(
            train_graphs, dim, projection_type="orthogonal", seed=seed,
        )
        test_HVs, _ = VSA_conversion(
            test_graphs, dim, projection_type="orthogonal", seed=seed,
            train_stats=node_train_stats,
        )
        if node_train_stats is not None:
            print(f"  [Seed {seed}] node train_stats col_min: "
                  f"{node_train_stats['col_min'].tolist()}")
            if len(train_HVs) > 0:
                norm_means = [
                    g.node_features.norm(dim=1).mean()
                    for g in train_HVs[:5]
                ]
                nk = len(norm_means)
                print(f"  [Seed {seed}] train HV norm mean (first {nk} graphs): "
                      f"{torch.stack(norm_means).mean():.4f}")

        for sigma_pi_orders, sigma_tag in sigma_configs:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            model_eq1 = GraphCNN(
                test_HVs[0].node_features.shape[1], 5, args.delta, 'sum', 'sum', device, args.equation,
                edge_feat_dim=5, edge_projection_type="orthogonal",
                use_reservoir=True, hop_decay=0.85, sigma_pi_orders=sigma_pi_orders,
                rng_seed=seed, binding_type=args.binding,
            )
            # edge_attr stays raw [E, 5]; VSA_conversion only overwrites node_features.
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
                      f"training edges. bond_length col min/range "
                      f"(idx {EDGE_MINMAX_COLS[1]} in full 5-d): "
                      f"{edge_col_min[1]:.4f} / {edge_col_range[1]:.4f}")
            else:
                print("  [Edge features] No edge attrs found; skipping.")
            if args.tsne:
                tsne_out = args.tsne_out_dir
                if args.save_results:
                    tsne_out = os.path.join(args.save_results, 'tsne')
                draw_tsne_pipeline(
                    model=model_eq1,
                    train_HVs=train_HVs,
                    test_HVs=test_HVs,
                    device=device,
                    split=args.tsne_split,
                    max_mols=args.tsne_max_mols,
                    out_dir=tsne_out,
                    tsne_perplexity=args.tsne_perplexity,
                    seed=seed,
                    dim=dim,
                    sigma_tag=sigma_tag,
                )
            orders_str = ','.join(str(x) for x in sigma_pi_orders)
            reg_tag    = args.ridge_type if args.use_ridge else 'xgboost'

            train_emb, train_labels = getEmbedding(
                model_eq1, device, train_HVs, use_size_aware=True, hop_alpha=1.0,
            )
            test_emb, test_labels = getEmbedding(
                model_eq1, device, test_HVs, use_size_aware=True, hop_alpha=1.0,
            )
            train_emb = train_emb.squeeze(0)
            test_emb  = test_emb.squeeze(0)

            # ── Optional GVFA-layer-only regression ──────────────────────────
            if args.gvfa_ridge:
                gvfa_train_emb, gvfa_train_labels = get_gvfa_layer_embedding(
                    model_eq1, train_HVs, device,
                )
                gvfa_test_emb, gvfa_test_labels = get_gvfa_layer_embedding(
                    model_eq1, test_HVs, device,
                )

                if args.ridge_type == 'ridgecv':
                    gvfa_reg = RidgeCV(
                        alphas=np.logspace(-2, 4, 100),
                        cv=5,
                        scoring='neg_mean_squared_error',
                    )
                else:
                    gvfa_reg = GridSearchCV(
                        KernelRidge(kernel='rbf'),
                        {'alpha': np.logspace(-4, 4, 40), 'gamma': np.logspace(-6, 2, 25)},
                        cv=5, scoring='neg_mean_squared_error', n_jobs=-1, refit=True,
                    )

                gvfa_reg.fit(gvfa_train_emb, gvfa_train_labels)
                gvfa_pred = gvfa_reg.predict(gvfa_test_emb)
                gvfa_m    = compute_metrics(gvfa_test_labels, gvfa_pred)

                print(
                    f"[GVFA-layers] Dim={dim}  sigma_pi=[{orders_str}]  ({sigma_tag})  "
                    f"head={reg_tag}  emb_dim={gvfa_train_emb.shape[1]}  "
                    f"RMSE={gvfa_m['rmse']:.4f}  STD_err={gvfa_m['std_err']:.4f}  "
                    f"MAE={gvfa_m['mae']:.4f}  R2_COD={gvfa_m['r2_cod']:.4f}  "
                    f"Pearson_R2={gvfa_m['pearson_r2']:.4f}",
                )

                if save:
                    gvfa_pred_path = os.path.join(
                        args.save_results,
                        f'predictions_gvfalayers_dim{dim}_{sigma_tag}_{reg_tag}.csv',
                    )
                    with open(gvfa_pred_path, 'w', newline='') as f:
                        w = csv.writer(f)
                        w.writerow(['y_true', 'y_pred', 'error'])
                        for yt, yp in zip(
                            np.asarray(gvfa_test_labels).ravel(),
                            np.asarray(gvfa_pred).ravel(),
                        ):
                            w.writerow([f'{yt:.6f}', f'{yp:.6f}', f'{yt - yp:.6f}'])

                    with open(summary_path, 'a', newline='') as f:
                        csv.DictWriter(f, fieldnames=summary_fields).writerow({
                            'dim':        dim,
                            'sigma_pi':   f'[{orders_str}]',
                            'sigma_tag':  sigma_tag,
                            'seed':       seed,
                            'regressor':  f'gvfalayers_{reg_tag}',
                            'RMSE':       f'{gvfa_m["rmse"]:.6f}',
                            'STD_err':    f'{gvfa_m["std_err"]:.6f}',
                            'MAE':        f'{gvfa_m["mae"]:.6f}',
                            'R2_COD':     f'{gvfa_m["r2_cod"]:.6f}',
                            'Pearson_R2': f'{gvfa_m["pearson_r2"]:.6f}',
                            'Pearson_R':  f'{gvfa_m["pearson_r"]:.6f}',
                        })
                    print(f"  [GVFA-layers] Saved predictions to {gvfa_pred_path}")
            # ── End GVFA-layer-only regression ───────────────────────────────

            if args.use_ridge:
                if args.ridge_type == 'ridgecv':
                    reg = RidgeCV(
                        alphas=np.logspace(-2, 4, 100),
                        cv=5,
                        scoring='neg_mean_squared_error',
                    )
                    reg.fit(train_emb, train_labels)
                    pred = reg.predict(test_emb)
                else:
                    kr = KernelRidge(kernel='rbf')
                    param_grid = {
                        'alpha': np.logspace(-4, 4, 40),
                        'gamma': np.logspace(-6, 2, 25),
                    }
                    reg = GridSearchCV(
                        kr,
                        param_grid,
                        cv=5,
                        scoring='neg_mean_squared_error',
                        n_jobs=-1,
                        refit=True,
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
                reg.fit(train_emb, train_labels, eval_set=[(test_emb, test_labels)], verbose=False)
                pred = reg.predict(test_emb)

            m = compute_metrics(test_labels, pred)
            print(
                f"Dim={dim}  sigma_pi=[{orders_str}]  ({sigma_tag})  head={reg_tag}  "
                f"RMSE={m['rmse']:.4f}  STD_err={m['std_err']:.4f}  MAE={m['mae']:.4f}  "
                f"R2_COD={m['r2_cod']:.4f}  Pearson_R2={m['pearson_r2']:.4f}",
            )

            if save:
                y_true = np.asarray(test_labels).ravel()
                y_pred = np.asarray(pred).ravel()
                pred_path = os.path.join(
                    args.save_results, f'predictions_dim{dim}_{sigma_tag}_{reg_tag}.csv',
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
                        'regressor': reg_tag,
                        'RMSE': f'{m["rmse"]:.6f}',
                        'STD_err': f'{m["std_err"]:.6f}',
                        'MAE': f'{m["mae"]:.6f}',
                        'R2_COD': f'{m["r2_cod"]:.6f}',
                        'Pearson_R2': f'{m["pearson_r2"]:.6f}',
                        'Pearson_R': f'{m["pearson_r"]:.6f}',
                    })
                print(f"  Saved predictions to {pred_path}")

            if args.export_analysis_dir:
                sub = os.path.join(
                    args.export_analysis_dir,
                    f"ridge_dim_{dim}_sigma_{sigma_tag}_{reg_tag}",
                )
                export_hypervector_analysis(
                    model_eq1, train_graphs, device, sub, "train", args.batch_size,
                )
                export_hypervector_analysis(
                    model_eq1, test_graphs, device, sub, "test", args.batch_size,
                )
                print(f"  Saved HV analysis under {sub}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    seeds = _parse_seeds(args.seeds) if args.seeds else [args.seed]

    print("Running for dataset: ", args.dataset)
    print(f"Seeds: {seeds}")

    train_data, test_data = load_data(dataset=args.dataset, seed=seeds[0])
    # train_data, test_data = train_data[200:400], test_data
    print("Building graph objects (ETKDGv3 conformers)...")
    train_graphs_base = create_graph_list(train_data)
    test_graphs_base = create_graph_list(test_data)
    print(f"  Train graphs: {len(train_graphs_base)}, Test graphs: {len(test_graphs_base)}")
    sample_ea = train_graphs_base[0].edge_attr
    if sample_ea is not None and sample_ea.numel() > 0:
        print(
            "  Sample edge_attr[0] (bond_type, conj, in_ring, length, stereo): "
            f"{sample_ea[0].tolist()}"
        )

    for seed in seeds:
        print("\n" + "=" * 80)
        print(f"SEED = {seed}")
        print("=" * 80)

        args.seed = seed

        if args.save_results:
            original_save = args.save_results
            args.save_results = os.path.join(original_save, f"seed_{seed}")

        run_gvfa_ridge(
            args, train_data, test_data, device,
            train_graphs_base, test_graphs_base,
        )

        if args.save_results:
            args.save_results = original_save


if __name__ == '__main__':
    main()
