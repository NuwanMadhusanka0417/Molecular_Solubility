"""
GVFA for molecular solubility prediction.

Pipeline: GVFA encoder -> embeddings -> RidgeCV on full training set -> train / test metrics.

Optional: K-fold CV on training embeddings (--k_folds >= 2). Default is full-dataset Ridge only.

Train: solubility_1.csv.  Test: testset_novel.csv.

Use --save-analysis to dump node-level tensors and logS (y_node) to .npz (+ JSON sign stats).
"""
import argparse
import json
import os

from src.create_graphs import create_graph_list
from src.load_data import load_data
from src.VSA_conversion import VSA_conversion, configure_other_binding
from src.embeddings import getEmbedding
from models.graphcnnVSA_Binding_FULL import GraphCNN

import torch
import numpy as np
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, RidgeCV


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _fit_ridge_cv(X, y):
    """Ridge with CV over alphas; safe for small training sets."""
    n = len(X)
    if n < 3:
        reg = Ridge(alpha=1.0)
        reg.fit(X, y)
        return reg
    cv_inner = min(5, n)
    reg = RidgeCV(alphas=np.logspace(-4, 2, 50), cv=cv_inner, scoring="neg_mean_squared_error")
    reg.fit(X, y)
    return reg


def _metrics_list_to_mean_std(rows):
    """rows: list of metric dicts -> dict of metric -> (mean, std)."""
    keys = ["rmse", "mae", "r2_cod", "pearson_r2"]
    out = {}
    for k in keys:
        vals = np.array([r[k] for r in rows], dtype=np.float64)
        out[k] = (float(np.mean(vals)), float(np.std(vals)))
    return out


def _print_metrics_line(prefix, m):
    print(
        f"  {prefix}  RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}  "
        f"R2_COD={m['r2_cod']:.4f}  Pearson_R2={m['pearson_r2']:.4f}"
    )


def _print_cv_mean_std(name, ms_std):
    print(f"\n--- {name} (mean ± std over folds) ---")
    for k in ["rmse", "mae", "r2_cod", "pearson_r2"]:
        mu, sd = ms_std[k]
        label = k.upper() if k != "pearson_r2" else "Pearson_R2"
        print(f"  {label}: {mu:.4f} ± {sd:.4f}")


def _capture_to_numpy(cap):
    return {k: (v.numpy() if isinstance(v, torch.Tensor) else v) for k, v in cap.items()}


def _merge_analysis_batches(batch_dicts):
    """Concatenate node dimension (axis 0) for each array key."""
    if not batch_dicts:
        return {}
    keys = batch_dicts[0].keys()
    out = {}
    for k in keys:
        parts = [b[k] for b in batch_dicts]
        out[k] = np.concatenate(parts, axis=0)
    return out


def _sign_distribution_stats(merged_np):
    """After torch.sign: fractions of +1, -1, 0 in layer_*_post_sign tensors."""
    stats = {}
    for k, arr in merged_np.items():
        if "post_sign" not in k:
            continue
        a = np.asarray(arr).ravel()
        stats[k] = {
            "frac_positive": float(np.mean(a > 0)),
            "frac_negative": float(np.mean(a < 0)),
            "frac_zero": float(np.mean(a == 0)),
        }
    return stats


@torch.no_grad()
def export_gvfa_analysis(model, graphs, device, out_npz_path, batch_size=8, meta=None):
    """
    Run GraphCNN with analysis_capture over batches; save merged .npz + JSON sidecar with sign stats.
    """
    model.to(device)
    model.eval()
    batch_dicts = []
    graph_offset = 0
    for start in range(0, len(graphs), batch_size):
        bg = graphs[start : start + batch_size]
        cap = {}
        model(bg, analysis_capture=cap)
        cap = _capture_to_numpy(cap)
        cap["graph_id"] = np.asarray(cap["graph_id"], dtype=np.int64) + graph_offset
        graph_offset += len(bg)
        batch_dicts.append(cap)

    merged = _merge_analysis_batches(batch_dicts)
    os.makedirs(os.path.dirname(out_npz_path) or ".", exist_ok=True)
    np.savez_compressed(out_npz_path, **merged)

    side = {"npz": os.path.basename(out_npz_path), "n_nodes": int(merged["y_node"].shape[0])}
    if meta:
        side.update(meta)
    side["sign_distribution_post_sign_layers"] = _sign_distribution_stats(merged)
    json_path = os.path.splitext(out_npz_path)[0] + "_meta.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(side, f, indent=2)
    print(f"  Saved analysis: {out_npz_path}  ({side['n_nodes']} nodes)  meta: {json_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="GVFA + Ridge regression for molecular solubility")
    p.add_argument(
        "--dataset",
        type=str,
        default="solubility_novel",
        choices=["old", "solubility_novel", "new"],
        help="solubility_novel: train solubility_1.csv, test testset_novel.csv",
    )
    p.add_argument(
        "--dim",
        type=int,
        default=1000,
        help="Single VSA dimension (used only when --dims is empty)",
    )
    p.add_argument(
        "--dims",
        type=str,
        default="1000,2000,5000,10000",
        help="Comma-separated VSA dimensions. Empty string uses --dim once.",
    )
    p.add_argument(
        "--k_folds",
        type=int,
        default=0,
        help="If >=2, run K-fold CV on training embeddings before the final Ridge fit. "
        "Default 0: only fit Ridge on full training data (no K-fold block).",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed for KFold shuffle (when --k_folds>=2)")
    p.add_argument(
        "--gvfa-binding",
        type=str,
        default="circular",
        choices=["circular", "elementwise"],
        help="Binding inside GraphCNN (GVFA): circular=FFT/HRR, elementwise=Hadamard",
    )
    p.add_argument(
        "--other-binding",
        type=str,
        default="elementwise",
        choices=["circular", "elementwise"],
        help="Default for hv_bind / vsa_message_passing (configure_other_binding)",
    )
    p.add_argument(
        "--save-analysis",
        action="store_true",
        help="Save node-level HV tensors + y (logS) to .npz for sigma-pi / binarization analysis",
    )
    p.add_argument(
        "--analysis-dir",
        type=str,
        default="gvfa_analysis_dump",
        help="Output directory for --save-analysis",
    )
    p.add_argument(
        "--analysis-batch-size",
        type=int,
        default=8,
        help="Batch size when running analysis export (lower if GPU OOM)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# GVFA + Ridge
# ---------------------------------------------------------------------------

def run_gvfa_ridge(args, train_data, test_data, device):
    dims = (
        [int(x.strip()) for x in args.dims.split(",") if x.strip()]
        if args.dims.strip()
        else [args.dim]
    )

    for dim in dims:
        train_graphs = create_graph_list(train_data)
        test_graphs = create_graph_list(test_data)
        test_HVs = VSA_conversion(test_graphs.copy(), dim, projection_type="orthogonal")
        train_HVs = VSA_conversion(train_graphs.copy(), dim, projection_type="orthogonal")

        n_train = len(train_HVs)
        kf_note = "off" if args.k_folds < 2 else str(args.k_folds)
        print(f"\n{'='*60}\nDim={dim}  |  training graphs: {n_train}  |  K-fold CV: {kf_note}\n{'='*60}")

        model_eq1 = GraphCNN(
            test_HVs[0].node_features.shape[1],
            5,
            1,
            "sum",
            "sum",
            device,
            10,
            edge_feat_dim=5,
            edge_projection_type="orthogonal",
            use_reservoir=True,
            hop_decay=0.85,
            sigma_pi_orders=[0, 1],
            gvfa_binding=args.gvfa_binding,
        )
        if args.save_analysis:
            os.makedirs(args.analysis_dir, exist_ok=True)
            meta = {
                "dataset": args.dataset,
                "dim": dim,
                "gvfa_binding": args.gvfa_binding,
                "sigma_pi_orders": [0, 1],
            }
            train_npz = os.path.join(
                args.analysis_dir, f"gvfa_analysis_train_dim{dim}.npz"
            )
            test_npz = os.path.join(args.analysis_dir, f"gvfa_analysis_test_dim{dim}.npz")
            print("\n--- Exporting GVFA analysis tensors ---")
            export_gvfa_analysis(
                model_eq1,
                train_HVs,
                device,
                train_npz,
                batch_size=args.analysis_batch_size,
                meta={**meta, "split": "train"},
            )
            export_gvfa_analysis(
                model_eq1,
                test_HVs,
                device,
                test_npz,
                batch_size=args.analysis_batch_size,
                meta={**meta, "split": "test"},
            )

        train_emb, train_labels_t = getEmbedding(
            model_eq1, device, train_HVs, use_size_aware=True, hop_alpha=1.0
        )
        test_emb, test_labels_t = getEmbedding(
            model_eq1, device, test_HVs, use_size_aware=True, hop_alpha=1.0
        )
        train_emb = train_emb.squeeze(0).cpu().numpy()
        test_emb = test_emb.squeeze(0).cpu().numpy()
        train_y = np.asarray(train_labels_t.cpu().numpy().ravel(), dtype=np.float64)
        test_y = np.asarray(test_labels_t.cpu().numpy().ravel(), dtype=np.float64)

        if args.k_folds < 2:
            print("\n--- K-fold CV: skipped (default: use --k_folds 2 or higher to enable) ---")
        else:
            k = min(args.k_folds, n_train)
            if k < 2:
                print(
                    "  Warning: not enough training graphs for K-fold (need at least 2); skipping CV."
                )
            else:
                kf = KFold(n_splits=k, shuffle=True, random_state=args.seed)
                fold_rows = []
                for _, (tr_idx, va_idx) in enumerate(kf.split(np.arange(n_train))):
                    X_tr, X_va = train_emb[tr_idx], train_emb[va_idx]
                    y_tr, y_va = train_y[tr_idx], train_y[va_idx]
                    reg_fold = _fit_ridge_cv(X_tr, y_tr)
                    pred_va = reg_fold.predict(X_va)
                    fold_rows.append(compute_metrics(y_va, pred_va))

                cv_ms_std = _metrics_list_to_mean_std(fold_rows)
                _print_cv_mean_std(f"K-fold CV (K={k}) validation", cv_ms_std)

        reg_full = _fit_ridge_cv(train_emb, train_y)
        pred_train = reg_full.predict(train_emb)
        pred_test = reg_full.predict(test_emb)

        m_train = compute_metrics(train_y, pred_train)
        m_test = compute_metrics(test_y, pred_test)

        print("\n--- Full training set (in-sample, Ridge on all training data) ---")
        _print_metrics_line("", m_train)
        print("\n--- Independent test set ---")
        _print_metrics_line("", m_test)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    configure_other_binding(args.other_binding)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running for dataset:", args.dataset)
    print(
        f"GVFA binding (GraphCNN): {args.gvfa_binding}  |  "
        f"other binding (hv_bind default): {args.other_binding}"
    )
    train_data, test_data = load_data(dataset=args.dataset)
    run_gvfa_ridge(args, train_data, test_data, device)


if __name__ == "__main__":
    main()
