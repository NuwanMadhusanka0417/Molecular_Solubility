"""
GVFA for molecular solubility — Llinas2020 cross-dataset evaluation.

Train  : Cui et al. (solubility_1.csv)
Test   : Llinas2020 set1  (Llinas2020_set1.csv)
         Llinas2020 set2  (Llinas2020_set2.csv)

The GVFA encoder is identical to GVFA_edge_main.py.
Ridge regression is trained ONCE on the Cui et al. training set
and then evaluated on BOTH Llinas test sets in the same run.

Data loading is handled by src/load_data_llinas.py — no existing
source files are modified.

Usage examples
──────────────
    # Default paths, single seed, single dim
    python GVFA_edge_llinas.py --dim 1000 --seed 42

    # Multiple dims and seeds
    python GVFA_edge_llinas.py --dims 1000,2000,5000,10000 --seeds 0-9

    # Custom dataset paths
    python GVFA_edge_llinas.py --train_path final_data/solubility_1.csv \\
                                --set1_path  final_data/Llinas2020_set1.csv \\
                                --set2_path  final_data/Llinas2020_set2.csv

    # Save results
    python GVFA_edge_llinas.py --dims 1000,2000,5000 --seeds 0-4 \\
                                --save_results results/llinas/
"""

import argparse
import csv
import os
import random

import torch
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV

# ── existing modules (unchanged) ──────────────────────────────────────────
from src.create_graphs import create_graph_list
from src.VSA_conversion import VSA_conversion
from src.embeddings import getEmbedding
from models.graphcnnVSA_Binding_FULL import GraphCNN

# ── new data loader (does not modify any existing file) ───────────────────
from src.load_data_llinas import load_llinas_data


# ---------------------------------------------------------------------------
# Helpers  (identical to GVFA_edge_main.py)
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred):
    """RMSE, MAE, std of residuals, R² (COD), Pearson R² — original logS units."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    err    = y_true - y_pred
    rmse   = np.sqrt(np.mean(err ** 2))
    std_err = float(np.std(err, ddof=0))
    mae    = np.mean(np.abs(err))
    sse    = np.sum(err ** 2)
    sst    = np.sum((y_true - np.mean(y_true)) ** 2)
    r2_cod = 1.0 - (sse / sst) if sst > 0 else 0.0
    pr, _  = pearsonr(y_true, y_pred) if len(y_true) >= 2 else (0.0, 1.0)
    return {
        "rmse":       rmse,
        "std_err":    std_err,
        "mae":        mae,
        "r2_cod":     r2_cod,
        "pearson_r2": pr ** 2,
        "pearson_r":  pr,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="GVFA — train on Cui et al., test on Llinas2020 set1 & set2"
    )

    # ── Dataset paths ───────────────────────────────────────────────────
    p.add_argument(
        "--train_path", type=str, default="final_data/solubility_1.csv",
        help="Path to training CSV (Cui et al., default: final_data/solubility_1.csv)",
    )
    p.add_argument(
        "--set1_path", type=str, default="final_data/Llinas2020_set1.csv",
        help="Path to Llinas2020 set1 CSV (default: final_data/Llinas2020_set1.csv)",
    )
    p.add_argument(
        "--set2_path", type=str, default="final_data/Llinas2020_set2.csv",
        help="Path to Llinas2020 set2 CSV (default: final_data/Llinas2020_set2.csv)",
    )

    # ── VSA / model ─────────────────────────────────────────────────────
    p.add_argument("--dim",  type=int, default=1000, help="Single VSA dimension")
    p.add_argument(
        "--dims", type=str, default="1000,2000,5000,10000",
        help="Comma-separated dimensions to sweep (overrides --dim)",
    )
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument(
        "--hop_decay", type=float, default=0.85,
        help="Hop decay factor for the GVFA reservoir (default: 0.85)",
    )
    p.add_argument(
        "--sigma_pi", type=str, default="all",
        help=(
            "Sigma-Pi order sets to sweep. "
            "Presets: 'all' = [0],[1],[2],[0,1],[0,1,2]; "
            "'legacy' = [0,1] only. "
            "Or a single set, e.g. '0,1,2'."
        ),
    )

    # ── Ridge ───────────────────────────────────────────────────────────
    p.add_argument("--use_ridge",  action="store_true",  default=True)
    p.add_argument("--no_ridge",   action="store_false", dest="use_ridge")

    # ── Seeds ───────────────────────────────────────────────────────────
    p.add_argument("--seed",  type=int, default=42,
                   help="Single RNG seed (use --seeds for multiple)")
    p.add_argument(
        "--seeds", type=str, default=None,
        help="Multiple seeds: range '0-49' or comma-separated '0,1,42'. Overrides --seed.",
    )

    # ── Output ──────────────────────────────────────────────────────────
    p.add_argument(
        "--save_results", type=str, default=None,
        help="If set, save CSV files (summary + per-molecule predictions) to this directory.",
    )

    return p.parse_args()


# ---------------------------------------------------------------------------
# Seed / sigma-pi parsing  (identical to GVFA_edge_main.py)
# ---------------------------------------------------------------------------

def _parse_seeds(s: str):
    """Parse '0-49' or '0,1,42,123' into a list of ints."""
    s = s.strip()
    if "-" in s and "," not in s:
        lo, hi = s.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_sigma_pi_arg(s: str):
    """Return list of (sigma_pi_orders, tag) for the ridge loop."""
    key = s.strip().lower()
    if key == "all":
        return [
            ([0],       "o0"),
            ([1],       "o1"),
            ([2],       "o2"),
            ([0, 1],    "o0_o1"),
            ([0, 1, 2], "o0_o1_o2"),
        ]
    if key == "legacy":
        return [([0, 1], "o0_o1")]
    orders = [int(x.strip()) for x in s.split(",") if x.strip() != ""]
    if not orders:
        raise ValueError(f"Invalid --sigma_pi: {s!r}")
    tag = "o" + "_o".join(str(t) for t in orders)
    return [(orders, tag)]


# ---------------------------------------------------------------------------
# Core GVFA + Ridge loop
# ---------------------------------------------------------------------------

def run_gvfa_ridge(args, train_data, test_datasets, device):
    """
    Train GVFA + Ridge on train_data and evaluate on every dataset in
    test_datasets.

    Parameters
    ----------
    args         : parsed CLI arguments
    train_data   : ZINCLikeCSV — Cui et al. training set
    test_datasets: list of (name: str, dataset: ZINCLikeCSV) tuples
                   e.g. [("Llinas_set1", set1_data), ("Llinas_set2", set2_data)]
    device       : torch.device
    """
    seed         = args.seed
    dims         = [int(x) for x in args.dims.split(",")]
    sigma_configs = _parse_sigma_pi_arg(args.sigma_pi)

    # ── Initialise CSV summary writer ──────────────────────────────────
    save = args.save_results is not None
    if save:
        os.makedirs(args.save_results, exist_ok=True)
        summary_path = os.path.join(args.save_results, "results_summary.csv")
        summary_fields = [
            "test_set", "dim", "sigma_pi", "sigma_tag", "seed",
            "RMSE", "STD_err", "MAE", "R2_COD", "Pearson_R2", "Pearson_R",
        ]
        with open(summary_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=summary_fields).writeheader()

    # ── Outer loop: VSA dimension ──────────────────────────────────────
    for dim in dims:
        # Fix all RNG state for this dim
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Build graph lists and project to hypervectors — once per dim
        print(f"\n{'─' * 70}")
        print(f"  DIM = {dim}   SEED = {seed}")
        print(f"{'─' * 70}")

        # Training set
        train_graphs = create_graph_list(train_data)
        train_HVs = VSA_conversion(
            train_graphs.copy(), dim, projection_type="orthogonal", seed=seed,
        )

        # Test sets — build graph list + HVs for each, store alongside name
        test_set_info = []                      # [(name, HVs, graphs), ...]
        for test_name, test_data in test_datasets:
            test_graphs_i = create_graph_list(test_data)
            test_HVs_i = VSA_conversion(
                test_graphs_i.copy(), dim, projection_type="orthogonal", seed=seed,
            )
            test_set_info.append((test_name, test_HVs_i, test_graphs_i))

        # ── Inner loop: sigma-pi configuration ────────────────────────
        for sigma_pi_orders, sigma_tag in sigma_configs:
            # Reset torch seed so model weights are identical across sigma-pi runs
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            # Build GVFA encoder (no learnable weights — seed controls fixed buffers)
            model_eq1 = GraphCNN(
                train_HVs[0].node_features.shape[1], 4, 0, "sum", "sum", device, 10,
                edge_feat_dim=5,
                edge_projection_type="orthogonal",
                use_reservoir=True,
                hop_decay=args.hop_decay,
                sigma_pi_orders=sigma_pi_orders,
                rng_seed=seed,
            )

            # ── Training embeddings + Ridge fit (done once per config) ─
            train_emb, train_labels = getEmbedding(
                model_eq1, device, train_HVs, use_size_aware=True, hop_alpha=1.0,
            )
            train_emb = train_emb.squeeze(0)

            if args.use_ridge:
                reg = RidgeCV(
                    alphas=np.logspace(-4, 2, 50),
                    cv=5,
                    scoring="neg_mean_squared_error",
                )
                reg.fit(train_emb, train_labels)
            else:
                from xgboost import XGBRegressor
                # Fit without a validation set since we have multiple test sets
                reg = XGBRegressor(
                    n_estimators=2000, learning_rate=0.03, max_depth=7,
                    subsample=0.8, colsample_bytree=0.8,
                    reg_lambda=1.0, reg_alpha=0.0,
                    random_state=seed, n_jobs=4, tree_method="hist",
                )
                reg.fit(train_emb, train_labels, verbose=False)

            orders_str = ",".join(str(x) for x in sigma_pi_orders)
            alpha_part = ""
            if args.use_ridge:
                best_alpha = float(np.asarray(reg.alpha_).ravel()[0])
                alpha_part = f"  Ridge_alpha={best_alpha:.6g}"

            # ── Evaluate on each test set ──────────────────────────────
            for test_name, test_HVs_i, _ in test_set_info:

                test_emb, test_labels = getEmbedding(
                    model_eq1, device, test_HVs_i, use_size_aware=True, hop_alpha=1.0,
                )
                test_emb = test_emb.squeeze(0)

                pred = reg.predict(test_emb)
                m    = compute_metrics(test_labels, pred)

                # Console output
                print(
                    f"  [{test_name}]  "
                    f"Dim={dim}  sigma_pi=[{orders_str}]  ({sigma_tag}){alpha_part}  "
                    f"RMSE={m['rmse']:.4f}  STD_err={m['std_err']:.4f}  "
                    f"MAE={m['mae']:.4f}  R2_COD={m['r2_cod']:.4f}  "
                    f"Pearson_R2={m['pearson_r2']:.4f}"
                )

                # CSV output
                if save:
                    y_true = np.asarray(test_labels).ravel()
                    y_pred = np.asarray(pred).ravel()

                    # Per-molecule predictions
                    pred_fname = f"predictions_{test_name}_dim{dim}_{sigma_tag}.csv"
                    pred_path  = os.path.join(args.save_results, pred_fname)
                    with open(pred_path, "w", newline="") as f:
                        w = csv.writer(f)
                        w.writerow(["y_true", "y_pred", "error"])
                        for yt, yp in zip(y_true, y_pred):
                            w.writerow([f"{yt:.6f}", f"{yp:.6f}", f"{yt - yp:.6f}"])

                    # Append to running summary
                    with open(summary_path, "a", newline="") as f:
                        csv.DictWriter(f, fieldnames=summary_fields).writerow({
                            "test_set":   test_name,
                            "dim":        dim,
                            "sigma_pi":   f"[{orders_str}]",
                            "sigma_tag":  sigma_tag,
                            "seed":       seed,
                            "RMSE":       f"{m['rmse']:.6f}",
                            "STD_err":    f"{m['std_err']:.6f}",
                            "MAE":        f"{m['mae']:.6f}",
                            "R2_COD":     f"{m['r2_cod']:.6f}",
                            "Pearson_R2": f"{m['pearson_r2']:.6f}",
                            "Pearson_R":  f"{m['pearson_r']:.6f}",
                        })
                    print(f"    Saved predictions → {pred_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Resolve --dims (takes precedence over --dim if user sets it explicitly;
    # if user only passes --dim, honour it by replacing the default --dims)
    if args.dim != 1000:          # user explicitly set --dim
        args.dims = str(args.dim)

    seeds = _parse_seeds(args.seeds) if args.seeds else [args.seed]
    print(f"Seeds : {seeds}")
    print(f"Dims  : {args.dims}")

    # ── Load data ONCE (same graphs are reused across seeds) ──────────
    train_data, set1_data, set2_data = load_llinas_data(
        train_path=args.train_path,
        set1_path=args.set1_path,
        set2_path=args.set2_path,
    )

    # Both Llinas sets tested in the same run
    test_datasets = [
        ("Llinas2020_set1", set1_data),
        ("Llinas2020_set2", set2_data),
    ]

    # ── Seed loop ─────────────────────────────────────────────────────
    for seed in seeds:
        print("\n" + "=" * 80)
        print(f"SEED = {seed}")
        print("=" * 80)

        args.seed = seed

        # Nest results under seed sub-directory when saving
        if args.save_results:
            original_save    = args.save_results
            args.save_results = os.path.join(original_save, f"seed_{seed}")

        run_gvfa_ridge(args, train_data, test_datasets, device)

        if args.save_results:
            args.save_results = original_save


if __name__ == "__main__":
    main()
