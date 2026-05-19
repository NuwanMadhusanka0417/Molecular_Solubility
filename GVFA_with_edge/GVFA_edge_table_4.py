"""
Table 4 Cross-Dataset Generalization Test Script - UPDATED
═══════════════════════════════════════════════════════════

Evaluate Enhanced GVFA on Llinas2020 set1 & set2 (SolTransNet).
- Train ridge regression ONCE on solubility_1.csv (Cui et al.)
- Test DIRECTLY on Llinas2020 datasets (no k-fold CV)
- Support for MULTIPLE SEEDS and MULTIPLE DIMENSIONS
- Report mean ± std across seeds

Key Features:
✓ Direct training (no k-fold CV)
✓ Multi-seed support (0-9, 0-4, etc.)
✓ Multi-dimension support (1000, 2000, 5000, 10000)
✓ Results saved as JSON and CSV
✓ Consistent with Table 3 methodology

Usage:
    # Single seed, single dimension
    python GVFA_edge_table_4.py --dataset_dir datasets/ --dim 5000 --seed 0
    
    # Multiple seeds (0-9)
    python GVFA_edge_table_4.py --dataset_dir datasets/ --dim 5000 --seeds 0-9
    
    # Multiple dimensions and seeds
    python GVFA_edge_table_4.py --dataset_dir datasets/ --dims 1000,2000,5000,10000 --seeds 0-9
    
    # Save results
    python GVFA_edge_table_4.py --dataset_dir datasets/ --dims 5000 --seeds 0-9 --save_results results/table4/
"""

import argparse
import csv
import os
import sys
import copy
import random
import json
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ✓ Import your existing code (unchanged)
from src.create_graphs import create_graph_list
from src.load_data import ZINCLikeCSV
from src.VSA_conversion import VSA_conversion
from src.embeddings import getEmbedding
from models.graphcnnVSA_Binding_FULL import GraphCNN


# ═══════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════

def compute_metrics(y_true, y_pred):
    """Compute RMSE, MAE, R², Pearson R²."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    err = y_true - y_pred
    rmse = np.sqrt(np.mean(err ** 2))
    mae = np.mean(np.abs(err))
    sse = np.sum(err ** 2)
    sst = np.sum((y_true - np.mean(y_true)) ** 2)
    r2_cod = 1.0 - (sse / sst) if sst > 0 else 0.0
    pr, _ = pearsonr(y_true, y_pred) if len(y_true) >= 2 else (0.0, 1.0)
    return {
        "rmse": rmse,
        "mae": mae,
        "r2_cod": r2_cod,
        "pearson_r2": pr ** 2,
        "pearson_r": pr,
    }


def save_predictions_csv(y_true, y_pred, out_path):
    """Save predictions to CSV."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['y_true', 'y_pred', 'error'])
        for yt, yp in zip(y_true, y_pred):
            w.writerow([f'{yt:.6f}', f'{yp:.6f}', f'{yt - yp:.6f}'])
    print(f"Predictions saved to {out_path}")


# ═══════════════════════════════════════════════════════════
# PHASE 1: Train on Cui et al.
# ═══════════════════════════════════════════════════════════

def phase1_train_projections(args, device, dim, seed):
    """
    Train on Cui et al. training set.
    Returns: embeddings and ridge model
    """
    print(f"\n{'─' * 80}")
    print(f"Training on Cui et al. (D={dim}, seed={seed})")
    print(f"{'─' * 80}")
    
    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Load Cui et al. data
    train_path = os.path.join(args.dataset_dir, "solubility_1.csv")
    
    train_df = pd.read_csv(train_path)
    train_df = train_df.dropna(subset=["SMILES", "logS"])
    
    # Create graph lists
    train_dataset = ZINCLikeCSV(train_df, smiles_col="SMILES", target_col="logS")
    train_graphs = create_graph_list(train_dataset)
    
    print(f"Train: {len(train_df)} molecules")
    
    # Project to hypervectors
    train_HVs = VSA_conversion(
        copy.deepcopy(train_graphs),
        dim,
        projection_type="orthogonal",
        seed=seed
    )
    
    # Build GVFA model
    model = GraphCNN(
        train_HVs[0].node_features.shape[1], 4, 0, 'sum', 'sum', device, 10,
        edge_feat_dim=5,
        edge_projection_type="orthogonal",
        use_reservoir=True,
        hop_decay=args.hop_decay,
        sigma_pi_orders=args.sigma_pi_orders,
        rng_seed=seed,
    )
    
    # Get embeddings
    train_emb, train_labels = getEmbedding(
        model, device, train_HVs, use_size_aware=True, hop_alpha=1.0
    )
    train_emb = train_emb.squeeze(0)
    
    # Train ridge regression
    ridge = RidgeCV(alphas=np.logspace(-4, 2, 50), cv=5, scoring='neg_mean_squared_error')
    ridge.fit(train_emb.cpu().numpy(), train_labels.cpu().numpy())
    
    return model, ridge, train_HVs


# ═══════════════════════════════════════════════════════════
# PHASE 2: Direct Test on Llinas2020 (No K-Fold)
# ═══════════════════════════════════════════════════════════

def phase2_test_llinas_direct(args, device, model, ridge, dim, seed, dataset_name, csv_path):
    """
    Test DIRECTLY on Llinas2020 dataset (no k-fold).
    Returns: metrics dictionary
    """
    
    # Load Llinas dataset
    llinas_df = pd.read_csv(csv_path)
    llinas_df = llinas_df.dropna(subset=["SMILES", "logS"])
    
    print(f"  {dataset_name}: {len(llinas_df)} molecules")
    
    # Create graph lists
    llinas_dataset = ZINCLikeCSV(llinas_df, smiles_col="SMILES", target_col="logS")
    llinas_graphs = create_graph_list(llinas_dataset)
    
    # Project using SAME seed (frozen projections)
    llinas_HVs = VSA_conversion(
        copy.deepcopy(llinas_graphs),
        dim,
        projection_type="orthogonal",
        seed=seed  # ← SAME SEED = FROZEN PROJECTIONS
    )
    
    # Get embeddings
    llinas_emb, llinas_labels = getEmbedding(
        model, device, llinas_HVs, use_size_aware=True, hop_alpha=1.0
    )
    llinas_emb = llinas_emb.squeeze(0)
    
    # Predict directly (no k-fold, just single prediction)
    y_pred = ridge.predict(llinas_emb.cpu().numpy())
    y_true = llinas_labels.cpu().numpy()
    
    # Compute metrics
    metrics = compute_metrics(y_true, y_pred)
    
    return metrics, y_true, y_pred


# ═══════════════════════════════════════════════════════════
# Main Evaluation Loop
# ═══════════════════════════════════════════════════════════

def run_evaluation(args, device):
    """
    Run evaluation for all dimensions and seeds.
    """
    
    dims = [int(x.strip()) for x in args.dims.split(',')]
    seeds = _parse_seeds(args.seeds) if args.seeds else [args.seed]
    
    print(f"\n{'=' * 80}")
    print(f"Table 4: SolTransNet Cross-Dataset Evaluation (Direct Training)")
    print(f"{'=' * 80}")
    print(f"Dimensions: {dims}")
    print(f"Seeds: {seeds}")
    print(f"Hop decay: {args.hop_decay}")
    print(f"Sigma-Pi orders: {args.sigma_pi_orders}")
    print(f"{'=' * 80}\n")
    
    # Storage for results
    results_by_dim = defaultdict(lambda: {
        "set1": {"rmses": [], "maes": [], "r2s": [], "pearson_r2s": []},
        "set2": {"rmses": [], "maes": [], "r2s": [], "pearson_r2s": []},
    })
    
    all_results_json = []
    
    # Run for each dimension
    for dim in dims:
        print(f"\n{'#' * 80}")
        print(f"DIMENSION = {dim}")
        print(f"{'#' * 80}")
        
        set1_path = os.path.join(args.dataset_dir, "Llinas2020_set1.csv")
        set2_path = os.path.join(args.dataset_dir, "Llinas2020_set2.csv")
        
        # Run for each seed
        for seed in seeds:
            print(f"\n{'=' * 80}")
            print(f"SEED = {seed}")
            print(f"{'=' * 80}")
            
            try:
                # PHASE 1: Train on Cui et al.
                model, ridge, train_HVs = phase1_train_projections(args, device, dim, seed)
                
                # PHASE 2A: Test on Llinas set1 (DIRECT, NO K-FOLD)
                metrics_set1, y_true_set1, y_pred_set1 = phase2_test_llinas_direct(
                    args, device, model, ridge, dim, seed, "Llinas2020 set1", set1_path
                )
                
                # PHASE 2B: Test on Llinas set2 (DIRECT, NO K-FOLD)
                metrics_set2, y_true_set2, y_pred_set2 = phase2_test_llinas_direct(
                    args, device, model, ridge, dim, seed, "Llinas2020 set2", set2_path
                )
                
                # Store results
                results_by_dim[dim]["set1"]["rmses"].append(metrics_set1["rmse"])
                results_by_dim[dim]["set1"]["maes"].append(metrics_set1["mae"])
                results_by_dim[dim]["set1"]["r2s"].append(metrics_set1["r2_cod"])
                results_by_dim[dim]["set1"]["pearson_r2s"].append(metrics_set1["pearson_r2"])
                
                results_by_dim[dim]["set2"]["rmses"].append(metrics_set2["rmse"])
                results_by_dim[dim]["set2"]["maes"].append(metrics_set2["mae"])
                results_by_dim[dim]["set2"]["r2s"].append(metrics_set2["r2_cod"])
                results_by_dim[dim]["set2"]["pearson_r2s"].append(metrics_set2["pearson_r2"])
                
                # Print results
                print(f"\nSet1 - RMSE: {metrics_set1['rmse']:.4f}, MAE: {metrics_set1['mae']:.4f}, "
                      f"R²: {metrics_set1['r2_cod']:.4f}")
                print(f"Set2 - RMSE: {metrics_set2['rmse']:.4f}, MAE: {metrics_set2['mae']:.4f}, "
                      f"R²: {metrics_set2['r2_cod']:.4f}")
                
                # Save predictions
                if args.save_results:
                    os.makedirs(args.save_results, exist_ok=True)
                    pred_path_1 = os.path.join(args.save_results, f"set1_dim{dim}_seed{seed}.csv")
                    pred_path_2 = os.path.join(args.save_results, f"set2_dim{dim}_seed{seed}.csv")
                    save_predictions_csv(y_true_set1, y_pred_set1, pred_path_1)
                    save_predictions_csv(y_true_set2, y_pred_set2, pred_path_2)
                
                # Store JSON results
                all_results_json.append({
                    "dim": dim,
                    "seed": seed,
                    "set1": metrics_set1,
                    "set2": metrics_set2,
                })
                
            except Exception as e:
                print(f"❌ Error for D={dim}, seed={seed}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    return results_by_dim, all_results_json


# ═══════════════════════════════════════════════════════════
# Print Final Results
# ═══════════════════════════════════════════════════════════

def print_final_results(results_by_dim, dims, seeds):
    """
    Print final Table 4 results aggregated by dimension and seed.
    """
    
    print("\n" + "=" * 120)
    print("TABLE 4: SolTransNet Cross-Dataset Generalization Results")
    print("(Direct Training, Multi-Seed Results)")
    print("=" * 120)
    
    for dim in dims:
        print(f"\n{'─' * 120}")
        print(f"DIMENSION = {dim}")
        print(f"{'─' * 120}")
        
        # Calculate mean ± std across seeds
        set1_rmse_mean = np.mean(results_by_dim[dim]["set1"]["rmses"])
        set1_rmse_std = np.std(results_by_dim[dim]["set1"]["rmses"])
        set1_r2_mean = np.mean(results_by_dim[dim]["set1"]["r2s"])
        set1_r2_std = np.std(results_by_dim[dim]["set1"]["r2s"])
        
        set2_rmse_mean = np.mean(results_by_dim[dim]["set2"]["rmses"])
        set2_rmse_std = np.std(results_by_dim[dim]["set2"]["rmses"])
        set2_r2_mean = np.mean(results_by_dim[dim]["set2"]["r2s"])
        set2_r2_std = np.std(results_by_dim[dim]["set2"]["r2s"])
        
        # Calculate improvements
        set1_improvement = (0.92 - set1_rmse_mean) / 0.92 * 100
        set2_improvement = (1.28 - set2_rmse_mean) / 1.28 * 100
        
        print(f"\n{'Test set model':<35} {'Llinas2020 set1':<40} {'Llinas2020 set2':<40}")
        print(f"{'─' * 35} {'─' * 40} {'─' * 40}")
        print(f"{'AttentiveFP (Ahmad et al.)':<35} {'0.92':<40} {'1.28':<40}")
        print(f"{'Enhanced GVFA (ours)':<35} {f'{set1_rmse_mean:.4f}±{set1_rmse_std:.4f}':<40} {f'{set2_rmse_mean:.4f}±{set2_rmse_std:.4f}':<40}")
        print(f"{'Improvement':<35} {f'{set1_improvement:+.1f}%':<40} {f'{set2_improvement:+.1f}%':<40}")
        
        print(f"\nR² (COD):")
        print(f"  Set1: {set1_r2_mean:.4f} ± {set1_r2_std:.4f}")
        print(f"  Set2: {set2_r2_mean:.4f} ± {set2_r2_std:.4f}")
        
        print(f"\nIndividual seed RMSEs:")
        print(f"  Set1: {[f'{r:.4f}' for r in results_by_dim[dim]['set1']['rmses']]}")
        print(f"  Set2: {[f'{r:.4f}' for r in results_by_dim[dim]['set2']['rmses']]}")
    
    print("\n" + "=" * 120)
    print("Note:")
    print("  - Results are mean ± std across multiple seeds")
    print("  - Projection matrices frozen per seed from Cui et al. training")
    print("  - Ridge regression trained once per seed on Cui et al.")
    print("  - Direct testing on Llinas2020 sets (NO k-fold CV)")
    print("  - Negative improvement = better than AttentiveFP baseline")
    print("=" * 120)


# ═══════════════════════════════════════════════════════════
# CLI & Main
# ═══════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Table 4: Direct Training with Multi-Seed & Multi-Dimension Support"
    )
    p.add_argument(
        '--dataset_dir', type=str, default='final_data/',
        help='Path to datasets directory'
    )
    p.add_argument(
        '--dim', type=int, default=5000,
        help='Single VSA dimension (default: 5000)'
    )
    p.add_argument(
        '--dims', type=str, default="1000,2000,5000,10000",
        help='Multiple dimensions: "1000,2000,5000,10000". Overrides --dim.'
    )
    p.add_argument(
        '--seed', type=int, default=0,
        help='Single random seed (default: 0)'
    )
    p.add_argument(
        '--seeds', type=str, default="0,1,2,3,4",
        help='Multiple seeds: "0-9" or "0,1,2,42". Overrides --seed.'
    )
    p.add_argument(
        '--sigma_pi_orders', type=str, default='0,1',
        help='Sigma-Pi polynomial orders (default: "0,1")'
    )
    p.add_argument(
        '--hop_decay', type=float, default=0.85,
        help='Hop decay factor for GVFA reservoir (default: 0.85)'
    )
    p.add_argument(
        '--save_results', type=str, default=None,
        help='If set, save results to this directory'
    )
    return p.parse_args()


def _parse_seeds(s: str):
    """Parse '0-9' or '0,1,2' into list of ints."""
    s = s.strip()
    if '-' in s and ',' not in s:
        lo, hi = s.split('-')
        return list(range(int(lo), int(hi) + 1))
    return [int(x.strip()) for x in s.split(',') if x.strip()]


def _parse_sigma_pi(s: str):
    """Parse sigma-pi orders from string."""
    return [int(x.strip()) for x in s.split(',') if x.strip()]


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Parse dimensions and seeds
    if args.dims:
        dims = [int(x.strip()) for x in args.dims.split(',')]
    else:
        dims = [args.dim]
    
    seeds = _parse_seeds(args.seeds) if args.seeds else [args.seed]
    args.sigma_pi_orders = _parse_sigma_pi(args.sigma_pi_orders)
    
    # Run evaluation
    results_by_dim, all_results_json = run_evaluation(args, device)
    
    # Print results
    print_final_results(results_by_dim, dims, seeds)
    
    # Save JSON results
    if args.save_results:
        os.makedirs(args.save_results, exist_ok=True)
        json_path = os.path.join(args.save_results, "table4_results.json")
        with open(json_path, 'w') as f:
            json.dump(all_results_json, f, indent=2)
        print(f"\n✓ Results saved to {json_path}")
        
        # Also save a summary CSV
        csv_path = os.path.join(args.save_results, "table4_summary.csv")
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=[
                'dim', 'seed', 'set1_rmse', 'set1_mae', 'set1_r2', 'set1_pearson_r2',
                'set2_rmse', 'set2_mae', 'set2_r2', 'set2_pearson_r2'
            ])
            w.writeheader()
            for result in all_results_json:
                w.writerow({
                    'dim': result['dim'],
                    'seed': result['seed'],
                    'set1_rmse': result['set1']['rmse'],
                    'set1_mae': result['set1']['mae'],
                    'set1_r2': result['set1']['r2_cod'],
                    'set1_pearson_r2': result['set1']['pearson_r2'],
                    'set2_rmse': result['set2']['rmse'],
                    'set2_mae': result['set2']['mae'],
                    'set2_r2': result['set2']['r2_cod'],
                    'set2_pearson_r2': result['set2']['pearson_r2'],
                })
        print(f"✓ Summary saved to {csv_path}")
    
    print("\n✓ Table 4 Evaluation Complete!")


if __name__ == '__main__':
    main()