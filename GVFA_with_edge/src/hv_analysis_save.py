"""
Experiment layout export: experiment_D{dim}_seed{seed}/
Train and test tensors are saved under train/ and test/ (never merged in RAM).

Enable with --save_analysis_data from GVFA_edge_main (gvfa_ridge).
"""
from __future__ import annotations

import gc
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch


def run_layout_capture(
    model: torch.nn.Module,
    graphs: list,
    device: torch.device,
    batch_size: int,
) -> List[Dict[str, np.ndarray]]:
    """Run forward with analysis_layout_batches; returns list of per-batch record dicts."""
    batches: List[Dict[str, np.ndarray]] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(graphs), batch_size):
            end = min(start + batch_size, len(graphs))
            model(graphs[start:end], analysis_layout_batches=batches)
    return batches


def merge_batch_records(records: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """Concatenate batch dicts; clears *records* in place to drop references early."""
    if not records:
        return {}
    keys = list(records[0].keys())
    out: Dict[str, np.ndarray] = {}
    for k in keys:
        out[k] = np.concatenate([r.pop(k) for r in records], axis=0)
    records.clear()
    return out


def _release_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _write_split_npz(split_dir: str, merged: Dict[str, np.ndarray]) -> None:
    os.makedirs(split_dir, exist_ok=True)

    np.savez_compressed(
        os.path.join(split_dir, "projected.npz"),
        node_vectors=merged["node_projected"].astype(np.float32, copy=False),
        edge_vectors=merged["edge_hv"].astype(np.float32, copy=False),
    )

    np.savez_compressed(
        os.path.join(split_dir, "gvfa_layers.npz"),
        layer_0=merged["layer_0"].astype(np.float32, copy=False),
        layer_1=merged["layer_1"].astype(np.float32, copy=False),
        layer_2=merged["layer_2"].astype(np.float32, copy=False),
        layer_3=merged["layer_3"].astype(np.float32, copy=False),
        layer_4=merged["layer_4"].astype(np.float32, copy=False),
    )

    np.savez_compressed(
        os.path.join(split_dir, "pooled_float32.npz"),
        pooled=merged["pooled_float32"].astype(np.float32, copy=False),
    )
    np.savez_compressed(
        os.path.join(split_dir, "pooled_binarized.npz"),
        pooled=merged["pooled_binarized"].astype(np.float32, copy=False),
    )
    np.savez_compressed(
        os.path.join(split_dir, "pooled_normalized.npz"),
        pooled=merged["pooled_normalized"].astype(np.float32, copy=False),
    )

    np.savez_compressed(
        os.path.join(split_dir, "memory_buffer.npz"),
        pre_l2=merged["tap_pre_l2"].astype(np.float32, copy=False),
        post_l2=merged["tap_post_l2"].astype(np.float32, copy=False),
    )

    for t in range(4):
        np.savez_compressed(
            os.path.join(split_dir, f"sigma_pi_order_{t}.npz"),
            vectors=merged[f"sigma_order_{t}"].astype(np.float32, copy=False),
        )

    np.savez_compressed(
        os.path.join(split_dir, "sigma_pi_combined.npz"),
        combined_bundle=merged["sigma_bundle"].astype(np.float32, copy=False),
        combined_concat=merged["sigma_concat"].astype(np.float32, copy=False),
    )


def write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_experiment_layout(
    base_dir: str,
    dim: int,
    seed: int,
    train_graphs: list,
    test_graphs: list,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    config: Dict[str, Any],
    y_train: np.ndarray,
    y_test: np.ndarray,
    train_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
    reg_info: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Writes experiment_D{dim}_seed{seed}/ with train/ and test/ subfolders.
    Does not merge train+test in memory (avoids huge peak RAM on sigma_concat).
    """
    exp_dir = os.path.join(base_dir, f"experiment_D{int(dim)}_seed{int(seed)}")
    os.makedirs(exp_dir, exist_ok=True)

    n_tr_g = len(train_graphs)
    n_te_g = len(test_graphs)

    train_recs = run_layout_capture(model, train_graphs, device, batch_size)
    tr = merge_batch_records(train_recs)
    del train_recs
    n_tr_n = int(tr["node_projected"].shape[0])
    _write_split_npz(os.path.join(exp_dir, "train"), tr)
    del tr
    _release_cuda_cache()

    test_recs = run_layout_capture(model, test_graphs, device, batch_size)
    te = merge_batch_records(test_recs)
    del test_recs
    n_te_n = int(te["node_projected"].shape[0])
    _write_split_npz(os.path.join(exp_dir, "test"), te)
    del te
    _release_cuda_cache()

    cfg = {
        **config,
        "dim": int(dim),
        "seed": int(seed),
        "binding_method": (
            f"gvfa={config.get('gvfa_binding_mode', 'circular_conv')};"
            f"other={config.get('other_binding_mode', 'circular_conv')}"
        ),
        "n_graphs_train": n_tr_g,
        "n_graphs_test": n_te_g,
        "n_nodes_train": n_tr_n,
        "n_nodes_test": n_te_n,
        "layout": "tensors under train/ and test/; not merged in RAM",
    }
    write_json(os.path.join(exp_dir, "config.json"), cfg)

    np.savez_compressed(
        os.path.join(exp_dir, "targets.npz"),
        y_train=y_train.astype(np.float32),
        y_test=y_test.astype(np.float32),
        train_idx=np.arange(n_tr_g, dtype=np.int64),
        test_idx=np.arange(n_te_g, dtype=np.int64),
    )

    write_json(
        os.path.join(exp_dir, "regression_results.json"),
        {
            "train": {k: float(v) for k, v in train_metrics.items()},
            "test": {k: float(v) for k, v in test_metrics.items()},
            "regressor": reg_info or {},
        },
    )

    readme = f"""experiment_D{dim}_seed{seed} layout

Root: config.json, targets.npz, regression_results.json

train/ and test/ (identical file names in each):
  projected.npz, gvfa_layers.npz, pooled_float32.npz, pooled_binarized.npz,
  pooled_normalized.npz, memory_buffer.npz, sigma_pi_order_0..3.npz,
  sigma_pi_combined.npz

Train and test are saved separately to reduce peak memory (no full-dataset concat).

Same dim+seed overwrites this folder; see config.json for last run.
"""
    with open(os.path.join(exp_dir, "README.txt"), "w", encoding="utf-8") as f:
        f.write(readme)

    return exp_dir


def save_train_test_results_json(
    path: str,
    train_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
    reg_info: Optional[Dict[str, Any]] = None,
) -> None:
    write_json(
        path,
        {
            "train": {k: float(v) for k, v in train_metrics.items()},
            "test": {k: float(v) for k, v in test_metrics.items()},
            "regressor": reg_info or {},
        },
    )
