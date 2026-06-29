"""
Solubility (logS) regression using Hyperdimensional Fingerprints (HDF).

Pipeline per seed:
  1. Encode train/test SMILES into HDF fingerprints (numpy backend).
  2. Fit RidgeCV on train fingerprints -> logS.
  3. Predict on the test set, report RMSE and R2.

The HDF `seed` controls the random fingerprint codebook, so seeds 0..4 give
5 independent fingerprint initialisations. The train/test split is fixed by
the two input files.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

# Use the local hyper_fingerprints package (numpy backend, no Rust needed).
sys.path.insert(0, ".")
from hyper_fingerprints import Encoder  # noqa: E402

RDLogger.DisableLog("rdApp.*")  # silence RDKit parse warnings

TRAIN_CSV = r"C:\Users\22390013@students.ltu.edu.au\OneDrive - LA TROBE UNIVERSITY\Projects\Molecular_Solubility\MolecularSolubility\GVFA_with_edge\final_data\solubility_1.csv"
TEST_CSV = r"C:\Users\22390013@students.ltu.edu.au\OneDrive - LA TROBE UNIVERSITY\Projects\Molecular_Solubility\MolecularSolubility\GVFA_with_edge\final_data\testset_novel.csv"

SEEDS = [0, 1, 2, 3, 4]
DIMENSION = 2000 #1024
DEPTH = 3
BATCH_SIZE = 128  # molecules per encode() call (keeps numpy memory bounded)


def load_dataset(path: str) -> tuple[list[Chem.Mol], np.ndarray, int]:
    """Load a CSV with SMILES + logS, return parsed mols, targets, n_dropped."""
    df = pd.read_csv(path)
    smiles = df["SMILES"].astype(str).tolist()
    logs = df["logS"].to_numpy(dtype=np.float64)

    mols, targets, dropped = [], [], 0
    for smi, y in zip(smiles, logs):
        mol = Chem.MolFromSmiles(smi)
        if mol is None or np.isnan(y):
            dropped += 1
            continue
        mols.append(mol)
        targets.append(y)
    return mols, np.asarray(targets, dtype=np.float64), dropped


def collect_atom_types(*mol_lists: list[Chem.Mol]) -> list[str]:
    """Build the union of element symbols present across all molecules."""
    symbols: set[str] = set()
    for mols in mol_lists:
        for mol in mols:
            for atom in mol.GetAtoms():
                symbols.add(atom.GetSymbol())
    return sorted(symbols)


def encode_mols(encoder: Encoder, mols: list[Chem.Mol]) -> np.ndarray:
    """Encode a list of RDKit mols into fingerprints, in mini-batches."""
    chunks = []
    for i in range(0, len(mols), BATCH_SIZE):
        chunks.append(encoder.encode(mols[i : i + BATCH_SIZE]))
    return np.vstack(chunks)


def build_regressor(model: str, scale: bool, seed: int):
    """Build the selected regressor, optionally with StandardScaler scaling.

    model : "ridge" | "svr" | "xgb"
    """
    if model == "ridge":
        reg = RidgeCV(
            alphas=np.logspace(-4, 2, 50), cv=5, scoring="neg_mean_squared_error"
        )
    elif model == "svr":
        reg = SVR(kernel="rbf", C=10.0, epsilon=0.1, gamma="scale")
    elif model == "xgb":
        reg = XGBRegressor(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model: {model!r}")

    if scale:
        return make_pipeline(StandardScaler(), reg)
    return reg


def model_info(model_name: str, fitted) -> str:
    """Short description of the fitted model (e.g. RidgeCV's chosen alpha)."""
    if model_name == "ridge":
        ridge = fitted.named_steps["ridgecv"] if hasattr(fitted, "named_steps") else fitted
        return f"alpha={float(ridge.alpha_):.4g}"
    return model_name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=["ridge", "svr", "xgb", "all"],
        default="ridge",
        help="Downstream regressor on top of HDF fingerprints. 'all' encodes "
        "once per seed and evaluates ridge, svr and xgb (default: ridge).",
    )
    feat = parser.add_mutually_exclusive_group()
    feat.add_argument(
        "--scale",
        dest="scale",
        action="store_true",
        help="StandardScaler feature scaling before RidgeCV (default).",
    )
    feat.add_argument(
        "--no-scale",
        dest="scale",
        action="store_false",
        help="Disable feature scaling; fit RidgeCV on raw fingerprints.",
    )
    parser.set_defaults(scale=True)
    args = parser.parse_args()

    print(f"Model: {args.model}")
    print(f"Feature scaling: {'ON (StandardScaler)' if args.scale else 'OFF'}")
    print("Loading datasets...")
    train_mols, y_train, train_dropped = load_dataset(TRAIN_CSV)
    test_mols, y_test, test_dropped = load_dataset(TEST_CSV)
    print(f"  train: {len(train_mols)} molecules ({train_dropped} dropped)")
    print(f"  test : {len(test_mols)} molecules ({test_dropped} dropped)")

    atom_types = collect_atom_types(train_mols, test_mols)
    print(f"  atom vocabulary ({len(atom_types)}): {atom_types}")

    models = ["ridge", "svr", "xgb"] if args.model == "all" else [args.model]
    # results[model] -> list of (seed, rmse, r2)
    results: dict[str, list] = {m: [] for m in models}

    for seed in SEEDS:
        t_enc = time.time()
        encoder = Encoder(
            dimension=DIMENSION,
            depth=DEPTH,
            atom_types=atom_types,
            seed=seed,
            backend="numpy",
        )
        X_train = encode_mols(encoder, train_mols)
        X_test = encode_mols(encoder, test_mols)
        print(f"  seed {seed}: encoded in {time.time() - t_enc:.1f}s")

        for m in models:
            t0 = time.time()
            reg = build_regressor(m, args.scale, seed)
            reg.fit(X_train, y_train)
            y_pred = reg.predict(X_test)

            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            r2 = float(r2_score(y_test, y_pred))
            results[m].append((seed, rmse, r2))
            print(
                f"    [{m}] RMSE={rmse:.4f}  R2={r2:.4f}  "
                f"({model_info(m, reg)}, {time.time() - t0:.1f}s)"
            )

    for m in models:
        rmses = np.array([r[1] for r in results[m]])
        r2s = np.array([r[2] for r in results[m]])
        print(f"\n=== Test-set results ({m} on HDF fingerprints) ===")
        print(f"{'Seed':<6}{'RMSE':<12}{'R2':<12}")
        for seed, rmse, r2 in results[m]:
            print(f"{seed:<6}{rmse:<12.4f}{r2:<12.4f}")
        print("-" * 30)
        print(f"{'Mean':<6}{rmses.mean():<12.4f}{r2s.mean():<12.4f}")
        print(f"{'Std':<6}{rmses.std():<12.4f}{r2s.std():<12.4f}")


if __name__ == "__main__":
    main()
