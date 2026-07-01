"""
Encode + regress across (dimension, seed) combinations, with memory
instrumentation and explicit cleanup to prevent cross-iteration
accumulation.

Usage:
    python solubility_regression_instrumented.py
"""

import gc
import psutil
import numpy as np
import pandas as pd

from hyper_fingerprints import Encoder
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score. mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor


TRAIN_CSV = "data/final_unique_train.csv" #"data/solubility_1.csv"
TEST_CSV = "data/final_unique_test.csv" #"data/testset_novel.csv"
TARGET_COL = "logS"
SMILES_COL = "SMILES"
SEEDS = [0, 1, 2, 3, 4]
DIMENSIONS = [1000, 2000, 5000, 10000]

_process = psutil.Process()


def peak_mem_gb():
    """Current RSS memory usage of this process, in GB.

    Note: unlike resource.getrusage's ru_maxrss, this is the CURRENT
    RSS, not the historical peak. It still works fine for step-by-step
    tracing since we print after every stage, but if you want the true
    peak, take the max of all printed values yourself.
    """
    return _process.memory_info().rss / (1024 ** 3)


def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[SMILES_COL, TARGET_COL]).reset_index(drop=True)
    smiles = df[SMILES_COL].tolist()
    y = df[TARGET_COL].to_numpy(dtype=float)
    print(f"{csv_path}: loaded {len(smiles)} molecules.")
    return smiles, y


def detect_atom_vocabulary(smiles_list):
    from rdkit import Chem
    vocab = set()
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        for atom in mol.GetAtoms():
            vocab.add(atom.GetSymbol())
    return sorted(vocab)


train_smiles, y_train = load_dataset(TRAIN_CSV)
test_smiles, y_test = load_dataset(TEST_CSV)
atom_vocab = detect_atom_vocabulary(train_smiles + test_smiles)
print(f"[mem] after data load: {peak_mem_gb():.2f} GB")


def build_models():
    """Fresh model instances each call -- never reuse a fitted model
    across iterations, since fitted attributes can retain references
    to training data."""
    return {
        "Ridge": RidgeCV(
            alphas=np.logspace(-6, 4, 80),
            cv=5,
            scoring="neg_mean_squared_error",
            fit_intercept=True,
        ),
        "SVR": SVR(kernel="rbf", C=10.0, epsilon=0.1, gamma="scale"),
        "XGBoost": XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        ),
    }


all_results = []

for dim in DIMENSIONS:
    for seed in SEEDS:
        print(f"\n=== dim={dim} seed={seed} ===")
        print(f"[mem] loop start: {peak_mem_gb():.2f} GB")

        encoder = Encoder(
            dimension=dim,
            depth=3,
            atom_types=atom_vocab,
            seed=seed,
            normalize=False,
            backend="auto",
        )

        X_train = encoder.encode(train_smiles)
        X_test = encoder.encode(test_smiles)
        print(f"[mem] after encode: {peak_mem_gb():.2f} GB  "
              f"(X_train {X_train.nbytes / 1e9:.3f} GB, "
              f"X_test {X_test.nbytes / 1e9:.3f} GB)")

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        print(f"[mem] after scaling: {peak_mem_gb():.2f} GB")

        # raw encodings no longer needed once scaled
        del X_train, X_test

        models = build_models()
        for name, model in models.items():
            model.fit(X_train_s, y_train)
            pred = model.predict(X_test_s)

            rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
            mae = mean_absolute_error(y_test, pred)
            r2 = r2_score(y_test, pred)
            all_results.append((dim, seed, name, rmse, r2))
            print(f"{name:10s} dim={dim} seed={seed}  "
                  f"MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}  "
                  f"[mem] {peak_mem_gb():.2f} GB")

            # fitted model may hold references to training data
            del model

        # explicit cleanup before next iteration
        del X_train_s, X_test_s, scaler, encoder, models
        gc.collect()
        print(f"[mem] after cleanup: {peak_mem_gb():.2f} GB")

results_df = pd.DataFrame(
    all_results, columns=["dimension", "seed", "model", "RMSE", "R2"]
)
results_df.to_csv("sweep_results.csv", index=False)
print("\nFinal peak memory:", f"{peak_mem_gb():.2f} GB")
print(results_df)