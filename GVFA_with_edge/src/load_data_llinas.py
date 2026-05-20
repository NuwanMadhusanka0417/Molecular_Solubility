"""
load_data_llinas.py
───────────────────
Data loading for Llinas2020 cross-dataset evaluation.

Trains on Cui et al. (solubility_1.csv) and tests on Llinas2020 set1 & set2
without touching any existing source files.

Reuses ZINCLikeCSV from src/load_data.py — no changes to that file.

Typical usage
─────────────
    from src.load_data_llinas import load_llinas_data

    train_data, set1_data, set2_data = load_llinas_data()
    # or with custom paths:
    train_data, set1_data, set2_data = load_llinas_data(
        train_path="final_data/solubility_1.csv",
        set1_path="final_data/Llinas2020_set1.csv",
        set2_path="final_data/Llinas2020_set2.csv",
    )
"""

import pandas as pd

# Reuse the existing ZINCLikeCSV class — no modification needed.
from src.load_data import ZINCLikeCSV


def load_llinas_data(
    train_path: str = "final_data/solubility_1.csv",
    set1_path:  str = "final_data/Llinas2020_set1.csv",
    set2_path:  str = "final_data/Llinas2020_set2.csv",
    train_smiles_col:  str = "SMILES",
    train_target_col:  str = "logS",
    llinas_smiles_col: str = "SMILES",
    llinas_target_col: str = "logS",
):
    """
    Load the Cui et al. training set and both Llinas2020 test sets.

    Parameters
    ----------
    train_path : str
        Path to training CSV (Cui et al., solubility_1.csv).
    set1_path : str
        Path to Llinas2020 set1 CSV.
    set2_path : str
        Path to Llinas2020 set2 CSV.
    train_smiles_col : str
        SMILES column name in the training CSV.
    train_target_col : str
        Target (logS) column name in the training CSV.
    llinas_smiles_col : str
        SMILES column name in the Llinas CSVs.
    llinas_target_col : str
        Target (logS) column name in the Llinas CSVs.

    Returns
    -------
    train_data : ZINCLikeCSV
        Training dataset — Cui et al. (solubility_1.csv).
    set1_data : ZINCLikeCSV
        Llinas2020 set1 test dataset.
    set2_data : ZINCLikeCSV
        Llinas2020 set2 test dataset.
    """
    # ── Training set (Cui et al.) ──────────────────────────────────────────
    train_df = pd.read_csv(train_path)
    train_df = train_df.dropna(subset=[train_smiles_col, train_target_col])
    train_data = ZINCLikeCSV(
        train_df,
        smiles_col=train_smiles_col,
        target_col=train_target_col,
    )

    # ── Llinas2020 set1 ───────────────────────────────────────────────────
    set1_df = pd.read_csv(set1_path)
    set1_df = set1_df.dropna(subset=[llinas_smiles_col, llinas_target_col])
    set1_data = ZINCLikeCSV(
        set1_df,
        smiles_col=llinas_smiles_col,
        target_col=llinas_target_col,
    )

    # ── Llinas2020 set2 ───────────────────────────────────────────────────
    set2_df = pd.read_csv(set2_path)
    set2_df = set2_df.dropna(subset=[llinas_smiles_col, llinas_target_col])
    set2_data = ZINCLikeCSV(
        set2_df,
        smiles_col=llinas_smiles_col,
        target_col=llinas_target_col,
    )

    print(f"[load_llinas_data] Train  (Cui et al.)   : {len(train_df):>5} molecules  ({train_path})")
    print(f"[load_llinas_data] Test   (Llinas set1)  : {len(set1_df):>5} molecules  ({set1_path})")
    print(f"[load_llinas_data] Test   (Llinas set2)  : {len(set2_df):>5} molecules  ({set2_path})")

    return train_data, set1_data, set2_data
