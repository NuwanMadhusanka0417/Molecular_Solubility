import sys
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize.rdMolStandardize import LargestFragmentChooser
 

from hyper_fingerprints import Encoder, cosine_similarity
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor
from hyper_fingerprints import Encoder, cosine_similarity


TRAIN_CSV = "data/solubility_1.csv"
TEST_CSV = "data/testset_novel.csv"
TARGET_COL = "logS"
SMILES_COL = "SMILES"
STRIP_SALTS = True          # remove counter-ions (e.g. .Cl) before encoding
TEST_SIZE = 0.2
SEEDS = 0,1,2,3,4
DIMENSIONS = 100,120


#------------------------ Load Data----------
def load_dataset(csv_path):
    """Load a CSV, return (smiles_list, y_array)."""
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[SMILES_COL, TARGET_COL]).reset_index(drop=True)
    smiles = df[SMILES_COL].tolist()
    y = df[TARGET_COL].to_numpy(dtype=float)
    print(f"{csv_path}: loaded {len(smiles)} molecules.")
    return smiles, y
 
def detect_atom_vocabulary(smiles_list):
    """
    Parse every SMILES with RDKit and collect the set of unique atom
    symbols actually present in the dataset. Use this to build an
    atom_types list that fully covers your data, instead of relying
    on the encoder's default vocabulary.
    """
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

#-------------------Regression Models------------------
models = {
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

for dim in DIMENSIONS:
    for seed in SEEDS:

        #--------------------------Encoder------------
        ENCODER_KWARGS = dict(
            dimension=dim,
            depth=3,
            atom_types=atom_vocab,   # set below based on vocabulary found in the data
            seed=seed,
            normalize=False,
            backend="auto",
        )
        encoder = Encoder(**ENCODER_KWARGS)

        X_train = encoder.encode(train_smiles)  # shape: (n_train, 512)
        X_test = encoder.encode(test_smiles)    # shape: (n_test, 512)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        results = []
        for name, model in models.items():
            model.fit(X_train_s, y_train)
            pred = model.predict(X_test_s)
        
            rmse = float(np.sqrt(mean_squared_error(y_test, pred))) #mean_squared_error(y_test, pred, squared=False)
            r2 = r2_score(y_test, pred)
            results.append((name, rmse, r2))
            print(f"{name:10s} {dim} {seed}  RMSE={rmse:.4f}  R2={r2:.4f}")

