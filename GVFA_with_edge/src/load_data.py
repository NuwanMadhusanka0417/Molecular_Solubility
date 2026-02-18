import pandas as pd, torch
from torch_geometric.data import InMemoryDataset, Data
from rdkit import Chem

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
# Map RDKit bond types to a single integer (edge_attr is 1D like ZINC prints)
_BOND2ID = {
    Chem.BondType.SINGLE: 0,
    Chem.BondType.DOUBLE: 1,
    Chem.BondType.TRIPLE: 2,
    Chem.BondType.AROMATIC: 3,
}

def smiles_to_data(smi, yval):
    mol = Chem.MolFromSmiles(smi)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    # Skip molecules with no bonds (single-atom or invalid); they break edge feature building.
    if mol.GetNumBonds() == 0:
        return None

    # x: [num_nodes, 1] (long) — single integer per atom (e.g., atomic number)
    x = torch.tensor([[a.GetAtomicNum()] for a in mol.GetAtoms()], dtype=torch.long)

    # Edges (both directions) and 1D edge_attr (long)
    src, dst, eattr = [], [], []
    for b in mol.GetBonds():
        u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        t = _BOND2ID.get(b.GetBondType(), 0)
        # add both directions
        src += [u, v]
        dst += [v, u]
        eattr += [t, t]

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr  = torch.tensor(eattr, dtype=torch.long)  # shape [E], same as ZINC print

    # y: [1] (float)
    y = torch.tensor([float(yval)], dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


class ZINCLikeCSV(InMemoryDataset):
    def __init__(self, csv_path=None, smiles_col="smiles_canon", target_col="LogS", df=None):
        """
        If df is provided, use it directly; otherwise load from csv_path.
        """
        if df is None and csv_path is None:
            raise ValueError("Either csv_path or df must be provided.")
        if df is None:
            df = pd.read_csv(csv_path)
        super().__init__('.')
        graphs = []
        for smi, y in zip(df[smiles_col], df[target_col]):
            g = smiles_to_data(smi, y)
            if g is not None:
                graphs.append(g)
        self.data, self.slices = self.collate(graphs)

def load_data(csv_path="final_data/solubility_1.csv", smiles_col="SMILES", target_col="logS", train_frac=0.9, random_state=42):
    """
    Load data from a single CSV (e.g., solubility_1.csv), shuffle it, and split
    into train and test according to train_frac (default 0.9 / 0.1).
    """
    df = pd.read_csv(csv_path)
    # Shuffle before splitting to avoid order bias
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    n_total = len(df)
    n_train = int(train_frac * n_total)

    df_train = df.iloc[:n_train].reset_index(drop=True)
    df_test = df.iloc[n_train:].reset_index(drop=True)

    dataset_train = ZINCLikeCSV(df=df_train, smiles_col=smiles_col, target_col=target_col)
    dataset_test = ZINCLikeCSV(df=df_test, smiles_col=smiles_col, target_col=target_col)

    return dataset_train, dataset_test  #, gl_train, 