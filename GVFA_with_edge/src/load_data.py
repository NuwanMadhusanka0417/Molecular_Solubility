import pandas as pd, torch
from torch_geometric.data import InMemoryDataset, Data
from rdkit import Chem
from sklearn.model_selection import train_test_split

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
    """Load graph data from a CSV path or a pandas DataFrame."""
    def __init__(self, csv_path_or_df, smiles_col="smiles_canon", target_col="LogS"):
        if isinstance(csv_path_or_df, pd.DataFrame):
            df = csv_path_or_df
        else:
            df = pd.read_csv(csv_path_or_df)
        super().__init__('.')
        graphs = []
        for smi, y in zip(df[smiles_col], df[target_col]):
            g = smiles_to_data(smi, y)
            if g is not None:
                graphs.append(g)
        self.data, self.slices = self.collate(graphs)

def load_data():
  """
  Load solubility_1.csv: SMILES and logS. Split ~90% train / ~10% test.
  Training: ~90% of compounds; Test: ~10% of compounds (fixed random_state for reproducibility).
  """
  df = pd.read_csv("final_data/solubility_1.csv")
  df = df.dropna(subset=["SMILES", "logS"])
  train_df, test_df = train_test_split(
      df, test_size=0.1, random_state=42, shuffle=True
  )
  dataset_train = ZINCLikeCSV(train_df, smiles_col="SMILES", target_col="logS")
  dataset_test  = ZINCLikeCSV(test_df,  smiles_col="SMILES", target_col="logS")
  return dataset_train, dataset_test 