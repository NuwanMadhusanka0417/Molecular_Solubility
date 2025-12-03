'''from typing import Optional, Tuple
import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset, Data
from rdkit import Chem

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

# ---- Bond type encoding (edge_attr is a 1D vector of ids) ----
_BOND2ID = {
    Chem.BondType.SINGLE: 0,
    Chem.BondType.DOUBLE: 1,
    Chem.BondType.TRIPLE: 2,
    Chem.BondType.AROMATIC: 3,
}
_ID2BOND = {v: k for k, v in _BOND2ID.items()}

# ---- Minimal cleaner to make sanitization robust ----
_NITRO  = Chem.MolFromSmarts('[N;X3](=O)=O')
_NITRO_OK = Chem.MolFromSmiles('[N+](=O)[O-]')

def _drop_orphan_H(m: Chem.Mol) -> Chem.Mol:
    to_del = [a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum() == 1 and a.GetDegree() == 0]
    if not to_del:
        return m
    em = Chem.EditableMol(m)
    for i in sorted(to_del, reverse=True):
        em.RemoveAtom(i)
    return em.GetMol()

def _largest_fragment(m: Chem.Mol) -> Chem.Mol:
    frags = Chem.GetMolFrags(m, asMols=True, sanitizeFrags=False)
    if not frags:
        return m
    return max(frags, key=lambda f: f.GetNumAtoms())

def clean_and_sanitize_smiles(smi: str) -> Optional[Chem.Mol]:
    m = Chem.MolFromSmiles(smi, sanitize=False)
    if m is None:
        return None
    # Fix common issues before sanitize
    m = _drop_orphan_H(m)
    # Nitro normalization (avoids N valence=4 warnings)
    try:
        reps = Chem.ReplaceSubstructs(m, _NITRO, _NITRO_OK, replaceAll=True)
        if reps and reps[0] is not None:
            m = reps[0]
    except Exception:
        pass
    # Keep largest fragment (drop salts)
    m = _largest_fragment(m)
    # Now sanitize
    try:
        Chem.SanitizeMol(m)
    except Exception:
        return None
    return m

# ---- Feature builder (no deprecated RDKit calls) ----
def atom_features_from_mol(mol: Chem.Mol) -> torch.Tensor:
    """
    Columns: [Z, degree, valence_electrons, total_valence, sp, sp2, sp3, aromatic]
    Shape: [N, 8] float32
    """
    pt = Chem.GetPeriodicTable()
    N = mol.GetNumAtoms()
    X = torch.zeros((N, 8), dtype=torch.float32)

    mol.UpdatePropertyCache(strict=False)

    try:
        Chem.SanitizeMol(mol)  # safe if already sanitized
    except Exception:
        pass

    for i, a in enumerate(mol.GetAtoms()):
        Z = a.GetAtomicNum()
        # base
        X[i, 0] = float(Z)                 # Z
        X[i, 1] = float(a.GetDegree())     # topological degree (undirected)
        # valence electrons (outer shell)
        X[i, 2] = float(pt.GetNOuterElecs(Z))
        # total valence (explicit + implicit; implicit may be -1)
        ev = a.GetExplicitValence()
        iv = a.GetImplicitValence()
        X[i, 3] = float(ev + (0 if iv < 0 else iv))
        # hybridization one-hot (sp, sp2, sp3)
        h = a.GetHybridization()
        if h == Chem.HybridizationType.SP:    X[i, 4] = 1.0
        elif h == Chem.HybridizationType.SP2: X[i, 5] = 1.0
        elif h == Chem.HybridizationType.SP3: X[i, 6] = 1.0
        # aromatic flag
        X[i, 7] = 1.0 if a.GetIsAromatic() else 0.0
        # If aromatic and no sp2 set, bias sp2
        if X[i, 7] == 1.0 and X[i, 5] == 0.0:
            X[i, 5] = 1.0

    # Optional light scaling of continuous cols [Z, degree, ve, tv]
    if N > 1:
        cont = X[:, :4]
        cmin = cont.min(0).values
        cmax = cont.max(0).values
        denom = torch.clamp(cmax - cmin, min=1e-6)
        X[:, :4] = (cont - cmin) / denom

    return X

def edges_from_mol(mol: Chem.Mol) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      edge_index: [2, 2*E] long (bidirected)
      edge_attr:  [2*E] long   (bond type id)
    """
    src, dst, ea = [], [], []
    for b in mol.GetBonds():
        u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        t = _BOND2ID.get(b.GetBondType(), 0)
        src += [u, v]
        dst += [v, u]
        ea  += [t, t]
    if src:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr  = torch.tensor(ea, dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr  = torch.empty((0,), dtype=torch.long)
    return edge_index, edge_attr

def smiles_to_enriched_data(smi: str, yval: float) -> Optional[Data]:
    mol = clean_and_sanitize_smiles(smi)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    x = atom_features_from_mol(mol)                   # [N, 8]
    edge_index, edge_attr = edges_from_mol(mol)
    y = torch.tensor([float(yval)], dtype=torch.float32)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.smiles = smi  # optional: keep SMILES for debugging
    return data

# ---- PyG dataset that reads CSV and builds enriched graphs ----
class EnrichedFromCSV(InMemoryDataset):
    def __init__(self, csv_path: str, smiles_col: str = "smiles_canon", target_col: str = "LogS"):
        self.df = pd.read_csv(csv_path)
        self.smiles_col = smiles_col
        self.target_col = target_col
        super().__init__('.')
        data_list, bad_idx = [], []
        for i, (smi, y) in enumerate(zip(self.df[smiles_col].astype(str), self.df[target_col])):
            g = smiles_to_enriched_data(smi, y)
            if g is None:
                bad_idx.append(i)
            else:
                data_list.append(g)
        if bad_idx:
            print(f"[INFO] Skipped {len(bad_idx)} invalid rows. First few: {bad_idx[:10]}")
        self.data, self.slices = self.collate(data_list)
'''
# ====== s2v_graph_loader.py ======
from typing import Optional, Tuple, Dict, List
import pandas as pd
import torch
import networkx as nx
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")  # quiet RDKit warnings

# ---------- Your S2VGraph container ----------
class S2VGraph(object):
    def __init__(self, g, label, mol, node_tags=None, node_features=None):
        """
        g: a networkx graph
        label: numeric graph label (float for regression)
        node_tags: list[int] (integer tag per node)
        node_features: torch.FloatTensor [N, F]
        edge_mat: torch.LongTensor [2, E] (undirected edges, one per bond)
        neighbors: list[list[int]]
        """
        self.label = label
        self.g = g
        self.node_tags = node_tags or []
        self.neighbors = []
        self.node_features = node_features
        self.edge_mat = torch.empty(2, 0, dtype=torch.long)
        self.mol = mol
        self.max_neighbor = 0

# ---------- Cleaner / sanitizer ----------
_NITRO   = Chem.MolFromSmarts('[N;X3](=O)=O')
_NITRO_OK= Chem.MolFromSmiles('[N+](=O)[O-]')

def _drop_orphan_H(m: Chem.Mol) -> Chem.Mol:
    to_del = [a.GetIdx() for a in m.GetAtoms()
              if a.GetAtomicNum() == 1 and a.GetDegree() == 0]
    if not to_del:
        return m
    em = Chem.EditableMol(m)
    for i in sorted(to_del, reverse=True):
        em.RemoveAtom(i)
    return em.GetMol()

def _largest_fragment(m: Chem.Mol) -> Chem.Mol:
    frags = Chem.GetMolFrags(m, asMols=True, sanitizeFrags=False)
    if not frags:
        return m
    return max(frags, key=lambda f: f.GetNumAtoms())

def clean_and_sanitize_smiles(smi: str) -> Optional[Chem.Mol]:
    m = Chem.MolFromSmiles(smi, sanitize=False)
    if m is None:
        return None
    m = _drop_orphan_H(m)
    try:
        reps = Chem.ReplaceSubstructs(m, _NITRO, _NITRO_OK, replaceAll=True)
        if reps and reps[0] is not None:
            m = reps[0]
    except Exception:
        pass
    m = _largest_fragment(m)
    try:
        Chem.SanitizeMol(m)
    except Exception:
        return None
    return m

# ---------- Atom features ----------
def atom_features_from_mol(mol: Chem.Mol) -> torch.Tensor:
    """
    Columns: [Z, degree, val_elec, total_val, sp, sp2, sp3, aromatic] -> [N, 8]
    """
    pt = Chem.GetPeriodicTable()
    N = mol.GetNumAtoms()
    X = torch.zeros((N, 8), dtype=torch.float32)

    mol.UpdatePropertyCache(strict=False)
    # already sanitized at load; keep try/except to be safe
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        pass

    for i, a in enumerate(mol.GetAtoms()):
        Z = a.GetAtomicNum()
        X[i, 0] = float(Z)                 # atomic number
        X[i, 1] = float(a.GetDegree())     # topological degree
        X[i, 2] = float(pt.GetNOuterElecs(Z))  # valence electrons
        ev = a.GetExplicitValence()
        iv = a.GetImplicitValence()        # -1 if unknown
        X[i, 3] = float(ev + (0 if iv < 0 else iv))  # total valence
        h = a.GetHybridization()
        if h == Chem.HybridizationType.SP:    X[i, 4] = 1.0
        elif h == Chem.HybridizationType.SP2: X[i, 5] = 1.0
        elif h == Chem.HybridizationType.SP3: X[i, 6] = 1.0
        X[i, 7] = 1.0 if a.GetIsAromatic() else 0.0
        if X[i, 7] == 1.0 and X[i, 5] == 0.0:
            X[i, 5] = 1.0  # aromatic ~ sp2

    # optional scaling for first 4 continuous cols
    if N > 1:
        cont = X[:, :4]
        cmin, cmax = cont.min(0).values, cont.max(0).values
        denom = torch.clamp(cmax - cmin, 1e-6)
        X[:, :4] = (cont - cmin) / denom

    return X

# ---------- Build NetworkX graph, tags, neighbors, edge_mat ----------
def nx_from_mol(mol: Chem.Mol, tag_vocab: Optional[Dict[str, int]] = None
               ) -> Tuple[nx.Graph, List[int], Dict[str, int], torch.Tensor, List[List[int]], int]:
    """
    Returns:
      G: networkx.Graph
      node_tags: list[int] (symbol -> id)
      tag_vocab: updated global mapping
      edge_mat: torch.LongTensor [2, E] (one undirected edge per bond)
      neighbors: list[list[int]]
      max_neighbor: int
    """
    if tag_vocab is None:
        tag_vocab = {}

    G = nx.Graph()
    node_tags: List[int] = []

    # nodes
    for i, a in enumerate(mol.GetAtoms()):
        sym = a.GetSymbol()
        if sym not in tag_vocab:
            tag_vocab[sym] = len(tag_vocab)
        tag_id = tag_vocab[sym]
        node_tags.append(tag_id)
        G.add_node(i, label=sym, tag=tag_id)

    # edges (undirected, add once)
    edges = []
    for b in mol.GetBonds():
        u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if u != v:
            G.add_edge(u, v)
            edges.append((u, v))

    if edges:
        edge_mat = torch.tensor(list(zip(*edges)), dtype=torch.long)  # shape [2, E]
    else:
        edge_mat = torch.empty((2, 0), dtype=torch.long)

    # neighbors
    neighbors: List[List[int]] = [[] for _ in range(mol.GetNumAtoms())]
    for u, v in edges:
        neighbors[u].append(v)
        neighbors[v].append(u)
    max_neighbor = max((len(n) for n in neighbors), default=0)

    return G, node_tags, tag_vocab, edge_mat, neighbors, max_neighbor

# ---------- Convert a single SMILES to S2VGraph ----------
def smiles_to_s2vgraph(smi: str, yval: float,
                       tag_vocab: Optional[Dict[str, int]] = None
                      ) -> Tuple[Optional[S2VGraph], Dict[str, int]]:
    mol = clean_and_sanitize_smiles(smi)
    if mol is None or mol.GetNumAtoms() == 0:
        return None, tag_vocab or {}
    X = atom_features_from_mol(mol)
    G, node_tags, tag_vocab, edge_mat, neighbors, max_neighbor = nx_from_mol(mol, tag_vocab)

    gobj = S2VGraph(
        g=G,
        label=float(yval),
        mol=mol,
        node_tags=node_tags,
        node_features=X
    )
    gobj.edge_mat = edge_mat
    gobj.neighbors = neighbors
    gobj.max_neighbor = max_neighbor
    return gobj, tag_vocab

# ---------- Load CSV -> list[S2VGraph] ----------
def load_s2v_from_csv(csv_path: str,
                      smiles_col: str = "smiles_canon",
                      target_col: str = "LogS"
                     ) -> Tuple[List[S2VGraph], Dict[str, int]]:
    df = pd.read_csv(csv_path)
    g_list: List[S2VGraph] = []
    tag_vocab: Dict[str, int] = {}
    bad_idx = []

    for i, (smi, y) in enumerate(zip(df[smiles_col].astype(str), df[target_col])):
        g, tag_vocab = smiles_to_s2vgraph(smi, y, tag_vocab)
        if g is None:
            bad_idx.append(i)
        else:
            g_list.append(g)

    if bad_idx:
        print(f"[INFO] Skipped {len(bad_idx)} invalid rows. First few: {bad_idx[:10]}")
    print(f"[INFO] Built {len(g_list)} graphs. tag_vocab size = {len(tag_vocab)}")
    return g_list, tag_vocab

# ------------- Example -------------
# g_train, vocab = load_s2v_from_csv("final_data/final_unique_train.csv")
# g_test, _     = load_s2v_from_csv("final_data/final_unique_test.csv")
# print(type(g_train[0]), g_train[0].node_features.shape, g_train[0].edge_mat.shape, g_train[0].label)
