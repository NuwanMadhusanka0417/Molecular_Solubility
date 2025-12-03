import torch
from rdkit import Chem
from rdkit.Chem import RWMol
import networkx as nx
from torch_geometric.data import Data
import numpy as np

BOND_TYPES  = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC]
BTYPE2IDX   = {bt:i for i,bt in enumerate(BOND_TYPES)}
STEREOS     = [Chem.BondStereo.STEREONONE, Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOE,
               Chem.BondStereo.STEREOCIS, Chem.BondStereo.STEREOTRANS, Chem.BondStereo.STEREOANY]
STEREO2IDX  = {s:i for i,s in enumerate(STEREOS)}

HYB_MAP = {
    Chem.rdchem.HybridizationType.SP: 0,
    Chem.rdchem.HybridizationType.SP2: 1,
    Chem.rdchem.HybridizationType.SP3: 2,
}
HYB_DIM = 3  # sp, sp2, sp3

def _hyb_onehot(atom):
    """return 3-dim onehot (sp, sp2, sp3). others -> zeros"""
    v = np.zeros(HYB_DIM, dtype=np.float32)
    idx = HYB_MAP.get(atom.GetHybridization(), None)
    if idx is not None:
        v[idx] = 1.0
    return v


def bond_feat(b: Chem.Bond) -> np.ndarray:
    t = BTYPE2IDX.get(b.GetBondType(), 0)
    one_hot_t = np.zeros(len(BOND_TYPES), dtype=np.float32); one_hot_t[t] = 1.0

    s = STEREO2IDX.get(b.GetStereo(), 0)
    one_hot_s = np.zeros(len(STEREOS), dtype=np.float32); one_hot_s[s] = 1.0

    extra = np.array([
        float(b.GetIsConjugated()),
        float(b.IsInRing()),
        float(b.GetIsAromatic()),
    ], dtype=np.float32)

    # 4) NEW: endpoint hybridizations (begin/end) -> 6 dims
    a_begin = b.GetBeginAtom()
    a_end   = b.GetEndAtom()
    hyb_begin = _hyb_onehot(a_begin)  # (3,)
    hyb_end   = _hyb_onehot(a_end)    # (3,)

    # total = 4 + 6 + 3 + 3 + 3 = 19 dims
    return np.concatenate([one_hot_t, one_hot_s, extra, hyb_begin, hyb_end], axis=0)


def expand_atomic_features(data):
    index_to_atomic_number = {0: 6, 1: 8, 2: 7, 3: 16, 4: 9}  
    num_nodes = data.x.shape[0]

    # Directly use atomic numbers if they are present
    atomic_numbers = data.x.reshape(-1, 1)  # No need for mapping
    # print(atomic_numbers)
    # print("Atiomic numbers ", atomic_numbers)
    # print("data.edge_index -  ", data.edge_index)

    # Compute degree (number of bonds)
    degrees = torch.zeros((num_nodes, 1))
    for i in range(data.edge_index.shape[1]):
        u, v = data.edge_index[:, i]
        degrees[u] += 1
        degrees[v] += 1
    # print("degrees ", degrees)

    # Additional atomic properties
    valence_electrons = torch.zeros((num_nodes, 1))
    hybridization = torch.zeros((num_nodes, 3))  # One-hot for (sp, sp2, sp3)
    aromaticity = torch.zeros((num_nodes, 1))

    valence_dict = {
        0:0,
        # Period 1
        1:1,  2:2,
        # Period 2
        3:1,  4:2,  5:3,  6:4,  7:5,  8:6,  9:7,  10:8,
        # Period 3
        11:1, 12:2, 13:3, 14:4, 15:5, 16:6, 17:7, 18:8,
        # Period 4
        19:1, 20:2, 21:2, 22:2, 23:2, 24:2, 25:2, 26:2, 27:2, 28:2, 29:2, 30:2,
        31:3, 32:4, 33:5, 34:6, 35:7, 36:8,
        # Period 5
        37:1, 38:2, 39:2, 40:2, 41:2, 42:2, 43:2, 44:2, 45:2, 46:2, 47:2, 48:2,
        49:3, 50:4, 51:5, 52:6, 53:7, 54:8,
        # Period 6
        55:1, 56:2,                         # Cs, Ba
        57:2, 58:2, 59:2, 60:2, 61:2, 62:2, 63:2, 64:2, 65:2, 66:2, 67:2, 68:2, 69:2, 70:2, 71:2,  # La–Lu
        72:2, 73:2, 74:2, 75:2, 76:2, 77:2, 78:2, 79:2, 80:2,     # Hf–Hg (you had 78,80 already)
        81:3, 82:4, 83:5, 84:6, 85:7, 86:8,                       # Tl–Rn (you had 83 already)
        # Period 7
        87:1, 88:2,                                               # Fr, Ra
        89:2, 90:2, 91:2, 92:2, 93:2, 94:2, 95:2, 96:2, 97:2, 98:2, 99:2, 100:2, 101:2, 102:2, 103:2,  # Ac–Lr
        104:2, 105:2, 106:2, 107:2, 108:2, 109:2, 110:2, 111:2, 112:2,  # Rf–Cn
        113:3, 114:4, 115:5, 116:6, 117:7, 118:8                 # Nh–Og
    }
    hybridization_dict = {}
    noble_gases   = {2, 10, 18, 36, 54, 86, 118}
    halogens      = {9, 17, 35, 53, 85, 117}            # F, Cl, Br, I, At, Ts
    chalcogens    = {8, 16, 34, 52, 84, 116}            # O, S, Se, Te, Po, Lv
    pnictogens    = {7, 15, 33, 51, 83, 115}            # N, P, As, Sb, Bi, Mc
    group14       = {6, 14, 32, 50, 82, 114}            # C, Si, Ge, Sn, Pb, Fl
    group13       = {5, 13, 31, 49, 81, 113}            # B, Al, Ga, In, Tl, Nh

    alkali        = {1, 3, 11, 19, 37, 55, 87}
    alkaline_earth= {4, 12, 20, 38, 56, 88}

    # d-block: periods 4–7, groups 3–12 (Sc–Zn, Y–Cd, Hf–Hg, Rf–Cn)
    d_block = set(range(21, 31)) | set(range(39, 49)) | set(range(72, 81)) | set(range(104, 113))

    # f-block: La–Lu (57–71), Ac–Lr (89–103)
    f_block = set(range(57, 72)) | set(range(89, 104))

    for Z in sorted(k for k in valence_dict.keys() if k > 0):
        if Z in noble_gases:
            hybridization_dict[Z] = [0, 0, 0]
        elif Z in d_block or Z in f_block:
            hybridization_dict[Z] = [0, 0, 1]
        elif Z in alkali or Z in alkaline_earth:
            hybridization_dict[Z] = [0, 0, 1]
        elif Z in halogens:
            hybridization_dict[Z] = [0, 0, 1]
        elif Z in chalcogens:
            hybridization_dict[Z] = [0, 1, 1]
        elif Z in pnictogens:
            hybridization_dict[Z] = [0, 1, 1]
        elif Z in group14:
            hybridization_dict[Z] = [1, 1, 1] if Z == 6 else [0, 1, 1]
        elif Z in group13:
            hybridization_dict[Z] = [0, 1, 1]
        elif Z == 1:
            hybridization_dict[Z] = [0, 0, 1]
        else:
            # catch-all for remaining p-block superheavies, etc.
            hybridization_dict[Z] = [0, 1, 1]
    
    aromatic_like = {6, 7, 8, 16, 34, 52}  # C, N, O, S, Se, Te

    aromaticity_dict = {0: 0}
    for Z in valence_dict:
        if Z == 0: 
            continue
        aromaticity_dict[Z] = 1 if Z in aromatic_like else 0
    # print(atomic_numbers)
    # Assign atomic properties
    for i, atomic_num in enumerate(atomic_numbers.squeeze(1).tolist()):
        
        valence_electrons[i] = valence_dict.get(int(atomic_num))

        # Hybridization one-hot encoding
        hybridization[i, :] = torch.tensor(hybridization_dict.get(int(atomic_num)))

        # Aromaticity assumption
        aromaticity[i] = aromaticity_dict.get(int(atomic_num),0)


    
    enhanced_features = torch.cat((atomic_numbers, degrees, valence_electrons, hybridization, aromaticity), dim=1)
    # print("valence_electrons ", valence_electrons)
    # print("hybridization ", hybridization)
    # Concatenate all features
    # print("enhanced_features.shape : ",enhanced_features.shape)
    # print("atomic_numbers, degrees, valence_electrons, hybridization, aromaticity")
    # print("enhanced_features : ",enhanced_features)

    return enhanced_features

class S2VGraph(object):
  def __init__(self, g, label,mol, node_tags=None, node_features=None, edge_mat=None, edge_features=None, neighbors=None):
    '''
        g: a networkx graph
        label: an integer graph label
        node_tags: a list of integer node tags
        node_features: a torch float tensor, one-hot representation of the tag that is used as input to neural nets
        edge_mat: a torch long tensor, contain edge list, will be used to create torch sparse tensor
        neighbors: list of neighbors (without self-loop)
    '''
    self.label = label
    self.g = g
    self.node_tags = node_tags
    self.neighbors = []
    self.node_features = node_features
    self.edge_mat = edge_mat
    self.mol = mol
    self.edge_features = edge_features          # [E, d_e] float32
    self.neighbors = neighbors or []
    self.max_neighbor = max((len(n) for n in self.neighbors), default=0)

def create_nodetags(mol):
  # node tags for one mol

  feat_dict = {}
  node_tags = []

  # Create a NetworkX graph
  nx_graph = nx.Graph()

  # Add nodes (atoms)
  for i, atom in enumerate(mol.GetAtoms()):
      atom_symbol = atom.GetSymbol()
      nx_graph.add_node(i, label=atom_symbol)

  # Add edges (bonds)
  for bond in mol.GetBonds():
      start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
      nx_graph.add_edge(start, end)

  node_tags = [atom.GetSymbol() for atom in mol.GetAtoms()]

  return nx_graph, node_tags


def edge_index_and_attr_from_mol(mol: Chem.Mol):
    rows, cols, feats = [], [], []
    for b in mol.GetBonds():
        i = b.GetBeginAtomIdx(); j = b.GetEndAtomIdx()
        f = bond_feat(b)
        # undirected → store both directions for message passing
        rows += [i, j]; cols += [j, i]
        feats += [f, f]
    edge_index = torch.tensor([rows, cols], dtype=torch.long)              # [2, E]
    edge_attr  = torch.tensor(np.stack(feats, axis=0), dtype=torch.float32) # [E, 13]
    return edge_index, edge_attr

def pyg_graph_to_mol(data):
    """
    Convert a PyG Data object to an RDKit Mol object.
    """
    mol = RWMol()

    # Add atoms (assuming first feature in node feature vector is atomic number)
    for atom_feature in data.x:
        atomic_num = int(atom_feature[0])  # Extract atomic number from node features
        mol.AddAtom(Chem.Atom(atomic_num))

    # Convert tensors to numpy arrays
    edge_index = data.edge_index.numpy()
    edge_attr = data.edge_attr.numpy() if data.edge_attr is not None else None

    # Add bonds only if they don't exist
    for i in range(edge_index.shape[1]):  # Loop through edges
        start, end = int(edge_index[0, i]), int(edge_index[1, i])

        # Check if bond already exists
        if mol.GetBondBetweenAtoms(start, end) is None:
            # Determine bond type (modify this based on your dataset's edge attributes)
            bond_type = Chem.BondType.SINGLE  # Default to single bond
            mol.AddBond(start, end, bond_type)

    return mol


def create_graph_list(dataset, device="cpu"):
  def _coerce_edge_attr_19(mol, edge_index, edge_attr):
    """Ensure per-edge features have shape [E, 19]."""
    TARGET_DE = 19

    if edge_attr is None:
        return edge_index.to(device), torch.zeros((0, TARGET_DE), dtype=torch.float32, device=device)

    edge_attr = edge_attr.to(device)

    # 1D -> likely flattened or scalar-per-edge; rebuild from RDKit to get 19-dim
    if edge_attr.dim() == 1:
        ei_rd, ea_rd = edge_index_and_attr_from_mol(mol)  # must return [E, 19]
        return ei_rd.to(device), ea_rd.to(device)

    # Transposed case: [19, E] -> [E, 19]
    if edge_attr.size(1) != TARGET_DE and edge_attr.size(0) == TARGET_DE:
        edge_attr = edge_attr.t().contiguous()

    # Wrong width (not just transposed)
    if edge_attr.size(1) != TARGET_DE:
        raise ValueError(f"edge_attr width {edge_attr.size(1)} != {TARGET_DE}")

    # Rows must match number of edges
    if edge_attr.size(0) != edge_index.size(1):
        raise ValueError(f"edge rows {edge_attr.size(0)} != edges {edge_index.size(1)}")

    return edge_index.to(device), edge_attr
  g_list = []
  for data in dataset:
    mol = pyg_graph_to_mol(data)

    node_features = expand_atomic_features(data)

    '''if getattr(data, "edge_index", None) is not None and getattr(data, "edge_attr", None) is not None:
        edge_index = data.edge_index.long()
        edge_attr  = data.edge_attr.float()
    else:
        edge_index, edge_attr = edge_index_and_attr_from_mol(mol) # from RDKit'''

    if not isinstance(node_features, torch.Tensor):
        node_features = torch.tensor(node_features, dtype=torch.float32)
    node_features = node_features.to(device).float().contiguous()

    # ---- edges & edge features ----
    if getattr(data, "edge_index", None) is not None and getattr(data, "edge_attr", None) is not None:
        edge_index = data.edge_index.long().to(device)          # [2, E]
        edge_attr  = data.edge_attr.to(device).float()          # [E, ?] or [E]
    else:
        edge_index, edge_attr = edge_index_and_attr_from_mol(mol)  # [2, E], [E, 19]

    # Coerce to [E, 19] robustly (rebuilds via RDKit if needed)
    edge_index, edge_attr = _coerce_edge_attr_19(mol, edge_index, edge_attr)


    # print("create_graph_list: edge_index.shape ", edge_index.shape)
    # print("create_graph_list: edge_attr.shape ", edge_attr.shape)

    N = node_features.size(0)
    neighbors = [[] for _ in range(N)]
    for (u, v) in edge_index.t().tolist():
        neighbors[u].append(v)

    # print("create_g_list : W ", node_features.shape)
    # edge_index = data.edge_index
    # edge_features = data.edge_attr #get_edge_index_and_features(mol)
    # graph = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features)

    g, node_tags = create_nodetags(mol)
    g_list.append(S2VGraph(
        g = g, label= data.y, mol= mol, 
        node_tags= node_tags, 
        node_features=torch.tensor(node_features, dtype=torch.float32),
        edge_mat=edge_index,
        edge_features=edge_attr,
        neighbors=neighbors      
        ))
    # break
  return g_list