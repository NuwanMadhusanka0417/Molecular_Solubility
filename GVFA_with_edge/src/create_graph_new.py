import torch
from rdkit import Chem
from rdkit.Chem import RWMol
import networkx as nx

_ID2BOND = {0: Chem.BondType.SINGLE, 1: Chem.BondType.DOUBLE,
            2: Chem.BondType.TRIPLE, 3: Chem.BondType.AROMATIC}

def _data_to_mol(data):
    """Reconstruct an RDKit Mol from a PyG Data if none is provided."""
    mol = RWMol()
    # data.x[:,0] is atomic number (long/float)
    for z in data.x.view(-1):
        mol.AddAtom(Chem.Atom(int(z.item())))
    ei = data.edge_index.detach().cpu().numpy()
    ea = (data.edge_attr.detach().cpu().numpy()
          if (data.edge_attr is not None and data.edge_attr.numel()) else None)
    seen = set()
    for k in range(ei.shape[1]):
        u, v = int(ei[0, k]), int(ei[1, k])
        a, b = (u, v) if u <= v else (v, u)
        if a == b or (a, b) in seen:
            continue
        seen.add((a, b))
        btype = _ID2BOND.get(int(ea[k]) if ea is not None else 0, Chem.BondType.SINGLE)
        mol.AddBond(a, b, btype)
    m = mol.GetMol()
    try:
        Chem.SanitizeMol(m)
    except Exception:
        m.UpdatePropertyCache(strict=False)
    return m

def _total_valence_no_depr(atom: Chem.Atom) -> float:
    """Total valence without using deprecated GetTotalValence()."""
    # explicit + (implicit if known)
    ev = atom.GetExplicitValence(which=AtomValencePart.Implicit)
    iv = atom.GetImplicitValence(getExplicit=False)  # may be -1 when unknown
    return float(ev + (0 if iv < 0 else iv))

def expand_atomic_features(data, mol=None, normalize_continuous=True):
    """
    Build [N, F] node feature matrix with environment-aware chemistry:
      cols = [Z, degree, valence_electrons, total_valence, sp, sp2, sp3, aromatic]
    Args:
      data: torch_geometric.data.Data with x=[N,1] (atomic numbers), edge_index, (edge_attr optional)
      mol:  optional RDKit Mol; if None, reconstructed from `data`
    """

    valence_dict = {
        0:0,
        # Period 1
        1:1, 2:2,
        # Period 2
        3:1, 4:2, 5:3, 6:4, 7:5, 8:6, 9:7, 10:8,
        # Period 3
        11:1, 12:2, 13:3, 14:4, 15:5, 16:6, 17:7, 18:8,
        # Period 4
        19:1, 20:2, 21:2, 22:2, 23:2, 24:2, 25:2, 26:2, 27:2, 28:2, 29:2, 30:2,
        31:3, 32:4, 33:5, 34:6, 35:7, 36:8,
        # Period 5
        37:1, 38:2, 39:2, 40:2, 41:2, 42:2, 43:2, 44:2, 45:2, 46:2, 47:2, 48:2,
        49:3, 50:4, 51:5, 52:6, 53:7, 54:8
    }
    dev = data.x.device
    N = data.num_nodes

    # Base continuous features
    Z   = data.x.view(-1, 1).to(dtype=torch.float32, device=dev)
    deg = torch.bincount(data.edge_index[0], minlength=N).to(device=dev, dtype=torch.float32).unsqueeze(1)
    # Ensure we have an RDKit molecule for env-aware props
    if mol is None:
        mol = _data_to_mol(data)

    pt   = Chem.GetPeriodicTable()
    ve   = torch.zeros((N, 1), dtype=torch.float32, device=dev)  # valence electrons (outer-shell)
    tv   = torch.zeros((N, 1), dtype=torch.float32, device=dev)  # total valence in this molecule
    hyb  = torch.zeros((N, 3), dtype=torch.float32, device=dev)  # [sp, sp2, sp3]
    arom = torch.zeros((N, 1), dtype=torch.float32, device=dev)

    for i, a in enumerate(mol.GetAtoms()):
        Z_i = a.GetAtomicNum()
        ve[i, 0] = float(pt.GetNOuterElecs(Z_i))
        # iv = a.GetImplicitValence()
        tv[i, 0] = valence_dict.get(int(Z_i)) #_total_valence_no_depr(a)
        h = a.GetHybridization()
        if h == Chem.HybridizationType.SP:   hyb[i, 0] = 1.0
        elif h == Chem.HybridizationType.SP2: hyb[i, 1] = 1.0
        elif h == Chem.HybridizationType.SP3: hyb[i, 2] = 1.0
        arom[i, 0] = float(a.GetIsAromatic())

    X = torch.cat([Z, deg, ve, tv, hyb, arom], dim=1)

    # # Optional light scaling of continuous cols (Z, degree, ve, tv) to keep VSA/FPE stable
    # if normalize_continuous and N > 1:
    #     cont = X[:, :4]
    #     cmin, cmax = cont.min(0).values, cont.max(0).values
    #     X[:, :4] = (cont - cmin) / torch.clamp(cmax - cmin, min=1e-6)

    return X, mol

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

class S2VGraph(object):
  def __init__(self, g, label,mol, node_tags=None, node_features=None):
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
    self.edge_mat = 0
    self.mol = mol
    self.max_neighbor = 0

def create_graph_list(dataset):
  g_list = []
  for data in dataset:
    # mol = pyg_graph_to_mol(data)

    node_features, mol = expand_atomic_features(data)
    # print("create_g_list : W ", node_features.shape)
    # edge_index = data.edge_index
    # edge_features = data.edge_attr #get_edge_index_and_features(mol)
    # graph = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features)

    g, node_tags = create_nodetags(mol)
    g_list.append(S2VGraph(g = g, label= data.y, mol= mol, node_tags= node_tags, node_features=torch.tensor(node_features, dtype=torch.float32)))
    # break
  return g_list