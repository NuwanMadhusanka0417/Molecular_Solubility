import torch
from rdkit import Chem
from rdkit.Chem import RWMol
import networkx as nx
from torch_geometric.data import Data
import numpy as np
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors

def _bond_stereo_to_float(stereo):
    """Map RDKit BondStereo to scalar: 0=None, 1=CIS, 2=TRANS, 3=Z, 4=E."""
    mapping = {
        Chem.rdchem.BondStereo.STEREONONE: 0.0,
        Chem.rdchem.BondStereo.STEREOCIS: 1.0,
        Chem.rdchem.BondStereo.STEREOTRANS: 2.0,
        Chem.rdchem.BondStereo.STEREOZ: 3.0,
        Chem.rdchem.BondStereo.STEREOE: 4.0,
    }
    return float(mapping.get(stereo, 0.0))


def bond_node_features_geognn(bond, pos):
    """
    Compute features for a single bond.

    Returns
    -------
    np.ndarray, shape [5]
        [bond_type, is_conjugated, in_ring, bond_length, stereo]
    """
    bt = bond.GetBondType()
    bond_type = {
        Chem.rdchem.BondType.SINGLE: 1,
        Chem.rdchem.BondType.DOUBLE: 2,
        Chem.rdchem.BondType.TRIPLE: 3,
        Chem.rdchem.BondType.AROMATIC: 4,
    }.get(bt, 0)

    is_conjugated = int(bond.GetIsConjugated())
    in_ring = int(bond.IsInRing())

    a = bond.GetBeginAtomIdx()
    b = bond.GetEndAtomIdx()
    length = float(np.linalg.norm(pos[a] - pos[b]))

    stereo = _bond_stereo_to_float(bond.GetStereo())

    return np.array([bond_type, 
                    is_conjugated, 
                    in_ring, 
                    length, 
                    stereo
                    ], dtype=np.float32)
'''
def build_edge_features_geognn_for_atom_graph(data, mol):
    """
    Use your existing bond_node_features_geognn(bond, pos)
    to build edge_attr aligned with data.edge_index.

    Returns: torch.FloatTensor [E, 4] or None if 3D embedding fails.
    """
    # Rebuild a clean SMILES → mol with Hs and 3D coords
    smiles = Chem.MolToSmiles(mol)
    mol3d = Chem.MolFromSmiles(smiles)
    if mol3d is None:
        return None

    mol3d = Chem.AddHs(mol3d)

    try:
        params = AllChem.ETKDGv3()
        params.randomSeed = 0xf00d
        if AllChem.EmbedMolecule(mol3d, params) != 0:
            return None
        AllChem.MMFFOptimizeMolecule(mol3d)
    except Exception:
        return None

    # 3D positions
    conf = mol3d.GetConformer()
    num_atoms = mol3d.GetNumAtoms()
    pos = np.zeros((num_atoms, 3), dtype=np.float32)
    for i in range(num_atoms):
        p = conf.GetAtomPosition(i)
        pos[i] = [p.x, p.y, p.z]

    # Build edge features aligned with data.edge_index
    E = data.edge_index.shape[1]
    edge_feats = []

    for e in range(E):
        u = int(data.edge_index[0, e])
        v = int(data.edge_index[1, e])

        bond = mol3d.GetBondBetweenAtoms(u, v)
        if bond is None:
            edge_feats.append(np.zeros(4, dtype=np.float32))
        else:
            bf = bond_node_features_geognn(bond, pos)   # ⬅️ your working function
            edge_feats.append(bf)

    edge_attr = np.stack(edge_feats, axis=0)  # [E, 4]
    return torch.from_numpy(edge_attr)        # float32


def build_edge_features_geognn_for_atom_graph(data, mol):
    """
    Use bond_node_features_geognn(bond, pos) to build edge_attr
    aligned with data.edge_index, preserving the original atom ordering.

    Returns: torch.FloatTensor [E, 4] or None if 3D embedding fails.
    """
    # Copy the original mol to avoid modifying it in-place
    mol3d = Chem.Mol(mol)

    # Make sure the atom order stays the same; AddHs will append H atoms
    mol3d = Chem.AddHs(mol3d)

    # Generate 3D coordinates
    try:
        params = AllChem.ETKDGv3()
        params.randomSeed = 0xf00d
        if AllChem.EmbedMolecule(mol3d, params) != 0:
            return None
        AllChem.MMFFOptimizeMolecule(mol3d)
    except Exception:
        return None

    # 3D positions
    conf = mol3d.GetConformer()
    num_atoms = mol3d.GetNumAtoms()
    pos = np.zeros((num_atoms, 3), dtype=np.float32)
    for i in range(num_atoms):
        p = conf.GetAtomPosition(i)
        pos[i] = [p.x, p.y, p.z]

    # Build edge features aligned with data.edge_index
    E = data.edge_index.shape[1]
    edge_feats = []

    for e in range(E):
        u = int(data.edge_index[0, e])
        v = int(data.edge_index[1, e])

        bond = mol3d.GetBondBetweenAtoms(u, v)
        if bond is None:
            # If you expect ONLY chemical bonds in edge_index, this shouldn't happen.
            # You can either assert or keep zeros as a fallback.
            edge_feats.append(np.zeros(4, dtype=np.float32))
        else:
            bf = bond_node_features_geognn(bond, pos)
            edge_feats.append(bf)

    edge_attr = np.stack(edge_feats, axis=0)  # [E, 4]
    return torch.from_numpy(edge_attr)
'''

def build_edge_features_geognn_for_atom_graph(data, mol):
    """
    Build edge_attr aligned with data.edge_index using
    bond_node_features_geognn(bond, pos).

    Returns: torch.FloatTensor [E, 5] or None if 3D embedding fails.
    Uses the *provided* mol (same atom order as data.edge_index) so (u,v) align.
    """
    E = data.edge_index.shape[1]
    if E == 0:
        # Single-atom or no-bond molecule: no edges to stack
        return torch.zeros((0, 5), dtype=torch.float32)

    mol3d = Chem.RWMol(mol)
    try:
        Chem.SanitizeMol(mol3d)
    except Exception:
        mol3d.UpdatePropertyCache(strict=False)
        # return None
    mol3d = Chem.AddHs(mol3d)
    try:
        params = AllChem.ETKDGv3()
        params.randomSeed = 0xF00D
        if AllChem.EmbedMolecule(mol3d, params) != 0:
            return None
        AllChem.MMFFOptimizeMolecule(mol3d)
    except Exception:
        return None

    conf = mol3d.GetConformer()
    num_atoms = mol3d.GetNumAtoms()
    pos = np.zeros((num_atoms, 3), dtype=np.float32)
    for i in range(num_atoms):
        p = conf.GetAtomPosition(i)
        pos[i] = [p.x, p.y, p.z]

    # Build edge features aligned with data.edge_index (E already set above)
    edge_feats = []
    for e in range(E):
        u = int(data.edge_index[0, e])
        v = int(data.edge_index[1, e])

        bond = mol3d.GetBondBetweenAtoms(u, v)
        if bond is None:
            edge_feats.append(np.zeros(5, dtype=np.float32))
        else:
            edge_feats.append(bond_node_features_geognn(bond, pos))

    edge_attr = np.stack(edge_feats, axis=0)  # [E, 5]
    return torch.from_numpy(edge_attr)

def expand_atomic_features(data, mol, precomputed_edge_attr=None):
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

    formal_charge = torch.zeros((num_nodes, 1))      # assuming neutral atoms
    hbond_flags  = torch.zeros((num_nodes, 2))       # [is_donor, is_acceptor]
    chirality    = torch.zeros((num_nodes, 2))       # [R, S] placeholder
    num_attached_h = torch.zeros((num_nodes, 1))     # number of H attached to each atom

    # New: Gasteiger partial charge, Crippen logP, TPSA, aromatic ring, ring size
    gasteiger_charge = torch.zeros((num_nodes, 1))
    crippen_logp = torch.zeros((num_nodes, 1))
    tpsa_contrib = torch.zeros((num_nodes, 1))
    is_in_aromatic_ring = torch.zeros((num_nodes, 1))
    smallest_ring_size = torch.zeros((num_nodes, 1))

    mol_h = Chem.RWMol(mol)
    try:
        Chem.SanitizeMol(mol_h)
        for i in range(num_nodes):
            num_attached_h[i] = float(mol_h.GetAtomWithIdx(i).GetTotalNumHs())
    except Exception:
        pass

    # Per-atom Gasteiger partial charge
    try:
        AllChem.ComputeGasteigerCharges(mol_h, throwOnParamFailure=False)
        for i in range(num_nodes):
            if mol_h.GetAtomWithIdx(i).HasProp("_GasteigerCharge"):
                q = float(mol_h.GetAtomWithIdx(i).GetProp("_GasteigerCharge"))
                gasteiger_charge[i] = q if not (np.isnan(q) or np.isinf(q)) else 0.0
    except Exception:
        pass

    # Per-atom Crippen logP and TPSA contributions
    try:
        crippen_contribs = rdMolDescriptors._CalcCrippenContribs(mol_h)
        for i, (logp, _) in enumerate(crippen_contribs):
            if i < num_nodes:
                crippen_logp[i] = float(logp) if not (np.isnan(logp) or np.isinf(logp)) else 0.0
    except Exception:
        pass

    try:
        tpsa_contribs = rdMolDescriptors._CalcTPSAContribs(mol_h)
        for i, tpsa in enumerate(tpsa_contribs):
            if i < num_nodes:
                tpsa_contrib[i] = float(tpsa) if not (np.isnan(tpsa) or np.isinf(tpsa)) else 0.0
    except Exception:
        pass

    # Is in aromatic ring and smallest ring size
    try:
        ri = mol_h.GetRingInfo()
        atom_ring_sizes = {}
        for ring in ri.AtomRings():
            size = len(ring)
            for aid in ring:
                if aid not in atom_ring_sizes or size < atom_ring_sizes[aid]:
                    atom_ring_sizes[aid] = size
        for i in range(num_nodes):
            atom = mol_h.GetAtomWithIdx(i)
            is_in_aromatic_ring[i] = float(atom.GetIsAromatic())
            smallest_ring_size[i] = float(atom_ring_sizes.get(i, 0))
    except Exception:
        pass

    valence_dict = {
        0:0,
        1:1,  2:2,
        3:1,  4:2,  5:3,  6:4,  7:5,  8:6,  9:7,  10:8,
        11:1, 12:2, 13:3, 14:4, 15:5, 16:6, 17:7, 18:8,
        19:1, 20:2, 21:2, 22:2, 23:2, 24:2, 25:2, 26:2, 27:2, 28:2, 29:2, 30:2,
        31:3, 32:4, 33:5, 34:6, 35:7, 36:8,
        37:1, 38:2, 39:2, 40:2, 41:2, 42:2, 43:2, 44:2, 45:2, 46:2, 47:2, 48:2,
        49:3, 50:4, 51:5, 52:6, 53:7, 54:8,
        55:1, 56:2,
        57:2, 58:2, 59:2, 60:2, 61:2, 62:2, 63:2, 64:2, 65:2, 66:2, 67:2, 68:2, 69:2, 70:2, 71:2,
        72:2, 73:2, 74:2, 75:2, 76:2, 77:2, 78:2, 79:2, 80:2,
        81:3, 82:4, 83:5, 84:6, 85:7, 86:8,
        87:1, 88:2,
        89:2, 90:2, 91:2, 92:2, 93:2, 94:2, 95:2, 96:2, 97:2, 98:2, 99:2, 100:2, 101:2, 102:2, 103:2,
        104:2, 105:2, 106:2, 107:2, 108:2, 109:2, 110:2, 111:2, 112:2,
        113:3, 114:4, 115:5, 116:6, 117:7, 118:8,
    }

    donor_like    = {7, 8, 16}                 # N, O, S
    acceptor_like = {7, 8, 9, 16, 17, 35, 53}  # N, O, F, S, Cl, Br, I

    for i, atomic_num in enumerate(atomic_numbers.squeeze(1).tolist()):
        atom = mol.GetAtomWithIdx(i)
        Z = int(atomic_num)

        valence_electrons[i] = valence_dict.get(Z, 0)

        # RDKit per-atom hybridization (context-specific, not element-based)
        hyb = atom.GetHybridization()
        hybridization[i, 0] = 1.0 if hyb == Chem.rdchem.HybridizationType.SP else 0.0
        hybridization[i, 1] = 1.0 if hyb == Chem.rdchem.HybridizationType.SP2 else 0.0
        hybridization[i, 2] = 1.0 if hyb == Chem.rdchem.HybridizationType.SP3 else 0.0

        # RDKit per-atom aromaticity (molecule-specific, not element-based)
        aromaticity[i] = float(atom.GetIsAromatic())

        formal_charge[i] = float(atom.GetFormalCharge())
        hbond_flags[i, 0] = 1.0 if Z in donor_like    else 0.0
        hbond_flags[i, 1] = 1.0 if Z in acceptor_like else 0.0

        if atom.HasProp('_CIPCode'):
            cip = atom.GetProp('_CIPCode')
            if cip == 'R':
                chirality[i, 0] = 1.0
                chirality[i, 1] = 0.0
            elif cip == 'S':
                chirality[i, 0] = 0.0
                chirality[i, 1] = 1.0

    enhanced_features = torch.cat((
        atomic_numbers,
        degrees,
        valence_electrons,
        hybridization,
        aromaticity,
        formal_charge,
        hbond_flags,
        chirality,
        num_attached_h,
        gasteiger_charge,
        crippen_logp,
        tpsa_contrib,
        is_in_aromatic_ring,
        smallest_ring_size,
    ), dim=1)

    return enhanced_features

class S2VGraph(object):
  def __init__(self, g, label, mol,
               node_tags=None,
               node_features=None,
               edge_index=None,
               edge_attr=None):
    '''
        g: a networkx graph
        label: an integer graph label
        node_tags: list of node labels
        node_features: torch.FloatTensor [N, F_node]
        edge_index: torch.LongTensor [2, E]
        edge_attr: torch.FloatTensor [E, F_edge]
    '''
    self.label = label
    self.g = g
    self.node_tags = node_tags
    self.neighbors = []
    self.node_features = node_features
    self.edge_mat = 0
    self.mol = mol
    self.max_neighbor = 0

    # NEW:
    self.edge_index = edge_index
    self.edge_attr = edge_attr

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

_ID2BOND = {
    0: Chem.BondType.SINGLE,
    1: Chem.BondType.DOUBLE,
    2: Chem.BondType.TRIPLE,
    3: Chem.BondType.AROMATIC,
}

def pyg_graph_to_mol(data):
    """Convert a PyG Data object to an RDKit Mol with correct bond orders."""
    mol = RWMol()

    for atom_feature in data.x:
        mol.AddAtom(Chem.Atom(int(atom_feature[0])))

    edge_index = data.edge_index.numpy()
    edge_attr = data.edge_attr.numpy() if data.edge_attr is not None else None

    added = set()
    for i in range(edge_index.shape[1]):
        u, v = int(edge_index[0, i]), int(edge_index[1, i])
        if (u, v) in added or (v, u) in added:
            continue
        if edge_attr is not None:
            bond_type = _ID2BOND.get(int(edge_attr[i]), Chem.BondType.SINGLE)
        else:
            bond_type = Chem.BondType.SINGLE
        mol.AddBond(u, v, bond_type)
        added.add((u, v))

    try:
        Chem.SanitizeMol(mol)
    except Exception:
        mol.UpdatePropertyCache(strict=False)
    return mol


def create_graph_list(dataset):
  g_list = []
  for data in dataset:
    mol = pyg_graph_to_mol(data)

    # Compute 3D edge features once (expensive: ETKDGv3 + MMFF)
    edge_attr = build_edge_features_geognn_for_atom_graph(data, mol)
    if edge_attr is None:
        E = data.edge_index.shape[1]
        edge_attr = torch.zeros((E, 5), dtype=torch.float32)

    node_features = expand_atomic_features(data, mol, precomputed_edge_attr=edge_attr)

    edge_index = data.edge_index.clone()                       # [2, E]

    g, node_tags = create_nodetags(mol)

    # Avoid copy-from-tensor warning: node_features is already a tensor
    nf = node_features.clone().detach().to(torch.float32)
    g_list.append(
        S2VGraph(
            g=g,
            label=data.y,
            mol=mol,
            node_tags=node_tags,
            node_features=nf,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )
    )
  return g_list
