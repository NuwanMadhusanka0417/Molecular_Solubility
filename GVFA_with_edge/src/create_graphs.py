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
                    in_ring 
                    # length, 
                    # stereo
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

    # Build edge features aligned with data.edge_index
    E = data.edge_index.shape[1]
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

def expand_atomic_features(data, mol):
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
    d_block       = set(range(21, 31)) | set(range(39, 49)) | set(range(72, 81)) | set(range(104, 113))
    f_block       = set(range(57, 72)) | set(range(89, 104))

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
    
    '''hybridization_dict = {
    1:  [0, 0, 1],  # Hydrogen (H) - sp3-like when bonded
    2:  [0, 0, 0],  # Helium (He) - noble gas, no bonding
    3:  [0, 0, 1],  # Lithium (Li) - single bond if bonded (ionic nature)
    4:  [0, 1, 0],  # Beryllium (Be) - linear (sp) but here considered as sp2 for simplicity
    5:  [0, 1, 1],  # Boron (B) - sp2 (BF3), sp3 in other cases
    6:  [1, 1, 1],  # Carbon (C) - sp, sp2, sp3
    7:  [0, 1, 1],  # Nitrogen (N) - sp2, sp3
    8:  [0, 1, 1],  # Oxygen (O) - sp2 (carbonyl), sp3 (alcohol)
    9:  [0, 0, 1],  # Fluorine (F) - typically sp3
    10: [0, 0, 0],  # Neon (Ne) - noble gas, no bonding
    11: [0, 0, 1],  # Sodium (Na) - ionic, but sp3-like if bonded
    12: [0, 0, 1],  # Magnesium (Mg) - sp3-like in complexes
    13: [0, 1, 1],  # Aluminum (Al) - sp2, sp3
    14: [0, 1, 1],  # Silicon (Si) - sp2, sp3
    15: [0, 1, 1],  # Phosphorus (P) - sp2, sp3
    16: [0, 1, 1],  # Sulfur (S) - sp2, sp3
    17: [0, 0, 1],  # Chlorine (Cl) - sp3
    18: [0, 0, 0],  # Argon (Ar) - noble gas, no bonding
    19: [0, 0, 1],  # Potassium (K) - ionic, sp3-like if bonded
    20: [0, 0, 1],  # Calcium (Ca) - ionic, sp3-like if bonded
    30: [0, 0, 1],  # Zinc (Zn) - sp3-like in complexes
    29: [0, 0, 1],  # Copper (Cu) - sp3-like in coordination
    26: [0, 0, 1],  # Iron (Fe) - sp3-like in coordination
    25: [0, 0, 1],  # Manganese (Mn) - sp3-like in coordination
    35: [0, 0, 1],  # Bromine (Br) - sp3
    53: [0, 0, 1],  # Iodine (I) - sp3
    }
    aromaticity_dict = {
    1: 0,    # Hydrogen (H)
    2: 0,    # Helium (He)
    3: 0,    # Lithium (Li)
    4: 0,    # Beryllium (Be)
    5: 0,    # Boron (B) - often part of π systems, but typically treated as non-aromatic in this context
    6: 1,    # Carbon (C) - Yes, in benzene, pyridine, etc.
    7: 1,    # Nitrogen (N) - Yes, in pyridine, imidazole
    8: 1,    # Oxygen (O) - Yes, in furan (but not always)
    9: 0,    # Fluorine (F)
    10: 0,   # Neon (Ne)
    11: 0,   # Sodium (Na)
    12: 0,   # Magnesium (Mg)
    13: 0,   # Aluminum (Al)
    14: 0,   # Silicon (Si)
    15: 0,   # Phosphorus (P) - rarely part of aromatic systems
    16: 1,   # Sulfur (S) - Yes, in thiophene
    17: 0,   # Chlorine (Cl)
    18: 0,   # Argon (Ar)
    19: 0,   # Potassium (K)
    20: 0,   # Calcium (Ca)
    30: 0,   # Zinc (Zn)
    29: 0,   # Copper (Cu)
    26: 0,   # Iron (Fe)
    25: 0,   # Manganese (Mn)
    35: 0,   # Bromine (Br)
    53: 0    # Iodine (I)
    }'''

    donor_like    = {7, 8, 16}                 # N, O, S
    acceptor_like = {7, 8, 9, 16, 17, 35, 53}  # N, O, F, S, Cl, Br, I

    aromatic_like = {6, 7, 8, 16, 34, 52}  # C, N, O, S, Se, Te

    # Chem.SanitizeMol(mol)
    # Chem.AssignAtomChiralTagsFromStructure(mol, replaceExistingTags=True)


    aromaticity_dict = {0: 0}
    for Z in valence_dict:
        if Z == 0: 
            continue
        aromaticity_dict[Z] = 1 if Z in aromatic_like else 0
    # print(atomic_numbers)
    # Assign atomic properties
    for i, atomic_num in enumerate(atomic_numbers.squeeze(1).tolist()):
        atom = mol.GetAtomWithIdx(i)  
        Z = int(atomic_num)    
        
        valence_electrons[i] = valence_dict.get(int(atomic_num))

        # Hybridization one-hot encoding
        hybridization[i, :] = torch.tensor(hybridization_dict.get(int(atomic_num)))

        # Aromaticity assumption
        aromaticity[i] = aromaticity_dict.get(int(atomic_num),0)

        formal_charge[i] = float(atom.GetFormalCharge())
        hbond_flags[i, 0] = 1.0 if Z in donor_like    else 0.0  # is_donor
        hbond_flags[i, 1] = 1.0 if Z in acceptor_like else 0.0  # is_acceptor

        if atom.HasProp('_CIPCode'):
            cip = atom.GetProp('_CIPCode')
            if cip == 'R':
                chirality[i, 0] = 1.0   # R
                chirality[i, 1] = 0.0
            elif cip == 'S':
                chirality[i, 0] = 0.0
                chirality[i, 1] = 1.0   # S

    # ================== NEW: edge features → node features ==================
    edge_attr = build_edge_features_geognn_for_atom_graph(data, mol)  # [E, 5] or None

    if edge_attr is not None:
        num_nodes = data.x.shape[0]
        num_edges, edge_feat_dim = edge_attr.shape

        node_edge_sum = torch.zeros((num_nodes, edge_feat_dim),
                                    dtype=edge_attr.dtype)

        for e in range(num_edges):
            u, v = data.edge_index[:, e]
            feat = edge_attr[e]
            node_edge_sum[u] += feat
            node_edge_sum[v] += feat

        # mean over incident bonds; atoms with degree 0 just stay zeros
        node_edge_mean = node_edge_sum / degrees.clamp(min=1.0)  # [N, 5]
    else:
        node_edge_mean = torch.zeros((num_nodes, 5), dtype=torch.float32)
    # =======================================================================



    
    enhanced_features = torch.cat((atomic_numbers, degrees, 
                                   valence_electrons, hybridization, aromaticity,
                                   formal_charge,
                                   hbond_flags,
                                   chirality,
                                   num_attached_h,
                                #    gasteiger_charge,
                                #    crippen_logp,
                                #    tpsa_contrib,
                                #    is_in_aromatic_ring,
                                #    smallest_ring_size,
                                   ), dim=1)
    # print("valence_electrons ", valence_electrons)
    # print("hybridization ", hybridization)
    # Concatenate all features
    # print("enhanced_features.shape : ",enhanced_features.shape)
    # print("atomic_numbers, degrees, valence_electrons, hybridization, aromaticity")
    # print("enhanced_features : ",enhanced_features)




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


def create_graph_list(dataset):
  g_list = []
  for data in dataset:
    mol = pyg_graph_to_mol(data)

    # node features (already include some edge-aggregated info)
    node_features = expand_atomic_features(data, mol)          # [N, F_node]

    # NEW: raw edge features per bond (E, F_edge)
    edge_attr = build_edge_features_geognn_for_atom_graph(data, mol)
    if edge_attr is None:
        # fallback: zero features, but keep shape consistent [E, 5]
        E = data.edge_index.shape[1]
        edge_attr = torch.zeros((E, 5), dtype=torch.float32)

    edge_index = data.edge_index.clone()                       # [2, E]

    g, node_tags = create_nodetags(mol)

    g_list.append(
        S2VGraph(
            g=g,
            label=data.y,
            mol=mol,
            node_tags=node_tags,
            node_features=torch.tensor(node_features, dtype=torch.float32),
            edge_index=edge_index,
            edge_attr=edge_attr,
        )
    )
  return g_list
