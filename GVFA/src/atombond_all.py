import torch
from rdkit import Chem
from rdkit.Chem import RWMol
import networkx as nx
from torch_geometric.data import Data
from rdkit import RDLogger
import pandas as pd, torch
from torch_geometric.data import InMemoryDataset, Data
import numpy as np
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



    
    enhanced_features = torch.cat((atomic_numbers, degrees, 
                                   valence_electrons, hybridization, aromaticity,
                                   formal_charge,
                                    hbond_flags,
                                    chirality,
                                    ), dim=1)
    # print("valence_electrons ", valence_electrons)
    # print("hybridization ", hybridization)
    # Concatenate all features
    # print("enhanced_features.shape : ",enhanced_features.shape)
    # print("atomic_numbers, degrees, valence_electrons, hybridization, aromaticity")
    # print("enhanced_features : ",enhanced_features)

    return enhanced_features

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

    node_features = expand_atomic_features(data, mol)
    # print("create_g_list : W ", node_features.shape)
    # edge_index = data.edge_index
    # edge_features = data.edge_attr #get_edge_index_and_features(mol)
    # graph = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features)

    g, node_tags = create_nodetags(mol)
    g_list.append(S2VGraph(g = g, label= data.y, mol= mol, node_tags= node_tags, node_features=torch.tensor(node_features, dtype=torch.float32)))
    # break
  return g_list

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
    def __init__(self, csv_path, smiles_col="smiles_canon", target_col="LogS"):
        df = pd.read_csv(csv_path)
        super().__init__('.')
        graphs = []
        for smi, y in zip(df[smiles_col], df[target_col]):
            g = smiles_to_data(smi, y)
            if g is not None:
                graphs.append(g)
        self.data, self.slices = self.collate(graphs)

def load_data():
  dataset_test  = ZINCLikeCSV("final_data/final_unique_test.csv")
  dataset_train  = ZINCLikeCSV("final_data/final_unique_train_fixed.csv")

  return dataset_train, dataset_test #, gl_train, 

# def project_node_features(g_list, original_feature_dim, new_dim):
#     # Set a random seed for reproducibility
#     torch.manual_seed(0)
#     # Generate a random projection matrix
#     # R = np.random.randn(original_feature_dim, new_dim) / np.sqrt(new_dim)
#     # Initialize a random weight matrix for projection
#     W = torch.randn(original_feature_dim, new_dim) / np.sqrt(new_dim)
#     print("W : ", W.shape)
#     # Project node features for each graph

#     print("g list item shape before : ", g_list[0].node_features.shape)
#     for g in g_list:
#         # Assuming g.node_features is a torch.Tensor
#         if g.node_features is not None:
#             # print(g.node_features)
#             g.node_features  = torch.matmul(g.node_features, W)
#             # print("new g.node_features : ",g.node_features.shape)
#     print("g list item shape after : ", g_list[0].node_features.shape)
#     return g_list

def project_node_features(g_list, W):
    for g in g_list:
        if g.node_features is not None:
            g.node_features = g.node_features @ W
    return g_list

def VSA_conversion(g_list, new_dim=None):
    # Add labels and edge_mat
    for g in g_list:
        g.neighbors = [[] for _ in range(len(g.g))]

        # Build neighbors list
        for i, j in g.g.edges():
            g.neighbors[i].append(j)
            g.neighbors[j].append(i)

        # Compute max degree
        degree_list = [len(g.neighbors[i]) for i in range(len(g.g))]
        g.max_neighbor = max(degree_list)

        # Create edge matrix
        edges = [list(pair) for pair in g.g.edges()]
        edges.extend([[j, i] for i, j in edges])
        g.edge_mat = torch.LongTensor(edges).transpose(0, 1)

    #Extracting unique tag labels
    # tagset = set([])
    # for g in g_list:
    #     tagset = tagset.union(set(g.node_tags))

    # tagset = list(tagset)
    # tag2index = {tagset[i]:i for i in range(len(tagset))}


    ########## This part make one hit encoding of each node as they contain different atoms
    # for g in g_list:
    #     g.node_features = torch.zeros(len(g.node_tags), len(tagset))
    #     g.node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1
            # hypervector[range(len(g.node_tags)), [tag2index[tag] for tag in node_tags if tag in tag2index]] = 1

    original_feature_dim = len(g_list[0].node_features[0])# len(tagset)
    # print(len(tagset))
    print("VSA_conversion",len(g_list[0].node_features[0]))


    # if new_dim:
    #     g_list = project_node_features(g_list, original_feature_dim, new_dim)
    return g_list

'''def getEmbedding( model, device, train_graphs, batch_size=100, SUM = True):

    model = model.to(device)
    # model.train()
    model.eval()

    combined_embeddings = []  # Initialize the total embedding
    all_labels = []

    # Create batches
    num_graphs = len(train_graphs)
    for start_idx in range(0, num_graphs, batch_size):
        end_idx = min(start_idx + batch_size, num_graphs)
        batch_graphs = train_graphs[start_idx:end_idx]
        # print("getEmbedding :: Before BBBBB")
        output = model(batch_graphs)
        # print(output.shape)

        ################### For regression taskuse TRUE and false both as sum ################

        
        if(SUM==True):    # allways use SUM
            # Sum all embeddings
            combined_embedding = output.sum(dim=0, keepdim=True)   #torch.sum(torch.stack(output), dim=0)  # Sum along the new batch dimension
            # 100 tensors
        else:
            #Concat
            combined_embedding = torch.cat(output, dim=1)
        
        # Add the summed embeddings of this batch to the total embedding
        #############################################################################
        # combined_embeddings.append(output) #combined_embedding)
        combined_embeddings.append(combined_embedding)

        ####################################Place conmvert label########
        # Collect labels
        labels = torch.FloatTensor([graph.label for graph in batch_graphs]).to(device)
        all_labels.append(labels)


    final_labels = torch.cat(all_labels, dim=0)
    final_embeddings = torch.cat(combined_embeddings, dim=0)

    # print("getEmbedding :: endo")
    return final_embeddings, final_labels'''

def getEmbedding(model, device, graphs, batch_size=100):
    """
    Returns:
        embeddings: [N, D]
        labels:     [N]
    """
    model = model.to(device)
    model.eval()

    all_emb = []
    all_y = []

    with torch.no_grad():
        for start in range(0, len(graphs), batch_size):
            batch = graphs[start:start + batch_size]
            out = model(batch)

            # normalize out to tensor [B, D]
            if isinstance(out, (list, tuple)):
                out = torch.stack(out, dim=0)
            elif out.dim() == 1:
                out = out.unsqueeze(0)

            all_emb.append(out.detach().cpu())
            all_y.append(torch.tensor([g.label for g in batch], dtype=torch.float32))

    embeddings = torch.cat(all_emb, dim=0)  # [N, D]
    labels = torch.cat(all_y, dim=0)        # [N]
    return embeddings, labels
