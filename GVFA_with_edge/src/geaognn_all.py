import numpy as np
import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem

import pandas as pd
from torch_geometric.data import InMemoryDataset
import math
import torch
import networkx as nx


def bond_node_features_geognn(bond, pos):
    """
    Features for each bond (node in the bond-angle graph).

    Returns a 1D numpy array:
        [bond_type, is_conjugated, in_ring, bond_length]
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
    length = float(np.linalg.norm(pos[a] - pos[b]))  # 3D bond length

    return np.array([bond_type, is_conjugated, in_ring, length],
                    dtype=np.float32)


def smiles_to_bond_angle_data_geognn(smiles, y):
    """
    Build a bond-angle graph from a SMILES string.

    Returns a PyG Data object with fields:
      - data.bond_x           : [num_bonds, F_bond]  (bond node features)
      - data.angle_edge_index : [2, num_angle_edges] (directed edges between bonds)
      - data.angle_edge_attr  : [num_angle_edges, 3] (angle features)
      - data.y                : [1] target value (e.g., LogS)

    Angle features are [theta, cos(theta), sin(theta)] where theta is
    the bond angle at the central atom (radians).
    """
    # ----- RDKit molecule + 3D coords -----
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None  # skip invalid SMILES

    mol = Chem.AddHs(mol)

    # Generate a 3D conformer
    try:
        params = AllChem.ETKDGv3()
        params.randomSeed = 0xf00d
        if AllChem.EmbedMolecule(mol, params) != 0:
            return None  # embedding failed
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        return None

    conf = mol.GetConformer()
    num_atoms = mol.GetNumAtoms()

    # Positions [N, 3]
    pos = np.zeros((num_atoms, 3), dtype=np.float32)
    for i in range(num_atoms):
        p = conf.GetAtomPosition(i)
        pos[i] = [p.x, p.y, p.z]

    # ----- Bonds and bond node features -----
    bonds = list(mol.GetBonds())
    num_bonds = len(bonds)
    if num_bonds == 0:
        return None

    bond_x_list = []        # node features
    bond_endpoints = []     # (a, b) for each bond id

    for bond in bonds:
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()
        bond_endpoints.append((a, b))
        bond_x_list.append(bond_node_features_geognn(bond, pos))

    bond_x = torch.tensor(np.vstack(bond_x_list), dtype=torch.float32)  # [B, F_bond]

    # ----- Bond-angle edges: nodes = bonds, edges = angles -----

    # For each atom v, get incident bonds (bond_id, neighbor_atom)
    atom_to_bonds = [[] for _ in range(num_atoms)]
    for bond_id, (a, b) in enumerate(bond_endpoints):
        atom_to_bonds[a].append((bond_id, b))
        atom_to_bonds[b].append((bond_id, a))

    angle_edge_pairs = []   # list of [bond_i, bond_j]
    angle_feat_list  = []   # list of [theta, cos(theta), sin(theta)]

    for v in range(num_atoms):
        inc = atom_to_bonds[v]
        if len(inc) < 2:
            continue

        # All unordered pairs of incident bonds at atom v
        for i in range(len(inc)):
            bond_id1, u = inc[i]
            for j in range(i + 1, len(inc)):
                bond_id2, w = inc[j]

                # Angle u - v - w at atom v
                vec1 = pos[u] - pos[v]
                vec2 = pos[w] - pos[v]
                norm1 = np.linalg.norm(vec1)
                norm2 = np.linalg.norm(vec2)
                if norm1 < 1e-6 or norm2 < 1e-6:
                    continue

                cos_theta = float(np.dot(vec1, vec2) / (norm1 * norm2))
                cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
                theta = float(np.arccos(cos_theta))  # radians

                angle_feat = np.array(
                    [theta, np.cos(theta), np.sin(theta)],
                    dtype=np.float32
                )

                # Add two directed edges for this angle: bond1→bond2 and bond2→bond1
                angle_edge_pairs.append([bond_id1, bond_id2])
                angle_feat_list.append(angle_feat)

                angle_edge_pairs.append([bond_id2, bond_id1])
                angle_feat_list.append(angle_feat)

    if len(angle_edge_pairs) > 0:
        angle_edge_index = torch.tensor(
            angle_edge_pairs, dtype=torch.long
        ).t().contiguous()                      # [2, num_angle_edges]
        angle_edge_attr = torch.tensor(
            np.vstack(angle_feat_list), dtype=torch.float32
        )                                      # [num_angle_edges, 3]
    else:
        angle_edge_index = torch.empty((2, 0), dtype=torch.long)
        angle_edge_attr  = torch.empty((0, 3), dtype=torch.float32)

    # ----- Wrap in a PyG Data object -----
    data = Data()
    data.bond_x           = bond_x
    data.angle_edge_index = angle_edge_index
    data.angle_edge_attr  = angle_edge_attr
    data.y                = torch.tensor([float(y)], dtype=torch.float32)

    return data


class ZINCLikeCSV_geognn(InMemoryDataset):
    """
    Loads a CSV with SMILES + target (LogS) and stores
    a list of PyG Data objects containing ONLY the bond-angle graph.
    """
    def __init__(self, csv_path, smiles_col="smiles_canon", target_col="LogS"):
        self.csv_path = csv_path
        self.smiles_col = smiles_col
        self.target_col = target_col
        self.good_idx = []

        df = pd.read_csv(csv_path)
        super().__init__('.')

        data_list = []
        
        for idx, (smi, y) in enumerate(zip(df[smiles_col], df[target_col])):
            d = smiles_to_bond_angle_data_geognn(smi, y)
            if d is not None:
                data_list.append(d)
                self.good_idx.append(idx)

        self.data, self.slices = self.collate(data_list)


def load_data_geognn():
    dataset_test  = ZINCLikeCSV_geognn("final_data/final_unique_test.csv")
    dataset_train = ZINCLikeCSV_geognn("final_data/final_unique_train_fixed.csv")
    return dataset_train, dataset_test


class S2VGraph_geognn(object):
  def __init__(self, g, label,mol, node_tags=None, node_features=None, bond_x=None,
               bond_endpoints=None,
               angle_edge_index=None,
               angle_edge_attr=None,
               ):
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
    self.bond_x=None,
    self.bond_endpoints=bond_x,
    self.angle_edge_index=bond_endpoints,
    self.angle_edge_attr=angle_edge_attr,


def data_to_S2VGraph_bond_geognn(data):
    """
    Convert a PyG Data (bond-angle graph) into an S2VGraph
    where nodes = bonds and edges = angles between bonds.
    """
    num_bonds = data.bond_x.size(0)

    # 1. Build NetworkX graph over bonds
    g_nx = nx.Graph()
    g_nx.add_nodes_from(range(num_bonds))

    ei = data.angle_edge_index
    if ei.numel() > 0:
        src = ei[0].tolist()
        dst = ei[1].tolist()
        edges = list(zip(src, dst))
        g_nx.add_edges_from(edges)

    # 2. Node features = bond_x
    node_features = data.bond_x.clone()  # [num_bonds, F_bond]

    # 3. Label (LogS)
    y = data.y
    label = y.item() if y.numel() == 1 else y

    # 4. Wrap in S2VGraph
    g_obj = S2VGraph_geognn(
        g=g_nx,
        label=label,
        mol=None,              # optional; you can pass RDKit Mol if you want
        node_tags=None,        # not used in your current VSA code
        node_features=node_features,
        bond_x=None,
        bond_endpoints=None,
        angle_edge_index=data.angle_edge_index,
        angle_edge_attr=data.angle_edge_attr,
    )

    return g_obj


def create_bond_graph_list_geognn(dataset):

    """
    Turn a PyG dataset into a list of S2VGraph objects
    for the bond-angle graphs.
    """
    graphs = []
    for d in dataset:
        g = data_to_S2VGraph_bond_geognn(d)
        if g is not None:
            graphs.append(g)
    return graphs


def project_node_features_geognn(g_list, new_dim):
    """
    Randomly project node_features of each graph in g_list
    from original_dim -> new_dim using a shared random matrix W.
    """
    if len(g_list) == 0:
        return g_list

    # Make sure node_features exist
    if g_list[0].node_features is None:
        raise ValueError("node_features is None for the first graph. "
                         "Make sure you set g.node_features before calling VSA_conversion.")

    original_feature_dim = g_list[0].node_features.size(1)
    print("Original feature dim:", original_feature_dim)
    print("Target HV dim       :", new_dim)

    # Set a random seed for reproducibility (within this call)
    torch.manual_seed(0)

    # Random projection matrix: [F_orig, new_dim]
    W = torch.randn(original_feature_dim, new_dim) / math.sqrt(new_dim)
    print("W shape:", W.shape)

    print("g_list[0].node_features shape BEFORE:", g_list[0].node_features.shape)
    for g in g_list:
        if g.node_features is not None:
            g.node_features = torch.matmul(g.node_features, W)
    print("g_list[0].node_features shape AFTER :", g_list[0].node_features.shape)

    return g_list


def VSA_conversion_geognn(g_list, new_dim=None, use_bond_x_if_none=True):
    """
    Prepare graphs for GVFA/VSA:
      - build neighbors and edge_mat from g.g (NetworkX graph)
      - ensure g.node_features exists
        (for bond graphs, we can copy from g.bond_x)
      - optionally project node_features to hypervector space (new_dim)

    Args:
        g_list : list of S2VGraph objects
        new_dim : if not None, project features to this HV dimension
        use_bond_x_if_none : if True and g.node_features is None but g.bond_x
                             exists, copy g.bond_x into g.node_features
    """
    # 1. Ensure node_features are set
    for g in g_list:
        if getattr(g, "node_features", None) is None:
            if use_bond_x_if_none and hasattr(g, "bond_x") and g.bond_x is not None:
                # For bond-angle graphs: use bond_x as the node features
                g.node_features = g.bond_x.clone()
            else:
                raise ValueError(
                    "g.node_features is None and no usable bond_x found. "
                    "Set g.node_features (or bond_x) before calling VSA_conversion."
                )

    # 2. Build neighbors and edge_mat from the NetworkX graph g.g
    for g in g_list:
        num_nodes = len(g.g)

        # Neighbors list (no self-loops)
        g.neighbors = [[] for _ in range(num_nodes)]
        for i, j in g.g.edges():
            g.neighbors[i].append(j)
            g.neighbors[j].append(i)

        # Max degree
        degree_list = [len(g.neighbors[i]) for i in range(num_nodes)]
        g.max_neighbor = max(degree_list) if degree_list else 0

        # Edge matrix for GVFA (undirected: add both directions)
        edges = [list(pair) for pair in g.g.edges()]
        edges.extend([[j, i] for i, j in edges])
        if len(edges) > 0:
            g.edge_mat = torch.LongTensor(edges).transpose(0, 1)  # [2, num_edges]
        else:
            g.edge_mat = torch.empty((2, 0), dtype=torch.long)

    # 3. Optional random projection to hypervector space
    if new_dim is not None:
        g_list = project_node_features_geognn(g_list, new_dim)

    return g_list


def getEmbedding_geognn( model, device, train_graphs, batch_size=100, SUM = True):

    model.to(device)
    model.train()

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
    final_embeddings = torch.cat(combined_embeddings, dim=1)

    # print("getEmbedding :: endo")
    return final_embeddings, final_labels
