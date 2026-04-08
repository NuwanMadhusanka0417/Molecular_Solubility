"""
GVFA with cross-layer attention for molecular solubility (logS) prediction.

Uses the existing GraphCNN from graphcnnVSA_Binding_FULL.py (basic GVFA:
parameter-free, returns [num_layers, num_graphs, feature_dim]). Adds an
INPUT-DEPENDENT attention over layers and a small MLP regressor; only the
attention + regressor are trained. GVFA runs inside torch.no_grad().

Dataset: Cui et al. 2020 style — training CSV (e.g. 9943 compounds) and
independent test set (e.g. 62 anticancer compounds). RDKit featurization:
node 92-dim, edge 10-dim. No PyTorch Geometric required.
"""

from __future__ import print_function

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

# RDKit for SMILES -> graph
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

# Add project root for imports; graphcnnVSA_Binding_FULL is unchanged
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.graphcnnVSA_Binding_FULL import GraphCNN


# -----------------------------------------------------------------------------
# Graph type compatible with GraphCNN.forward(): .g, .node_features, .edge_mat
# -----------------------------------------------------------------------------

class MolGraph(object):
    """Simple graph for GraphCNN: .g (networkx), .node_features [N, F], .edge_mat [2, E], .label (logS)."""
    def __init__(self, g, node_features, edge_mat, label):
        self.g = g
        self.node_features = node_features  # [N, 92]
        self.edge_mat = edge_mat          # [2, E] LongTensor
        self.label = label


# -----------------------------------------------------------------------------
# RDKit featurization: 92-dim atom, 10-dim edge (Cui / Ahmad et al. style)
# -----------------------------------------------------------------------------

# Atom symbol -> index for one-hot (common elements; pad rest to "Other")
ATOM_SYMBOLS = [
    'C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'B', 'Si', 'H',
    'Na', 'K', 'Ca', 'Fe', 'Zn', 'Cu', 'Mg', 'Mn', 'Se', 'As', 'Other'
]
ATOM_SYMBOL_DIM = 60  # one-hot size (pad to 60 for fixed size)

def _atom_features_92(mol):
    """One-hot style atom features: symbol, degree, charge, radical, hybridization, aromaticity, hydrogens, chirality. Total 92."""
    num_atoms = mol.GetNumAtoms()
    # Pre-allocate
    symbol_onehot = np.zeros((num_atoms, 60), dtype=np.float32)  # we'll use first len(ATOM_SYMBOLS) then pad
    degree = np.zeros((num_atoms, 11), dtype=np.float32)       # 0-10
    formal_charge = np.zeros((num_atoms, 3), dtype=np.float32)  # -1, 0, 1
    radical = np.zeros((num_atoms, 4), dtype=np.float32)        # 0,1,2,3
    hybridization = np.zeros((num_atoms, 6), dtype=np.float32)  # SP, SP2, SP3, SP3D, SP3D2, other
    aromatic = np.zeros((num_atoms, 1), dtype=np.float32)
    num_h = np.zeros((num_atoms, 5), dtype=np.float32)          # 0-4
    chirality = np.zeros((num_atoms, 2), dtype=np.float32)      # R, S (none -> both 0)

    HYBRIDIZATION_MAP = {
        Chem.rdchem.HybridizationType.SP: 0,
        Chem.rdchem.HybridizationType.SP2: 1,
        Chem.rdchem.HybridizationType.SP3: 2,
        Chem.rdchem.HybridizationType.SP3D: 3,
        Chem.rdchem.HybridizationType.SP3D2: 4,
    }

    for i in range(num_atoms):
        atom = mol.GetAtomWithIdx(i)
        sym = atom.GetSymbol()
        idx_sym = ATOM_SYMBOLS.index(sym) if sym in ATOM_SYMBOLS else len(ATOM_SYMBOLS) - 1
        symbol_onehot[i, min(idx_sym, 59)] = 1.0

        d = min(atom.GetDegree(), 10)
        degree[i, d] = 1.0

        fc = atom.GetFormalCharge()
        if fc == -1:
            formal_charge[i, 0] = 1.0
        elif fc == 0:
            formal_charge[i, 1] = 1.0
        else:
            formal_charge[i, 2] = 1.0

        rad = min(atom.GetNumRadicalElectrons(), 3)
        radical[i, rad] = 1.0

        hyb = atom.GetHybridization()
        hyb_idx = HYBRIDIZATION_MAP.get(hyb, 5)
        hybridization[i, hyb_idx] = 1.0

        aromatic[i, 0] = 1.0 if atom.GetIsAromatic() else 0.0
        num_h[i, min(atom.GetTotalNumHs(), 4)] = 1.0

        if atom.HasProp('_CIPCode'):
            cip = atom.GetProp('_CIPCode')
            if cip == 'R':
                chirality[i, 0] = 1.0
            elif cip == 'S':
                chirality[i, 1] = 1.0

    out = np.hstack([
        symbol_onehot[:, :60],  # 60
        degree,                  # 11
        formal_charge,           # 3
        radical,                 # 4
        hybridization,          # 6
        aromatic,               # 1
        num_h,                  # 5
        chirality               # 2 -> total 92
    ])
    assert out.shape[1] == 92, out.shape
    return out


def _bond_features_10(mol):
    """Bond features per edge: bond type, conjugation, ring, stereo. Total 10 (one-hot / binary)."""
    bond_type_onehot = np.zeros((0, 4), dtype=np.float32)   # single, double, triple, aromatic
    conjugated = np.zeros((0, 1), dtype=np.float32)
    in_ring = np.zeros((0, 1), dtype=np.float32)
    stereo_onehot = np.zeros((0, 4), dtype=np.float32)       # none, cis, trans, other

    BT_MAP = {
        Chem.rdchem.BondType.SINGLE: 0,
        Chem.rdchem.BondType.DOUBLE: 1,
        Chem.rdchem.BondType.TRIPLE: 2,
        Chem.rdchem.BondType.AROMATIC: 3,
    }
    STEREO_MAP = {
        Chem.rdchem.BondStereo.STEREONONE: 0,
        Chem.rdchem.BondStereo.STEREOCIS: 1,
        Chem.rdchem.BondStereo.STEREOTRANS: 2,
    }

    feats = []
    for bond in mol.GetBonds():
        bt = BT_MAP.get(bond.GetBondType(), 0)
        bt_vec = np.zeros(4, dtype=np.float32)
        bt_vec[bt] = 1.0
        conj = np.array([1.0 if bond.GetIsConjugated() else 0.0], dtype=np.float32)
        ring = np.array([1.0 if bond.IsInRing() else 0.0], dtype=np.float32)
        st = bond.GetStereo()
        st_idx = STEREO_MAP.get(st, 0)
        if st_idx == 0 and st not in STEREO_MAP:
            st_idx = 3  # other
        st_vec = np.zeros(4, dtype=np.float32)
        st_vec[min(st_idx, 3)] = 1.0
        feats.append(np.concatenate([bt_vec, conj, ring, st_vec]))
    if not feats:
        return np.zeros((0, 10), dtype=np.float32)
    return np.stack(feats, axis=0)


def smiles_to_mol_graph(smiles, logS):
    """Convert one SMILES and logS to MolGraph with .g, .node_features [N,92], .edge_mat [2,E], .label."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    import networkx as nx
    g_nx = nx.Graph()
    for i in range(mol.GetNumAtoms()):
        g_nx.add_node(i)
    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        g_nx.add_edge(u, v)

    node_features = _atom_features_92(mol)
    node_features = torch.tensor(node_features, dtype=torch.float32)

    edges = []
    for u, v in g_nx.edges():
        edges.append([u, v])
        edges.append([v, u])
    if not edges:
        edge_mat = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_mat = torch.tensor(edges, dtype=torch.long).t()  # [2, E]

    label = torch.tensor([float(logS)], dtype=torch.float32)
    return MolGraph(g_nx, node_features, edge_mat, label)


def load_cui_style_data(train_csv, test_csv, smiles_col="SMILES", target_col="logS"):
    """
    Load training and test CSVs (e.g. Cui et al. 2020: 9943 train, 62 test).
    Returns list of MolGraph for train and test.
    """
    train_df = pd.read_csv(train_csv)
    train_df = train_df.dropna(subset=[smiles_col, target_col])
    test_df = pd.read_csv(test_csv)
    test_df = test_df.dropna(subset=[smiles_col, target_col])

    train_graphs = []
    for _, row in train_df.iterrows():
        g = smiles_to_mol_graph(row[smiles_col], row[target_col])
        if g is not None:
            train_graphs.append(g)
    test_graphs = []
    for _, row in test_df.iterrows():
        g = smiles_to_mol_graph(row[smiles_col], row[target_col])
        if g is not None:
            test_graphs.append(g)
    return train_graphs, test_graphs


def load_solubility1_data(csv_path="final_data/solubility_1.csv", smiles_col="SMILES", target_col="logS",
                          test_size=0.1, seed=42):
    """
    Load solubility_1.csv and split into train/test (90/10), matching load_data.py 'old' protocol.
    Returns train_graphs, test_graphs.
    """
    from sklearn.model_selection import train_test_split
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[smiles_col, target_col])
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed, shuffle=True)
    train_graphs = []
    for _, row in train_df.iterrows():
        g = smiles_to_mol_graph(row[smiles_col], row[target_col])
        if g is not None:
            train_graphs.append(g)
    test_graphs = []
    for _, row in test_df.iterrows():
        g = smiles_to_mol_graph(row[smiles_col], row[target_col])
        if g is not None:
            test_graphs.append(g)
    return train_graphs, test_graphs


# -----------------------------------------------------------------------------
# GVFAWithAttention: cross-layer attention + regressor (only these train)
# -----------------------------------------------------------------------------

class GVFAWithAttention(nn.Module):
    """
    Wraps the existing GraphCNN (basic GVFA). Runs GVFA in no_grad; applies
    INPUT-DEPENDENT attention over the layer dimension (scores from embedding
    content), then a small MLP regressor. Only attn_vector and regressor are
    trained — this is why attention weights can differ per molecule (scores
    are computed from the layer embeddings, not fixed scalars).
    """
    def __init__(self, gvfa_encoder, feature_dim, regressor_hidden=64, dropout=0.1):
        super(GVFAWithAttention, self).__init__()
        self.encoder = gvfa_encoder
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.feature_dim = feature_dim
        # Scores each layer per molecule: input-dependent (true attention)
        self.attn_vector = nn.Linear(feature_dim, 1)
        self.regressor = nn.Sequential(
            nn.Linear(feature_dim, regressor_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(regressor_hidden, 1),
        )

    def forward(self, batch_graph, return_attention_weights=False):
        """
        batch_graph: list of MolGraph (or S2VGraph with .node_features, .edge_mat, .g).
        Returns: pred [B, 1]. If return_attention_weights, also (attn_weights [B, num_layers], embedding [B, D]).
        """
        with torch.no_grad():
            # H_stack: [num_layers, B, 2*D] (FHRR real/imag interleaved)
            H_stack = self.encoder(batch_graph)

        num_layers, B, D = H_stack.shape
        # Input-dependent scores: each layer embedding -> scalar (different per molecule)
        scores = self.attn_vector(H_stack).squeeze(-1)  # [num_layers, B]
        attn_weights = F.softmax(scores, dim=0)          # softmax over layers (dim=0)
        # Weighted sum over layers: [num_layers, B, D] * [num_layers, B, 1] -> sum -> [B, D]
        embedding = (H_stack * attn_weights.unsqueeze(-1)).sum(dim=0)  # [B, D]
        pred = self.regressor(embedding)  # [B, 1]

        if return_attention_weights:
            return pred, (attn_weights.t(), embedding)  # attn_weights [B, num_layers]
        return pred


# -----------------------------------------------------------------------------
# Training with 10-fold CV, early stopping; evaluation; attention visualization
# -----------------------------------------------------------------------------

def compute_metrics_ahmad(y_true, y_pred):
    """
    Ahmad et al. (ACS Omega 2023) metrics on same arrays (original logS units).
    A) RMSE (Eq. 18): sqrt(mean((y - yhat)^2))
    B) R2 (Eq. 17) coefficient-of-determination: 1 - SSE/SST = 1 - sum((y-yhat)^2)/sum((y-ybar)^2)
    C) Pearson R²: (corr(y, yhat))^2
    y_true, y_pred: 1D arrays in original logS units (no z-score un-inverted).
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    n = len(y_true)
    if n == 0:
        return {"rmse": float("nan"), "r2_cod": float("nan"), "pearson_r2": float("nan"), "pearson_r": float("nan")}
    # RMSE (Eq. 18)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    if n < 2:
        return {"rmse": rmse, "r2_cod": 0.0, "pearson_r2": 0.0, "pearson_r": 0.0}
    # R2 Eq. 17: 1 - SSE/SST
    sse = np.sum((y_true - y_pred) ** 2)
    sst = np.sum((y_true - np.mean(y_true)) ** 2)
    r2_cod = 1.0 - (sse / sst) if sst > 0 else 0.0
    # Pearson R²
    pearson_r, _ = pearsonr(y_true, y_pred)
    pearson_r2 = pearson_r ** 2
    return {"rmse": rmse, "r2_cod": r2_cod, "pearson_r2": pearson_r2, "pearson_r": pearson_r}


def evaluate(model, dataloader_list, device, return_arrays=False):
    """
    Compute Ahmad et al. 2023 metrics on the same (y_true, y_pred) arrays.
    Returns rmse, r2_cod (Eq. 17), pearson_r2, pearson_r. If return_arrays=True, also returns y_true, y_pred
    for summary stats (both in original logS units).
    """
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_graphs, labels in dataloader_list:
            labels = labels.to(device)
            pred = model(batch_graphs)
            all_preds.append(pred.cpu().numpy().ravel())
            all_labels.append(labels.cpu().numpy().ravel())
    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    out = compute_metrics_ahmad(y_true, y_pred)
    if return_arrays:
        out["y_true"] = y_true
        out["y_pred"] = y_pred
    return out


def train_gvfa_attention(
    train_graphs,
    val_graphs,
    device,
    feature_dim=92,
    num_layers=4,
    delta=1,
    equation=10,
    graph_pooling_type="average",
    neighbor_pooling_type="sum",
    lr=1e-3,
    epochs=200,
    batch_size=64,
    patience=10,
    regressor_hidden=64,
    dropout=0.1,
    save_path=None,
    verbose=True,
):
    """
    Train GVFAWithAttention: optimizer is Adam on ONLY attn_vector and regressor.
    Early stopping with patience=10; save best by validation RMSE. Print R² and RMSE each epoch.
    """
    # Build encoder (basic GVFA: no reservoir, no edge features in encoder for minimal training)
    encoder = GraphCNN(
        input_dim=feature_dim,
        num_layers=num_layers,
        delta=delta,
        graph_pooling_type=graph_pooling_type,
        neighbor_pooling_type=neighbor_pooling_type,
        device=device,
        equation=equation,
        edge_feat_dim=0,
        use_reservoir=False,
    )
    # FHRR encoder output is real/imag interleaved -> 2 * hypervector dim
    encoder_out_dim = feature_dim * 2
    model = GVFAWithAttention(
        encoder,
        feature_dim=encoder_out_dim,
        regressor_hidden=regressor_hidden,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(
        list(model.attn_vector.parameters()) + list(model.regressor.parameters()),
        lr=lr,
    )
    criterion = nn.MSELoss()

    def _batches(graphs, shuffle=True):
        idx = np.arange(len(graphs))
        if shuffle:
            np.random.shuffle(idx)
        for start in range(0, len(idx), batch_size):
            batch_idx = idx[start : start + batch_size]
            batch_graphs = [graphs[i] for i in batch_idx]
            labels = torch.tensor(
                [float(torch.as_tensor(g.label).item()) for g in batch_graphs],
                dtype=torch.float32,
                device=device,
            ).unsqueeze(1)
            yield batch_graphs, labels

    best_val_rmse = float("inf")
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for batch_graphs, labels in _batches(train_graphs):
            optimizer.zero_grad()
            pred = model(batch_graphs)
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1
        train_rmse = np.sqrt(train_loss / n_batches) if n_batches else 0.0

        val_batches = list(_batches(val_graphs, shuffle=False))
        val_metrics = evaluate(model, val_batches, device)
        val_rmse, val_r2 = val_metrics["rmse"], val_metrics["r2_cod"]

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if verbose and (epoch + 1) % 1 == 0:
            print(
                f"Epoch {epoch+1} train_rmse={train_rmse:.4f} val_rmse={val_rmse:.4f} val_R2={val_r2:.4f}"
            )

        if wait >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        torch.save({"model_state_dict": model.state_dict(), "feature_dim": feature_dim}, save_path)
    return model


def run_10fold_cv(
    train_graphs,
    device,
    feature_dim=92,
    num_layers=4,
    delta=1,
    equation=10,
    graph_pooling_type="average",
    neighbor_pooling_type="sum",
    lr=1e-3,
    epochs=200,
    batch_size=64,
    patience=10,
    regressor_hidden=64,
    dropout=0.1,
    seed=42,
):
    """
    10-fold CV on training set. Returns (results, best_fold_ix) where best_fold_ix is the fold with
    MINIMUM validation RMSE (Ahmad et al.: select best fold by RMSE, then retrain on full data).
    """
    kf = KFold(n_splits=10, shuffle=True, random_state=seed)
    results = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_graphs)):
        tr = [train_graphs[i] for i in train_idx]
        val = [train_graphs[i] for i in val_idx]
        torch.manual_seed(seed + fold)
        np.random.seed(seed + fold)
        model = train_gvfa_attention(
            tr, val, device,
            feature_dim=feature_dim,
            num_layers=num_layers,
            delta=delta,
            equation=equation,
            graph_pooling_type=graph_pooling_type,
            neighbor_pooling_type=neighbor_pooling_type,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            regressor_hidden=regressor_hidden,
            dropout=dropout,
            save_path=None,
            verbose=True,
        )
        def _mk_batches(gs):
            return list(_eval_batches(gs, batch_size=batch_size))
        train_metrics = evaluate(model, _mk_batches(tr), device)
        val_metrics = evaluate(model, _mk_batches(val), device)
        results.append((train_metrics, val_metrics))
        print(f"Fold {fold+1} Val RMSE={val_metrics['rmse']:.4f} Val R2_COD={val_metrics['r2_cod']:.4f}")
    # Select fold with minimum (best) validation RMSE
    val_rmses = [r[1]["rmse"] for r in results]
    best_fold_ix = int(np.argmin(val_rmses))
    print(f"Best fold by Val RMSE: fold {best_fold_ix + 1} (RMSE={val_rmses[best_fold_ix]:.4f})")
    return results, best_fold_ix


def print_independent_test_summary(y_true, y_pred, expected_test_range=(-6.52, -2.36)):
    """
    Print y_test and y_pred summary stats for debug. Compare y_test to paper expected range (Fig. 1b).
    All in original logS units.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    print("\n--- Independent test set summary (Ahmad et al. Fig. 1b) ---")
    print(f"  y_test:  min={y_true.min():.4f} max={y_true.max():.4f} mean={y_true.mean():.4f} std={y_true.std():.4f}")
    print(f"  y_pred:  min={y_pred.min():.4f} max={y_pred.max():.4f} mean={y_pred.mean():.4f} std={y_pred.std():.4f}")
    print(f"  Paper expected test logS range: [{expected_test_range[0]}, {expected_test_range[1]}]")
    if y_true.min() >= expected_test_range[0] - 0.5 and y_true.max() <= expected_test_range[1] + 0.5:
        print("  [OK] y_test range matches paper (62 anticancer compounds).")
    else:
        print("  [CHECK] y_test range may differ from paper; confirm independent test set is 62 compounds.")


def _eval_batches(graphs, batch_size=64):
    """Yield (batch_graphs, labels) for evaluation; labels on CPU (evaluate() moves to device)."""
    for start in range(0, len(graphs), batch_size):
        batch_graphs = graphs[start : start + batch_size]
        labels = torch.tensor(
            [float(torch.as_tensor(g.label).item()) for g in batch_graphs],
            dtype=torch.float32,
        ).unsqueeze(1)
        yield batch_graphs, labels


def visualize_attention(model, batch_graphs, device, molecule_index=0):
    """
    For a given molecule in the batch, print the attention weight per layer (hop depth).
    This shows which message-passing depth the model focuses on for that molecule —
    useful for explainability (e.g. small molecules may focus on shallow layers,
    large/aromatic may focus on deeper layers).
    """
    model.eval()
    with torch.no_grad():
        pred, (attn_weights, _) = model(batch_graphs, return_attention_weights=True)
    attn_weights = attn_weights.cpu().numpy()  # [B, num_layers]
    w = attn_weights[molecule_index]
    print("Attention weight per layer (hop depth):")
    for layer_idx, weight in enumerate(w):
        print(f"  Layer {layer_idx}: {weight:.4f}")
    return w


def run_ahmad_protocol(
    train_graphs,
    test_graphs,
    device,
    feature_dim=92,
    num_layers=4,
    delta=1,
    equation=10,
    graph_pooling_type="average",
    neighbor_pooling_type="sum",
    lr=1e-3,
    epochs=200,
    batch_size=64,
    patience=10,
    regressor_hidden=64,
    dropout=0.1,
    seed=42,
    expected_test_range=(-6.52, -2.36),
):
    """
    Ahmad et al. (ACS Omega 2023) full protocol:
    1) 10-fold CV on training set (e.g. 9,943 Cui compounds); select fold with MINIMUM Val RMSE.
    2) Retrain that configuration on the full training set (90% train / 10% val for early stopping).
    3) Evaluate on independent test set (62 anticancer compounds). Print RMSE, r2_cod, pearson_r2
       and y_test/y_pred summary stats (same arrays for all metrics; original logS units).
    """
    from sklearn.model_selection import train_test_split
    # Sanity: training logS range (paper Fig. 1a: -18.21 to 1.7)
    train_logs = np.array([float(torch.as_tensor(g.label).item()) for g in train_graphs])
    print(f"Training logS range: [{train_logs.min():.2f}, {train_logs.max():.2f}] (paper: -18.21 to 1.7)")
    print("Step 1: 10-fold CV on training set; select best fold by minimum Val RMSE.")
    results, best_fold_ix = run_10fold_cv(
        train_graphs,
        device,
        feature_dim=feature_dim,
        num_layers=num_layers,
        delta=delta,
        equation=equation,
        graph_pooling_type=graph_pooling_type,
        neighbor_pooling_type=neighbor_pooling_type,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        regressor_hidden=regressor_hidden,
        dropout=dropout,
        seed=seed,
    )
    print(f"\nStep 2: Retrain selected configuration on full training set ({len(train_graphs)} compounds).")
    # Use 90% train / 10% val for early stopping (configuration from best fold; no test data used)
    tr_idx, val_idx = train_test_split(
        np.arange(len(train_graphs)), test_size=0.1, random_state=seed, shuffle=True
    )
    tr_graphs = [train_graphs[i] for i in tr_idx]
    val_graphs = [train_graphs[i] for i in val_idx]
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = train_gvfa_attention(
        tr_graphs,
        val_graphs,
        device,
        feature_dim=feature_dim,
        num_layers=num_layers,
        delta=delta,
        equation=equation,
        graph_pooling_type=graph_pooling_type,
        neighbor_pooling_type=neighbor_pooling_type,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        regressor_hidden=regressor_hidden,
        dropout=dropout,
        save_path=None,
        verbose=True,
    )
    print(f"\nStep 3: Evaluate on independent test set ({len(test_graphs)} compounds).")
    test_batches = list(_eval_batches(test_graphs, batch_size=batch_size))
    test_metrics = evaluate(model, test_batches, device, return_arrays=True)
    y_true = test_metrics["y_true"]
    y_pred = test_metrics["y_pred"]
    # All metrics on the SAME arrays (original logS units)
    print("\n--- Ahmad et al. metrics on 62-compound independent test (Table 3 style) ---")
    print(f"  RMSE (Eq. 18):     {test_metrics['rmse']:.4f}")
    print(f"  R2 Eq. 17 (COD):   {test_metrics['r2_cod']:.4f}   (1 - SSE/SST)")
    print(f"  Pearson R²:         {test_metrics['pearson_r2']:.4f}   (corr(y,yhat)^2)")
    print(f"  (Paper AttentiveFP: RMSE=0.61, R2=0.52)")
    print_independent_test_summary(y_true, y_pred, expected_test_range=expected_test_range)
    return model, test_metrics


# -----------------------------------------------------------------------------
# Main: default solubility_1.csv (90/10 split); optional Ahmad protocol (9943 + 62)
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="GVFA with cross-layer attention for logS (Ahmad et al. ACS Omega 2023 protocol)"
    )
    parser.add_argument("--train_csv", type=str, default="final_data/solubility_1.csv",
                        help="Training CSV (Cui 9,943 for Ahmad protocol); or single file for 90/10 split")
    parser.add_argument("--test_csv", type=str, default="",
                        help="Test CSV; if empty, split train_csv 90/10. For Ahmad: 62 anticancer compounds.")
    parser.add_argument("--protocol", type=str, default="single", choices=["single", "ahmad"],
                        help="single: one CSV 90/10 split. ahmad: 10-fold CV on train, best fold, retrain on full train, eval on 62 test.")
    parser.add_argument("--smiles_col", type=str, default="SMILES")
    parser.add_argument("--target_col", type=str, default="logS")
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--delta", type=int, default=1)
    parser.add_argument("--equation", type=int, default=10)
    parser.add_argument("--graph_pooling_type", type=str, default="average")
    parser.add_argument("--neighbor_pooling_type", type=str, default="sum")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--regressor_hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--cv", type=int, default=0, help="If 10, run 10-fold CV on train only (no retrain)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_dim = 92

    use_single_csv = not (args.test_csv and args.test_csv.strip())
    if args.protocol == "ahmad":
        if use_single_csv or not os.path.isfile(args.test_csv):
            raise SystemExit("Ahmad protocol requires --test_csv pointing to 62 anticancer compounds.")
        if not os.path.isfile(args.train_csv):
            raise SystemExit("Ahmad protocol requires --train_csv (Cui 9,943 compounds).")
        train_graphs, test_graphs = load_cui_style_data(
            args.train_csv, args.test_csv,
            smiles_col=args.smiles_col,
            target_col=args.target_col,
        )
        print(f"Train graphs: {len(train_graphs)} (Cui), Test graphs: {len(test_graphs)} (62 independent)")
        run_ahmad_protocol(
            train_graphs,
            test_graphs,
            device,
            feature_dim=feature_dim,
            num_layers=args.num_layers,
            delta=args.delta,
            equation=args.equation,
            graph_pooling_type=args.graph_pooling_type,
            neighbor_pooling_type=args.neighbor_pooling_type,
            lr=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            regressor_hidden=args.regressor_hidden,
            dropout=args.dropout,
            seed=args.seed,
        )
        return

    if not os.path.isfile(args.train_csv):
        print("Train CSV not found. Creating dummy data for structure test.")
        train_graphs, test_graphs = [], []
        for smi, log_s in [("CCO", -0.77), ("c1ccccc1", -1.52), ("CC(=O)O", -0.17)]:
            g = smiles_to_mol_graph(smi, log_s)
            if g is not None:
                train_graphs.append(g)
        for smi, log_s in [("CC(C)C", -0.5)]:
            g = smiles_to_mol_graph(smi, log_s)
            if g is not None:
                test_graphs.append(g)
        if not train_graphs or not test_graphs:
            raise FileNotFoundError("Provide --train_csv (e.g. final_data/solubility_1.csv).")
    elif use_single_csv:
        # solubility_1.csv: single file with 90/10 train/test split
        train_graphs, test_graphs = load_solubility1_data(
            args.train_csv,
            smiles_col=args.smiles_col,
            target_col=args.target_col,
            test_size=0.1,
            seed=args.seed,
        )
    else:
        train_graphs, test_graphs = load_cui_style_data(
            args.train_csv, args.test_csv,
            smiles_col=args.smiles_col,
            target_col=args.target_col,
        )
    print(f"Train graphs: {len(train_graphs)}, Test graphs: {len(test_graphs)}")

    if args.cv == 10:
        results, best_fold_ix = run_10fold_cv(
            train_graphs,
            device,
            feature_dim=feature_dim,
            num_layers=args.num_layers,
            delta=args.delta,
            equation=args.equation,
            graph_pooling_type=args.graph_pooling_type,
            neighbor_pooling_type=args.neighbor_pooling_type,
            lr=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            regressor_hidden=args.regressor_hidden,
            dropout=args.dropout,
            seed=args.seed,
        )
        mean_val_rmse = np.mean([r[1]["rmse"] for r in results])
        mean_val_r2 = np.mean([r[1]["r2_cod"] for r in results])
        print(f"10-fold CV: mean Val RMSE={mean_val_rmse:.4f}, mean Val R2_COD={mean_val_r2:.4f}")
        return

    # Single train/val split then train and evaluate on test
    from sklearn.model_selection import train_test_split
    indices = np.arange(len(train_graphs))
    tr_idx, val_idx = train_test_split(indices, test_size=0.1, random_state=args.seed, shuffle=True)
    tr_graphs = [train_graphs[i] for i in tr_idx]
    val_graphs = [train_graphs[i] for i in val_idx]

    save_path = os.path.join("checkpoints", "gvfa_attention_best.pt")
    model = train_gvfa_attention(
        tr_graphs,
        val_graphs,
        device,
        feature_dim=feature_dim,
        num_layers=args.num_layers,
        delta=args.delta,
        equation=args.equation,
        graph_pooling_type=args.graph_pooling_type,
        neighbor_pooling_type=args.neighbor_pooling_type,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        regressor_hidden=args.regressor_hidden,
        dropout=args.dropout,
        save_path=save_path,
        verbose=True,
    )

    test_batches = list(_eval_batches(test_graphs))
    test_metrics = evaluate(model, test_batches, device, return_arrays=True)
    print("\n--- Test metrics (same y_true, y_pred; original logS units) ---")
    print(f"  RMSE (Eq. 18):   {test_metrics['rmse']:.4f}")
    print(f"  R2 Eq. 17 (COD): {test_metrics['r2_cod']:.4f}")
    print(f"  Pearson R²:       {test_metrics['pearson_r2']:.4f}")
    if "y_true" in test_metrics and "y_pred" in test_metrics:
        print_independent_test_summary(test_metrics["y_true"], test_metrics["y_pred"])

    # Visualize attention for first test molecule
    if test_graphs:
        visualize_attention(model, test_graphs[:1], device, molecule_index=0)


if __name__ == "__main__":
    main()
