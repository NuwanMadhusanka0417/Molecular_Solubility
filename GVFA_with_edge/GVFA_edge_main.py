'''
Graph-based VSA model with edge-conditioned message passing.
Node features: 7 atom features. Edge features: bond type, conjugated, ring, length.
Use "sol" python environment.

Models:
  gvfa_ridge (default): GVFA encoder + Ridge/XGBoost readout.
  attn_gvfa: GVFA encoder (frozen) + learned attention readout + MLP regressor (trained).
'''
import argparse
import os

from src.create_graphs import create_graph_list
from src.load_data import load_data
from src.VSA_conversion import VSA_conversion
from src.embeddings import getEmbedding
from models.graphcnnVSA_Binding_FULL import GraphCNN
from models.attn_gvfa_readout import AttnGVFARegressor

import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def parse_args():
    p = argparse.ArgumentParser(description='GVFA for molecular solubility')
    p.add_argument('--model', type=str, default='gvfa_ridge', choices=['gvfa_ridge', 'attn_gvfa'],
                   help='gvfa_ridge: GVFA + Ridge/XGBoost; attn_gvfa: GVFA + attention readout + MLP (trained)')
    p.add_argument('--dataset', type=str, default='old', choices=['old', 'new'], help='Dataset: old or new')
    p.add_argument('--dim', type=int, default=1000, help='VSA dimension (and single dim when --model attn_gvfa)')
    p.add_argument('--dims', type=str, default='1000,2000,5000,10000', help='Comma-separated dims for gvfa_ridge loop')
    p.add_argument('--epochs', type=int, default=80, help='Epochs for attn_gvfa')
    p.add_argument('--batch_size', type=int, default=64, help='Batch size for attn_gvfa')
    p.add_argument('--lr', type=float, default=1e-3, help='Learning rate for attn_gvfa')
    p.add_argument('--regressor_hidden', type=int, default=64, help='MLP hidden size for attn_gvfa')
    p.add_argument('--dropout', type=float, default=0.2, help='Dropout in regressor')
    p.add_argument('--save_dir', type=str, default='checkpoints', help='Where to save best attn_gvfa checkpoint')
    p.add_argument('--use_ridge', action='store_true', default=True, help='Use Ridge (else XGBoost) for gvfa_ridge')
    p.add_argument('--no_ridge', action='store_false', dest='use_ridge')
    p.add_argument('--attn_heads', type=int, default=1, help='Multi-head attention (1 = single head)')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def run_gvfa_ridge(args, train_data, test_data, device):
    """Existing pipeline: GVFA encoder -> embeddings -> Ridge or XGBoost."""
    dims = [int(x) for x in args.dims.split(',')]
    for dim in dims:
        train_graphs = create_graph_list(train_data)
        test_graphs = create_graph_list(test_data)
        ts_graph = test_graphs.copy()
        tr_graph = train_graphs.copy()
        test_HVs = VSA_conversion(ts_graph, dim, projection_type="orthogonal")
        train_HVs = VSA_conversion(tr_graph, dim, projection_type="orthogonal")

        model_eq1 = GraphCNN(
            test_HVs[0].node_features.shape[1], 5, 1, 'sum', 'sum', device, 10,
            edge_feat_dim=5, edge_projection_type="orthogonal",
            use_reservoir=True, hop_decay=0.85, sigma_pi_orders=[0, 1],
        )
        train_emb, train_labels = getEmbedding(model_eq1, device, train_HVs, use_size_aware=True, hop_alpha=1.0)
        test_emb, test_labels = getEmbedding(model_eq1, device, test_HVs, use_size_aware=True, hop_alpha=1.0)
        train_emb = train_emb.squeeze(0)
        test_emb = test_emb.squeeze(0)

        if args.use_ridge:
            reg = RidgeCV(alphas=np.logspace(-4, 2, 50), cv=5, scoring='neg_mean_squared_error')
            reg.fit(train_emb, train_labels)
            pred = reg.predict(test_emb)
        else:
            from xgboost import XGBRegressor
            reg = XGBRegressor(
                n_estimators=2000, learning_rate=0.03, max_depth=7,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, reg_alpha=0.0,
                random_state=42, n_jobs=4, tree_method="hist"
            )
            reg.fit(train_emb, train_labels, eval_set=[(test_emb, test_labels)], verbose=False)
            pred = reg.predict(test_emb)

        rmse = np.sqrt(mean_squared_error(test_labels, pred))
        mae = mean_absolute_error(test_labels, pred)
        r2 = r2_score(test_labels, pred)
        print(f"Dimention,{dim},MAE,{mae},RMSE,{rmse},R2,{r2}")


def run_attn_gvfa(args, train_data, test_data, device):
    """Attention-based GVFA: frozen encoder + trainable attention readout + MLP."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_graphs = create_graph_list(train_data)
    test_graphs = create_graph_list(test_data)
    train_HVs = VSA_conversion(train_graphs, args.dim, projection_type="orthogonal")
    test_HVs = VSA_conversion(test_graphs, args.dim, projection_type="orthogonal")

    D = train_HVs[0].node_features.shape[1]
    encoder = GraphCNN(
        D, 5, 1, 'sum', 'sum', device, 10,
        edge_feat_dim=5, edge_projection_type="orthogonal",
        use_reservoir=True, hop_decay=0.85, sigma_pi_orders=[0, 1],
    )
    encoder.to(device)

    # When use_reservoir, node-level F_v has shape [N, D] with D = args.dim
    model = AttnGVFARegressor(
        encoder, args.dim,
        readout_hidden=max(args.dim // 2, 32),
        regressor_hidden=args.regressor_hidden,
        dropout=args.dropout,
        use_layernorm=True,
        num_heads=args.attn_heads,
    )
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.get_trainable_parameters())
    print(f"Total parameters: {total_params}, Trainable (readout+MLP): {trainable_params}")

    # Sanity: forward on small batch
    batch_graphs = train_HVs[: min(4, len(train_HVs))]
    with torch.no_grad():
        H, batch = encoder(batch_graphs, return_node_rep=True)
    g, yhat = model(batch_graphs, return_embedding=True)
    B_batch = len(batch_graphs)
    assert H.shape[1] == args.dim, f"H shape {H.shape}"
    assert g.shape == (B_batch, args.dim), f"g shape {g.shape}"
    assert yhat.shape == (B_batch, 1), f"yhat shape {yhat.shape}"
    print("Sanity shapes OK: H [N,D], g [B,D], yhat [B,1]")

    optimizer = torch.optim.Adam(model.get_trainable_parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    os.makedirs(args.save_dir, exist_ok=True)
    best_rmse = float('inf')
    best_path = os.path.join(args.save_dir, f"attn_gvfa_dim{args.dim}_best.pt")

    n_train = len(train_HVs)
    for epoch in range(args.epochs):
        model.train()
        perm = np.random.permutation(n_train)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_train, args.batch_size):
            idx = perm[start : start + args.batch_size]
            batch_graphs = [train_HVs[i] for i in idx]
            labels = torch.tensor(
                [float(torch.as_tensor(train_HVs[i].label).item()) for i in idx],
                dtype=torch.float32, device=device
            ).unsqueeze(1)
            optimizer.zero_grad()
            yhat = model(batch_graphs)
            loss = criterion(yhat, labels)
            loss.backward()
            if epoch == 0 and start == 0:
                for name, p in model.named_parameters():
                    if p.requires_grad and p.grad is None:
                        raise RuntimeError(f"Trainable param {name} has no grad")
                encoder_has_grad = any(p.grad is not None for p in model.encoder.parameters())
                if encoder_has_grad:
                    raise RuntimeError("Encoder should have no gradients (frozen)")
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        train_rmse = np.sqrt(epoch_loss / n_batches) if n_batches else 0

        # Eval on test set (held-out)
        model.eval()
        test_preds = []
        test_labels_list = []
        with torch.no_grad():
            for start in range(0, len(test_HVs), args.batch_size):
                batch_graphs = test_HVs[start : start + args.batch_size]
                yhat = model(batch_graphs)
                test_preds.append(yhat.cpu().numpy())
                test_labels_list.append(
                    torch.tensor([float(torch.as_tensor(g.label).item()) for g in batch_graphs], dtype=torch.float32)
                )
        test_pred = np.concatenate(test_preds, axis=0).ravel()
        test_y = torch.cat(test_labels_list).numpy()
        test_rmse = np.sqrt(mean_squared_error(test_y, test_pred))
        test_mae = mean_absolute_error(test_y, test_pred)
        test_r2 = r2_score(test_y, test_pred)

        if test_rmse < best_rmse:
            best_rmse = test_rmse
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_rmse': test_rmse, 'test_mae': test_mae, 'test_r2': test_r2,
            }, best_path)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{args.epochs} train_rmse={train_rmse:.4f} test_rmse={test_rmse:.4f} test_mae={test_mae:.4f} test_r2={test_r2:.4f}")

    # Load best and report final metrics
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    test_preds = []
    test_labels_list = []
    with torch.no_grad():
        for start in range(0, len(test_HVs), args.batch_size):
            batch_graphs = test_HVs[start : start + args.batch_size]
            yhat = model(batch_graphs)
            test_preds.append(yhat.cpu().numpy())
            test_labels_list.append(
                torch.tensor([float(torch.as_tensor(g.label).item()) for g in batch_graphs], dtype=torch.float32)
            )
    test_pred = np.concatenate(test_preds, axis=0).ravel()
    test_y = torch.cat(test_labels_list).numpy()
    rmse = np.sqrt(mean_squared_error(test_y, test_pred))
    mae = mean_absolute_error(test_y, test_pred)
    r2 = r2_score(test_y, test_pred)
    print(f"Best model (saved to {best_path}): MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_data, test_data = load_data(dataset=args.dataset)

    if args.model == 'attn_gvfa':
        run_attn_gvfa(args, train_data, test_data, device)
    else:
        run_gvfa_ridge(args, train_data, test_data, device)


if __name__ == '__main__':
    main()
