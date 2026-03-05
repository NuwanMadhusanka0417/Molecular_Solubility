"""
Role-Separated Context Encoding for molecular graphs.

Pure VSA architecture — zero trainable parameters.

Pipeline:
  1. Base Encoding:    L2-normalize node features and edge features
  2. Context Building: K rounds of edge-conditioned message passing (continuous, no sign())
  3. Role Combination: bind each hop-context with a unique role vector, weighted sum
  4. Graph Pooling:    degree-weighted bundling + Sigma-Pi expansion
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import fft, ifft
import math


class GraphCNN(nn.Module):

    def __init__(self, input_dim, num_layers, delta, graph_pooling_type,
                 neighbor_pooling_type, device, equation, edge_feat_dim=5,
                 edge_projection_type="orthogonal", use_reservoir=False,
                 reservoir_iters=7, reservoir_alpha=0.8,
                 reservoir_polynomial_order=2, reservoir_history_weight=0.75,
                 use_resonator=False, resonator_iters=7, resonator_beta=0.75,
                 hop_decay=0.85, sigma_pi_orders=None):
        super(GraphCNN, self).__init__()
        print("Input feature size: ", input_dim)
        self.device = device
        self.num_layers = num_layers
        self.edge_feat_dim = edge_feat_dim if edge_feat_dim else 0
        self.hop_decay = hop_decay
        self.sigma_pi_orders = sigma_pi_orders if sigma_pi_orders is not None else [0, 1]

        # Edge projection (fixed random, seeded) — same generation logic as before
        if self.edge_feat_dim > 0:
            g = torch.Generator().manual_seed(0)
            W_edge = torch.randn(self.edge_feat_dim, input_dim, generator=g)
            if edge_projection_type == "orthogonal" and input_dim >= self.edge_feat_dim:
                A = torch.randn(input_dim, self.edge_feat_dim, generator=g)
                Q, _ = torch.linalg.qr(A)
                W_edge = Q[:, :self.edge_feat_dim].T
            else:
                W_edge = W_edge / math.sqrt(self.edge_feat_dim)
            self.register_buffer("W_edge", W_edge)

        # Role vectors for hop-specific context tagging (fixed random unit-norm, seeded)
        roles = []
        for k in range(num_layers):
            rg = torch.Generator().manual_seed(k + 100)
            r = torch.randn(input_dim, generator=rg)
            r = r / r.norm()
            roles.append(r)
        self.register_buffer("roles", torch.stack(roles))  # [num_layers, D]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _preprocess_edges(self, batch_graph):
        """Batch edge_index and edge_attr across graphs with node-offset alignment."""
        start_idx = [0]
        for i, g in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(g.g))
        ei_list, ea_list = [], []
        for i, g in enumerate(batch_graph):
            ei = getattr(g, "edge_index", None)
            ea = getattr(g, "edge_attr", None)
            if ei is None or ea is None or ei.numel() == 0 or ea.numel() == 0:
                continue
            ei_list.append(ei.to(self.device) + start_idx[i])
            ea_list.append(ea.to(self.device))
        if not ei_list:
            return None, None
        return torch.cat(ei_list, dim=1), torch.cat(ea_list, dim=0)

    def bind(self, x, y):
        """Circular convolution via FFT — core VSA binding operation."""
        fft_x = fft(x, dim=1)
        fft_y = fft(y, dim=1)
        return torch.real(ifft(fft_x * fft_y, dim=1))

    def _pi(self, x, shift=None):
        """Sigma-Pi permutation: cyclic roll by D/3."""
        s = shift if shift is not None else max(1, x.shape[1] // 3)
        return torch.roll(x, shifts=int(s), dims=1)

    def sigma_pi_expansion(self, F1, eps=1e-8):
        """Sigma-Pi polynomial expansion over self.sigma_pi_orders."""
        D = F1.shape[1]
        result = torch.zeros_like(F1)
        ast_prev = F1
        for t in sorted(self.sigma_pi_orders):
            if t == 0:
                ast_t = F1
            else:
                ast_t = self.bind(self._pi(ast_prev, shift=max(1, D // 3)), F1)
                ast_t = F.normalize(ast_t, p=2, dim=1, eps=eps)
            ast_prev = ast_t
            result = result + ast_t
        return F.normalize(result, p=2, dim=1, eps=eps)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, batch_graph, return_embedding=False, return_node_rep=False):
        """
        Role-Separated Context Encoding.

        Returns
        -------
        return_node_rep=True  : (H [N, D], batch [N])  — for attention readout
        otherwise             : g [1, B, D]            — graph-level embeddings
        """
        start_idx = [0]
        for g in batch_graph:
            start_idx.append(start_idx[-1] + len(g.g))
        B = len(batch_graph)
        N = start_idx[-1]

        # --- Stage 1: Base Encoding ---
        X_concat = torch.cat(
            [g.node_features for g in batch_graph], 0
        ).to(self.device)
        feat = F.normalize(X_concat, p=2, dim=1, eps=1e-8)
        D = feat.shape[1]

        batched_ei, batched_ea = self._preprocess_edges(batch_graph)
        edge_index, edge_H = None, None
        if (batched_ei is not None and batched_ea is not None
                and self.edge_feat_dim > 0 and hasattr(self, "W_edge")):
            edge_index = batched_ei
            edge_H = F.normalize(
                torch.mm(batched_ea.to(feat.dtype), self.W_edge),
                p=2, dim=1, eps=1e-8,
            )

        # --- Stage 2: Context Building ---
        # Each context_k captures pure k-hop neighbourhood info.
        # No bind(self, messages), no sign(), no residual mixing.
        contexts = [feat]
        prev_ctx = feat
        K = self.num_layers - 1

        if edge_index is not None and edge_H is not None:
            src, dst = edge_index[0], edge_index[1]
            for _ in range(K):
                messages = self.bind(edge_H, prev_ctx[src])
                agg = torch.zeros(N, D, device=feat.device, dtype=feat.dtype)
                agg.index_add_(0, dst, messages)
                ctx_k = F.normalize(agg, p=2, dim=1, eps=1e-8)
                contexts.append(ctx_k)
                prev_ctx = ctx_k

        # --- Stage 3: Role-Separated Combination ---
        # bind(role_k, ctx_k) places each hop in an orthogonal subspace.
        h = torch.zeros(N, D, device=feat.device, dtype=feat.dtype)
        for k, ctx_k in enumerate(contexts):
            weight = self.hop_decay ** k
            ctx_k_enriched = self.sigma_pi_expansion(ctx_k)
            role_k = self.roles[k].unsqueeze(0).expand(N, -1)
            h = h + weight * self.bind(role_k, ctx_k_enriched)
        h = F.normalize(h, p=2, dim=1, eps=1e-8)

        if return_node_rep:
            batch_vec = torch.zeros(N, dtype=torch.long, device=h.device)
            for b in range(B):
                batch_vec[start_idx[b]:start_idx[b + 1]] = b
            return (h, batch_vec)

        # --- Stage 4: Multi-stat pooling (mean | max | variance) ---
        g_mean = torch.zeros(B, D, device=h.device, dtype=h.dtype)
        g_max = torch.full((B, D), float('-inf'), device=h.device, dtype=h.dtype)

        for i in range(B):
            lo, hi = start_idx[i], start_idx[i + 1]
            atoms = h[lo:hi]
            g_mean[i] = atoms.mean(dim=0)
            g_max[i] = atoms.max(dim=0).values

        g_var = torch.zeros(B, D, device=h.device, dtype=h.dtype)
        for i in range(B):
            lo, hi = start_idx[i], start_idx[i + 1]
            atoms = h[lo:hi]
            g_var[i] = (atoms ** 2).mean(dim=0) - g_mean[i] ** 2

        g = torch.cat([g_mean, g_max, g_var], dim=1)  # [B, 3D]
        return g.unsqueeze(0)                          # [1, B, 3D]
