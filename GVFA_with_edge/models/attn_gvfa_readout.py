"""
Attention-based readout for GVFA: learned attention pooling over node hypervectors
plus MLP regressor. GVFA encoder is frozen; only readout + MLP are trained.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.utils import softmax as pyg_softmax
except ImportError:
    pyg_softmax = None


def _grouped_softmax(logits, batch, dim=0):
    """Grouped softmax: normalize logits per graph. logits [N], batch [N]."""
    if pyg_softmax is not None:
        return pyg_softmax(logits.unsqueeze(1), batch, dim=dim).squeeze(1)
    # Fallback: manual grouped softmax (subtract max per group for stability)
    B = batch.max().item() + 1
    max_per_graph = torch.full((B,), float('-inf'), device=logits.device, dtype=logits.dtype)
    for b in range(B):
        mask = batch == b
        if mask.any():
            max_per_graph[b] = logits[mask].max()
    logits_stable = logits - max_per_graph[batch]
    exp = torch.exp(logits_stable.clamp(max=50.0))
    sum_per_graph = torch.zeros(B, device=logits.device, dtype=logits.dtype)
    for b in range(B):
        mask = batch == b
        if mask.any():
            sum_per_graph[b] = exp[mask].sum()
    return exp / (sum_per_graph[batch] + 1e-8)


class AttnReadoutPool(nn.Module):
    """
    Learned attention pooling over node hypervectors.
    Input: H [N, D], batch [N] (graph id per node).
    Output: g [B, D] (one embedding per graph).
    """
    def __init__(self, D, hidden_dim=None, use_layernorm=False, num_heads=1):
        super().__init__()
        self.D = D
        self.num_heads = num_heads
        hidden_dim = hidden_dim or max(D // 2, 32)
        self.use_layernorm = use_layernorm
        if use_layernorm:
            self.ln = nn.LayerNorm(D)

        if num_heads <= 1:
            # s_i = MLP(H_i) -> scalar: Linear(D, hidden) -> tanh -> Linear(hidden, 1)
            self.W = nn.Linear(D, hidden_dim)
            self.v = nn.Linear(hidden_dim, 1)
        else:
            # Multi-head: K heads, each produces g_k [B, D], concat -> [B, K*D], project to D
            self.head_W = nn.ModuleList([nn.Linear(D, hidden_dim) for _ in range(num_heads)])
            self.head_v = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(num_heads)])
            self.head_proj = nn.Linear(num_heads * D, D)

    def forward(self, H, batch):
        """
        H: [N, D], batch: [N] long, values in [0, B-1]
        Returns: g [B, D]
        """
        if self.use_layernorm and hasattr(self, 'ln'):
            H = self.ln(H)
        N, D = H.shape
        B = batch.max().item() + 1

        if self.num_heads <= 1:
            # logits: [N, 1] -> [N]
            h = torch.tanh(self.W(H))
            logits = self.v(h).squeeze(-1)
            alpha = _grouped_softmax(logits, batch)
            weighted = alpha.unsqueeze(1) * H
            g = torch.zeros(B, D, device=H.device, dtype=H.dtype)
            g.index_add_(0, batch, weighted)
            return g

        # Multi-head
        head_outs = []
        for k in range(self.num_heads):
            h = torch.tanh(self.head_W[k](H))
            logits = self.head_v[k](h).squeeze(-1)
            alpha = _grouped_softmax(logits, batch)
            weighted = alpha.unsqueeze(1) * H
            gk = torch.zeros(B, D, device=H.device, dtype=H.dtype)
            gk.index_add_(0, batch, weighted)
            head_outs.append(gk)
        g_cat = torch.cat(head_outs, dim=1)
        return self.head_proj(g_cat)


class RegressorHead(nn.Module):
    """MLP regressor. Supports 1-layer (legacy) or deeper architecture via hidden_dims list."""
    def __init__(self, D, hidden_dim=64, dropout=0.2, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [hidden_dim]
        layers = []
        in_d = D
        for hd in hidden_dims:
            layers.append(nn.Linear(in_d, hd))
            layers.append(nn.BatchNorm1d(hd))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_d = hd
        layers.append(nn.Linear(in_d, 1))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, g):
        return self.net(g)


class AttnGVFARegressor(nn.Module):
    """
    GVFA encoder (frozen) -> node H [N,D] -> AttnReadoutPool -> g [B,D] -> MLP -> yhat [B,1].
    Only readout + MLP parameters are trained.
    """
    def __init__(self, gvfa_encoder, D, readout_hidden=None, regressor_hidden=64, dropout=0.2,
                 use_layernorm=False, num_heads=1, regressor_hidden_dims=None):
        super().__init__()
        self.encoder = gvfa_encoder
        for p in self.encoder.parameters():
            p.requires_grad = False
        for buf in self.encoder.buffers():
            buf.requires_grad = False

        self.attn_pool = AttnReadoutPool(D, hidden_dim=readout_hidden, use_layernorm=use_layernorm, num_heads=num_heads)
        self.regressor = RegressorHead(D, hidden_dim=regressor_hidden, dropout=dropout,
                                       hidden_dims=regressor_hidden_dims)

    def forward(self, batch_graph, return_embedding=False):
        """
        batch_graph: list of S2VGraph (same interface as GraphCNN).
        Returns: yhat [B, 1]. If return_embedding, also return (g, yhat).
        """
        with torch.no_grad():
            H, batch = self.encoder(batch_graph, return_node_rep=True)
        g = self.attn_pool(H, batch)
        yhat = self.regressor(g)
        if return_embedding:
            return g, yhat
        return yhat

    def get_trainable_parameters(self):
        return list(self.attn_pool.parameters()) + list(self.regressor.parameters())


def build_attn_gvfa_regressor(encoder, D, readout_hidden=None, regressor_hidden=64, dropout=0.2,
                              use_layernorm=True, num_heads=1, regressor_hidden_dims=None):
    """Factory: encoder is already built (e.g. GraphCNN). D = node HV dimension from encoder."""
    return AttnGVFARegressor(
        encoder, D,
        readout_hidden=readout_hidden,
        regressor_hidden=regressor_hidden,
        dropout=dropout,
        use_layernorm=use_layernorm,
        num_heads=num_heads,
        regressor_hidden_dims=regressor_hidden_dims,
    )
