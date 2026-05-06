import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import fft, ifft
import math
import sys
sys.path.append("models/")
from models.mlp import MLP

# Edge attr [E, 5]: bond_type, conjugated, in_ring, bond_length, stereo — map to ~[-1,+1] before projection
EDGE_BINARY_COLS = [1, 2]
EDGE_MINMAX_COLS = [0, 3, 4]

class GraphCNN(nn.Module):
    def __init__(self, input_dim, num_layers, delta, graph_pooling_type, neighbor_pooling_type, device, equation, edge_feat_dim=5, edge_projection_type="orthogonal", use_reservoir=False, reservoir_iters=7, reservoir_alpha=0.8, reservoir_polynomial_order=2, reservoir_history_weight=0.75, use_resonator=False, resonator_iters=7, resonator_beta=0.75, hop_decay=0.25, sigma_pi_orders=None, rng_seed=0, binding_type="circular"):
        '''
            use_reservoir: VSA-RC (VSA Reservoir Computing): tap buffer + Sigma-Pi polynomial expansion
            reservoir_iters, reservoir_alpha, reservoir_history_weight: unused (kept for compat)
            reservoir_polynomial_order: unused, use sigma_pi_orders instead
            hop_decay: lambda in [0.6, 0.95], decay for far-hop mixing in tap buffer
            sigma_pi_orders: list of orders T, e.g. [0,1] for 1st+2nd order (recommended)
            use_resonator: legacy fallback (deprecated)
            binding_type: "circular" (FFT-based circular convolution, default) or
                          "hadamard" (elementwise multiplication — preserves geometry better)
        '''

        super(GraphCNN, self).__init__()
        print("Input feature size: ", input_dim)
        self.device = device
        self.num_layers = num_layers
        self.graph_pooling_type = graph_pooling_type
        self.neighbor_pooling_type = neighbor_pooling_type
        self.learn_eps = True
        self.delta = delta
        self.equation = equation
        self.edge_feat_dim = edge_feat_dim if edge_feat_dim else 0
        self.use_reservoir = use_reservoir
        self.reservoir_iters = reservoir_iters
        self.reservoir_alpha = reservoir_alpha
        self.reservoir_polynomial_order = reservoir_polynomial_order
        self.reservoir_history_weight = reservoir_history_weight
        self.use_resonator = use_resonator
        self.resonator_iters = resonator_iters
        self.resonator_beta = resonator_beta
        self.hop_decay = hop_decay if use_reservoir else 1.0
        self.sigma_pi_orders = sigma_pi_orders if sigma_pi_orders is not None else [0, 1]
        if binding_type not in ("circular", "hadamard"):
            raise ValueError(f"binding_type must be 'circular' or 'hadamard', got {binding_type!r}")
        self.binding_type = binding_type

        if self.edge_feat_dim > 0:
            g = torch.Generator().manual_seed(rng_seed)
            W_edge = torch.randn(self.edge_feat_dim, input_dim, generator=g)
            if edge_projection_type == "orthogonal" and input_dim >= self.edge_feat_dim:
                # Orthonormal columns: preserves norms of projected edge vectors (info-preserving)
                A = torch.randn(input_dim, self.edge_feat_dim, generator=g)
                Q, _ = torch.linalg.qr(A)
                W_edge = Q[:, :self.edge_feat_dim].T  # (edge_feat_dim, input_dim)
            else:
                W_edge = W_edge / math.sqrt(self.edge_feat_dim)
            self.register_buffer("W_edge", W_edge)

    def set_edge_stats(self, edge_col_min: torch.Tensor, edge_col_range: torch.Tensor):
        """
        Store training-set min/range for EDGE_MINMAX_COLS only; binary cols remapped in forward.

        edge_col_min:   [3] — min of (bond_type, bond_length, stereo) over train edges
        edge_col_range: [3] — max-min, clamped ≥ 1e-6
        """
        cmin = edge_col_min.to(self.device).clone()
        crng = edge_col_range.to(self.device).clone()
        if "edge_col_min" in self._buffers:
            self.edge_col_min = cmin
            self.edge_col_range = crng
        else:
            self.register_buffer("edge_col_min", cmin)
            self.register_buffer("edge_col_range", crng)

    def __preprocess_neighbors_sumavepool(self, batch_graph):
        ###create block diagonal sparse matrix

        edge_mat_list = []
        start_idx = [0]
        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.g))
            edge_mat_list.append(graph.edge_mat + start_idx[i])
        Adj_block_idx = torch.cat(edge_mat_list, 1)
        Adj_block_elem = torch.ones(Adj_block_idx.shape[1])

        #Add self-loops in the adjacency matrix if learn_eps is False, i.e., aggregate center nodes and neighbor nodes altogether.

        if not self.learn_eps:
            num_node = start_idx[-1]
            self_loop_edge = torch.LongTensor([range(num_node), range(num_node)])
            elem = torch.ones(num_node)
            Adj_block_idx = torch.cat([Adj_block_idx, self_loop_edge], 1)
            Adj_block_elem = torch.cat([Adj_block_elem, elem], 0)

        Adj_block = torch.sparse.FloatTensor(Adj_block_idx, Adj_block_elem, torch.Size([start_idx[-1],start_idx[-1]]))

        return Adj_block.to(self.device)
    



    def __preprocess_graphpool(self, batch_graph):
        ###create sum or average pooling sparse matrix over entire nodes in each graph (num graphs x num nodes)
        
        start_idx = [0]

        #compute the padded neighbor list
        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.g))

        idx = []
        elem = []
        for i, graph in enumerate(batch_graph):
            ###average pooling
            if self.graph_pooling_type == "average":
                elem.extend([1./len(graph.g)]*len(graph.g))
            
            else:
            ###sum pooling
                elem.extend([1]*len(graph.g))

            idx.extend([[i, j] for j in range(start_idx[i], start_idx[i+1], 1)])
        elem = torch.FloatTensor(elem)
        idx = torch.LongTensor(idx).transpose(0,1)
        graph_pool = torch.sparse.FloatTensor(idx, elem, torch.Size([len(batch_graph), start_idx[-1]]))
        
        return graph_pool.to(self.device)

    def __preprocess_edges(self, batch_graph):
        """Batched edge_index [2, E_total] and edge_attr [E_total, F_edge], aligned. start_idx for node offsets."""
        start_idx = [0]
        for i, g in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(g.g))
        ei_list, ea_list = [], []
        for i, g in enumerate(batch_graph):
            ei = getattr(g, "edge_index", None)
            ea = getattr(g, "edge_attr", None)
            if ei is None or ea is None or ei.numel() == 0 or ea.numel() == 0:
                continue
            off = start_idx[i]
            ei_list.append(ei.to(self.device) + off)
            ea_list.append(ea.to(self.device))
        if not ei_list:
            return None, None, start_idx
        batched_ei = torch.cat(ei_list, dim=1)
        batched_ea = torch.cat(ea_list, dim=0)
        return batched_ei, batched_ea, start_idx

    def _edge_message_pool(self, h_to_pool, edge_index, edge_H, num_nodes):
        """
        Edge messages: bind(neighbor_h, edge_H), sum at dst, then L2-normalize per node.
        Normalizing after aggregation prevents degree-dependent norm blow-up.
        """
        src, dst = edge_index[0], edge_index[1]
        neighbor_h = h_to_pool[src]
        messages = self.bind(neighbor_h, edge_H)
        D = h_to_pool.shape[1]
        pooled = torch.zeros(num_nodes, D, device=h_to_pool.device, dtype=h_to_pool.dtype)
        pooled.index_add_(0, dst, messages)
        pooled = F.normalize(pooled, p=2, dim=1, eps=1e-8)
        return pooled

    def maxpool(self, h, padded_neighbor_list):
        ###Element-wise minimum will never affect max-pooling

        dummy = torch.min(h, dim = 0)[0]
        h_with_dummy = torch.cat([h, dummy.reshape((1, -1)).to(self.device)])
        pooled_rep = torch.max(h_with_dummy[padded_neighbor_list], dim = 1)[0]
        return pooled_rep
    
    def permutation_to_matrix(self, perm):
        """Converts a permutation vector to its corresponding permutation matrix."""
        n = len(perm)
        matrix = torch.zeros(n, n, dtype=torch.float32)
        matrix[torch.arange(n), perm] = 1
        return matrix
    
    def bind(self, x, y, eps=1e-8):
        """
        VSA binding operator. Two modes selected at construction time via binding_type:
          'circular'  — circular convolution via FFT (default; equivalent to HRR binding).
                        Produces a near-random direction, maximally dissimilar to inputs.
          'hadamard'  — elementwise multiplication (MAP / FHRR-style).
                        Stays close to both inputs; preserves cluster geometry better.
        Both outputs are L2-normalised so norms stay unit after binding.
        """
        if self.binding_type == "hadamard":
            result = x * y
        else:
            fft_self = fft(x, dim=1)
            fft_other = fft(y, dim=1)
            product = torch.mul(fft_self, fft_other)
            result = torch.real(ifft(product, dim=1))
        return F.normalize(result, p=2, dim=1, eps=eps)

    def permute_hv(self, x, shift=1):
        """Cyclic permutation to encode structural/temporal relationships (P_bef in Gayler 2023)."""
        return torch.roll(x, shifts=shift, dims=1)

    def weighted_bundle(self, x, y, weight_x, weight_y, eps=1e-8):
        """Weighted VSA bundling with L2 normalization for feature fading control."""
        result = weight_x * x + weight_y * y
        norm = result.norm(p=2, dim=1, keepdim=True).clamp(min=eps)
        return result / norm

    def reservoir_message_passing(self, node_H, edge_H, edge_index, eps=1e-8):
        """VSA message passing: bind neighbor states with edge features, aggregate at dst."""
        src, dst = edge_index[0], edge_index[1]
        neighbor_h = node_H[src]
        messages = self.bind(neighbor_h, edge_H)
        aggregated = torch.zeros_like(node_H)
        aggregated.index_add_(0, dst, self.reservoir_alpha * messages)
        norm = aggregated.norm(p=2, dim=1, keepdim=True).clamp(min=eps)
        return aggregated / norm

    def _rho_k(self, x, k):
        """Hop-specific permutation (rho^k): cyclic roll by k positions."""
        return torch.roll(x, shifts=int(k), dims=1)

    def _pi(self, x, shift=None):
        """Sigma-Pi permutation (different from rho): used for polynomial recursion."""
        s = shift if shift is not None else max(1, x.shape[1] // 3)
        return torch.roll(x, shifts=int(s), dims=1)

    def tap_buffer(self, hidden_rep, eps=1e-8):
        """
        RC-style tap buffer per node: F_v^(1) = sum_k lambda^k * rho^k(h_v^(k)).
        hidden_rep: list of [N, D] tensors, one per hop k=0..K.
        """
        F1 = torch.zeros_like(hidden_rep[0], device=hidden_rep[0].device, dtype=hidden_rep[0].dtype)
        for k, h_k in enumerate(hidden_rep):
            weight = self.hop_decay ** k
            permuted = self._rho_k(h_k, k)
            F1 = F1 + weight * permuted
        return F.normalize(F1, p=2, dim=1, eps=eps)

    def sigma_pi_expansion_and_terms(self, F1, eps=1e-8):
        """
        Sigma-Pi polynomial expansion with per-order terms.
        *_0(F)=F, *_t(F)= pi(*_{t-1}(F)) circ F.

        Always builds the full recursive chain 0..max(sigma_pi_orders) so that
        higher orders use the correct *_{t-1} even when lower orders are not
        in self.sigma_pi_orders.  Only requested orders are summed into result.
        """
        D = F1.shape[1]
        max_order = max(self.sigma_pi_orders)
        wanted = set(self.sigma_pi_orders)

        result = torch.zeros_like(F1)
        terms = {}
        ast_prev = F1

        for t in range(max_order + 1):
            if t == 0:
                ast_t = F1
            else:
                ast_t = self.bind(self._pi(ast_prev, shift=max(1, D // 3)), F1)
                ast_t = F.normalize(ast_t, p=2, dim=1, eps=eps)
            ast_prev = ast_t

            if t in wanted:
                result = result + ast_t
                terms[t] = ast_t.clone()

        F_v = F.normalize(result, p=2, dim=1, eps=eps)
        return F_v, terms

    def sigma_pi_expansion(self, F1, eps=1e-8):
        """
        Sigma-Pi polynomial expansion: F_v = sum_{t in T} *_t(F_v^(1)).
        """
        F_v, _ = self.sigma_pi_expansion_and_terms(F1, eps=eps)
        return F_v

    def multi_stat_pool(self, F_v, graph_pool, start_idx, eps=1e-8):
        """
        Multi-stat pooling: g = [mean_v(F_v) | max_v(F_v) | mean_v(F_v circ F_v)].
        graph_pool: [num_graphs, total_nodes] sparse; start_idx from preprocess.
        Returns [num_graphs, 3*D] (concatenation).
        """
        num_graphs = graph_pool.shape[0]
        D = F_v.shape[1]
        g_mean = torch.spmm(graph_pool, F_v)
        g_max = self._segment_max(F_v, graph_pool, start_idx, num_graphs)
        F_sq = self.bind(F_v, F_v)
        g_mean_sq = torch.spmm(graph_pool, F_sq)
        num_nodes = torch.tensor(
            [start_idx[i + 1] - start_idx[i] for i in range(num_graphs)],
            dtype=F_v.dtype, device=F_v.device
        ).view(-1, 1).clamp(min=1)
        if self.graph_pooling_type == "sum":
            g_mean = g_mean / num_nodes
            g_mean_sq = g_mean_sq / num_nodes
        g = torch.cat([g_mean, g_max, g_mean_sq], dim=1)
        return g

    def _segment_max(self, x, graph_pool, start_idx, num_graphs):
        """Compute max over nodes per graph. x: [total_nodes, D]."""
        total_nodes = x.shape[0]
        D = x.shape[1]
        out = torch.full((num_graphs, D), float('-inf'), device=x.device, dtype=x.dtype)
        for i in range(num_graphs):
            lo, hi = start_idx[i], start_idx[i + 1]
            if hi > lo:
                out[i] = x[lo:hi].max(dim=0)[0]
        return out

    def reservoir_update(self, node_H, edge_H, edge_index):
        """
        Legacy: old reservoir-style update (deprecated). Use tap_buffer + sigma_pi_expansion instead.
        """
        x_state = node_H.clone()
        for _ in range(self.reservoir_iters):
            messages = self.reservoir_message_passing(x_state, edge_H, edge_index)
            x_permuted = self.permute_hv(x_state, shift=1)
            interaction = self.bind(messages, x_permuted)
            if self.reservoir_polynomial_order >= 2:
                w_msg = (1 - self.reservoir_history_weight) * 0.5
                w_state = self.reservoir_history_weight * 0.5
                w_interaction = 0.5
                total = w_msg + w_state + w_interaction
                x_state = (w_msg / total) * messages + (w_state / total) * x_permuted + (w_interaction / total) * interaction
            else:
                x_state = self.weighted_bundle(x_permuted, messages, self.reservoir_history_weight, 1 - self.reservoir_history_weight)
            x_state = F.normalize(x_state, p=2, dim=1)
        return x_state

    def resonator_consensus(self, node_H, edge_H, edge_index, iterations=7, beta=0.75):
        """Legacy resonator (deprecated, use reservoir_update instead)."""
        N, D = node_H.shape
        current = node_H.clone()
        src, dst = edge_index[0], edge_index[1]
        for _ in range(iterations):
            messages = torch.zeros_like(current)
            neighbor_h = current[src]
            msg = self.bind(edge_H, neighbor_h)
            messages.index_add_(0, dst, msg)
            msg_norm = messages.norm(p=2, dim=1, keepdim=True).clamp(min=1e-8)
            messages = messages / msg_norm
            current = beta * current + (1 - beta) * messages
            curr_norm = current.norm(p=2, dim=1, keepdim=True).clamp(min=1e-8)
            current = current / curr_norm
        return current
    def invert_permutation(self, perm):
        """Generate the inverse of a permutation."""
        inverse = [0] * len(perm)
        for i, p in enumerate(perm):
            inverse[p] = i
        return inverse
    def _pool_neighbors(self, h_pool, Adj_block, padded_neighbor_list, edge_index, edge_H, num_nodes):
        """Dispatch to edge-conditioned pool or adjacency-based pool."""
        use_edges = edge_index is not None and edge_H is not None and num_nodes is not None
        avg = (self.neighbor_pooling_type == "average")
        if use_edges:
            return self._edge_message_pool(h_pool, edge_index, edge_H, num_nodes)
        if self.neighbor_pooling_type == "max":
            return self.maxpool(h_pool, padded_neighbor_list)
        pooled = torch.spmm(Adj_block, h_pool)
        if avg:
            degree = torch.spmm(Adj_block, torch.ones((Adj_block.shape[0], 1)).to(self.device))
            pooled = pooled / degree
        # Degree-invariant unit norm (same idea as _edge_message_pool after sum/average)
        return F.normalize(pooled, p=2, dim=1, eps=1e-8)

    def next_layer_eps(self, h, layer, padded_neighbor_list=None, Adj_block=None, delta=1, equation=10,
                       edge_index=None, edge_H=None, num_nodes=None, return_pre_norm=False):
        shift = 1
        eps_n = 1e-8
        pre_norm = None

        if equation == 10:
            rotated = torch.roll(h.clone(), shifts=shift, dims=1)
            pooled_nb = self._pool_neighbors(rotated, Adj_block, padded_neighbor_list, edge_index, edge_H, num_nodes)
            if delta == 1:
                bound = self.bind(h, pooled_nb)
                bundled = bound + h
                pre_norm = bundled
                pooled = F.normalize(bundled, p=2, dim=1, eps=eps_n)
            elif delta == 2:
                bound = self.bind(h, pooled_nb)
                bundled = bound + h + pooled_nb
                pre_norm = bundled
                pooled = F.normalize(bundled, p=2, dim=1, eps=eps_n)
            else:
                bundled = pooled_nb + h
                pre_norm = bundled
                pooled = F.normalize(bundled, p=2, dim=1, eps=eps_n)

        elif equation == 11:
            pooled_nb = self._pool_neighbors(h, Adj_block, padded_neighbor_list, edge_index, edge_H, num_nodes)
            if delta == 1:
                bound = self.bind(h, pooled_nb)
                bundled = bound + h
                pre_norm = bundled
                pooled = F.normalize(bundled, p=2, dim=1, eps=eps_n)
                pooled = torch.roll(pooled, shifts=shift, dims=1)
            elif delta == 2:
                bound = self.bind(h, pooled_nb)
                bundled = bound + h + pooled_nb
                pre_norm = bundled
                pooled = F.normalize(bundled, p=2, dim=1, eps=eps_n)
                pooled = torch.roll(pooled, shifts=shift, dims=1)
            else:
                bundled = pooled_nb + h
                pre_norm = bundled
                pooled = F.normalize(bundled, p=2, dim=1, eps=eps_n)
                pooled = torch.roll(pooled, shifts=shift, dims=1)

        else:
            rotated = torch.roll(h.clone(), shifts=shift, dims=1)
            pooled_nb = self._pool_neighbors(rotated, Adj_block, padded_neighbor_list, edge_index, edge_H, num_nodes)
            if delta == 1:
                bound = self.bind(h, pooled_nb)
                bundled = bound + h
                pre_norm = bundled
                pooled = F.normalize(bundled, p=2, dim=1, eps=eps_n)
                pooled = torch.roll(pooled, shifts=shift, dims=1)
            elif delta == 2:
                bound = self.bind(h, pooled_nb)
                bundled = bound + h + pooled_nb
                pre_norm = bundled
                pooled = F.normalize(bundled, p=2, dim=1, eps=eps_n)
                pooled = torch.roll(pooled, shifts=shift, dims=1)
            else:
                bundled = pooled_nb + h
                pre_norm = bundled
                pooled = F.normalize(bundled, p=2, dim=1, eps=eps_n)
                pooled = torch.roll(pooled, shifts=shift, dims=1)

        if return_pre_norm:
            return pooled, pre_norm
        return pooled




    def forward(self, batch_graph, return_embedding=False, return_node_rep=False, capture_aux=False):
        """
        return_node_rep: if True, return (H, batch) with H [N, D] node hypervectors and batch [N]
                         for use with attention readout. Only one of graph-level or node-level is returned.
        capture_aux: if True, fill self._aux with layer pre/post L2-normalization HV and sigma-pi tensors
                     (same forward math as capture_aux=False).
        """
        self._aux = None
        start_idx = [0]
        for g in batch_graph:
            start_idx.append(start_idx[-1] + len(g.g))
        B = len(batch_graph)
        N = start_idx[-1]

        X_concat = torch.cat([g.node_features for g in batch_graph], 0).to(self.device)
        X_concat = F.normalize(X_concat, p=2, dim=1, eps=1e-8)
        graph_pool = self.__preprocess_graphpool(batch_graph)
        Adj_block = self.__preprocess_neighbors_sumavepool(batch_graph)

        batched_ei, batched_ea, _ = self.__preprocess_edges(batch_graph)
        num_nodes = start_idx[-1]
        edge_index = None
        edge_H = None
        if batched_ei is not None and batched_ea is not None and self.edge_feat_dim > 0 and hasattr(self, "W_edge"):
            edge_index = batched_ei
            ea = batched_ea.to(X_concat.dtype).float()
            if "edge_col_min" in self._buffers and "edge_col_range" in self._buffers:
                ea = ea.clone()
                ea[:, EDGE_BINARY_COLS] = ea[:, EDGE_BINARY_COLS] * 2.0 - 1.0
                cmin = self.edge_col_min
                crng = self.edge_col_range
                ea[:, EDGE_MINMAX_COLS] = ((ea[:, EDGE_MINMAX_COLS] - cmin) / crng) * 2.0 - 1.0
            edge_H = torch.mm(ea, self.W_edge)
            edge_H = F.normalize(edge_H, p=2, dim=1, eps=1e-8)

        hidden_rep = [X_concat]
        h = X_concat
        layer_pre_bin, layer_post_bin = [], []
        for layer in range(self.num_layers - 1):
            if capture_aux:
                h, pre = self.next_layer_eps(
                    h, layer,
                    Adj_block=Adj_block,
                    delta=self.delta,
                    equation=self.equation,
                    edge_index=edge_index,
                    edge_H=edge_H,
                    num_nodes=num_nodes,
                    return_pre_norm=True,
                )
                layer_pre_bin.append(pre.detach())
                layer_post_bin.append(h.detach())
            else:
                h = self.next_layer_eps(
                    h, layer,
                    Adj_block=Adj_block,
                    delta=self.delta,
                    equation=self.equation,
                    edge_index=edge_index,
                    edge_H=edge_H,
                    num_nodes=num_nodes,
                )
            hidden_rep.append(h)

        # VSA-RC: tap buffer + Sigma-Pi; then either node-level (H, batch) or graph-level g
        if self.use_reservoir:
            F1 = self.tap_buffer(hidden_rep)
            if capture_aux:
                F_v, sigma_terms = self.sigma_pi_expansion_and_terms(F1)
            else:
                F_v = self.sigma_pi_expansion(F1)
            if capture_aux:
                self._aux = {
                    "layer_pre_bin": layer_pre_bin,
                    "layer_post_bin": layer_post_bin,
                    "F1_tap": F1.detach(),
                    "sigma_pi_terms": {t: v.detach() for t, v in sigma_terms.items()},
                    "sigma_pi_combined": F_v.detach(),
                    "start_idx": [int(x) for x in start_idx],
                }
            if return_node_rep:
                batch = torch.zeros(N, dtype=torch.long, device=F_v.device)
                for b in range(B):
                    batch[start_idx[b] : start_idx[b + 1]] = b
                if capture_aux:
                    self._aux["batch_node_graph_id"] = batch.detach().cpu()
                return (F_v, batch)
            g = self.multi_stat_pool(F_v, graph_pool, start_idx)
            return g.unsqueeze(0)
        if capture_aux:
            self._aux = {
                "layer_pre_bin": layer_pre_bin,
                "layer_post_bin": layer_post_bin,
                "F1_tap": None,
                "sigma_pi_terms": None,
                "sigma_pi_combined": None,
                "start_idx": [int(x) for x in start_idx],
            }
        # Legacy resonator: refine final layer only
        if edge_index is not None and edge_H is not None and self.use_resonator:
            hidden_rep[-1] = self.resonator_consensus(
                hidden_rep[-1], edge_H, edge_index,
                iterations=self.resonator_iters,
                beta=self.resonator_beta,
            )

        if return_node_rep:
            H = hidden_rep[-1]
            batch = torch.zeros(N, dtype=torch.long, device=H.device)
            for b in range(B):
                batch[start_idx[b] : start_idx[b + 1]] = b
            if capture_aux and self._aux is not None:
                self._aux["batch_node_graph_id"] = batch.detach().cpu()
            return (H, batch)

        pooled_hS = []
        for layer, h in enumerate(hidden_rep):
            pooled_h = torch.spmm(graph_pool, h)
            pooled_hS.append(pooled_h)
        return torch.stack(pooled_hS, dim=0)

    
