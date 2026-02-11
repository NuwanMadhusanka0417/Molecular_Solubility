import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import fft, ifft
import math
import sys
sys.path.append("models/")
from models.mlp import MLP


class GraphCNN(nn.Module):
    def __init__(self, input_dim, num_layers, delta, graph_pooling_type, neighbor_pooling_type, device, equation,
                 edge_feat_dim=5, edge_projection_type="orthogonal",
                 use_hier_khop=False, max_hops=2, hop_alpha=0.8, skip_gcnn_after_hier=False,
                 use_edge_strength=True, use_positional_encoding=True, use_adaptive_pooling=False,
                 use_resonator=False, resonator_iters=7, resonator_beta=0.75,
                 # k-hop controls
                 khop_edge_reduce="sum",          # "sum" or "mean" for k-hop aggregation (recommended: "sum")
                 khop_postprocess="l2"):          # "l2" or "sign" after bundling (recommended: "l2")
        '''
            use_resonator: refine node HVs through neighbor consensus (5-10% MAE improvement)
            resonator_iters: 7 recommended; resonator_beta: 0.75 for stable convergence
        '''

        super(GraphCNN, self).__init__()
        print("Input feature size: ", input_dim)
        print("  Improvements: edge_strength={}, positional={}, adaptive_pool={}, resonator={}".format(
            use_edge_strength, use_positional_encoding, use_adaptive_pooling, use_resonator))
        self.device = device
        self.num_layers = num_layers
        self.graph_pooling_type = graph_pooling_type
        self.neighbor_pooling_type = neighbor_pooling_type
        self.learn_eps = True
        self.delta = delta
        self.equation = equation
        self.edge_feat_dim = edge_feat_dim if edge_feat_dim else 0
        self.use_hier_khop = use_hier_khop
        self.max_hops = max_hops
        self.hop_alpha = hop_alpha
        self.skip_gcnn_after_hier = skip_gcnn_after_hier
        self.use_edge_strength = use_edge_strength
        self.use_positional_encoding = use_positional_encoding
        self.use_adaptive_pooling = use_adaptive_pooling
        self.use_resonator = use_resonator
        self.resonator_iters = resonator_iters
        self.resonator_beta = resonator_beta
        self.khop_edge_reduce = khop_edge_reduce
        self.khop_postprocess = khop_postprocess

        if self.edge_feat_dim > 0:
            g = torch.Generator().manual_seed(0)
            W_edge = torch.randn(self.edge_feat_dim, input_dim, generator=g)
            if edge_projection_type == "orthogonal" and input_dim >= self.edge_feat_dim:
                # Orthonormal columns: preserves norms of projected edge vectors (info-preserving)
                A = torch.randn(input_dim, self.edge_feat_dim, generator=g)
                Q, _ = torch.linalg.qr(A)
                W_edge = Q[:, :self.edge_feat_dim].T  # (edge_feat_dim, input_dim)
            else:
                W_edge = W_edge / math.sqrt(self.edge_feat_dim)
            self.register_buffer("W_edge", W_edge)

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

    def _compute_edge_strengths(self, edge_attr_raw, eps=1e-8):
        """
        Stronger edge differentiation: 0.6-1.5 range (no sigmoid).
        Bond type lookup: single=0.7, double=1.0, triple=1.3, aromatic=1.2.
        """
        E = edge_attr_raw.shape[0]
        device = edge_attr_raw.device
        dtype = edge_attr_raw.dtype
        if edge_attr_raw.shape[1] >= 5:
            bond_type = edge_attr_raw[:, 0]
            conjugated = edge_attr_raw[:, 1]
            in_ring = edge_attr_raw[:, 2]
            length = edge_attr_raw[:, 3].clamp(min=0.5, max=3.0)
            # Convert 1-4 to 0-3 (data uses 1=single,2=double,3=triple,4=aromatic; 0=fallback)
            bt = bond_type - 1
            # Bond type lookup: 0=single, 1=double, 2=triple, 3=aromatic (0.5,1.5,2.5 separate)
            bond_strength = torch.where(
                bt < 0.5, torch.tensor(0.7, device=device, dtype=dtype),   # single
                torch.where(
                    bt < 1.5, torch.tensor(1.0, device=device, dtype=dtype),  # double
                    torch.where(
                        bt < 2.5, torch.tensor(1.3, device=device, dtype=dtype),  # triple
                        torch.tensor(1.2, device=device, dtype=dtype)   # aromatic
                    )
                )
            )
            length_factor = (1.5 / length).clamp(0.8, 1.3)
            conjugation_bonus = conjugated * 0.15
            ring_bonus = in_ring * 0.15
            strength = bond_strength * length_factor + conjugation_bonus + ring_bonus
            strength = strength.clamp(min=0.6, max=1.5)
        elif edge_attr_raw.shape[1] >= 3:
            bond_type = edge_attr_raw[:, 0]
            conjugated = edge_attr_raw[:, 1]
            in_ring = edge_attr_raw[:, 2]
            bt = bond_type - 1
            bond_strength = torch.where(
                bt < 0.5, torch.tensor(0.7, device=device, dtype=dtype),
                torch.where(
                    bt < 1.5, torch.tensor(1.0, device=device, dtype=dtype),
                    torch.where(
                        bt < 2.5, torch.tensor(1.3, device=device, dtype=dtype),
                        torch.tensor(1.2, device=device, dtype=dtype)
                    )
                )
            )
            strength = bond_strength + conjugated * 0.15 + in_ring * 0.15
            strength = strength.clamp(min=0.6, max=1.5)
        else:
            strength = torch.ones(E, device=device, dtype=dtype)
        return strength

    def _edge_message_pool(self, h_to_pool, edge_index, edge_H, num_nodes, average=False, reduce=None):
        """
        Edge-conditioned message passing. When use_adaptive_pooling, mix sum and mean by degree.
        """
        E = edge_index.shape[1]
        D = h_to_pool.shape[1]
        src, dst = edge_index[0], edge_index[1]
        neighbor_h = h_to_pool[src]
        messages = self.bind(neighbor_h, edge_H)
        pooled_sum = torch.zeros(num_nodes, D, device=h_to_pool.device, dtype=h_to_pool.dtype)
        pooled_sum.index_add_(0, dst, messages)
        degree = torch.zeros(num_nodes, 1, device=h_to_pool.device, dtype=h_to_pool.dtype)
        degree.index_add_(0, dst, torch.ones(E, 1, device=h_to_pool.device, dtype=h_to_pool.dtype))
        degree = degree.clamp(min=1.0)
        pooled_mean = pooled_sum / degree

        # Explicit override (used by k-hop so we can force "sum" and avoid averaging)
        if reduce == "sum":
            return pooled_sum
        if reduce == "mean":
            return pooled_mean

        if self.use_adaptive_pooling:
            degree_norm = torch.clamp(degree / 10.0, 0.0, 1.0)
            alpha = 0.5 * (1.0 - 0.5 * degree_norm)
            pooled = alpha * pooled_sum + (1.0 - alpha) * pooled_mean
        elif average:
            pooled = pooled_mean
        else:
            pooled = pooled_sum
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
    
    def bind(self, x, y):
        # Perform FFT on each hypervector in the tensors
        fft_self = fft(x, dim=1)
        fft_other = fft(y, dim=1)

        # Multiply element-wise in the frequency domain
        product = torch.mul(fft_self, fft_other)

        # Perform inverse FFT to get back to the spatial domain
        result = ifft(product, dim=1)

        # Return the real part of the result as the final bound hypervectors
        return torch.real(result)
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
            return self._edge_message_pool(h_pool, edge_index, edge_H, num_nodes, average=avg, reduce=None)
        if self.neighbor_pooling_type == "max":
            return self.maxpool(h_pool, padded_neighbor_list)
        pooled = torch.spmm(Adj_block, h_pool)
        if avg:
            degree = torch.spmm(Adj_block, torch.ones((Adj_block.shape[0], 1)).to(self.device))
            pooled = pooled / degree
        return pooled

    def _resonator_consensus(self, node_H, edge_index, edge_H, num_nodes, iterations=None, beta=None, eps=1e-8):
        """
        Iterative resonator: refine node HVs through neighbor agreement.
        Message from src to dst: bind(edge_H[e], current[src]); mix with beta.
        """
        iters = iterations if iterations is not None else self.resonator_iters
        beta_val = beta if beta is not None else self.resonator_beta
        current = node_H.clone()
        E = edge_index.shape[1]
        src, dst = edge_index[0], edge_index[1]
        for _ in range(iters):
            neighbor_h = current[src]
            messages = self.bind(edge_H, neighbor_h)
            pooled = torch.zeros_like(current)
            pooled.index_add_(0, dst, messages)
            msg_norm = pooled.norm(p=2, dim=1, keepdim=True).clamp(min=eps)
            messages = pooled / msg_norm
            current = beta_val * current + (1.0 - beta_val) * messages
            curr_norm = current.norm(p=2, dim=1, keepdim=True).clamp(min=eps)
            current = current / curr_norm
        return current

    def _add_positional_encoding(self, X_concat, edge_index, num_nodes, eps=1e-8):
        """
        Bind degree and inverse-degree into initial node features (structural context).
        """
        N, D = X_concat.shape
        degrees = torch.zeros(N, device=X_concat.device, dtype=X_concat.dtype)
        if edge_index is not None:
            for e in range(edge_index.shape[1]):
                u = int(edge_index[0, e])
                degrees[u] += 1
        pos_features = torch.zeros(N, 2, device=X_concat.device, dtype=X_concat.dtype)
        pos_features[:, 0] = degrees / (degrees.max() + eps)
        pos_features[:, 1] = 1.0 / (degrees + 1.0)
        torch.manual_seed(42)
        W_pos = torch.randn(2, D, device=X_concat.device, dtype=X_concat.dtype)
        W_pos = F.normalize(W_pos, p=2, dim=1)
        pos_hvs = torch.matmul(pos_features, W_pos)
        pos_hvs = F.normalize(pos_hvs, p=2, dim=1)
        enhanced = X_concat + 0.3 * pos_hvs  # Additive (not multiplicative bind)
        enhanced = F.normalize(enhanced, p=2, dim=1)
        return enhanced

    def _hier_khop_encode(self, X_concat, edge_index, edge_H, num_nodes, Adj_block, hop_shift_prime=13, eps=1e-8):
        """
        VSA-consistent hierarchical k-hop encoding at node level.
        Per hop k:
          - A^(k)_u = aggregate over neighbors: bind(H^(k-1)_v, b_vu)
          - normalize: A^(k)_u /= (||A^(k)_u||_2 + eps)
          - hop tag: S^(k)_u = Roll(A^(k)_u, 13*k)  [prime shift separates hop bands]
        Bundle with decay: H_enc = sum_k alpha^k S^(k), then harden with sign.
        Returns enriched node HVs [N, D].
        """
        sigs = []
        # Hop 0: original nodes (optionally L2-normalize for consistency)
        s0 = X_concat / (X_concat.norm(p=2, dim=1, keepdim=True).clamp(min=eps))
        sigs.append(torch.roll(s0, shifts=0, dims=1))  # shift 0 for hop 0

        h_curr = X_concat
        for k in range(1, self.max_hops + 1):
            if edge_index is not None and edge_H is not None and num_nodes is not None:
                # IMPORTANT: don't average in k-hop unless explicitly requested
                agg = self._edge_message_pool(
                    h_curr, edge_index, edge_H, num_nodes,
                    average=False,
                    reduce=self.khop_edge_reduce,
                )
            else:
                agg = self._pool_neighbors(h_curr, Adj_block, None, edge_index, edge_H, num_nodes)
            # Normalize after aggregation (clean HV behavior)
            agg = agg / (agg.norm(p=2, dim=1, keepdim=True).clamp(min=eps))
            # Hop-distinct permutation: prime shift separates hop bands (e.g. 13*k)
            shift = hop_shift_prime * k
            sig_k = torch.roll(agg, shifts=shift, dims=1)
            sigs.append(sig_k)
            h_curr = agg

        # Bundle with decay weights
        enriched = torch.zeros_like(X_concat)
        for k, sig in enumerate(sigs):
            enriched = enriched + (self.hop_alpha ** k) * sig

        # Post-process:
        # - "sign": strict bipolar HVs (old behavior, most lossy)
        # - "l2"  : continuous, L2-normalized
        # - "multi": multi-threshold binarization (5-level quantization)
        if self.khop_postprocess == "sign":
            enriched = torch.sign(enriched)
            enriched[enriched == 0] = 1.0
        elif self.khop_postprocess == "multi":
            # L2 normalize first to keep stable scale
            enriched = enriched / (enriched.norm(p=2, dim=1, keepdim=True).clamp(min=eps))
            hi = 0.7
            lo = 0.3
            out = torch.zeros_like(enriched)
            out = torch.where(enriched >= hi,  torch.tensor(1.0,  device=enriched.device, dtype=enriched.dtype), out)
            out = torch.where((enriched >= lo) & (enriched < hi),
                              torch.tensor(0.5,  device=enriched.device, dtype=enriched.dtype), out)
            out = torch.where(enriched <= -hi, torch.tensor(-1.0, device=enriched.device, dtype=enriched.dtype), out)
            out = torch.where((enriched <= -lo) & (enriched > -hi),
                              torch.tensor(-0.5, device=enriched.device, dtype=enriched.dtype), out)
            enriched = out
        else:  # default: L2
            enriched = enriched / (enriched.norm(p=2, dim=1, keepdim=True).clamp(min=eps))
        return enriched

    def next_layer_eps(self, h, layer, padded_neighbor_list=None, Adj_block=None, delta=1, equation=10,
                       edge_index=None, edge_H=None, num_nodes=None, edge_attr_raw=None):
        shift = 1
        torch.manual_seed(0)

        if equation == 10:
            rotated = torch.roll(h.clone(), shifts=shift, dims=1)
            pooled = self._pool_neighbors(rotated, Adj_block, padded_neighbor_list, edge_index, edge_H, num_nodes)
            if delta == 1:
                pooled = self.bind(h, pooled) + h
            elif delta == 2:
                pooled = self.bind(h, pooled) + h + pooled
            else:
                pooled = pooled + h

        elif equation == 11:
            pooled = self._pool_neighbors(h, Adj_block, padded_neighbor_list, edge_index, edge_H, num_nodes)
            if delta == 1:
                pooled = self.bind(h, pooled) + h
            elif delta == 2:
                pooled = self.bind(h, pooled) + h + pooled
            else:
                pooled = pooled + h
            pooled = torch.roll(pooled, shifts=shift, dims=1)

        else:
            rotated = torch.roll(h.clone(), shifts=shift, dims=1)
            pooled = self._pool_neighbors(rotated, Adj_block, padded_neighbor_list, edge_index, edge_H, num_nodes)
            if delta == 1:
                pooled = self.bind(h, pooled) + h
            elif delta == 2:
                pooled = self.bind(h, pooled) + h + pooled
            else:
                pooled = pooled + h
            pooled = torch.roll(pooled, shifts=shift, dims=1)

        # pooled = torch.sign(pooled)
        return pooled




    def forward(self, batch_graph, return_embedding=False):
        """
        Returns a single graph embedding [batch, D] (no per-layer stack).
        If use_hier_khop is True, first build hierarchical k-hop node encodings,
        then optionally run remaining GraphCNN layers, and pool once.
        """
        X_concat = torch.cat([g.node_features for g in batch_graph], 0).to(self.device)
        graph_pool = self.__preprocess_graphpool(batch_graph)
        Adj_block = self.__preprocess_neighbors_sumavepool(batch_graph)

        batched_ei, batched_ea, start_idx = self.__preprocess_edges(batch_graph)
        num_nodes = start_idx[-1]
        edge_index = None
        edge_H = None
        edge_attr_raw = None
        if batched_ei is not None and batched_ea is not None and self.edge_feat_dim > 0 and hasattr(self, "W_edge"):
            edge_index = batched_ei
            edge_attr_raw = batched_ea.to(X_concat.dtype)
            edge_H = torch.mm(edge_attr_raw, self.W_edge)
            if self.use_edge_strength and edge_attr_raw is not None:
                strengths = self._compute_edge_strengths(edge_attr_raw)
                edge_H = edge_H * strengths.unsqueeze(1)

        # Optional: bind degree into initial node features (structural context)
        if self.use_positional_encoding and edge_index is not None:
            X_enhanced = self._add_positional_encoding(X_concat, edge_index, num_nodes)
        else:
            X_enhanced = X_concat

        # Hierarchical k-hop encoding (node-level)
        if self.use_hier_khop:
            h_init = self._hier_khop_encode(X_enhanced, edge_index, edge_H, num_nodes, Adj_block)
        else:
            h_init = X_enhanced

        # Optionally run GraphCNN layers after hier encoding
        if self.skip_gcnn_after_hier or self.num_layers <= 1:
            h_final = h_init
        else:
            h = h_init
            for layer in range(self.num_layers - 1):
                h = self.next_layer_eps(
                    h, layer,
                    Adj_block=Adj_block,
                    delta=self.delta,
                    equation=self.equation,
                    edge_index=edge_index,
                    edge_H=edge_H,
                    num_nodes=num_nodes,
                    edge_attr_raw=edge_attr_raw,
                )
            h_final = h

        # Optional: resonator consensus to refine node HVs through neighbor agreement
        if self.use_resonator and edge_index is not None and edge_H is not None:
            h_final = self._resonator_consensus(h_final, edge_index, edge_H, num_nodes)

        # Pool once to get graph embedding [batch, D]
        graph_emb = torch.spmm(graph_pool, h_final)
        return graph_emb

    
