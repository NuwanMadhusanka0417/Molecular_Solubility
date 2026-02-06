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
                 use_hier_khop=False, max_hops=2, hop_alpha=0.8, skip_gcnn_after_hier=False):
        '''
            num_layers: number of layers (INCLUDING input)
            input_dim: node HV dim D
            delta: binding usage
            neighbor_pooling_type: sum, average, or max
            graph_pooling_type: sum or average
            device: device
            edge_feat_dim: raw edge feature dim; 0 = no edge conditioning
            edge_projection_type: "orthogonal" (info-preserving) or "gaussian" for edge_attr -> HV
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
        # Hierarchical k-hop encoding flags
        self.use_hier_khop = use_hier_khop
        self.max_hops = max_hops
        self.hop_alpha = hop_alpha
        self.skip_gcnn_after_hier = skip_gcnn_after_hier

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

    def _edge_message_pool(self, h_to_pool, edge_index, edge_H, num_nodes, average=False):
        """
        Edge-conditioned message passing: for each edge (src, dst), message = bind(h_to_pool[src], edge_H[e]),
        then aggregate at dst. Caller passes rotated or plain h as h_to_pool. Physically: combine
        neighbour atom with the bond along that edge, send message along that bond.
        """
        E = edge_index.shape[1]
        D = h_to_pool.shape[1]
        src, dst = edge_index[0], edge_index[1]
        neighbor_h = h_to_pool[src]
        messages = self.bind(neighbor_h, edge_H)
        pooled = torch.zeros(num_nodes, D, device=h_to_pool.device, dtype=h_to_pool.dtype)
        pooled.index_add_(0, dst, messages)
        if average:
            degree = torch.zeros(num_nodes, 1, device=h_to_pool.device, dtype=h_to_pool.dtype)
            degree.index_add_(0, dst.unsqueeze(1), torch.ones(E, 1, device=h_to_pool.device, dtype=h_to_pool.dtype))
            degree = degree.clamp(min=1.0)
            pooled = pooled / degree
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
            return self._edge_message_pool(h_pool, edge_index, edge_H, num_nodes, average=avg)
        if self.neighbor_pooling_type == "max":
            return self.maxpool(h_pool, padded_neighbor_list)
        pooled = torch.spmm(Adj_block, h_pool)
        if avg:
            degree = torch.spmm(Adj_block, torch.ones((Adj_block.shape[0], 1)).to(self.device))
            pooled = pooled / degree
        return pooled

    def _hier_khop_encode(self, X_concat, edge_index, edge_H, num_nodes, Adj_block):
        """
        Hierarchical k-hop encoding at node level.
        For k=0..K:
          - aggregate neighbor info via edge-conditioned binding
          - apply hop-specific roll (shift=k) to tag hop distance
          - accumulate with decay lambda_k = hop_alpha^k
        Returns enriched node HVs [N, D].
        """
        h_curr = X_concat
        sigs = []

        # Hop 0: original nodes
        sigs.append(X_concat)

        for k in range(1, self.max_hops + 1):
            if edge_index is not None and edge_H is not None and num_nodes is not None:
                avg = (self.neighbor_pooling_type == "average")
                agg = self._edge_message_pool(h_curr, edge_index, edge_H, num_nodes, average=avg)
            else:
                # fallback to adjacency pooling
                agg = self._pool_neighbors(h_curr, Adj_block, None, edge_index, edge_H, num_nodes)
            # tag hop distance via roll
            sig_k = torch.roll(agg, shifts=k, dims=1)
            sigs.append(sig_k)
            h_curr = agg

        # Weighted sum with decay
        enriched = torch.zeros_like(X_concat)
        for k, sig in enumerate(sigs):
            enriched = enriched + (self.hop_alpha ** k) * sig
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

        pooled = torch.sign(pooled)
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

        # Hierarchical k-hop encoding (node-level)
        if self.use_hier_khop:
            h_init = self._hier_khop_encode(X_concat, edge_index, edge_H, num_nodes, Adj_block)
        else:
            h_init = X_concat

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

        # Pool once to get graph embedding [batch, D]
        graph_emb = torch.spmm(graph_pool, h_final)
        return graph_emb

    
