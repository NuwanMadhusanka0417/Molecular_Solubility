"""
GraphCNN with VSA binding and multiple equation variants.

Equations (parameter `equation`):
  10: Original - rotate(h), pool, output = bind(h, pooled) + h (fixed shift=1).
  11: Original - pool(h), then rotate; bind+residual.
  12: Adaptive rotation - shift = 1 + layer (better k-hop distinction). Expected +5-8%.
  13: Edge strength - weight messages by bond_type/conjugated/in_ring/length. Expected +10-15%.
  14: Directional binding - src/dst rotations, triple bind. Expected +8-12%.
  15: Full - directional + edge strength + attention-like aggregation. Expected +15-25%.

Recommended: start with 12 (zero cost); then 13 for solubility (edge features matter).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import fft, ifft
import math
import sys
sys.path.append("models/")
from models.mlp import MLP

class GraphCNN(nn.Module):
    def __init__(self, input_dim, num_layers, delta, graph_pooling_type, neighbor_pooling_type, device, equation, edge_feat_dim=5, edge_projection_type="orthogonal"):
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

    def _compute_edge_strengths(self, edge_attr_raw):
        """
        Compute edge importance from raw features [bond_type, conjugated, in_ring, length?, stereo?].
        Returns [E] tensor of strength values in [0.5, 1.5].
        """
        E = edge_attr_raw.shape[0]
        device = edge_attr_raw.device
        dtype = edge_attr_raw.dtype
        if edge_attr_raw.shape[1] >= 5:
            bond_type = edge_attr_raw[:, 0] / 4.0
            conjugated = edge_attr_raw[:, 1]
            in_ring = edge_attr_raw[:, 2]
            length = edge_attr_raw[:, 3].clamp(min=1e-6)
            stereo = edge_attr_raw[:, 4] / 4.0
            strength = (
                0.35 * (1.0 + bond_type)
                + 0.20 * conjugated
                + 0.15 * in_ring
                + 0.20 * (1.5 / (1.0 + length))
                + 0.10 * stereo
            )
            strength = 0.5 + torch.sigmoid(strength - 0.5)
        elif edge_attr_raw.shape[1] >= 3:
            bond_type = edge_attr_raw[:, 0] / 4.0 if edge_attr_raw.shape[1] > 0 else 0.0
            conjugated = edge_attr_raw[:, 1] if edge_attr_raw.shape[1] > 1 else 0.0
            in_ring = edge_attr_raw[:, 2] if edge_attr_raw.shape[1] > 2 else 0.0
            strength = 0.35 * (1.0 + bond_type) + 0.30 * conjugated + 0.25 * in_ring
            strength = 0.5 + torch.sigmoid(strength - 0.5)
        else:
            strength = torch.ones(E, device=device, dtype=dtype)
        return strength

    def next_layer_eps(self, h, layer, padded_neighbor_list=None, Adj_block=None, delta=1, equation=10,
                       edge_index=None, edge_H=None, num_nodes=None, edge_attr_raw=None):
        """
        Dispatch to equation variant. equation: 10,11 (original), 12 (adaptive rotation),
        13 (edge strength), 14 (directional), 15 (full improvements).
        """
        torch.manual_seed(0)

        # Equation 12: Adaptive rotation (shift = 1 + layer)
        if equation == 12:
            shift = 1 + layer
            rotated = torch.roll(h.clone(), shifts=shift, dims=1)
            pooled = self._pool_neighbors(rotated, Adj_block, padded_neighbor_list, edge_index, edge_H, num_nodes)
            output = self.bind(h, pooled) + h
            return torch.sign(output)

        # Equation 13: Edge-strength modulated message passing
        if equation == 13:
            shift = 1 + layer
            if edge_index is not None and edge_H is not None and num_nodes is not None:
                E, D = edge_index.shape[1], h.shape[1]
                src, dst = edge_index[0], edge_index[1]
                rotated = torch.roll(h.clone(), shifts=shift, dims=1)
                neighbor_h = rotated[src]
                edge_H_mod = edge_H * self._compute_edge_strengths(edge_attr_raw).unsqueeze(1) if edge_attr_raw is not None else edge_H
                messages = self.bind(neighbor_h, edge_H_mod)
                pooled = torch.zeros(num_nodes, D, device=h.device, dtype=h.dtype)
                pooled.index_add_(0, dst, messages)
                if self.neighbor_pooling_type == "average":
                    degree = torch.zeros(num_nodes, 1, device=h.device, dtype=h.dtype)
                    degree.index_add_(0, dst.unsqueeze(1), torch.ones(E, 1, device=h.device, dtype=h.dtype))
                    degree = degree.clamp(min=1.0)
                    pooled = pooled / degree
            else:
                rotated = torch.roll(h.clone(), shifts=shift, dims=1)
                pooled = self._pool_neighbors(rotated, Adj_block, padded_neighbor_list, edge_index, edge_H, num_nodes)
            output = self.bind(h, pooled) + h
            return torch.sign(output)

        # Equation 14: Directional binding (src/dst rotations)
        if equation == 14:
            shift_src = 1 + layer * 2
            shift_dst = -(1 + layer * 2)
            if edge_index is not None and edge_H is not None and num_nodes is not None:
                E, D = edge_index.shape[1], h.shape[1]
                src, dst = edge_index[0], edge_index[1]
                h_src = torch.roll(h[src], shifts=shift_src, dims=1)
                h_dst = torch.roll(h[dst], shifts=shift_dst, dims=1)
                edge_H_mod = edge_H * self._compute_edge_strengths(edge_attr_raw).unsqueeze(1) if edge_attr_raw is not None else edge_H
                messages = self.bind(h_src, edge_H_mod)
                messages = self.bind(messages, h_dst)
                pooled = torch.zeros(num_nodes, D, device=h.device, dtype=h.dtype)
                pooled.index_add_(0, dst, messages)
                if self.neighbor_pooling_type == "average":
                    degree = torch.zeros(num_nodes, 1, device=h.device, dtype=h.dtype)
                    degree.index_add_(0, dst.unsqueeze(1), torch.ones(E, 1, device=h.device, dtype=h.dtype))
                    degree = degree.clamp(min=1.0)
                    pooled = pooled / degree
            else:
                rotated = torch.roll(h.clone(), shifts=shift_src, dims=1)
                pooled = self._pool_neighbors(rotated, Adj_block, padded_neighbor_list, edge_index, edge_H, num_nodes)
            output = pooled + h
            return torch.sign(output)

        # Equation 15: Full improvements (directional + edge strength + attention-like aggregation)
        if equation == 15:
            shift = 1 + layer
            if edge_index is not None and edge_H is not None and num_nodes is not None:
                E, D = edge_index.shape[1], h.shape[1]
                src, dst = edge_index[0], edge_index[1]
                h_src = torch.roll(h[src], shifts=shift, dims=1)
                h_dst = torch.roll(h[dst], shifts=-shift, dims=1)
                strengths = self._compute_edge_strengths(edge_attr_raw) if edge_attr_raw is not None else torch.ones(E, device=h.device, dtype=h.dtype)
                edge_H_mod = edge_H * strengths.unsqueeze(1)
                messages = self.bind(h_src, edge_H_mod)
                messages = self.bind(messages, h_dst)
                receiver_hvs = h[dst]
                similarities = F.cosine_similarity(messages, receiver_hvs, dim=1)
                attention_logits = similarities * strengths
                attention_weights = torch.zeros(E, device=h.device, dtype=h.dtype)
                for node in range(num_nodes):
                    mask = (dst == node)
                    if mask.any():
                        logits = attention_logits[mask]
                        attention_weights[mask] = F.softmax(logits * 3.0, dim=0)
                weighted_messages = messages * attention_weights.unsqueeze(1)
                pooled_sum = torch.zeros(num_nodes, D, device=h.device, dtype=h.dtype)
                pooled_sum.index_add_(0, dst, weighted_messages)
                degree = torch.zeros(num_nodes, 1, device=h.device, dtype=h.dtype)
                degree.index_add_(0, dst.unsqueeze(1), torch.ones(E, 1, device=h.device, dtype=h.dtype))
                degree = degree.clamp(min=1.0)
                pooled_mean = pooled_sum / degree
                degree_norm = torch.clamp(degree / 10.0, 0.0, 1.0)
                alpha = 0.5 * (1.0 - 0.5 * degree_norm)
                pooled = alpha * pooled_sum + (1.0 - alpha) * pooled_mean
            else:
                rotated = torch.roll(h.clone(), shifts=shift, dims=1)
                pooled = self._pool_neighbors(rotated, Adj_block, padded_neighbor_list, edge_index, edge_H, num_nodes)
            gate = torch.sigmoid(torch.norm(pooled, p=2, dim=1, keepdim=True) * 2.0 - 1.0)
            gate = 0.3 + 0.6 * gate
            output = h + gate * pooled
            return torch.sign(output)

        # Original equations 10, 11, and default
        shift = 1
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

        hidden_rep = [X_concat]
        h = X_concat
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
            hidden_rep.append(h)

        pooled_hS = []
        for layer, h in enumerate(hidden_rep):
            pooled_h = torch.spmm(graph_pool, h)
            pooled_hS.append(pooled_h)
        return torch.stack(pooled_hS, dim=0)

    
