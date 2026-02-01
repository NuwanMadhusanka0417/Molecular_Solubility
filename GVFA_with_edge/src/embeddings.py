import torch

def getEmbedding(model, device, train_graphs, batch_size=100, SUM=True, use_size_aware=True, hop_alpha=1.0):
    """
    Get graph-level embeddings for regression.

    use_size_aware: if True (default), applies two changes to help solubility prediction:
        1) Scale each graph's pooled vector at every layer by 1/√(num_nodes), so larger
           molecules don't dominate the sum-pooled representation.
        2) Append num_nodes as an extra feature (last column). XGBoost then gets D+1
           dimensions; the last is atom count, which is important for solubility.

    hop_alpha: topologically decaying hop weights. When combining layers, applies
        weights = alpha ** layer_ids so nearer hops (lower layer_id) get higher weight.
        hop_alpha=1.0 (default) means all layers weighted equally (original behavior).
        hop_alpha<1 (e.g. 0.9) decays weight with hop distance.
    """
    model.to(device)
    model.train()

    combined_embeddings = []
    all_labels = []

    num_graphs = len(train_graphs)
    for start_idx in range(0, num_graphs, batch_size):
        end_idx = min(start_idx + batch_size, num_graphs)
        batch_graphs = train_graphs[start_idx:end_idx]

        # output: [num_layers, batch_size, D]
        output = model(batch_graphs)

        # Size-aware: scale each graph's representation by 1/√(num_nodes) at every layer
        if use_size_aware:
            num_nodes = torch.tensor(
                [len(g.g) for g in batch_graphs],
                dtype=output.dtype,
                device=output.device,
            )
            # [batch_size] -> [1, batch_size, 1] for broadcasting over [num_layers, batch_size, D]
            scale = (num_nodes ** 0.5).clamp(min=1e-6).view(1, -1, 1)
            output = output / scale

        if SUM:
            # Topologically decaying hop weights: weights = alpha ** layer_ids
            num_layers_out = output.shape[0]
            layer_ids = torch.arange(
                num_layers_out, dtype=output.dtype, device=output.device
            )
            weights = (hop_alpha ** layer_ids).view(-1, 1, 1)  # [num_layers, 1, 1]
            combined_embedding = (output * weights).sum(dim=0, keepdim=True)  # [1, batch_size, D]
        else:
            combined_embedding = torch.cat(output, dim=1)

        # Append num_nodes as extra column so XGBoost gets explicit size feature (D+1 input)
        if use_size_aware:
            n = torch.tensor(
                [len(g.g) for g in batch_graphs],
                dtype=combined_embedding.dtype,
                device=combined_embedding.device,
            ).view(1, -1, 1)
            combined_embedding = torch.cat([combined_embedding, n], dim=2)  # [1, batch_size, D+1]

        combined_embeddings.append(combined_embedding)
        labels = torch.FloatTensor([graph.label for graph in batch_graphs]).to(device)
        all_labels.append(labels)

    final_labels = torch.cat(all_labels, dim=0)
    final_embeddings = torch.cat(combined_embeddings, dim=1)
    return final_embeddings, final_labels