import torch

def getEmbedding(model, device, train_graphs, batch_size=100, SUM=True, use_size_aware=True, hop_alpha=1.0):
    """
    Get graph-level embeddings for regression.

    use_size_aware: if True (default), applies two changes to help solubility prediction:
        1) Scale each graph's pooled vector by 1/√(num_nodes) to prevent large molecules
           from dominating the representation.
           NOTE: this scaling is skipped when model.use_reservoir=True because
           multi_stat_pool already divides g_mean and g_mean_sq by num_nodes (true
           per-atom means). Applying 1/√N on top would result in an effective 1/N^1.5
           weighting, unfairly amplifying small molecules.
        2) Append soft-scaled molecule size log1p(num_atoms)/log1p(10) as an extra
           column (similar scale to bipolar HV dims). Always appended regardless of
           use_reservoir, as it provides useful information to Ridge.

    hop_alpha: topologically decaying hop weights. When combining layers, applies
        weights = alpha ** layer_ids so nearer hops (lower layer_id) get higher weight.
        hop_alpha=1.0 (default) means all layers weighted equally (original behavior).
        hop_alpha<1 (e.g. 0.9) decays weight with hop distance.
    """
    model.to(device)
    model.eval()

    # When use_reservoir=True, multi_stat_pool already returns true per-atom means
    # (it explicitly divides by num_nodes for g_mean and g_mean_sq). Adding
    # another 1/√N here would double-normalise those stats. Skip it.
    reservoir_active = getattr(model, 'use_reservoir', False)

    combined_embeddings = []
    all_labels = []

    num_graphs = len(train_graphs)
    for start_idx in range(0, num_graphs, batch_size):
        end_idx = min(start_idx + batch_size, num_graphs)
        batch_graphs = train_graphs[start_idx:end_idx]

        # output: [num_layers, batch_size, D]  (non-reservoir)
        #      or [1, batch_size, 3*D]          (reservoir — multi_stat_pool output)
        output = model(batch_graphs)

        if use_size_aware:
            num_nodes = torch.tensor(
                [len(g.g) for g in batch_graphs],
                dtype=output.dtype,
                device=output.device,
            )

            # Only apply 1/√N when the model returns raw sums (non-reservoir path).
            # Reservoir path: multi_stat_pool already divided by N → skip to avoid N^1.5.
            if not reservoir_active:
                # [batch_size] → [1, batch_size, 1] broadcasts over [num_layers, batch_size, D]
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

        # Append log1p-scaled molecule size as an extra feature.
        # Appended regardless of reservoir_active — it always provides useful info to Ridge.
        if use_size_aware:
            n_raw = num_nodes.to(combined_embedding.device).to(combined_embedding.dtype)
            n_scaled = (
                torch.log1p(n_raw)
                / torch.log1p(torch.tensor(10.0, device=combined_embedding.device, dtype=combined_embedding.dtype))
            ).view(1, -1, 1)
            combined_embedding = torch.cat([combined_embedding, n_scaled], dim=2)  # [1, batch_size, D+1]

        combined_embeddings.append(combined_embedding)
        labels = torch.FloatTensor([graph.label for graph in batch_graphs]).to(device)
        all_labels.append(labels)

    final_labels = torch.cat(all_labels, dim=0)
    final_embeddings = torch.cat(combined_embeddings, dim=1)
    return final_embeddings, final_labels