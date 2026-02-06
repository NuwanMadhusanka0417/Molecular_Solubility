import torch

def getEmbedding(model, device, train_graphs, batch_size=100, use_size_aware=True):
    """
    Get graph-level embeddings for regression.

    The model now returns a single graph embedding per graph [batch, D].

    use_size_aware: if True (default), applies two changes:
        1) Scale each graph's pooled vector by 1/√(num_nodes), so larger molecules
           don't dominate the sum-pooled representation.
        2) Append num_nodes as an extra feature (last column). XGBoost then gets D+1
           dimensions; the last is atom count.
    """
    model.to(device)
    model.train()

    combined_embeddings = []
    all_labels = []

    num_graphs = len(train_graphs)
    for start_idx in range(0, num_graphs, batch_size):
        end_idx = min(start_idx + batch_size, num_graphs)
        batch_graphs = train_graphs[start_idx:end_idx]

        # output: [batch_size, D]
        output = model(batch_graphs)

        # Size-aware scaling by 1/sqrt(num_nodes)
        if use_size_aware:
            num_nodes = torch.tensor(
                [len(g.g) for g in batch_graphs],
                dtype=output.dtype,
                device=output.device,
            ).view(-1, 1)  # [batch, 1]
            scale = (num_nodes ** 0.5).clamp(min=1e-6)
            output = output / scale

        combined_embedding = output.unsqueeze(0)  # [1, batch, D]

        # Append num_nodes as extra column so XGBoost gets explicit size feature (D+1 input)
        if use_size_aware:
            n = num_nodes.view(1, -1, 1)
            combined_embedding = torch.cat([combined_embedding, n], dim=2)  # [1, batch, D+1]

        combined_embeddings.append(combined_embedding)
        labels = torch.FloatTensor([graph.label for graph in batch_graphs]).to(device)
        all_labels.append(labels)

    final_labels = torch.cat(all_labels, dim=0)
    final_embeddings = torch.cat(combined_embeddings, dim=1)  # [1, N, D] (or D+1)
    return final_embeddings, final_labels