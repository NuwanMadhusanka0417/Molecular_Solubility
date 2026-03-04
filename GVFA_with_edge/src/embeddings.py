import torch


def getEmbedding(model, device, train_graphs, batch_size=100, SUM=True,
                 use_size_aware=True, hop_alpha=1.0):
    """
    Get graph-level embeddings for regression.

    The model returns [1, B, D] with role-separated hop combination already
    applied internally, so no layer-wise weighting is needed here.

    use_size_aware: if True (default):
        1) Scale each graph's embedding by 1/sqrt(num_nodes).
        2) Append num_nodes as an extra feature (last column) → D+1 dims.
    """
    model.to(device)
    model.eval()

    combined_embeddings = []
    all_labels = []

    num_graphs = len(train_graphs)
    for start_idx in range(0, num_graphs, batch_size):
        end_idx = min(start_idx + batch_size, num_graphs)
        batch_graphs = train_graphs[start_idx:end_idx]

        output = model(batch_graphs)  # [1, B, D]

        # if use_size_aware:
        #     num_nodes = torch.tensor(
        #         [len(g.g) for g in batch_graphs],
        #         dtype=output.dtype,
        #         device=output.device,
        #     )
        #     scale = (num_nodes ** 0.5).clamp(min=1e-6).view(1, -1, 1)
        #     output = output / scale

        # combined_embedding = output  # [1, B, D]

        # if use_size_aware:
        #     n = torch.tensor(
        #         [len(g.g) for g in batch_graphs],
        #         dtype=combined_embedding.dtype,
        #         device=combined_embedding.device,
        #     ).view(1, -1, 1)
        #     combined_embedding = torch.cat([combined_embedding, n], dim=2)

        n = torch.tensor(
            [len(g.g) for g in batch_graphs],
            dtype=output.dtype, device=output.device,
        ).view(1, -1, 1)
        combined_embedding = torch.cat([output, n], dim=2)  # [1, B, D+1]
        
        combined_embeddings.append(combined_embedding)
        labels = torch.FloatTensor([graph.label for graph in batch_graphs]).to(device)
        all_labels.append(labels)

    final_labels = torch.cat(all_labels, dim=0)
    final_embeddings = torch.cat(combined_embeddings, dim=1)
    return final_embeddings, final_labels