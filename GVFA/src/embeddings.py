import torch

'''def getEmbedding( model, device, train_graphs, batch_size=100, SUM = True):

    model.to(device)
    model.train()

    combined_embeddings = []  # Initialize the total embedding
    all_labels = []

    # Create batches
    num_graphs = len(train_graphs)
    for start_idx in range(0, num_graphs, batch_size):
        end_idx = min(start_idx + batch_size, num_graphs)
        batch_graphs = train_graphs[start_idx:end_idx]
        # print("getEmbedding :: Before BBBBB")
        output = model(batch_graphs)
        
        # print(output.shape)

        ################### For regression taskuse TRUE and false both as sum ################

        
        if(SUM==True):    # allways use SUM
            # Sum all embeddings
            combined_embedding = output.sum(dim=0, keepdim=True)   #torch.sum(torch.stack(output), dim=0)  # Sum along the new batch dimension
            # 100 tensors
        else:
            #Concat
            combined_embedding = torch.cat(output, dim=1)
        
        # Add the summed embeddings of this batch to the total embedding
        #############################################################################
        # combined_embeddings.append(output) #combined_embedding)
        combined_embeddings.append(combined_embedding)

        ####################################Place conmvert label########
        # Collect labels
        labels = torch.FloatTensor([graph.label for graph in batch_graphs]).to(device)
        all_labels.append(labels)


    final_labels = torch.cat(all_labels, dim=0)
    final_embeddings = torch.cat(combined_embeddings, dim=1)

    # print("getEmbedding :: endo")
    return final_embeddings, final_labels'''


def getEmbedding(model, device, graphs, batch_size=100, layer_reduce="sum", debug=False):
    """
    Extract per-graph embeddings.

    GraphCNN in your project returns: [L, B, D] (L = num_layers)
    This function reduces over L to produce a stable:
      - [B, D] for sum/mean/last
      - [B, L*D] for concat

    Args:
        layer_reduce: "sum" | "mean" | "last" | "concat"
        debug: print shapes for troubleshooting

    Returns:
        embeddings: [N, D] or [N, L*D]
        labels:     [N]
    """
    model = model.to(device)
    model.eval()

    all_emb = []
    all_y = []

    with torch.no_grad():
        for start in range(0, len(graphs), batch_size):
            batch = graphs[start:start + batch_size]
            out = model(batch)

            # Normalize list/tuple -> tensor
            if isinstance(out, (list, tuple)):
                out = torch.stack(out, dim=0)

            if not torch.is_tensor(out):
                raise TypeError(f"Model output must be a tensor/list/tuple, got {type(out)}")

            # Handle GraphCNN outputs
            # Expected: [L, B, D]
            if out.dim() == 3:
                if layer_reduce == "sum":
                    out = out.sum(dim=0)            # [B, D]
                elif layer_reduce == "mean":
                    out = out.mean(dim=0)           # [B, D]
                elif layer_reduce == "last":
                    out = out[-1]                   # [B, D]
                elif layer_reduce == "concat":
                    # [L, B, D] -> [B, L, D] -> [B, L*D]
                    out = out.permute(1, 0, 2).reshape(out.size(1), -1)
                else:
                    raise ValueError("layer_reduce must be one of: sum, mean, last, concat")

            elif out.dim() == 2:
                # Already [B, D] (fine)
                pass
            elif out.dim() == 1:
                # Single vector [D] -> [1, D]
                out = out.unsqueeze(0)
            else:
                raise RuntimeError(f"Unexpected model output shape: {tuple(out.shape)}")

            if debug:
                print(f"start={start} batch_size={len(batch)} out_shape={tuple(out.shape)}")

            all_emb.append(out.detach().cpu())
            all_y.append(torch.tensor([g.label for g in batch], dtype=torch.float32))

    embeddings = torch.cat(all_emb, dim=0)  # [N, D] (or [N, L*D])
    labels = torch.cat(all_y, dim=0)        # [N]
    return embeddings, labels
