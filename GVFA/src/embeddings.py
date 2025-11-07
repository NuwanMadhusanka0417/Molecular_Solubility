import torch

def getEmbedding( model, device, train_graphs, batch_size=100, SUM = True):

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
    return final_embeddings, final_labels