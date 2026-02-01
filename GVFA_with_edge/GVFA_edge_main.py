'''
Graph-based VSA model with edge-conditioned message passing.
Node features: 7 atom features. Edge features: bond type, conjugated, ring, length.
For each bond (u,v), messages combine neighbour atom with that bond and are sent
along the bond. Use "sol" python environment.
'''


from src.create_graphs import create_graph_list
from src.load_data import load_data
from src.VSA_conversion import VSA_conversion
from src.embeddings import getEmbedding
from models.graphcnnVSA_Binding_FULL import GraphCNN
import torch
from xgboost import XGBRegressor
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


train_data, test_data = load_data()

# print(train_data[0].)

train_graphs = create_graph_list(train_data)
test_graphs = create_graph_list(test_data)



num_layers = 5
delta_eq1 = 1
equation_eq1 = 10
graph_pooling_type = 'sum'  # sum, average
neighbor_pooling_type = 'sum' # sum, average, max
device = 1  # help='if delta is 1 will be the model with binding, if 0 model will have be without binding (default: 1)'
device = torch.device('cpu')

dims = [1000, 2000, 5000, 10000]

# projection_type: "orthogonal" (info-preserving) or "gaussian"
# edge_projection_type: same for GraphCNN edge_attr -> HV (single place for edge conditioning)
for dim in dims:
    train_graphs = create_graph_list(train_data)
    test_graphs = create_graph_list(test_data)
    ts_graph = test_graphs.copy()
    tr_graph = train_graphs.copy()
    test_HVs = VSA_conversion(ts_graph, dim, projection_type="orthogonal")
    train_HVs = VSA_conversion(tr_graph, dim, projection_type="orthogonal")

    model_eq1 = GraphCNN(
        test_HVs[0].node_features.shape[1], num_layers, delta_eq1,
        graph_pooling_type, neighbor_pooling_type, device, equation_eq1,
        edge_feat_dim=5,
        edge_projection_type="orthogonal",
    )
    train_embeddings_eq1, train_labels_eq1 = getEmbedding(model_eq1, device, train_HVs)
    test_embeddings_eq1, test_labels_eq1 = getEmbedding(model_eq1, device, test_HVs)

    train_embeddings_eq1 = train_embeddings_eq1.squeeze(0)

    test_embeddings_eq1 = test_embeddings_eq1.squeeze(0)

    # print(len(test_graphs))
    # print(test_graphs[0].node_features)
    # print(train_embeddings_eq1.shape, train_labels_eq1.shape)

    xgb = XGBRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        random_state=42,
        n_jobs=4,
        tree_method="hist"   # fast on CPU; use "gpu_hist" if you have GPU
    )

    xgb.fit(
        train_embeddings_eq1, train_labels_eq1,
        eval_set=[(test_embeddings_eq1, test_labels_eq1)],
        # early_stopping_rounds=100,
        verbose=False
    )

    # ---------- Evaluate ----------
    pred = xgb.predict(test_embeddings_eq1)
    rmse = mean_squared_error(test_labels_eq1, pred)
    mae  = mean_absolute_error(test_labels_eq1, pred)
    r2   = r2_score(test_labels_eq1, pred)

    print(f"Dimention,{dim},MAE,{mae},RMSE,{rmse},R2,{r2}")

    del xgb
    del train_embeddings_eq1
    del test_embeddings_eq1
    del test_labels_eq1
    del train_labels_eq1
    del model_eq1
    del test_HVs
    del train_HVs