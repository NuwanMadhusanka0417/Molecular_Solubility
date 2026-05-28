'''
This code use 7 features of atoms as node features. 
USe "sol" python environment 
'''


from src.create_graphs import create_graph_list
from src.load_data import load_data
from src.embeddings import getEmbedding
from models.graphcnnVSA_Binding_FULL import GraphCNN
import torch
from xgboost import XGBRegressor
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from src.VSA_conversion import VSA_conversion, make_random_projection, project_node_features_with_W
import math


train_data, test_data = load_data(
    dataset="solubility_novel",
    train_path="final_data/solubility_1.csv",
    test_path="final_data/testset_novel.csv",
)


train_graphs = create_graph_list(train_data)
test_graphs = create_graph_list(test_data)



num_layers = 5
delta_eq1 = 1
equation_eq1 = 10
graph_pooling_type = 'sum'  # sum, average
neighbor_pooling_type = 'sum' # sum, average, max
device = 1  # help='if delta is 1 will be the model with binding, if 0 model will have be without binding (default: 1)'
device = torch.device('cpu')

dims = [2000, 5000, 10000]

for dim in dims:
    train_graphs = create_graph_list(train_data)
    test_graphs = create_graph_list(test_data)
    ts_graph = test_graphs.copy()
    tr_graph = train_graphs.copy()

    train_graphs  = VSA_conversion(tr_graph, dim)
    test_graphs   = VSA_conversion(ts_graph, dim)

    # create W ONCE per HV_dim
    F_in = train_graphs[0].node_features.shape[1]
    W = make_random_projection(F_in, dim, seed=0, device="cpu")

    # apply SAME W to train and test
    train_graphs = project_node_features_with_W(train_graphs, W)
    test_graphs  = project_node_features_with_W(test_graphs,  W)

    model_eq1 = GraphCNN(test_graphs[0].node_features.shape[1], num_layers, delta_eq1, graph_pooling_type, neighbor_pooling_type, device, equation_eq1) #.to(device)
    # train_embeddings_eq1, train_labels_eq1 = getEmbedding(model_eq1, device, train_HVs)
    # test_embeddings_eq1, test_labels_eq1 = getEmbedding(model_eq1, device, test_HVs)
    train_embeddings_eq1, train_labels_eq1 = getEmbedding(model_eq1, device, train_graphs, layer_reduce="sum")
    test_embeddings_eq1,  test_labels_eq1  = getEmbedding(model_eq1, device, test_graphs,  layer_reduce="sum")



    # train_embeddings_eq1 = train_embeddings_eq1.squeeze(0)

    # test_embeddings_eq1 = test_embeddings_eq1.squeeze(0)

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

    # xgb.fit(
    #     train_embeddings_eq1, train_labels_eq1,
    #     eval_set=[(test_embeddings_eq1, test_labels_eq1)],
    #     # early_stopping_rounds=100,
    #     verbose=False
    # )
    xgb.fit(train_embeddings_eq1.numpy(), train_labels_eq1.numpy())

    # ---------- Evaluate on test set ----------
    y_true = test_labels_eq1.numpy().ravel()
    pred = xgb.predict(test_embeddings_eq1.numpy())
    mae = mean_absolute_error(y_true, pred)
    rmse = math.sqrt(mean_squared_error(y_true, pred))
    r2 = r2_score(y_true, pred)
    pearson_r, _ = pearsonr(y_true, pred)

    print(f"Dimension {dim} | MAE {mae:.4f} | RMSE {rmse:.4f} | Pearson r {pearson_r:.4f} | R2 {r2:.4f}")

    del xgb
    del train_embeddings_eq1
    del test_embeddings_eq1
    del test_labels_eq1
    del train_labels_eq1
    del model_eq1