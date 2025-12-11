import numpy as np
import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem
import math
import torch
import pandas as pd
from torch_geometric.data import InMemoryDataset
import networkx as nx
import torch
import numpy as np
import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from models.graphcnnVSA_Binding_FULL import GraphCNN


# from src.geaognn_all import load_data_geognn
from src.geaognn_all import (
    load_data_geognn,
    create_bond_graph_list_geognn,
    VSA_conversion_geognn,
    getEmbedding_geognn,
    bond_node_features_geognn,
    smiles_to_bond_angle_data_geognn,
    ZINCLikeCSV_geognn,
    S2VGraph_geognn,
    data_to_S2VGraph_bond_geognn,
    project_node_features_geognn
)

from src.atombond_all import (create_graph_list,
                              VSA_conversion,
                              getEmbedding,
                              load_data)



HV_dim = 2000
num_layers = 5
delta_eq1 = 1
equation_eq1 = 10
graph_pooling_type = 'sum'  # sum, average
neighbor_pooling_type = 'sum' # sum, average, max
device = 1  # help='if delta is 1 will be the model with binding, if 0 model will have be without binding (default: 1)'
device = torch.device('cpu')

##################################   GEO
train_data_geognn, test_data_geognn = load_data_geognn()
train_graphs_bond = create_bond_graph_list_geognn(train_data_geognn)
test_graphs_bond  = create_bond_graph_list_geognn(test_data_geognn)
train_graphs_bond = VSA_conversion_geognn(train_graphs_bond, new_dim=HV_dim)
test_graphs_bond  = VSA_conversion_geognn(test_graphs_bond,  new_dim=HV_dim)

model_eq1 = GraphCNN(train_graphs_bond[0].node_features.shape[1], num_layers, delta_eq1, graph_pooling_type, neighbor_pooling_type, device, equation_eq1)

train_embeddings_eq1_geognn, train_labels_eq1_geognn = getEmbedding_geognn(model_eq1, device, train_graphs_bond)
test_embeddings_eq1_geognn, test_labels_eq1_geognn = getEmbedding_geognn(model_eq1, device, test_graphs_bond)
train_embeddings_eq1_geognn = train_embeddings_eq1_geognn.squeeze(0)
test_embeddings_eq1_geognn = test_embeddings_eq1_geognn.squeeze(0)

####################################### ATOm - BOND

train_data, test_data = load_data()
train_graphs = create_graph_list(train_data)
test_graphs = create_graph_list(test_data)
ts_graph = test_graphs.copy()
tr_graph = train_graphs.copy()
test_HVs_ = VSA_conversion(ts_graph, HV_dim)
train_HVs_ = VSA_conversion(tr_graph, HV_dim)
train_HVs = [train_HVs_[i] for i in train_data_geognn.good_idx]
test_HVs = [test_HVs_[i] for i in test_data_geognn.good_idx]

model_eq1 = GraphCNN(test_HVs[0].node_features.shape[1], num_layers, delta_eq1, graph_pooling_type, neighbor_pooling_type, device, equation_eq1) #.to(device)
train_embeddings_eq1, train_labels_eq1 = getEmbedding(model_eq1, device, train_HVs)
test_embeddings_eq1, test_labels_eq1 = getEmbedding(model_eq1, device, test_HVs)
train_embeddings_eq1 = train_embeddings_eq1.squeeze(0)
test_embeddings_eq1 = test_embeddings_eq1.squeeze(0)


######################################  Classifiers
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

xgb_geognn = XGBRegressor(
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

xgb_all = XGBRegressor(
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

###################################    Clssify ATOM - BOND


xgb.fit(
    train_embeddings_eq1, train_labels_eq1,
    eval_set=[(test_embeddings_eq1, test_labels_eq1)],
    # early_stopping_rounds=100,
    verbose=False
)
pred = xgb.predict(test_embeddings_eq1)
rmse = mean_squared_error(test_labels_eq1, pred)
mae  = mean_absolute_error(test_labels_eq1, pred)
r2   = r2_score(test_labels_eq1, pred)
print("MAE      RMSE      R2")
print(mae,"     ",rmse, "       ",r2)

###################################    Clssify GEO

xgb_geognn.fit(
    train_embeddings_eq1_geognn, train_labels_eq1_geognn,
    eval_set=[(test_embeddings_eq1_geognn, test_labels_eq1_geognn)],
    # early_stopping_rounds=100,
    verbose=False
)
pred_geognn = xgb_geognn.predict(test_embeddings_eq1_geognn)
rmse = mean_squared_error(test_labels_eq1_geognn, pred_geognn)
mae  = mean_absolute_error(test_labels_eq1_geognn, pred_geognn)
r2   = r2_score(test_labels_eq1_geognn, pred_geognn)
print("MAE      RMSE      R2")
print(mae,"     ",rmse, "       ",r2)

###################################    Clssify BOTH

combined_train_embeddings = torch.cat(
    [train_embeddings_eq1, train_embeddings_eq1_geognn],
    dim=1
)  # shape [17916, D1 + D2]

combined_test_embeddings = torch.cat(
    [test_embeddings_eq1, test_embeddings_eq1_geognn],
    dim=1
)

xgb.fit(
    combined_train_embeddings, train_labels_eq1,
    eval_set=[(combined_test_embeddings, test_labels_eq1)],
    # early_stopping_rounds=100,
    verbose=False
)
pred = xgb.predict(combined_test_embeddings)
rmse = mean_squared_error(test_labels_eq1, pred)
mae  = mean_absolute_error(test_labels_eq1, pred)
r2   = r2_score(test_labels_eq1, pred)
print("MAE      RMSE      R2")
print(mae,"     ",rmse, "       ",r2)