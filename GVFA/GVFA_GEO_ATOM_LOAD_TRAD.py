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
from sklearn.preprocessing import StandardScaler
from models.graphcnnVSA_Binding_FULL import GraphCNN
# from src import utilities

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



# HV_dim = 2000
HV_dims = [100, 500, 1000, 5000, 10000]
for HV_dim in HV_dims:
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

    train_geognn, train_labels_geognn = getEmbedding_geognn(model_eq1, device, train_graphs_bond)
    test_geognn, test_labels_geognn = getEmbedding_geognn(model_eq1, device, test_graphs_bond)
    train_geognn = train_geognn.squeeze(0)
    test_geognn = test_geognn.squeeze(0)

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
    train_atom_bond, train_labels_atom_bond = getEmbedding(model_eq1, device, train_HVs)
    test_atom_bond, test_labels_atom_bond = getEmbedding(model_eq1, device, test_HVs)
    train_atom_bond = train_atom_bond.squeeze(0)
    test_atom_bond = test_atom_bond.squeeze(0)

    combined_train_atom_bond_geognn = torch.cat(
        [train_atom_bond, train_geognn],
        dim=1
    )  # shape [17916, D1 + D2]

    combined_test_atom_bond_geognn = torch.cat(
        [test_atom_bond, test_geognn],
        dim=1
    )

    #######################   Traditional Feature set
    '''
    # train_set=pd.read_csv('final_data/final_unique_train_fixed.csv')
    # test_set=pd.read_csv('final_data/final_unique_test.csv')

    # train_smiles_list=train_set[['smiles_canon']]
    # test_smiles_list=test_set[['smiles_canon']]


    # ### Generate 4 descriptors ....
    # df4_train=utilities.generate4(train_set.smiles_canon)
    # df4_test=utilities.generate4(test_set.smiles_canon)
    # ### Generate 17 descriptors ....
    # df17_train=utilities.generate17(train_set.smiles_canon)
    # df17_test=utilities.generate17(test_set.smiles_canon)
    # ### Generate 123 descriptors ....
    # df123_train=utilities.generate123(train_set.smiles_canon)
    # df123_test=utilities.generate123(test_set.smiles_canon)
    # ### Generate 38 feature engineered based on the structure of the smiles ....
    # df38_train=utilities.generate_features38(train_set.smiles_canon)
    # df38_test=utilities.generate_features38(test_set.smiles_canon)
    # ### Generate 7 funnctional groups
    # df7_train=utilities.get_functional_groups(train_set.smiles_canon)
    # df7_test=utilities.get_functional_groups(test_set.smiles_canon)
    # ### Fingerprint 128....
    # df128_train=utilities.fingerprint(train_set.smiles_canon,2,128)
    # df128_test=utilities.fingerprint(test_set.smiles_canon,2,128)

    # df298_train=pd.concat([df123_train, df128_train, df7_train, df38_train], axis=1)
    # df298_test=pd.concat([df123_test, df128_test, df7_test, df38_test], axis=1)

    # scaler_298 = StandardScaler()
    # scaler_298.fit(df298_train.values) 

    # df298_train_scaled = scaler_298.transform(df298_train.values)
    # df298_test_scaled  = scaler_298.transform(df298_test.values)'''

    df298_train_scaled = np.load("data/X298_train_scaled.npy")   # shape [N_train, 298]
    df298_test_scaled  = np.load("data/X298_test_scaled.npy")    # shape [N_test, 298]


    df_torch_train = torch.from_numpy(df298_train_scaled.astype(np.float32))
    df_torch_test  = torch.from_numpy(df298_test_scaled.astype(np.float32))

    df_torch_train = [df_torch_train[i] for i in train_data_geognn.good_idx]
    df_torch_test = [df_torch_test[i] for i in test_data_geognn.good_idx]

    train_traditional = torch.stack(df_torch_train, dim=0)  # [N, 298]
    test_traditional  = torch.stack(df_torch_test,  dim=0)

    combined_train_atom_geognn_trad = torch.cat([train_traditional, combined_train_atom_bond_geognn], axis=1)

    # X_train = pd.concat([df_t, train_embeddings_eq1], axis=1)
    combined_test_atom_geognn_trad = torch.cat([test_traditional, combined_test_atom_bond_geognn], axis=1)

    combined_train_atom_trad = torch.cat([train_traditional, train_atom_bond], axis=1)
    combined_test_atom_trad = torch.cat([test_traditional, test_atom_bond], axis=1)


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

    xgb_atom_angle = XGBRegressor(
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

    xgb_atom_trad = XGBRegressor(
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

    xgb_atom_angle_trad = XGBRegressor(
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


    ###################################    Clssify Atom - Bond
    '''xgb.fit(
        train_atom_bond, train_labels_atom_bond,
        eval_set=[(test_atom_bond, test_labels_atom_bond)],
        # early_stopping_rounds=100,
        verbose=False
    )
    pred = xgb.predict(test_atom_bond)
    rmse = mean_squared_error(test_labels_atom_bond, pred)
    mae  = mean_absolute_error(test_labels_atom_bond, pred)
    r2   = r2_score(test_labels_atom_bond, pred)
    print("Dimention: ", HV_dim)
    print("Atom-Bond")
    print("MAE      RMSE      R2")
    print(mae,"     ",rmse, "       ",r2)'''

    ###################################    Clssify GEO

    xgb_geognn.fit(
        train_geognn, train_labels_geognn,
        eval_set=[(test_geognn, test_labels_geognn)],
        # early_stopping_rounds=100,
        verbose=False
    )
    pred_geognn = xgb_geognn.predict(test_geognn)
    rmse = mean_squared_error(test_labels_geognn, pred_geognn)
    mae  = mean_absolute_error(test_labels_geognn, pred_geognn)
    r2   = r2_score(test_labels_geognn, pred_geognn)
    print("Dimention: ", HV_dim)
    print("Atom-Angle")
    print("MAE      RMSE      R2")
    print(mae,"     ",rmse, "       ",r2)

    ###################################    Clssify BOTH

    xgb_atom_angle.fit(
        combined_train_atom_bond_geognn, train_labels_geognn,
        eval_set=[(combined_test_atom_bond_geognn, test_labels_geognn)],
        # early_stopping_rounds=100,
        verbose=False
    )
    pred = xgb_atom_angle.predict(combined_test_atom_bond_geognn)
    rmse = mean_squared_error(test_labels_geognn, pred)
    mae  = mean_absolute_error(test_labels_geognn, pred)
    r2   = r2_score(test_labels_geognn, pred)

    print("Dimention: ", HV_dim)
    print("Atom-Bond + Atom-Angle")

    print("MAE      RMSE      R2")
    print(mae,"     ",rmse, "       ",r2)


    
    ###################################    Clssify ATom-bond + traditional

    '''xgb_atom_trad.fit(
        combined_train_atom_trad, train_labels_geognn,
        eval_set=[(combined_test_atom_trad, test_labels_geognn)],
        # early_stopping_rounds=100,
        verbose=False
    )
    pred = xgb_atom_trad.predict(combined_test_atom_trad)
    rmse = mean_squared_error(test_labels_geognn, pred)
    mae  = mean_absolute_error(test_labels_geognn, pred)
    r2   = r2_score(test_labels_geognn, pred)

    print("Dimention: ", HV_dim)
    print("Atom-Bond + Traditional")

    print("MAE      RMSE      R2")
    print(mae,"     ",rmse, "       ",r2)'''
    

    ###################################    Atom-bond + bond-angle + Tradisional

    xgb_atom_angle_trad.fit(
        combined_train_atom_geognn_trad, train_labels_geognn,
        eval_set=[(combined_test_atom_geognn_trad, test_labels_geognn)],
        # early_stopping_rounds=100,
        verbose=False
    )

    pred = xgb_atom_angle_trad.predict(combined_test_atom_geognn_trad)
    rmse = mean_squared_error(test_labels_geognn, pred)
    mae  = mean_absolute_error(test_labels_geognn, pred)
    r2   = r2_score(test_labels_geognn, pred)

    print("Dimention: ", HV_dim)
    print("Atom-bond + bond-angle + Tradisional")

    print("MAE      RMSE      R2")
    print(mae,"     ",rmse, "       ",r2)