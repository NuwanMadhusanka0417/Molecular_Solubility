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
import numpy as np
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ----- Dataset choice -----
# "old": final_data/solubility_1.csv (SMILES, logS), 90/10 train/test split
# "new": final_data/final_unique_train.csv + final_unique_test.csv (smiles_canon, LogS)
#        Override paths/columns via load_data(train_path=..., test_path=..., smiles_col=..., target_col=...)
DATASET = "old"

train_data, test_data = load_data(dataset=DATASET)

# print(train_data[0].)

train_graphs = create_graph_list(train_data)
test_graphs = create_graph_list(test_data)



num_layers = 5
delta_eq1 = 1
equation_eq1 = 10
graph_pooling_type = 'sum'
neighbor_pooling_type = 'sum'
use_reservoir = True   # VSA-RC: tap buffer + Sigma-Pi polynomial expansion
hop_decay = 0.85      # lambda in [0.6, 0.95] for tap buffer (far-hop decay)
sigma_pi_orders = [0, 1,2]  # T={0,1}: 1st + 2nd order (sweet spot); {0,1,2} for 3rd order
use_ridge = True      # Ridge regression (RC standard); False = XGBoost
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
        use_reservoir=use_reservoir,
        hop_decay=hop_decay,
        sigma_pi_orders=sigma_pi_orders,
    )
    # use_size_aware: scale by 1/√(num_nodes) + append num_nodes; hop_alpha=1 when use_reservoir (single vector)
    train_embeddings_eq1, train_labels_eq1 = getEmbedding(
        model_eq1, device, train_HVs, use_size_aware=True, hop_alpha=1.0
    )
    test_embeddings_eq1, test_labels_eq1 = getEmbedding(
        model_eq1, device, test_HVs, use_size_aware=True, hop_alpha=1.0
    )

    train_embeddings_eq1 = train_embeddings_eq1.squeeze(0)  # [N_train, D] or [N_train, D+1]
    test_embeddings_eq1 = test_embeddings_eq1.squeeze(0)     # [N_test, D] or [N_test, D+1]

    # print(len(test_graphs))
    # print(test_graphs[0].node_features)
    # print(train_embeddings_eq1.shape, train_labels_eq1.shape)

    if use_ridge:
        # Ridge regression (RC standard): w = (G'G + alpha*I)^{-1} G' y
        reg = RidgeCV(alphas=np.logspace(-4, 2, 50), cv=5, scoring='neg_mean_squared_error')
        reg.fit(train_embeddings_eq1, train_labels_eq1)
        pred = reg.predict(test_embeddings_eq1)
    else:
        from xgboost import XGBRegressor
        reg = XGBRegressor(
            n_estimators=2000,
            learning_rate=0.03,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            reg_alpha=0.0,
            random_state=42,
            n_jobs=4,
            tree_method="hist"
        )
        reg.fit(
            train_embeddings_eq1, train_labels_eq1,
            eval_set=[(test_embeddings_eq1, test_labels_eq1)],
            verbose=False
        )
        pred = reg.predict(test_embeddings_eq1)

    # ---------- Evaluate ----------
    rmse = mean_squared_error(test_labels_eq1, pred)
    mae  = mean_absolute_error(test_labels_eq1, pred)
    r2   = r2_score(test_labels_eq1, pred)

    print(f"Dimention,{dim},MAE,{mae},RMSE,{rmse},R2,{r2}")

    del reg
    del train_embeddings_eq1
    del test_embeddings_eq1
    del test_labels_eq1
    del train_labels_eq1
    del model_eq1
    del test_HVs
    del train_HVs