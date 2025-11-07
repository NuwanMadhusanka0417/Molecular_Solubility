from src.new_data_load import load_s2v_from_csv #EnrichedFromCSV
from src.VSA_conversion import VSA_conversion
from src.embeddings import getEmbedding
from models.graphcnnVSA_Binding_FULL import GraphCNN
import torch
import numpy as np
from xgboost import XGBRegressor
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Example:
# dataset_train = EnrichedFromCSV("final_data/final_unique_train.csv")
# dataset_test  = EnrichedFromCSV("final_data/final_unique_test.csv")

g_train, vocab = load_s2v_from_csv("final_data/final_unique_train_fixed.csv")
g_test, _     = load_s2v_from_csv("final_data/final_unique_test.csv")
print(type(g_train[0]), g_train[0].node_features.shape, g_train[0].edge_mat.shape, g_train[0].label)

print(g_test[1].node_features)
print(g_test[1].edge_mat)
print(g_test[1].label)
print(g_test[1].max_neighbor)

train_HVs = VSA_conversion(g_train, 1000)
test_HVs = VSA_conversion(g_test, 1000)

num_layers = 4
delta_eq1 = 1
equation_eq1 = 11
graph_pooling_type = 'sum'  # sum, average
neighbor_pooling_type = 'sum' # sum, average, max
device = 1  # help='if delta is 1 will be the model with binding, if 0 model will have be without binding (default: 1)'
device = torch.device('cpu')

model_eq1 = GraphCNN(test_HVs[0].node_features.shape[1], num_layers, delta_eq1, graph_pooling_type, neighbor_pooling_type, device, equation_eq1) #.to(device)
train_embeddings_eq1, train_labels_eq1 = getEmbedding(model_eq1, device, train_HVs)
test_embeddings_eq1, test_labels_eq1 = getEmbedding(model_eq1, device, test_HVs)

train_embeddings_eq1 = train_embeddings_eq1.squeeze(0)
test_embeddings_eq1 = test_embeddings_eq1.squeeze(0)

print(test_embeddings_eq1.shape, test_labels_eq1.shape)


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

print("RMSE: ", rmse)
print("MAE: ", mae)
print("R2: ", r2)