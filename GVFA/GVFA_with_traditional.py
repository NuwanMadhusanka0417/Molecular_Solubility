from src.create_graphs import create_graph_list
from src.load_data import load_data
from src.VSA_conversion import VSA_conversion
from src.embeddings import getEmbedding
from models.graphcnnVSA_Binding_FULL import GraphCNN
import torch
from xgboost import XGBRegressor
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors,rdMolDescriptors

from rdkit import Chem,DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Descriptors
from rdkit.Chem import Lipinski
from rdkit.Chem import Crippen
import argparse

### Importing the required library 

import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import MolStandardize
import joblib
from src import utilities

parser = argparse.ArgumentParser()
parser.add_argument("--hv_dim", type=int, required=True,
                    help="Hypervector dimension")
args = parser.parse_args()

HV_Dimention = args.hv_dim

train_set=pd.read_csv('final_data/final_unique_train_fixed.csv')
test_set=pd.read_csv('final_data/final_unique_test.csv')

train_smiles_list=train_set[['smiles_canon']]
test_smiles_list=test_set[['smiles_canon']]


### Generate 4 descriptors ....
# df4_train=utilities.generate4(train_set.smiles_canon)
# df4_test=utilities.generate4(test_set.smiles_canon)
# ### Generate 17 descriptors ....
# df17_train=utilities.generate17(train_set.smiles_canon)
# df17_test=utilities.generate17(test_set.smiles_canon)
### Generate 123 descriptors ....'''
df123_train=utilities.generate123(train_set.smiles_canon)
df123_test=utilities.generate123(test_set.smiles_canon)
### Generate 38 feature engineered based on the structure of the smiles ....
df38_train=utilities.generate_features38(train_set.smiles_canon)
df38_test=utilities.generate_features38(test_set.smiles_canon)
### Generate 7 funnctional groups
df7_train=utilities.get_functional_groups(train_set.smiles_canon)
df7_test=utilities.get_functional_groups(test_set.smiles_canon)
### Fingerprint 128....
df128_train=utilities.fingerprint(train_set.smiles_canon,2,128)
df128_test=utilities.fingerprint(test_set.smiles_canon,2,128)


## Prof. Ulf proposed data
df96_train=utilities.generate_desc_96(train_set.smiles_canon)
df96_test=utilities.generate_desc_96(test_set.smiles_canon)

## Prof. Ulf + chatgpt suggested data
df193_train=utilities.generate_desc_193(train_set.smiles_canon)
df193_test=utilities.generate_desc_193(test_set.smiles_canon)

df298_train=pd.concat([df123_train, df128_train, df7_train, df38_train], axis=1)
df298_test=pd.concat([df123_test, df128_test, df7_test, df38_test], axis=1)

# HV_Dimentions = [100, 500, 1000, 2000, 5000, 10000]

scaler_298 = StandardScaler()
scaler_298.fit(df298_train.values)   # each column: its own mean/std


# for HV_Dimention in HV_Dimentions:

train_data, test_data = load_data()
print(train_data[0].edge_attr)

# train_graphs = create_graph_list(train_data)
# test_graphs = create_graph_list(test_data)


num_layers = 5
delta_eq1 = 1
equation_eq1 = 10
graph_pooling_type = 'sum'  # sum, average
neighbor_pooling_type = 'sum' # sum, average, max
device = 1  # help='if delta is 1 will be the model with binding, if 0 model will have be without binding (default: 1)'
device = torch.device('cpu')

train_graphs = create_graph_list(train_data)
test_graphs = create_graph_list(test_data)
ts_graph = test_graphs.copy()
tr_graph = train_graphs.copy()

test_HVs = VSA_conversion(ts_graph, HV_Dimention)
train_HVs = VSA_conversion(tr_graph, HV_Dimention)

model_eq1 = GraphCNN(test_HVs[0].node_features.shape[1], num_layers, delta_eq1, graph_pooling_type, neighbor_pooling_type, device, equation_eq1) #.to(device)
train_embeddings_eq1, train_labels_eq1 = getEmbedding(model_eq1, device, train_HVs)
test_embeddings_eq1, test_labels_eq1 = getEmbedding(model_eq1, device, test_HVs)

train_embeddings_eq1 = train_embeddings_eq1.squeeze(0)
test_embeddings_eq1 = test_embeddings_eq1.squeeze(0)

df298_train_scaled = scaler_298.transform(df298_train.values)
df298_test_scaled  = scaler_298.transform(df298_test.values)


df_torch_train = torch.from_numpy(df298_train_scaled.astype(np.float32))
df_torch_test  = torch.from_numpy(df298_test_scaled.astype(np.float32))


X_train = torch.cat([df_torch_train, train_embeddings_eq1], axis=1)

# X_train = pd.concat([df_t, train_embeddings_eq1], axis=1)
X_test = torch.cat([df_torch_test, test_embeddings_eq1], axis=1)

print(X_train.shape)
print(X_test.shape)

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
    X_train, train_labels_eq1,
    eval_set=[(X_test, test_labels_eq1)],
    # early_stopping_rounds=100,
    verbose=False
)

pred_xgb = xgb.predict(X_test)

xgb_298=utilities.get_errors1(test_labels_eq1,pred_xgb,f"XGB_298 concatinate GVFA({HV_Dimention})")
xgb_298['Descriptors_Detail']='125 features + 128 fingerprint 7 f_group+38 fe features'
print(xgb_298)