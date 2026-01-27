import numpy as np
import torch


df298_train_scaled = np.load("data/X298_train_scaled.npy")   # shape [N_train, 298]
df298_test_scaled  = np.load("data/X298_test_scaled.npy")    # shape [N_test, 298]


df_torch_train = torch.from_numpy(df298_train_scaled.astype(np.float32))
df_torch_test  = torch.from_numpy(df298_test_scaled.astype(np.float32))

print(df_torch_test)