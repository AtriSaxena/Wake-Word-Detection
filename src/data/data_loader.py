import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys 
sys.path.append("..")
from utils.util import *
from torch.utils.data import DataLoader
BATCH_SIZE = 32

def torch_dataloader(X_train, y_train, X_test, y_test):
    train_dataloader = DataLoader(list(zip(X_train, y_train)), batch_size= BATCH_SIZE, shuffle=True) 
    test_dataloader = DataLoader(list(zip(X_test, y_test)), batch_size= BATCH_SIZE, shuffle=False)

    print(f"Length of train dataloader: {len(train_dataloader)} batches of {BATCH_SIZE}")
    print(f"Length of validation dataloader: {len(test_dataloader)} batches of {BATCH_SIZE}")

    return train_dataloader, test_dataloader

# train_dataloader, test_dataloader = torch_dataloader() 
# train_feature_batch, train_labels_batch = next(iter(train_dataloader))
# print(train_feature_batch.shape, train_labels_batch.shape)