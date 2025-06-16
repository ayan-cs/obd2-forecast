import numpy as np
import torch, os, sys, gc
from torch.utils.data import Dataset

from models import *

def createSplit(X_l, X_r, test_size):
    # X_l = np.array(X_l)
    # X_r = np.array(X_r)
    num_samples = len(X_l)
    test_size = int(num_samples * test_size)
    random_idx = np.random.permutation(num_samples)
    X_train_left = []
    X_train_right = []
    X_val_left = []
    X_val_right = []

    for idx in random_idx[test_size:]:
        X_train_left.append(X_l[idx])
        X_train_right.append(X_r[idx])
    
    for idx in random_idx[:test_size]:
        X_val_left.append(X_l[idx])
        X_val_right.append(X_r[idx])

    return X_train_left, X_train_right, X_val_left, X_val_right

def createOBDchunks(obd_path, context=99, test_size=0.15, verbose=True):
    # parent = os.path.abspath('')
    # obd_path = os.path.join(parent, 'datasets', 'obd_driverwise_data')
    # all_files = os.listdir(obd_path)
    all_files = [c for c in os.listdir(obd_path) if 'driver' in c]
    X_left = []
    X_right = []
    for f in all_files:
        if 'driver' not in f:
            continue
        if 'driver6' in f:
            continue
        if verbose:
            print(f"Loading \"{f}\" ...", end='\t', flush=True)
        X = np.load(os.path.join(obd_path, f), mmap_mode='r').tolist()
        if context > len(X):
            raise ValueError(f"Context value cannot exceed minimum length data. Retry with a value <={len(X)}.", flush=True)
        if context < 2 :
            raise ValueError(f"Context value should have minimum length = 2. Retry with a value >= 2.", flush=True)
        context_l = context - 1
        
        for i in range(len(X) - context + 1):
            X_left.append(X[i : i+context_l])
            X_right.append(X[i+context_l : i+context])
        if verbose:
            print(f"Complete!", flush=True)
        del X
        gc.collect()
    print(f"X_left : {len(X_left),len(X_left[0]),len(X_left[0][0])}\tX_right : {len(X_right),len(X_right[0]),len(X_right[0][0])}", flush=True)
    X_train_left, X_train_right, X_test_left, X_test_right = createSplit(X_left, X_right, test_size)
    return X_train_left, X_train_right, X_test_left, X_test_right

class EarlyStopper :
    def __init__(self, patience=100, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_loss = np.inf

    def early_stop(self, loss):
        if loss < self.min_loss:
            self.min_loss = loss
            self.counter = 0
        elif loss > (self.min_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

class TimeSeriesDataset(Dataset):
    def __init__(self, X_left, X_right):
        self.X_left = X_left      # Input sequences
        self.X_right = X_right    # Ground truth values
        assert len(X_left) == len(X_right), "Mismatch in input-output lengths"

    def __len__(self):
        return len(self.X_left)

    def __getitem__(self, idx):
        # Convert only when accessed (Lazy Loading)
        X = torch.FloatTensor(self.X_left[idx])
        y = torch.FloatTensor(self.X_right[idx])
        return X, y

# Load model from checkpoint
def load_model(model_type, artifact_path, metadata):
    if model_type == 'lstm':
        model = LSTM(
            input_size=metadata['model']['input_size'],
            hidden_size=metadata['model']['hidden_size'],
            num_layers=metadata['model']['num_layers'],
            bidirectional=metadata['model']['bidirectional']
        ).cuda()
        model.load_state_dict(torch.load(os.path.join(artifact_path, 'checkpoint.pth'), weights_only=True))
    
    elif model_type == 'tf':
        model = TransformerTSF(
            input_size=metadata['model']['input_size'],
            d_model=metadata['model']['d_model'],
            nhead=metadata['model']['nhead'],
            num_layers=metadata['model']['num_layers']
        ).cuda()
        model.load_state_dict(torch.load(os.path.join(artifact_path, 'checkpoint.pth'), weights_only=True))
    
    else:
        raise TypeError(f"Model type '{model_type}' is invalid.")
    
    return model