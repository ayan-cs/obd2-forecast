import torch, json, os, time, sys, gc
import numpy as np
from itertools import product
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from utils import epoch_time, getModelName, load_data
from models import LSTM
from services import EarlyStopper, TimeSeriesDataset

def train_epoch(model, train_dl, val_dl, criterion, optimizer):
    total_loss = 0
    val_loss = 0
    model.train()
    for X, y in train_dl:
        X = X.cuda()
        y = y.cuda()
        optimizer.zero_grad()
        y_pred = model(X)
        loss = criterion(y_pred, y.squeeze(1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
    train_loss = total_loss / len(train_dl.dataset)
    torch.cuda.empty_cache()

    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X, y in val_dl:
            X, y = X.cuda(), y.cuda()
            y_pred = model(X).squeeze(1)
            loss = criterion(y_pred, y.squeeze(1))
            total_loss += loss.item() * X.size(0)
    val_loss = total_loss / len(val_dl.dataset)
    torch.cuda.empty_cache()
    
    return np.sqrt(train_loss), np.sqrt(val_loss)

def grid_search_trainer(dataset_name, param_grid, X_train_left1, X_train_right1, X_val_left1, X_val_right1):
    parent = os.path.abspath('')

    if not os.path.exists(os.path.join(parent, 'artifacts_lstm')):
        os.mkdir(os.path.join(parent, 'artifacts_lstm'))
    model_name = getModelName(dataset=dataset_name, type='lstm')
    artifact_path = os.path.join(parent, 'artifacts_lstm', model_name)
    os.mkdir(os.path.join(artifact_path))

    sys.stdout = open(os.path.join(artifact_path, 'train.log'), 'w')

    input_size = len(X_train_left1[0][0]) # X_train_left.shape[-1]
    seq_len = len(X_train_left1[0]) + 1 # X_train_left.shape[-2] + 1

    train_dataset = TimeSeriesDataset(X_train_left1, X_train_right1)
    val_dataset = TimeSeriesDataset(X_val_left1, X_val_right1)

    param_combinations = list(product(
    param_grid['lr'],
    param_grid['batch_size'],
    param_grid['hidden_size'],
    param_grid['num_layers'],
    param_grid['bidirectional']
    ))

    best_val_loss = np.inf
    epochs = 2
    start_gs = time.time()
    for lr, batch_size, hidden_size, num_layers, bidirectional in param_combinations:
        print(f"\nTraining with combination :\nInitial LR : {lr}\tBatch size : {batch_size}\thidden_size : {hidden_size}\tnum_layers : {num_layers}\tbidirectional : {bidirectional}", flush=True)
        train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dl = DataLoader(val_dataset, batch_size=batch_size)

        model = LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional
        ).cuda()
        
        patience = 50
        step_size = 10
        optimizer = Adam(model.parameters(), lr = lr, weight_decay=0)
        criterion = nn.MSELoss().cuda()
        scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
        earlystopper = EarlyStopper(patience=patience)

        train_loss_list = []
        val_loss_list = []

        step_counter = 1
        start = time.time()
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}\tLearning rate : {scheduler.get_last_lr()}\n", flush=True)
            start_ep = time.time()
            train_loss, val_loss = train_epoch(model, train_dl, val_dl, criterion, optimizer)
            end_ep = time.time()
            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)

            print(f"Train loss : {train_loss:10.6f}\nValidation loss : {val_loss:10.6f}", flush=True)
            _, mn, sc = epoch_time(start_ep, end_ep)
            print(f"Epoch execution time : {mn}min. {sc:.6f}sec.")

            if val_loss < best_val_loss :
                best_val_loss = val_loss
                step_counter = 1
                metadata = {
                    'dataset' : dataset_name,
                    'n_dim' : input_size,
                    'seq_len' : seq_len,
                    'artifact' : model_name,
                    'model' : {
                        'batch_size' : batch_size,
                        'input_size' : input_size,
                        'hidden_size' : hidden_size,
                        'num_layers' : num_layers,
                        'bidirectional' : bidirectional
                    },
                    'max_epochs' : epoch,
                    'initial_lr' : lr,
                    'earlystopper_patience' : patience,
                    'lr_step' : step_size
                }
                torch.save(model.state_dict(), os.path.join(artifact_path, 'checkpoint.pth'))
                print(f"Model recorded with Val loss : {val_loss}", flush=True)
                best_loss = val_loss_list[-1]
                best_epoch = epoch
            else:
                step_counter += 1
            
            if step_counter == step_size:
                scheduler.step()
                step_counter = 1
            
            if earlystopper.early_stop(val_loss):
                print(f"Model not improving. Moving on ...", flush=True)
                break
        
        del model, train_dl, val_dl
        gc.collect()
        
        end = time.time()
        h, m, s = epoch_time(start, end)
        metadata['final_epoch'] = epoch
        metadata['optimal_epoch'] = best_epoch
        metadata['best_val_loss'] = best_loss
        metadata['training_time'] = {'hr' : h, 'mins' : m, 'sec' : s}
        metadata['avg_epoch_sec'] = (end - start)/(epoch+1)
        metadata['train_loss_list'] = train_loss_list
        metadata['val_loss_list'] = val_loss_list
        print(f"Total training time : {h}hrs. {m}mins. {s}sec.", flush=True)
        print("\n"+"#"*100+"\n"+"#"*100+"\n"+"#"*100+"\n", flush=True)
    
    end_gs = time.time()
    h, m, s = epoch_time(start_gs, end_gs)
    print(f"Total Grid Search training time : {h}hrs. {m}mins. {s}sec.", flush=True)
    metadata['grid_search_time'] = {'hr' : h, 'mins' : m, 'sec' : s}
    sys.stdout = sys.__stdout__
    return metadata


if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"]="1"
    torch.manual_seed(42)
    np.random.seed(42)

    parent = os.path.abspath('')
    
    data = 'data' # For plain normalized data
    # data = 'mix' # For normalized + DWT data
    dataset_name = "obd"
    
    context = 1000
    # context = 100

    print(f"Loading dataset : {dataset_name} ...", flush=True)
    X_train_left = load_data(os.path.join(parent, 'datasets', f'obd_driverwise_{data}', f'obd___X_train_left_{context}.h5'))
    X_train_right = load_data(os.path.join(parent, 'datasets', f'obd_driverwise_{data}', f'obd___X_train_right_{context}.h5'))
    X_val_left = load_data(os.path.join(parent, 'datasets', f'obd_driverwise_{data}', f'obd___X_val_left_{context}.h5'))
    X_val_right = load_data(os.path.join(parent, 'datasets', f'obd_driverwise_{data}', f'obd___X_val_right_{context}.h5'))
    print(f"Dataset loaded.", flush=True)

    # param_grid = {
    # "lr" : [0.0001, 0.001, 0.01, 0.1],
    # "batch_size" : [128, 256, 512],
    # "hidden_size" : [64, 128, 256, 512],
    # "num_layers" : [1, 2, 3, 4],
    # "bidirectional" : [True, False]
    # }
    param_grid = {
    "lr" : [0.001],
    "batch_size" : [128],
    "hidden_size" : [256],
    "num_layers" : [1],
    "bidirectional" : [False]
    }

    metadata = grid_search_trainer(
        dataset_name, param_grid,
        X_train_left, X_train_right, X_val_left, X_val_right,
    )

    with open(os.path.join(parent, 'artifacts_lstm', metadata['artifact'], 'train_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)