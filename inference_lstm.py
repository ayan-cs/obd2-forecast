import torch, os, sys, json, time, gc
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from utils import epoch_time, load_data, load_single_sample
from services import load_model, TimeSeriesDataset

def batchInference(model, dl):
    criterion = nn.MSELoss()
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X_l, X_r in dl:
            X_l = X_l.cuda()
            X_r = X_r.cuda()
            pred = model(X_l)
            loss = criterion(pred, X_r.squeeze(1))
            total_loss += loss.item() * X_l.shape[0]
    return np.sqrt(total_loss / len(dl.dataset))

def singleInference(model, x_l, x_r):
    criterion = nn.MSELoss().cuda()
    model.eval()
    with torch.no_grad():
        x_l = x_l.unsqueeze(0).cuda()
        x_r = x_r.unsqueeze(0)
        pred = model(x_l).detach().unsqueeze(0)
        loss = criterion(pred, x_r)
    return np.sqrt(loss.item()), pred.squeeze(0).cpu()

if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    parent = os.path.abspath('')
    
    # Replace with your artifact name
    artifact_name = 'lstm___obd___17-06-2025_01-50-29'
    
    model_type = artifact_name.split('_')[0]
    artifact_path = os.path.join(parent, f'artifacts_{model_type}', artifact_name)
    with open(os.path.join(artifact_path, 'train_metadata.json'), 'r') as f:
        metadata = json.load(f)

    num_args = len(sys.argv)
    if num_args > 3:
        raise ValueError(f"Too many arguments : Expected atmost 2, got {num_args-1}")
    if num_args < 2 :
        raise ValueError(f"Expected atleast 1 argument\nUsage :\npython inference.py <mode : batch | single <sample_name>")
    if num_args == 2:
        if sys.argv[1] not in ['batch', 'single']:
            raise ValueError(f"Mode '{sys.argv[1]}' is invalid!")

    model = load_model(model_type=model_type, artifact_path=artifact_path, metadata=metadata)

    # if 'dwt' in metadata['dataset']:
    #     data = 'dwt' # For normalized+DWT data
    #     dataset_name = "obddwt" # Put 'obddwt'
    # elif 'mix' in metadata['dataset']:
    #     data = 'mix' # For normalized+DWT -> normalized data
    #     dataset_name = 'obdmix' # Put 'obdmix'
    # else:
    data = 'data'
    dataset_name =  'obd'

    context = metadata['seq_len']

    mode = sys.argv[1]

    if mode=='batch':
        print(f"Loading dataset : {dataset_name} ...", flush=True)
        X_train_left = load_data(os.path.join(parent, 'datasets', f'obd_driverwise_{data}', f'obd___X_train_left_{context}.h5'))
        X_train_right = load_data(os.path.join(parent, 'datasets', f'obd_driverwise_{data}', f'obd___X_train_right_{context}.h5'))
        X_val_left = load_data(os.path.join(parent, 'datasets', f'obd_driverwise_{data}', f'obd___X_val_left_{context}.h5'))
        X_val_right = load_data(os.path.join(parent, 'datasets', f'obd_driverwise_{data}', f'obd___X_val_right_{context}.h5'))
        print(f"Dataset loaded.", flush=True)

        train_dataset = TimeSeriesDataset(X_train_left, X_train_right)
        val_dataset = TimeSeriesDataset(X_val_left, X_val_right)

        train_dl = DataLoader(train_dataset, batch_size=128, shuffle=True)
        del X_train_left, X_train_right
        gc.collect()

        val_dl = DataLoader(val_dataset, batch_size=128)
        del X_val_left, X_val_right
        gc.collect()

        inference_metadata = {}
        start = time.time()
        train_loss = batchInference(model, train_dl)
        end = time.time()
        h, m, s = epoch_time(start, end)
        inference_metadata['train_split_time'] = {'hr' : h, 'min' : m, 'sec' : s}
        avg = (end - start)/len(train_dl)
        h, m, s = epoch_time(0, avg)
        inference_metadata['train_loss'] = train_loss
        inference_metadata['train_split_avg'] = {'hr' : h, 'min' : m, 'sec' : s}

        start = time.time()
        val_loss = batchInference(model, val_dl)
        end = time.time()
        h, m, s = epoch_time(start, end)
        inference_metadata['val_split_time'] = {'hr' : h, 'min' : m, 'sec' : s}
        avg = (end - start)/len(val_dl)
        h, m, s = epoch_time(0, avg)
        inference_metadata['val_loss'] = val_loss
        inference_metadata['val_split_avg'] = {'hr' : h, 'min' : m, 'sec' : s}

        with open(os.path.join(artifact_path, 'inference_metadata.json'), 'w') as f:
            json.dump(inference_metadata, f, indent=4)
        
        print(f"Batch inference complete!")
    
    elif mode=='single':
        print(f"Single sample inference")
        if num_args != 3:
            raise ValueError(f"Mode 'single' requires sample filename")
        sample_name = sys.argv[2]
        if not os.path.exists(os.path.join(parent, 'datasets', 'sample_data', f'{sample_name}.h5')):
            raise FileNotFoundError(f"No data found with name '{sample_name}'")
        x_l, x_r = load_single_sample(os.path.join(parent, 'datasets', 'sample_data', f'{sample_name}.h5'))
        x_l = torch.FloatTensor(x_l).cuda()
        x_r = torch.FloatTensor(x_r).cuda()
        print(f"Data loaded")
        start = time.time()
        loss, x_pred = singleInference(model, x_l, x_r)
        end = time.time()
        _, m, s = epoch_time(start, end)
        print(f"Results came : {loss}\t{x_pred.shape}")
        metadata = {
            'loss' : loss,
            'elapsed_time' : f'{m}m {s:.6f}s',
            'sample_name' : sample_name,
            'n_dim' : x_l.shape[-1],
            'seq_len' : x_l.shape[0],
            'x_l' : x_l.tolist(),
            'x_r' : x_r.tolist(),
            'x_pred' : x_pred.tolist()
        }

        with open(os.path.join(artifact_path, f'inference_{sample_name}.json'), 'w') as f:
            json.dump(metadata, f, indent=4)
        
        print(f"Inference complete!")