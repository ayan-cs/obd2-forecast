from datetime import datetime
import h5py
import numpy as np

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_hrs = int(elapsed_time / 3600)
    elapsed_mins = int((elapsed_time - elapsed_hrs * 3600) / 60)
    elapsed_secs = elapsed_time - (elapsed_mins * 60 + elapsed_hrs * 3600)
    return elapsed_hrs, elapsed_mins, elapsed_secs

def getModelName(dataset, type):
    now = str(datetime.now())
    date, time = now.split()[0], now.split()[1]
    date = date.split('-')
    date.reverse()
    date = '-'.join(date)
    time = time.replace(':', '-')[:8]

    model_name = f"{type}___{dataset}___{date}_{time}"
    return model_name

# Load H5PY data
def load_data(path):
    with h5py.File(path, 'r') as f:
        data = f['data']
        return list(data)

def load_single_sample(path):
    with h5py.File(path, 'r') as f:
        x_l = list(f['x_l'])
        x_r = list(f['x_r'])
        return np.array(x_l), np.array(x_r)