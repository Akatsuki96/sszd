import os
import torch
import numpy as np
from tqdm import tqdm
from scipy.optimize import minimize

import pandas as pd
from sklearn.model_selection import train_test_split

def read_htru2_data(fpath, test_size = 0.2, rnd_state = np.random.RandomState(seed=12131415)):

    data = pd.read_csv(fpath, header=None)

    X = data.values[:, :-1]
    Y = data.values[:, -1]

    X = (X - X.mean(0))/X.std(0)

    Y[Y == 0] = -1.0
    return train_test_split(X, Y, test_size=test_size, random_state=rnd_state, shuffle=True, stratify=Y)



def run_scipy_agorithm(out_path, algo_name, method, target, x0, sample_z, T, reps=1):
    os.makedirs(out_path, exist_ok=True)
    f_values = []    
    iterator = tqdm(range(reps))
    for r in iterator:
        z_k = [sample_z()]
        vals = [target(x0).item()]
        def target_surrogate(x):
            return target(torch.from_numpy(x), z_k[0])

        def callback(x):
            z_k[0] = sample_z()
            vals.append(target(torch.from_numpy(x)).item())
        result = minimize(target_surrogate, x0.numpy(), callback=callback, method = method, options={'maxiter': T, 'rhobeg' : 0.01}, tol=1e-50)
        for i in range(len(vals), T):
            vals.append(vals[-1])
        f_values.append(vals)
        
    f_values = np.array(f_values).reshape(reps, -1)
    mu, sigma = np.mean(f_values, axis=0), np.std(f_values, axis=0)
    write_result("{}/{}.log".format(out_path, algo_name), mu, sigma)


def write_result(fname, mu, sigma):
    with open(fname, 'w') as f:
        for i in range(len(mu)):
            f.write("{},{}\n".format(mu[i], sigma[i]))
            f.flush()

def run_algorithm(out_path, algo_name, opt, x0, sample_z, T, reps=1):
    os.makedirs(out_path, exist_ok=True)
    f_values = []
    iterator = tqdm(range(reps))
    for r in iterator:
        result = opt.optimize(x0=x0.clone(), sample_z=sample_z, T=T, verbose=True, return_trace = True)
        values = []
        for i in range(0, len(result['fun_values'])):
            values += [result['fun_values'][i] for _ in range(result['num_evals'][i])]
      #  print(len(values))
        f_values.append(values)

    f_values = np.array(f_values).reshape(reps, -1)
    mu, sigma = np.mean(f_values, axis=0), np.std(f_values, axis=0)
    write_result("{}/{}.log".format(out_path, algo_name), mu, sigma)

