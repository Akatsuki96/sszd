import torch
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

import os
import numpy as np
import pandas as pd

def store_result(results, out):
    os.makedirs("{}".format(out), exist_ok = True)
    mean, std, mu_val, std_val, ct_mean, ct_std = results
    with open("{}/tr_err.log".format(out), "w") as f:
        for i in range(mean.shape[0]):
            f.write("{},{}\n".format(mean[i], std[i]))
    with open("{}/vl_err.log".format(out), "w") as f:
        for i in range(mu_val.shape[0]):
            f.write("{},{}\n".format(mu_val[i], std_val[i]))
    with open("{}/ctime.log".format(out), "w") as f:
        for i in range(ct_mean.shape[0]):
            f.write("{},{}\n".format(ct_mean[i], ct_std[i]))
            
def load_htru2():
    with open("../data/HTRU_2.csv", 'r') as f:
        lines = f.readlines()
        data = []
        for line in lines:
            data.append([float(elem) for elem in line.split(",")])
    data = np.asarray(data)
    X, y = data[:, :-1], data[:, -1]
    y[y == 0.0] = -1.0
    Xtr, Xte, ytr, yte = train_test_split(X, y, train_size=0.7)
    Xm = Xtr.mean(axis=0)
    Xstd = Xtr.std(axis=0)
    Xtr = (Xtr - Xm) / Xstd
    Xte = (Xte - Xm) / Xstd
    return torch.from_numpy(Xtr), torch.from_numpy(Xte), torch.from_numpy(ytr), torch.from_numpy(yte)
            
def standardize(X):
    return (X - X.mean())/X.std()
            
def load_casp():
    data = pd.read_csv("../data/CASP.csv",dtype=np.float32).values[1:,:]

    X, y = data[:, 1:], data[:, 0]

    Xtr, Xte, ytr, yte = train_test_split(standardize(X), y, train_size=0.8)

    return torch.from_numpy(Xtr), torch.from_numpy(Xte), torch.from_numpy(ytr), torch.from_numpy(yte)
            
def build_dataset():
    X, y = datasets.fetch_california_housing(return_X_y=True)
    #X = (X - X.min(axis=0))/(X.max(axis=0) - X.min(axis=0))
    Xtr, Xte, ytr, yte = train_test_split(X, y, train_size=0.7, random_state=np.random.RandomState(1234))
#    print(ytr)
    Xm = Xtr.mean(axis=0)
    Xstd = Xtr.std(axis=0)
#    
    Xtr = (Xtr - Xm) / Xstd
    Xte = (Xte - Xm) / Xstd

    return torch.from_numpy(Xtr), torch.from_numpy(Xte), torch.from_numpy(ytr), torch.from_numpy(yte)