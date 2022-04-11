import torch
import numpy as np
import pandas as pd

from falkon import Falkon
from falkon.kernels import GaussianKernel
from falkon.options import FalkonOptions

from sklearn.model_selection import train_test_split, KFold

MAXITER = 20

def build_model(param_config, lam, M):
    kernel = GaussianKernel(torch.from_numpy(param_config))
    penalty = lam
    config ={
        'kernel' : kernel,
        'penalty' : penalty,
        'M' : M,
        'seed' : 12,
        'maxiter' : MAXITER,
        'options' : FalkonOptions(cg_tolerance=np.sqrt(1e-4))
    }
    return Falkon(**config)
    
def evaluate_model(Xtr, ytr, Xte, yte, loss, params, lam, M):
    model = build_model(params, lam, M)
      
    model.fit(X_train, y_train)

    te_pred = model.predict(Xte)
    return loss(te_pred.reshape(-1), yte.reshape(-1))
    
def objective_fun(Xtr, ytr, loss, params, lam, M, val_size=0.2):
     
    X_train, X_val, y_train, y_val = train_test_split(Xtr, ytr, test_size=val_size)
    
    model = build_model(params, lam, M)
      
    model.fit(X_train, y_train)
        
    tr_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    tr_err = loss(tr_pred.reshape(-1), y_train.reshape(-1))
    val_err = loss(val_pred.reshape(-1), y_val.reshape(-1))
        
    return tr_err, val_err
    