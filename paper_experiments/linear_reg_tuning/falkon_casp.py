import torch
import numpy as np
import pandas as pd

from falkon import Falkon
from falkon.kernels import GaussianKernel
from falkon.options import FalkonOptions
from falkon.hopt.objectives.transforms import PositiveTransform

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader 


from szo import SZO

from datasets import CustomLoader

def mse_err(y_true, y_pred):
    return ((y_true - y_pred)**2).mean()

def class_err(y_true, y_pred):
    return torch.sum(y_true != np.sign(y_pred.reshape(-1))) / y_true.shape[0]

def build_falkon(parameters):
    #parameter[sigma, lam]
    config = {
        'kernel' : GaussianKernel(trsf(torch.from_numpy(parameters[:-1]))),
        'penalty' : trsf(torch.tensor(parameters[-1])),
        'M' : M,
        'options' : FalkonOptions(keops_active="no", debug=False)
    }
    return Falkon(**config)

def standardize(X):
    return (X - X.mean())/X.std()

def load_data():
    data = pd.read_csv("/data/mrando/CASP/CASP.csv",dtype=np.float32).values[1:,:]

    X, y = data[:, 1:], data[:, 0]

    Xtr, Xte, ytr, yte = train_test_split(standardize(X), y, train_size=0.8)

    return Xtr, Xte, ytr, yte

def evaluate_configuration(config, data):
    Xtr, Xte, ytr, yte = data
    model = build_falkon(config)
    model.fit(torch.from_numpy(Xtr), torch.from_numpy(ytr))
    return mse_err(torch.from_numpy(yte), model.predict(torch.from_numpy(Xte)))

def szo_experiment(Xtr, Xte, ytr, yte, directions, l):
    d = Xtr.shape[1]
    assert l <= d
    alpha = lambda k: l/d * 1/k
    h = lambda k : 1/(np.log(k)**2) if k > 1 else 3
    optimizer = SZO(directions, d + 1, l, alpha, h)
    
    init_config = np.array([1.5 for _ in range(d)] + [1e-6])
    
    for t in range(T):
        data = train_test_split(Xtr, ytr, train_size=0.7)
        init_config, grad, fx = optimizer.stochastic_step(evaluate_configuration, init_config, data)
        print("[--] verr: {}\t|g|^2: {}".format(fx, np.linalg.norm(grad)**2))
        #if np.linalg.norm(grad)**2 < 1e-5:
        #    break
    print("[--] config: ", trsf(torch.tensor(init_config)))
    print("[--] te_err: {}".format(evaluate_configuration(init_config, (Xtr, Xte, ytr, yte))))
    
l = 2
M = 200
T = 50
trsf = PositiveTransform(1e-9)


Xtr, Xte, ytr, yte = load_data()
print("X shape: ", Xtr.shape[1])
szo_experiment(Xtr, Xte, ytr, yte, "coordinate", l)





