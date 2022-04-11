import numpy as np

import torch
import time

from szo import SZO
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from falkon import Falkon
from falkon.center_selection import FixedSelector
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from datasets import DigitDataset
from falkon.kernels import GaussianKernel

from sklearn import datasets

from utils import plot_results
from other_methods import CompassSearch
from falkon.hopt.objectives.transforms import PositiveTransform
import falkon.hopt
from falkon import FalkonOptions
from falkon.hopt.objectives import CompReg, HoldOut

loss = torch.nn.MSELoss()

def mclass_loss(true, pred):
    true = torch.argmax(true, dim=1)
    pred = torch.argmax(pred, dim=1)
    return torch.mean((true != pred).to(torch.float32))



def generate_dataset(random_state):    
    X, Y = datasets.load_digits(return_X_y = True)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=random_state, shuffle=True)

    X_train = torch.from_numpy(X_train).to(dtype=torch.float32)
    X_test = torch.from_numpy(X_test).to(dtype=torch.float32)
    Y_train = torch.from_numpy(Y_train)
    Y_test = torch.from_numpy(Y_test)

    train_mean, train_std = X_train.mean(), X_train.std()
    X_train -= train_mean
    X_train /= train_std
    X_test -= train_mean
    X_test /= train_std
    eye = torch.eye(10, dtype=torch.float32)
    Y_train = eye[Y_train]
    Y_test = eye[Y_test]

    return DigitDataset(X_train, Y_train), DigitDataset(X_test, Y_test)

rnd_state = np.random.RandomState(12)
split_state = np.random.RandomState(21)



def szo_experiment(Xtr, ytr, test_data, centers_init, directions, l, reps=10):

   # X, y, w_true = generate_dataset(data_dim, rnd_state, size=100)
    d = 65

    alpha = lambda t : l/d * 1/t
    h = lambda t : 1/np.log(t)**4 if t > 1 else 0.5 
    trsf = PositiveTransform(1e-8)

    optimizer = SZO(directions, d, l, alpha, h, seed=12, dtype=np.float32)

    T = 30
    flk_opt = FalkonOptions(use_cpu=not torch.cuda.is_available(), keops_active="no")
    
    observed_x = np.zeros((reps, T), dtype=np.float64)
    observed_y = np.zeros((reps, T), dtype=np.float64)
    exec_time = np.zeros((reps, T), dtype=np.float64)
    te_err = np.zeros(reps, dtype=np.float64)
    sigma_init = torch.tensor([3.050] * 64, dtype=torch.float32)#.requires_grad_()
    kernel = falkon.kernels.GaussianKernel(sigma=sigma_init)
    penalty_init = torch.tensor(1e-2, dtype=torch.float32)
    model = Falkon(
        kernel=kernel, 
        penalty=penalty_init, 
        center_selection=FixedSelector(centers_init),
        M=centers_init.shape[0],
        options = flk_opt
        )          
    def target(params, batch):
        model = Falkon(
            kernel=GaussianKernel(params[:-1]), 
            penalty=trsf(params[-1]), 
            center_selection=FixedSelector(centers_init),
            M=centers_init.shape[0],
            options = flk_opt
            )
#        model.kernel.sigma = torch.nn.Parameter(params[:-1])
#        model.penalty = torch.nn.Parameter(params[-1])
        model.fit(batch[0], batch[1])
        
        return torch.mean(torch.square(batch[3] - model.predict(batch[2])))
    
    params = torch.hstack((sigma_init, penalty_init))
    for i in range(reps):
        for t in range(T):
            X_train, Xval, y_train, yval = train_test_split(Xtr, ytr, train_size=0.8)
            exec_time[i, t] = time.time()
            params, grad = optimizer.stochastic_step(target, params, (X_train, y_train, Xval, yval))
            exec_time[i, t] = time.time() - exec_time[i, t]
            model = Falkon(
                kernel=GaussianKernel(params[:-1]), 
                penalty=trsf(params[-1]), 
                center_selection=FixedSelector(centers_init),
                M=centers_init.shape[0],
                options = flk_opt
            )
            model.fit(X_train, y_train)
            print("[--] validation error : {}".format(
                mclass_loss(yval, model.predict(Xval)) * 100
                ))
        optimizer.reset()
   
   

l = 2
batch_size = 128

train_data, test_data = generate_dataset(rnd_state)

Xtr = train_data.X
#shape = train_data.X.shape
#print(shape)
#train_data = DataLoader(train_data, batch_size=batch_size)
#test_data = DataLoader(train_data, batch_size=batch_size)
#def szo_experiment(Xtr, ytr, test_data, centers_init, directions, l, reps=10):

centers_init = Xtr[np.random.choice(1437, size=(500, ), replace=False)].clone()
szo_experiment(train_data.X, train_data.y, test_data, centers_init, "spherical", l, reps=1)

#ts_preds = model.predict(X_test)
#print(f"Test error: {mclass_loss(Y_test, ts_preds) * 100:.2f}%")

#coo_val_err_mean, coo_val_err_std, coo_exec_time_mean, coo_exec_time_std = szo_experiment(X, y, "coordinate",  reps=100)
#sph_val_err_mean, sph_val_err_std, sph_exec_time_mean, sph_exec_time_std = szo_experiment(X, y, "spherical",  reps=100)
#comp_val_err_mean, comp_val_err_std, comp_exec_time_mean, comp_exec_time_std = comp_experiment(X, y,   reps=100)
#
##(val_err_mean, val_err_std, label, color)
#val_err_results = [
#    (comp_val_err_mean, comp_val_err_std, "Direct Search", "darkblue"),
#    (coo_val_err_mean, coo_val_err_std, "SZO (coordinate)", "black"),
#    (sph_val_err_mean, sph_val_err_std, "SZO (spherical)", "darkred"),
#]
#
#time_results=[
#    (comp_exec_time_mean, comp_exec_time_std, "Direct Search", "darkblue"),
#    (coo_exec_time_mean, coo_exec_time_std, "SZO (coordinate)", "black"),
#    (sph_exec_time_mean, sph_exec_time_std, "SZO (spherical)", "darkred"),
#]
#
#
#plot_results(val_err_results, "Validation error", "MSE", "./linreg_valerr_50",  logscale=True)
#plot_results(time_results, "Cumulative Time", "seconds", "./linreg_time_50", logscale=True)
#
#
#