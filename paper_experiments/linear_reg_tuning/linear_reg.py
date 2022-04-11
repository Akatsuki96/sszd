import numpy as np

import torch
import time

from szo import SZO
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

from utils import plot_results
from other_methods import CompassSearch

loss = torch.nn.MSELoss()

def evaluate_lam(lam, batch):
    X_train, X_val, y_train, y_val = batch
    w = np.linalg.inv(X_train.T.dot(X_train) + lam * X_train.shape[0] * np.eye(X_train.shape[1])).dot(X_train.T.dot(y_train))
       
    val_err = mean_squared_error(y_val, X_val.dot(w))

    return val_err


def generate_dataset(d, rnd_state, size=1000):
    X = np.linspace([0 for _ in range(d)], [1 for _ in range(d)], size)
    w_true = rnd_state.rand(d, d)
    y = X.dot(w_true) + rnd_state.normal(0, 0.001, (X.shape[0], 1))
    
    return X, y, w_true


rnd_state = np.random.RandomState(12)
split_state = np.random.RandomState(21)

def comp_experiment(X, y, reps=10):

    d = 1

    alpha = lambda t : 1/t
    h = lambda t : 1/(np.log(t)**d) if t > 1 else 1

    optimizer = CompassSearch(d, init_alpha=1.0)

    T = 30
    
    observed_x = np.zeros((reps, T), dtype=np.float64)
    observed_y = np.zeros((reps, T), dtype=np.float64)
    exec_time = np.zeros((reps, T), dtype=np.float64)
    te_err = np.zeros(reps, dtype=np.float64)
    
    for i in range(reps):
        lam = rnd_state.rand(1) * 9 + 1
        Xtr, Xte, ytr, yte = train_test_split(X, y, train_size=0.8)
        for t in range(T):
            exec_time[i, t] = time.time()
            batch = train_test_split(Xtr, ytr, train_size=0.7)
            lam, grad = optimizer.stochastic_step(evaluate_lam, lam, batch)
            exec_time[i, t] = time.time() - exec_time[i, t]

            yt = evaluate_lam(lam, batch)
            observed_x[i, t] = lam[0]
            observed_y[i, t] = yt
            print("[--] t: {}/{}\tlam: {}\tvalidation error: {}\t||grad||^2: {}".format(t, T, round(lam[0],4), round(yt, 5), round(np.linalg.norm(grad)**2,5)) )
        optimizer.reset()
        w = np.linalg.inv(Xtr.T.dot(Xtr) + lam * Xtr.shape[0] * np.eye(Xtr.shape[1])).dot(Xtr.T.dot(ytr))
        te_err[i] = mean_squared_error(yte, Xte.dot(w))

    with open("./test_err.log", "a") as f:
        f.write("[--] method: compass\ttest error: {} \pm {}\n".format(te_err.mean(), te_err.std()))
    val_err_mean = observed_y.mean(axis=0)
    val_err_std = observed_y.std(axis=0)
    
    print("[--] val err: ", val_err_mean, val_err_std)
    exec_time_mean = np.cumsum(exec_time, axis=1).mean(axis=0)
    exec_time_std = np.cumsum(exec_time, axis=1).std(axis=0)
   
    return val_err_mean, val_err_std, exec_time_mean, exec_time_std


def szo_experiment(X, y, directions, reps=10):

   # X, y, w_true = generate_dataset(data_dim, rnd_state, size=100)
    d = 1
    l = 1

    alpha = lambda t : 1/(np.log(t)) if t > 1 else 1
    h = lambda t : 1/(t**5) 

    optimizer = SZO(directions, d, l, alpha, h, seed=12, dtype=np.float32)

    T = 30
    
    observed_x = np.zeros((reps, T), dtype=np.float64)
    observed_y = np.zeros((reps, T), dtype=np.float64)
    exec_time = np.zeros((reps, T), dtype=np.float64)
    te_err = np.zeros(reps, dtype=np.float64)
    
    for i in range(reps):
        lam = rnd_state.rand(1) * 9 + 1
        Xtr, Xte, ytr, yte = train_test_split(X, y, train_size=0.8)
        for t in range(T):
            exec_time[i, t] = time.time()
            batch = train_test_split(Xtr, ytr, train_size=0.7)
            lam, grad = optimizer.stochastic_step(evaluate_lam, lam, batch)
            exec_time[i, t] = time.time() - exec_time[i, t]
            yt = evaluate_lam(lam, batch)
            observed_x[i, t] = lam[0]
            observed_y[i, t] = yt
            print("[--] t: {}/{}\tlam: {}\tvalidation error: {}\t||grad||^2: {}".format(t, T, round(lam[0],4), round(yt, 5), round(np.linalg.norm(grad)**2,5)) )
        optimizer.reset()
        w = np.linalg.inv(Xtr.T.dot(Xtr) + lam * Xtr.shape[0] * np.eye(Xtr.shape[1])).dot(Xtr.T.dot(ytr))
        te_err[i] = mean_squared_error(yte, Xte.dot(w))

    with open("./test_err.log", "a") as f:
        f.write("[--] method: {}\ttest error: {} \pm {}\n".format(directions, te_err.mean(), te_err.std()))
    val_err_mean = observed_y.mean(axis=0)
    val_err_std = observed_y.std(axis=0)
    
    print("[--] val err: ", val_err_mean, val_err_std)
    exec_time_mean = np.cumsum(exec_time, axis=1).mean(axis=0)
    exec_time_std = np.cumsum(exec_time, axis=1).std(axis=0)
   
    return val_err_mean, val_err_std, exec_time_mean, exec_time_std
   
#def plot_results(results, title, out_file_name):

X, y, w_true = generate_dataset(50, rnd_state, size=500)
coo_val_err_mean, coo_val_err_std, coo_exec_time_mean, coo_exec_time_std = szo_experiment(X, y, "coordinate",  reps=100)
sph_val_err_mean, sph_val_err_std, sph_exec_time_mean, sph_exec_time_std = szo_experiment(X, y, "spherical",  reps=100)
comp_val_err_mean, comp_val_err_std, comp_exec_time_mean, comp_exec_time_std = comp_experiment(X, y,   reps=100)

#(val_err_mean, val_err_std, label, color)
val_err_results = [
    (comp_val_err_mean, comp_val_err_std, "Direct Search", "darkblue"),
    (coo_val_err_mean, coo_val_err_std, "SZO (coordinate)", "black"),
    (sph_val_err_mean, sph_val_err_std, "SZO (spherical)", "darkred"),
]

time_results=[
    (comp_exec_time_mean, comp_exec_time_std, "Direct Search", "darkblue"),
    (coo_exec_time_mean, coo_exec_time_std, "SZO (coordinate)", "black"),
    (sph_exec_time_mean, sph_exec_time_std, "SZO (spherical)", "darkred"),
]


plot_results(val_err_results, "Validation error", "MSE", "./linreg_valerr_50",  logscale=True)
plot_results(time_results, "Cumulative Time", "seconds", "./linreg_time_50", logscale=True)


