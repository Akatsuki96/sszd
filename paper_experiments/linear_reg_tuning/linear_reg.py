import numpy as np

import torch
import time

from szo import SZO
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

from falkon.hopt.objectives.transforms import PositiveTransform

from utils import plot_results
from other_methods import GDS, GDSOptions

loss = torch.nn.MSELoss()

trsf = PositiveTransform(1e-10)

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



def ds_experiment(X, y, options, reps=10):
    rnd_state = np.random.RandomState(12)
    split_state = np.random.RandomState(21)

    d = 1

    alpha = lambda t : 1/t
    h = lambda t : 1/(np.log(t)**d) if t > 1 else 1

    optimizer = GDS(init_alpha=1.0, options=options)

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
        f.write("[--] method: gss[{}]\ttest error: {} \pm {}\n".format(optimizer.options.gen_strat, te_err.mean(), te_err.std()))
    val_err_mean = observed_y.mean(axis=0)
    val_err_std = observed_y.std(axis=0)
    
    print("[--] val err: ", val_err_mean, val_err_std)
    exec_time_mean = np.cumsum(exec_time, axis=1).mean(axis=0)
    exec_time_std = np.cumsum(exec_time, axis=1).std(axis=0)
   
    return val_err_mean, val_err_std, exec_time_mean, exec_time_std

def szo_experiment(X, y, directions, reps=10):
    rnd_state = np.random.RandomState(12)
    split_state = np.random.RandomState(21)

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
            lam, grad, fx = optimizer.stochastic_step(evaluate_lam, lam, batch)
            exec_time[i, t] = time.time() - exec_time[i, t]
            yt = fx
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

comp_options =  GDSOptions(1, alpha_max = 1.0, exp_factor=1, cont_factor=0.5, gen_strat="compass")
n1_options = GDSOptions(1, alpha_max = 1.0, exp_factor=1, cont_factor=0.5, gen_strat="np1")

comp_options_e =  GDSOptions(1, alpha_max = 1000.0, exp_factor=2, cont_factor=0.5, gen_strat="compass")
n1_options_e = GDSOptions(1, alpha_max = 1000.0, exp_factor=2, cont_factor=0.5, gen_strat="np1")

randunit_options =  GDSOptions(1, alpha_max = 1.0, exp_factor=1, cont_factor=0.5, gen_strat="random_unit")
randunit_e_options =  GDSOptions(1, alpha_max = 1000.0, exp_factor=2, cont_factor=0.5, gen_strat="random_unit")

randorth_options =  GDSOptions(1, alpha_max = 1.0, exp_factor=1, cont_factor=0.5, gen_strat="random_orth")
randorth_e_options =  GDSOptions(1, alpha_max = 1000.0, exp_factor=2, cont_factor=0.5, gen_strat="random_orth")

stp_options = GDSOptions(1, alpha_max = 1.0, exp_factor=1, cont_factor=0.5, gen_strat="random_orth")

rnd_state = np.random.RandomState(12)
split_state = np.random.RandomState(21)


X, y, w_true = generate_dataset(50, rnd_state, size=5000)

randunit_val_err_mean, randunit_val_err_std, randunit_exec_time_mean, randunit_exec_time_std = ds_experiment(X, y, randunit_options, reps=10)
randunit_e_val_err_mean, randunit_e_val_err_std, randunit_e_exec_time_mean, randunit_e_exec_time_std = ds_experiment(X, y, randunit_e_options, reps=10)

randorth_val_err_mean, randorth_val_err_std,   randorth_exec_time_mean,   randorth_exec_time_std = ds_experiment(X, y, randorth_options, reps=10)
randorth_e_val_err_mean, randorth_e_val_err_std, randorth_e_exec_time_mean, randorth_e_exec_time_std = ds_experiment(X, y, randorth_e_options, reps=10)


comp_val_err_mean, comp_val_err_std, comp_exec_time_mean, comp_exec_time_std = ds_experiment(X, y, comp_options, reps=10)
n1_val_err_mean, n1_val_err_std, n1_exec_time_mean, n1_exec_time_std = ds_experiment(X, y, n1_options, reps=10)

comp_e_val_err_mean, comp_e_val_err_std, comp_e_exec_time_mean, comp_e_exec_time_std = ds_experiment(X, y, comp_options_e, reps=10)
n1_e_val_err_mean, n1_e_val_err_std, n1_e_exec_time_mean, n1_e_exec_time_std = ds_experiment(X, y, n1_options_e, reps=10)


coo_val_err_mean, coo_val_err_std, coo_exec_time_mean, coo_exec_time_std = szo_experiment(X, y, "coordinate",  reps=10)
sph_val_err_mean, sph_val_err_std, sph_exec_time_mean, sph_exec_time_std = szo_experiment(X, y, "spherical",  reps=10)

colormap={
    'comp' : "blue",
    'comp_e' : "midnightblue",
    'n1' : "lightskyblue",
    'n1_e' : "steelblue",
    "randunit" : 'lightsalmon',
    'randunit_e' : 'saddlebrown',
    'randorth' : 'mediumseagreen',
    'randorth_e' : 'darkgreen',
    'szo_sph' : 'darkorange',
    'szo_coo' : 'slateblue',
    'sds' : 'tan'
}


#(val_err_mean, val_err_std, label, color)
val_err_results = [
    (comp_val_err_mean, comp_val_err_std, "Compass Search", colormap['comp']),
    (comp_e_val_err_mean, comp_e_val_err_std, "Compass Search /w expansion", colormap['comp_e']),

    (n1_val_err_mean, n1_val_err_std, "N1", colormap['n1']),
    (n1_e_val_err_mean, n1_e_val_err_std, "N1 /w expansion ", colormap['n1_e']),

    (randunit_val_err_mean, randunit_val_err_std, "Random Unit", colormap['randunit']),    
    (randunit_e_val_err_mean, randunit_e_val_err_std, "Random Unit /w expansion ", colormap['randunit_e']),
    
    (randorth_val_err_mean, randorth_val_err_std, "Random Orthogonal", colormap['randorth']),
    (randorth_e_val_err_mean, randorth_e_val_err_std, "Random Orthogonal /w expansion", colormap['randorth_e']),

    (coo_val_err_mean, coo_val_err_std, "SZO (coordinate)", colormap['szo_coo']),
    (sph_val_err_mean, sph_val_err_std, "SZO (spherical)", colormap['szo_sph']),


]



time_results=[
    (comp_exec_time_mean, comp_exec_time_std, "Compass Search", colormap['comp']),
    (comp_e_exec_time_mean, comp_e_exec_time_std, "Compass Search /w expansion", colormap['comp_e']),

    (n1_exec_time_mean, n1_exec_time_std, "N1", colormap['n1']),
    (n1_e_exec_time_mean, n1_e_exec_time_std, "N1 /w expansion", colormap['n1_e']),

    (randunit_exec_time_mean, randunit_exec_time_std, "Random Unit", colormap['randunit']),    
    (randunit_e_exec_time_mean, randunit_e_exec_time_std, "Random Unit /w expansion ", colormap['randunit_e']),

    (randorth_exec_time_mean, randorth_exec_time_std, "Random Orthogonal", colormap['randorth']),
    (randorth_e_exec_time_mean, randorth_e_exec_time_std, "Random Orthogonal /w expansion", colormap['randorth_e']),

    (coo_exec_time_mean, coo_exec_time_std, "SZO (coordinate)", colormap['szo_coo']),
    (sph_exec_time_mean, sph_exec_time_std, "SZO (spherical)", colormap['szo_sph']),
]


plot_results(val_err_results, "Validation error", "MSE", "./linreg_valerr_50",  logscale=True)
plot_results(time_results, "Cumulative Time", "seconds", "./linreg_time_50", logscale=True)


val_err_results = [
    (comp_e_val_err_mean, comp_e_val_err_std, "Compass Search /w expansion", colormap['comp_e']),

    (n1_e_val_err_mean, n1_e_val_err_std, "N1 /w expansion ", colormap['n1_e']),

    (randunit_e_val_err_mean, randunit_e_val_err_std, "Random Unit /w expansion ", colormap['randunit_e']),
    
    (randorth_e_val_err_mean, randorth_e_val_err_std, "Random Orthogonal /w expansion", colormap['randorth_e']),


    (coo_val_err_mean, coo_val_err_std, "SZO (coordinate)", colormap['szo_coo']),
    (sph_val_err_mean, sph_val_err_std, "SZO (spherical)", colormap['szo_sph']),


]



time_results=[
    (comp_e_exec_time_mean, comp_e_exec_time_std, "Compass Search /w expansion", colormap['comp_e']),

    (n1_e_exec_time_mean, n1_e_exec_time_std, "N1 /w expansion", colormap['n1_e']),

    (randunit_e_exec_time_mean, randunit_e_exec_time_std, "Random Unit /w expansion ", colormap['randunit_e']),

    (randorth_e_exec_time_mean, randorth_e_exec_time_std, "Random Orthogonal /w expansion", colormap['randorth_e']),

    (coo_exec_time_mean, coo_exec_time_std, "SZO (coordinate)", colormap['szo_coo']),
    (sph_exec_time_mean, sph_exec_time_std, "SZO (spherical)", colormap['szo_sph']),
]


plot_results(val_err_results, "Validation error", "MSE", "./linreg_valerr_only_exp_50",  logscale=True)
plot_results(time_results, "Cumulative Time", "seconds", "./linreg_time_only_exp_50", logscale=True)


