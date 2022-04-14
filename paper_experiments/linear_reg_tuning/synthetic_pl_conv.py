import numpy as np

import torch
import time

from szo import SZO
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

from utils import plot_results
from other_methods import GDS, GDSOptions

loss = torch.nn.MSELoss()

d = 10


A = np.random.normal(0.0, 0.10, size=(d, d))
u, s, vh = np.linalg.svd(A, full_matrices=False)
#s[2] = 0.0
#A = np.dot(u, np.dot(np.diag(s), vh))

#eig, _ = np.linalg.eig(A)

#Lam = 1/np.max(eig)

def F(x, line_idx):
    return np.linalg.norm(A.dot(x))**2

def ds_experiment(options, reps=10):
    rnd_state = np.random.RandomState(12)
    split_state = np.random.RandomState(21)

    optimizer = GDS(init_alpha=1.0, options=options)

    T = 3000
    
    observed_x = np.zeros((reps, T), dtype=np.float64)
    observed_y = np.zeros((reps, T), dtype=np.float64)
    exec_time = np.zeros((reps, T), dtype=np.float64)
    te_err = np.zeros(reps, dtype=np.float64)
    
    for i in range(reps):
        x = (rnd_state.rand(d) * 5 + 1).reshape(-1, 1)
        for t in range(T):
            line_idx = split_state.randint(A.shape[0])
            exec_time[i, t] = time.time()
            x, grad = optimizer.stochastic_step(F, x, line_idx)
            exec_time[i, t] = time.time() - exec_time[i, t]

            yt = F(x, line_idx)
            observed_x[i, t] = x[0]
            observed_y[i, t] = yt
            print("[--] t: {}/{}\tF(x, line): {}\tf(x): {}\t||grad||^2: {}".format(t, T, round(yt, 5), np.linalg.norm(A.dot(x))**2, round(np.linalg.norm(grad)**2,5)) )
        optimizer.reset()
      
    val_mean = observed_y.mean(axis=0)
    val_std = observed_y.std(axis=0)
    
    exec_time_mean = np.cumsum(exec_time, axis=1).mean(axis=0)
    exec_time_std = np.cumsum(exec_time, axis=1).std(axis=0)
   
    return val_mean, val_std, exec_time_mean, exec_time_std

def szo_experiment(directions, l, reps=10):
    rnd_state = np.random.RandomState(12)
    split_state = np.random.RandomState(21)

    alpha = lambda t : 1/t# l/(d * np.sqrt(t) * Lam)  
    h = lambda t :  1/(np.log(t)**2) if t > 1 else 1
#    l/(d * (np.log(t)**2) * Lam)   

    optimizer = SZO(directions, d, l, alpha, h, seed=12, dtype=np.float64)

    T = 3000
    
    observed_x = np.zeros((reps, T), dtype=np.float64)
    observed_y = np.zeros((reps, T), dtype=np.float64)
    exec_time = np.zeros((reps, T), dtype=np.float64)
    te_err = np.zeros(reps, dtype=np.float64)
    
    for i in range(reps):
        x = (rnd_state.rand(d) * 5 + 1).reshape(-1, 1)
        for t in range(T):
            print("T")
            line_idx = split_state.randint(A.shape[0])
            exec_time[i, t] = time.time()
            x, grad, fx = optimizer.stochastic_step(F, x, line_idx)
            exec_time[i, t] = time.time() - exec_time[i, t]
            yt = fx
#            observed_x[i, t] = x[0]
            observed_y[i, t] = yt
            print("[--] t: {}/{}\tF(x,line): {}\tf(x): {}\t||grad||^2: {}".format(t, T, round(yt, 5), np.linalg.norm(A.dot(x))**2, round(np.linalg.norm(grad)**2,5)) )
        optimizer.reset()

    val_mean = observed_y.mean(axis=0)
    val_std = observed_y.std(axis=0)
    
    exec_time_mean = np.cumsum(exec_time, axis=1).mean(axis=0)
    exec_time_std = np.cumsum(exec_time, axis=1).std(axis=0)
   
#    print(val_mean)
#    exit()
    return val_mean, val_std, exec_time_mean, exec_time_std
   
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

coo_val_err_mean, coo_val_err_std, coo_exec_time_mean, coo_exec_time_std = szo_experiment("coordinate", 5, reps=10)
sph_val_err_mean, sph_val_err_std, sph_exec_time_mean, sph_exec_time_std = szo_experiment("spherical",  5, reps=10)


#randunit_val_err_mean, randunit_val_err_std, randunit_exec_time_mean, randunit_exec_time_std = ds_experiment(randunit_options, reps=10)
#randunit_e_val_err_mean, randunit_e_val_err_std, randunit_e_exec_time_mean, randunit_e_exec_time_std = ds_experiment(randunit_e_options, reps=10)
#
#randorth_val_err_mean, randorth_val_err_std,   randorth_exec_time_mean,   randorth_exec_time_std = ds_experiment(randorth_options, reps=10)
#randorth_e_val_err_mean, randorth_e_val_err_std, randorth_e_exec_time_mean, randorth_e_exec_time_std = ds_experiment(randorth_e_options, reps=10)
#
#
#comp_val_err_mean, comp_val_err_std, comp_exec_time_mean, comp_exec_time_std = ds_experiment(comp_options, reps=10)
#n1_val_err_mean, n1_val_err_std, n1_exec_time_mean, n1_exec_time_std = ds_experiment(n1_options, reps=10)
#
#comp_e_val_err_mean, comp_e_val_err_std, comp_e_exec_time_mean, comp_e_exec_time_std = ds_experiment(comp_options_e, reps=10)
#n1_e_val_err_mean, n1_e_val_err_std, n1_e_exec_time_mean, n1_e_exec_time_std = ds_experiment(n1_options_e, reps=10)
#
#
#
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
   # (comp_val_err_mean, comp_val_err_std, "Compass Search", colormap['comp']),
   # (comp_e_val_err_mean, comp_e_val_err_std, "Compass Search /w expansion", colormap['comp_e']),
#
   # (n1_val_err_mean, n1_val_err_std, "N1", colormap['n1']),
   # (n1_e_val_err_mean, n1_e_val_err_std, "N1 /w expansion ", colormap['n1_e']),
#
   # (randunit_val_err_mean, randunit_val_err_std, "Random Unit", colormap['randunit']),    
   # (randunit_e_val_err_mean, randunit_e_val_err_std, "Random Unit /w expansion ", colormap['randunit_e']),
   # 
   # (randorth_val_err_mean, randorth_val_err_std, "Random Orthogonal", colormap['randorth']),
   # (randorth_e_val_err_mean, randorth_e_val_err_std, "Random Orthogonal /w expansion", colormap['randorth_e']),

    (coo_val_err_mean, coo_val_err_std, "SZO (coordinate)", colormap['szo_coo']),
    (sph_val_err_mean, sph_val_err_std, "SZO (spherical)", colormap['szo_sph']),


]



time_results=[
 #   (comp_exec_time_mean, comp_exec_time_std, "Compass Search", colormap['comp']),
 #   (comp_e_exec_time_mean, comp_e_exec_time_std, "Compass Search /w expansion", colormap['comp_e']),
#
 #   (n1_exec_time_mean, n1_exec_time_std, "N1", colormap['n1']),
 #   (n1_e_exec_time_mean, n1_e_exec_time_std, "N1 /w expansion", colormap['n1_e']),
#
 #   (randunit_exec_time_mean, randunit_exec_time_std, "Random Unit", colormap['randunit']),    
 #   (randunit_e_exec_time_mean, randunit_e_exec_time_std, "Random Unit /w expansion ", colormap['randunit_e']),
#
 #   (randorth_exec_time_mean, randorth_exec_time_std, "Random Orthogonal", colormap['randorth']),
 #   (randorth_e_exec_time_mean, randorth_e_exec_time_std, "Random Orthogonal /w expansion", colormap['randorth_e']),

    (coo_exec_time_mean, coo_exec_time_std, "SZO (coordinate)", colormap['szo_coo']),
    (sph_exec_time_mean, sph_exec_time_std, "SZO (spherical)", colormap['szo_sph']),
]


plot_results(val_err_results, "Function values", "$f(x)$", "./conv_pl_values",  logscale=True)
plot_results(time_results, "Cumulative Time", "seconds", "./conv_pl_time", logscale=True)


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


plot_results(val_err_results, "Function values", "$f(x)$", "./conv_pl_values_onlyexp",  logscale=True)
plot_results(time_results, "Cumulative Time", "seconds", "./conv_pl_time_onlyexp", logscale=True)


