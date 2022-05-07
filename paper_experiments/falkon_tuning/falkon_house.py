import sys
sys.path.append("../")

import torch
import time

import numpy as np
import pandas as pd

from falkon import Falkon
from falkon.kernels import GaussianKernel
from falkon.options import FalkonOptions
from falkon.hopt.objectives.transforms import PositiveTransform


from sklearn import datasets
from sklearn.model_selection import train_test_split

from opt_result import OptResult

from szo import SZO

from other_methods import GDS, GDSOptions, STP

def mse_err(y_true, y_pred):
    return ((y_true - y_pred)**2).mean()
def mclass_loss(true, pred):
    true = torch.argmax(true, dim=1)
    pred = torch.argmax(pred, dim=1)
    return torch.mean((true != pred).to(torch.float32))
def class_err(y_true, y_pred):
    signs = np.sign(y_pred.reshape(-1).numpy()).reshape(-1)
    #print(y_true.reshape(-1).numpy()[:4])
    #print(signs[:4])
    #print(np.sum(y_true.reshape(-1).numpy() != signs))
    
    return np.sum(y_true.reshape(-1).numpy() != signs)/y_true.shape[0]


def build_falkon(parameters):
    #parameter[sigma, lam]
    config = {
        'kernel' : GaussianKernel(trsf(torch.from_numpy(parameters[:-1]))),
        'penalty' : trsf(torch.tensor(parameters[-1])),
        'M' : M,
        'maxiter' : 10,
        'options' : FalkonOptions(keops_active="no", debug=False)
    }
    return Falkon(**config)

def standardize(X):
    return (X - X.mean())/X.std()

def load_data():
    #data = pd.read_csv("/data/mrando/house/house.data",dtype=str).values[1:,:]

    X, y = datasets.fetch_california_housing(return_X_y=True)
    #y = np.eye(10, dtype=np.float32)[Y]
    Xtr, Xte, ytr, yte = train_test_split(X, y, train_size=0.8)
    Xm = Xtr.mean()
    Xstd = Xtr.std()
    
    Xtr = (Xtr - Xm) / Xstd
    Xte = (Xte - Xm) / Xstd

#    X, y = data[:, :-1].astype(np.float32), data[:, -1]
    
 #   y[y=='g'] = -1.
 #   y[y=='h'] = 1.
 #   y = y.astype(np.float32)

   
    return Xtr, Xte, ytr, yte

def evaluate_configuration(config, data):
    Xtr, Xte, ytr, yte = data
    model = build_falkon(config)
    model.fit(torch.from_numpy(Xtr).to(torch.float32), torch.from_numpy(ytr).to(torch.float32))
    return mse_err(torch.from_numpy(yte).to(torch.float32), model.predict(torch.from_numpy(Xte).to(torch.float32)))

def store_results(fname, result):
    # it_time, tr_err, fx
    avg_ctime, std_ctime, avg_terr, std_terr, avg_verr, std_verr = result.get_mean_std()
    with open(fname, "a") as f:
        for i in range(avg_ctime.shape[0]):
            f.write("{},{},{},{},{},{}\n".format(avg_ctime[i], std_ctime[i], avg_terr[i], std_terr[i], avg_verr[i], std_verr[i]))

def stp_experiment(Xtr, Xte, ytr, yte, options, reps=5):
    d = Xtr.shape[1]     
    init_config_state = np.random.RandomState(12)
    result = OptResult(T, reps)
    
    optimizer = STP(init_alpha=1.0, options=options)
    te_err = np.zeros(reps)
    for i in range(reps):
        init_sigmas = init_config_state.rand(d) * (5.0 - 1.001) + 0.001
        init_lam = init_config_state.rand(1) * (1e-3 - 1e-6) + 1e-6
        
        init_config = np.hstack((init_sigmas, init_lam))
        
        
        for t in range(T):
            data = train_test_split(Xtr, ytr, train_size=0.7)
            it_time = time.time()
            new_config, grad, v_err = optimizer.stochastic_step(evaluate_configuration, init_config, data)
            it_time = time.time() - it_time
            tr_err = evaluate_configuration(init_config, (data[0], data[0], data[2], data[2]))
            init_config = new_config
            result.append_result(i, t, it_time, tr_err, v_err)
            print("[--] t: {}/{}\ttr_err: {}\tverr: {}\t|g|^2: {}".format(t, T, tr_err, v_err, np.linalg.norm(grad)**2))
        optimizer.reset()
        print("-"*33)
        te_err[i] = evaluate_configuration(init_config, (Xtr, Xte, ytr, yte))
    return result, te_err 

def ds_experiment(Xtr, Xte, ytr, yte, options, reps=5):
    d = Xtr.shape[1] 
    init_config_state = np.random.RandomState(12)
    
    optimizer = GDS(init_alpha=1.0, options=options)
    
    result = OptResult(T, reps)
    te_err = np.zeros(reps)
    for i in range(reps):
        init_sigmas = init_config_state.rand(d) * (5.0 - 1.001) + 0.001
        init_lam = init_config_state.rand(1) * (1e-3 - 1e-6) + 1e-6
        
        init_config = np.hstack((init_sigmas, init_lam))
        
        
        for t in range(T):
            data = train_test_split(Xtr, ytr, train_size=0.7)
            it_time = time.time()
            new_config, grad = optimizer.stochastic_step(evaluate_configuration, init_config, data)
            it_time = time.time() - it_time
            tr_err = evaluate_configuration(init_config, (data[0], data[0], data[2], data[2]))
            v_err = evaluate_configuration(init_config, data)
            init_config = new_config
            result.append_result(i, t, it_time, tr_err, v_err)
            print("[--] t: {}/{}\ttr_err: {}\tverr: {}\t|g|^2: {}".format(t, T, tr_err, v_err, np.linalg.norm(grad)**2))
        optimizer.reset()
        print("-"*33)
        te_err[i] = evaluate_configuration(init_config, (Xtr, Xte, ytr, yte))
    return result, te_err 



def szo_experiment(Xtr, Xte, ytr, yte, directions, l, reps=5):
    d = Xtr.shape[1] 
    assert l <= d
    alpha = lambda k:   l/d * 1/(np.sqrt(k)) #* 1e-1
    h = lambda k :  l/d * (1/k ** 2) #* 1e-1
    
    init_config_state = np.random.RandomState(12)
    optimizer = SZO(directions, d + 1, l, alpha, h)   
    result = OptResult(T, reps)
    te_err = np.zeros(reps)
    for i in range(reps):   
        init_sigmas = init_config_state.rand(d) * (5.0 - 1.001) + 0.001
        init_lam = init_config_state.rand(1) * (1e-3 - 1e-6) + 1e-6
        init_config = np.hstack((init_sigmas, init_lam))
        
        for t in range(T):
            data = train_test_split(Xtr, ytr, train_size=0.7)
            it_time = time.time()
            new_config, grad, fx = optimizer.stochastic_step(evaluate_configuration, init_config, data)
            it_time = time.time() - it_time
            tr_err = evaluate_configuration(init_config, (data[0], data[0], data[2], data[2]))
            init_config = new_config
            result.append_result(i, t, it_time, tr_err, fx)
            print("[--] t: {}/{}\ttr_err: {}\tverr: {}".format(t, T, tr_err, fx))
        optimizer.reset()
        print("-"*33)
        te_err[i] = evaluate_configuration(init_config, (Xtr, Xte, ytr, yte))
    return result, te_err
l = 3
M = 500
T = 100
trsf = PositiveTransform(1e-9)

reps = 10
Xtr, Xte, ytr, yte = load_data()

_ = evaluate_configuration(np.array([1.0 for _ in range(Xtr.shape[1] + 1)]), (Xtr, Xte, ytr, yte))


comp_options = GDSOptions(9, alpha_max = 10000.0, exp_factor=2, cont_factor=0.5, gen_strat="compass")
stp_options = GDSOptions(9, gen_strat="random_unit")
sketch_unit_options = GDSOptions(9, alpha_max = 10000.0, exp_factor=10, cont_factor=0.25, gen_strat="random_unit", sketch=("gaussian", l))
sketch_orth_options = GDSOptions(9, alpha_max = 10000.0, exp_factor=10, cont_factor=0.25, gen_strat="random_orth", sketch=("orthogonal", l))
unit_options = GDSOptions(9, alpha_max = 10000.0, exp_factor=10, cont_factor=0.25, gen_strat="random_unit")
orth_options = GDSOptions(9, alpha_max = 10000.0, exp_factor=10, cont_factor=0.25, gen_strat="random_orth")
n_half_options = GDSOptions(9, alpha_max = 10000.0, exp_factor=10, cont_factor=0.25, gen_strat="n_half")


szo_sph_ris, te_err = szo_experiment(Xtr, Xte, ytr, yte, "spherical", l, reps=reps)
store_results("szo_sph_house.log", szo_sph_ris)
with open("szo_sph_te_house.log", "a") as f:
    for err in te_err:
        f.write("{}\n".format(err))

stp_ris, te_err = stp_experiment(Xtr, Xte, ytr, yte, stp_options, reps=reps)
store_results("stp_house.log", stp_ris)
with open("stp_te_house.log", "a") as f:
    for err in te_err:
        f.write("{}\n".format(err))
#
szo_sph_ris, te_err = szo_experiment(Xtr, Xte, ytr, yte, "coordinate", l, reps=reps)
store_results("szo_coo_house.log", szo_sph_ris)
with open("szo_coo_te_house.log", "a") as f:
    for err in te_err:
        f.write("{}\n".format(err))
      
sketch_unit_ris, te_err = ds_experiment(Xtr, Xte, ytr, yte, sketch_unit_options, reps=reps)
store_results("sk_gauss_house.log", sketch_unit_ris)
with open("sk_gauss_te_house.log", "a") as f:
    for err in te_err:
        f.write("{}\n".format(err))

sketch_orth_ris, te_err = ds_experiment(Xtr, Xte, ytr, yte, sketch_orth_options, reps=reps)
store_results("sk_orth_house.log", sketch_orth_ris)
with open("sk_orth_te_house.log", "a") as f:
    for err in te_err:
        f.write("{}\n".format(err))
unit_ris, te_err = ds_experiment(Xtr, Xte, ytr, yte, unit_options, reps=reps)
store_results("gauss_house.log", unit_ris)
with open("gauss_te_house.log", "a") as f:
    for err in te_err:
        f.write("{}\n".format(err))

orth_ris, te_err = ds_experiment(Xtr, Xte, ytr, yte, orth_options, reps=reps)
store_results("orth_house.log", orth_ris)
with open("orth_te_house.log", "a") as f:
    for err in te_err:
        f.write("{}\n".format(err))

n_half_ris, te_err = ds_experiment(Xtr, Xte, ytr, yte, n_half_options, reps=reps)
store_results("n_half_house.log", n_half_ris)
with open("n_half_te_house.log", "a") as f:
    for err in te_err:
        f.write("{}\n".format(err))



comp_ris, te_err = ds_experiment(Xtr, Xte, ytr, yte, comp_options, reps=reps)
store_results("compass_house.log", comp_ris)
with open("compass_te_house.log", "a") as f:
    for err in te_err:
        f.write("{}\n".format(err))
