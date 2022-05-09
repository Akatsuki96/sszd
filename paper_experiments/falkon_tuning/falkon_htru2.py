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


from sklearn.model_selection import train_test_split

from opt_result import OptResult

from szo import SZO

from other_methods import GDS, GDSOptions, STP

max_init = 20.0
min_init = 18.0



def mse_err(y_true, y_pred):
    return ((y_true - y_pred)**2).mean()

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
    data = pd.read_csv("/data/mrando/HTRU2/HTRU_2.csv",dtype=np.float32).values[1:,:]

    X, y = data[:, :-1], data[:, -1]
    
    y[y==0] = -1
#    print(y[:10])
#    exit()

    Xtr, Xte, ytr, yte = train_test_split(standardize(X), y, train_size=0.8, stratify=y)

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
        init_sigmas = init_config_state.rand(d) * (max_init - min_init) + min_init
        init_lam = init_config_state.rand(1) * (1e-5 - 1e-7) + 1e-7
        
        init_config = np.hstack((init_sigmas, init_lam))
        
        
        for t in range(T):
            data = train_test_split(Xtr, ytr, train_size=0.7, stratify=ytr)
            it_time = time.time()
            new_config, grad, v_err = optimizer.stochastic_step(evaluate_configuration, init_config, data)
            it_time = time.time() - it_time
            tr_err = evaluate_configuration(init_config, (data[0], data[0], data[2], data[2]))
            init_config = new_config
            result.append_result(i, t, it_time, tr_err, v_err)
            print("[STP] t: {}/{}\ttr_err: {}\tverr: {}\t|g|^2: {}".format(t, T, tr_err, v_err, np.linalg.norm(grad)**2))
        optimizer.reset()
        print("-"*33)
        te_err[i] = evaluate_configuration(init_config, (Xtr, Xte, ytr, yte))
    return result, te_err 

def ds_experiment(Xtr, Xte, ytr, yte, options, reps=5, mname="DS"):
    d = Xtr.shape[1] 
    init_config_state = np.random.RandomState(12)
    
    optimizer = GDS(init_alpha=1.0, options=options)
    
    result = OptResult(T, reps)
    te_err = np.zeros(reps)
    for i in range(reps):
        init_sigmas = init_config_state.rand(d) * (max_init - min_init) + min_init
        init_lam = init_config_state.rand(1) * (1e-5 - 1e-7) + 1e-7
        
        init_config = np.hstack((init_sigmas, init_lam))
        
        
        for t in range(T):
            data = train_test_split(Xtr, ytr, train_size=0.7, stratify=ytr)
            it_time = time.time()
            new_config, grad = optimizer.stochastic_step(evaluate_configuration, init_config, data)
            it_time = time.time() - it_time
            tr_err = evaluate_configuration(init_config, (data[0], data[0], data[2], data[2]))
            v_err = evaluate_configuration(init_config, data)
            init_config = new_config
            result.append_result(i, t, it_time, tr_err, v_err)
            print("[{}] t: {}/{}\ttr_err: {}\tverr: {}\t|g|^2: {}".format(mname, t, T, tr_err, v_err, np.linalg.norm(grad)**2))
        optimizer.reset()
        print("-"*33)
        te_err[i] = evaluate_configuration(init_config, (Xtr, Xte, ytr, yte))
    return result, te_err 



def szo_experiment(Xtr, Xte, ytr, yte, directions, l, reps=5, mname="SZD"):
    d = Xtr.shape[1] 
    assert l <= d
    alpha = lambda k: 100 * (k**(-1/2)) #* 100
    h = lambda k :  1e-2#1/np.sqrt(k) #**2 #* 1e-1
    
    init_config_state = np.random.RandomState(12)
    optimizer = SZO(directions, d + 1, l, alpha, h)   
    result = OptResult(T, reps)
    te_err = np.zeros(reps)
    for i in range(reps):   
        init_sigmas = init_config_state.rand(d) * (max_init - min_init) + min_init
        init_lam = init_config_state.rand(1) * (1e-5 - 1e-7) + 1e-7
        init_config = np.hstack((init_sigmas, init_lam))
        
        for t in range(T):
            data = train_test_split(Xtr, ytr, train_size=0.7, stratify=ytr)
            it_time = time.time()
            new_config, grad, fx = optimizer.stochastic_step(evaluate_configuration, init_config, data)
            it_time = time.time() - it_time
            tr_err = evaluate_configuration(init_config, (data[0], data[0], data[2], data[2]))
            init_config = new_config
            result.append_result(i, t, it_time, tr_err, fx)
            print("[{}] t: {}/{}\ttr_err: {}\tverr: {}".format(mname, t, T, tr_err, fx))
        optimizer.reset()
        print("-"*33)
        te_err[i] = evaluate_configuration(init_config, (Xtr, Xte, ytr, yte))
    return result, te_err
l = 4
M = 20
T = 30#100
trsf = PositiveTransform(1e-9)

reps = 2
Xtr, Xte, ytr, yte = load_data()

_ = evaluate_configuration(np.array([1.0 for _ in range(Xtr.shape[1] + 1)]), (Xtr, Xte, ytr, yte))
print("[--] Started...")

comp_options = GDSOptions(9, alpha_max = 10.0, exp_factor=2, cont_factor=0.5, gen_strat="compass")
stp_options = GDSOptions(9, gen_strat="random_unit")
sketch_unit_options = GDSOptions(9, alpha_max = 10.0, exp_factor=2, cont_factor=0.5, gen_strat="random_unit", sketch=("gaussian", l))
sketch_orth_options = GDSOptions(9, alpha_max = 10.0, exp_factor=2, cont_factor=0.5, gen_strat="random_orth", sketch=("orthogonal", l))
unit_options = GDSOptions(9, alpha_max = 10.0, exp_factor=2, cont_factor=0.5, gen_strat="random_unit")
orth_options = GDSOptions(9, alpha_max = 10.0, exp_factor=2, cont_factor=0.5, gen_strat="random_orth")
n_half_options = GDSOptions(9, alpha_max = 10.0, exp_factor=2, cont_factor=0.5, gen_strat="n_half")


szo_sph_ris, te_err = szo_experiment(Xtr, Xte, ytr, yte, "spherical", l, reps=reps, mname="SZD-SP")
store_results("szo_sph_htru2.log", szo_sph_ris)
with open("szo_sph_te_htru2.log", "a") as f:
    for err in te_err:
        f.write("{}\n".format(err))


unit_ris, te_err = ds_experiment(Xtr, Xte, ytr, yte, unit_options, reps=reps, mname="PDS (gauss)")
store_results("gauss_htru2.log", unit_ris)
with open("gauss_te_htru2.log", "a") as f:
    for err in te_err:
        f.write("{}\n".format(err))

sketch_unit_ris, te_err = ds_experiment(Xtr, Xte, ytr, yte, sketch_unit_options, reps=reps, mname="PDS-RD (gauss)")
store_results("sk_gauss_htru2.log", sketch_unit_ris)
with open("sk_gauss_te_htru2.log", "a") as f:
    for err in te_err:
        f.write("{}\n".format(err))

sketch_orth_ris, te_err = ds_experiment(Xtr, Xte, ytr, yte, sketch_orth_options, reps=reps, mname="PDS-RD (orth)")
store_results("sk_orth_htru2.log", sketch_orth_ris)
with open("sk_orth_te_htru2.log", "a") as f:
    for err in te_err:
        f.write("{}\n".format(err))


szo_sph_ris, te_err = szo_experiment(Xtr, Xte, ytr, yte, "coordinate", l, reps=reps, mname="SZD-CO")
store_results("szo_coo_htru2.log", szo_sph_ris)
with open("szo_coo_te_htru2.log", "a") as f:
    for err in te_err:
        f.write("{}\n".format(err))

stp_ris, te_err = stp_experiment(Xtr, Xte, ytr, yte, stp_options, reps=reps)
store_results("stp_htru2.log", stp_ris)
with open("stp_te_htru2.log", "a") as f:
    for err in te_err:
        f.write("{}\n".format(err))



orth_ris, te_err = ds_experiment(Xtr, Xte, ytr, yte, orth_options, reps=reps, mname="PDS (orth)")
store_results("orth_htru2.log", orth_ris)
with open("orth_te_htru2.log", "a") as f:
    for err in te_err:
        f.write("{}\n".format(err))

n_half_ris, te_err = ds_experiment(Xtr, Xte, ytr, yte, n_half_options, reps=reps, mname="PDS d/2")
store_results("n_half_htru2.log", n_half_ris)
with open("n_half_te_htru2.log", "a") as f:
    for err in te_err:
        f.write("{}\n".format(err))


comp_ris, te_err = ds_experiment(Xtr, Xte, ytr, yte, comp_options, reps=reps, mname="Compass")
store_results("compass_htru2.log", comp_ris)
with open("compass_te_htru2.log", "a") as f:
    for err in te_err:
        f.write("{}\n".format(err))
