import numpy as np
import torch
from datasets import CASP
from utils import build_model, objective_fun, evaluate_model
import time
import os

from sklearn.model_selection import train_test_split

from szo import SZO


device = "cuda" if torch.cuda.is_available() else "cpu"



train_size = 0.7
rnd_state = np.random.RandomState(12)

loss = torch.nn.MSELoss().to(device)

dataset_path = "/data/mrando/CASP/CASP.csv"
#Xtr, Xte, ytr, yte = train_test_split(X, y, train_size=train_size, shuffle=True, random_state=rnd_state)

dataset = CASP(dataset_path, train_size)


Xtr, Xte, ytr, yte = dataset.build_dataset()

Xtr, ytr = torch.from_numpy(Xtr), torch.from_numpy(ytr)
Xte, yte = torch.from_numpy(Xte), torch.from_numpy(yte)


#    def __init__(self, dir_build, d, l,  alpha, h,  dtype = np.float32, seed : int = 12):
#    def step(self, fun, x, verbose = False):

def target(x):
    tr_err, val_err = objective_fun(Xtr, ytr, loss, x, lam, M, val_size=0.2)
    return val_err.item()

DELIM = "---------"

d = 9
l = 1
seed =12
rnd_state = np.random.RandomState(12)
T = 20

alpha = lambda t : 1/(t)
h = lambda t : 1/(np.log(t)**2) if t > 1 else 1

M = 1000
lam = 1e-5

xt = rnd_state.rand(d) #* (1 - 1) + 1 #np.asarray([5.5 for _ in range(d)])

print("[--] x0: ", xt)

optimizer = SZO("spherical", d, l, alpha, h, dtype=np.float64, seed= seed)

tr_errs = []
val_errs = []
times = []

out_path = "./results/CASP"

os.makedirs(out_path, exist_ok=True)

for t in range(T):
    it_time = time.time()    
    xt, grad = optimizer.step(target, xt, verbose=True)
    it_time = time.time() - it_time
    times.append(it_time)
    print("[--] t: {}/{}\tx: {}\tgrad: {}".format(t, T, [round(x, 2) for x in xt], np.linalg.norm(grad)**2))    
    tr_err, val_err = objective_fun(Xtr, ytr, loss, xt, lam, M, val_size=0.2)
    tr_errs.append(tr_err)
    val_errs.append(val_err)
    with open(out_path + "/deterministic_{}.log".format(l), "a") as f:
        f.write("{},{},{}\n".format(tr_err, val_err, it_time))

with open(out_path + "/deterministic_{}.log".format(l), "a") as f:
    f.write("{}\n".format(DELIM))

#te_err = evaluate_model()
   
#print("[--] tr: {}\tval: {}".format(round(tr_err.item(), 2), round(val_err, 2)))

