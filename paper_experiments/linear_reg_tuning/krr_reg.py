import numpy as np

from szo import SZO

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.kernel_ridge import KernelRidge


def get_dataset():
    X, y = load_digits(return_X_y = True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, train_size=0.8)
    eye = np.eye(10)
    ytr, yte = eye[ytr], eye[yte]
    train_mean, train_std = Xtr.mean(), Xtr.std()
    
    Xtr -= train_mean
    Xte -= train_mean
    Xtr /= train_std
    Xte /= train_std
    return Xtr, Xte, ytr, yte

def mclass_loss(true, pred):
    true = np.argmax(true, axis=1)
    pred = np.argmax(pred, axis=1)
    return np.mean(true != pred)

def hold_out(params, batch):
    Xtr, Xval, ytr,  yval = batch
    model = KernelRidge(params[0], kernel="rbf", gamma=params[1])
    model.fit(Xtr, ytr)
    return np.mean((yval - model.predict(Xval))**2)

def test_model(Xtr, Xte, ytr, yte, params):
    model = KernelRidge(params[0], kernel="rbf", gamma=params[1])
    model.fit(Xtr, ytr)
    return mclass_loss(yte, model.predict(Xte))

def szo_experiment(Xtr, ytr, Xte, yte, directions, d, l):
    alpha = lambda t : l/d * 1/np.log(t) if t > 1 else l/d
    h = lambda t : 1/(t**2) 
    optimizer = SZO(directions, d, l, alpha, h, seed=12, dtype=np.float32)
    T = 10
    init_param = np.array([1.0, 2.0])
    for t in range(T):
        batch = train_test_split(Xtr, ytr, train_size=0.8)
        init_param, grd = optimizer.stochastic_step(hold_out, init_param, batch)
        print("[--] init_param: {}\tgrd: {}\tval: {}".format(init_param, grd, test_model(*batch, init_param)))
    print("[--] Test error: ", test_model(Xtr, Xte, ytr, yte, init_param))

Xtr, Xte, ytr, yte = get_dataset()

szo_experiment(Xtr, ytr, Xte, yte, "spherical", 2, 1)