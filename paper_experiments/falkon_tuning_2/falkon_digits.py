from sklearn import datasets, model_selection
import numpy as np
np.random.seed(30)
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import falkon.hopt
from falkon import FalkonOptions
from falkon.hopt.objectives import HoldOut
from falkon.hopt.objectives.transforms import PositiveTransform


from szo import SZO
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

flk_opt = FalkonOptions(use_cpu=not torch.cuda.is_available())


class DigitDataset(Dataset):
    
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __getitem__(self, idx):
        return self.X[idx, :].reshape(-1, self.X.shape[1]), self.y[idx, :].reshape(-1, self.y.shape[1])

    def __len__(self):
        return self.X.shape[0]

trsf = PositiveTransform(1e-8)

def mclass_loss(true, pred):
    true = torch.argmax(true, dim=1)
    pred = torch.argmax(pred, dim=1)
    return torch.mean((true != pred).to(torch.float32))

def zero_th_target(parameters, data):
    model, batch = data
    model.penalty_ = nn.Parameter(trsf(torch.tensor([parameters[-1]]).to(dtype=torch.float32)))
    model.kernel = falkon.kernels.GaussianKernel(trsf(torch.from_numpy(parameters[:-1])).to(dtype=torch.float32), opt=flk_opt) 
    return model(batch[0], batch[1])


X, Y = datasets.load_digits(return_X_y=True)

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
    X, Y, test_size=0.2, random_state=10, shuffle=True)

X_train = torch.from_numpy(X_train).to(dtype=torch.float32)
X_test = torch.from_numpy(X_test).to(dtype=torch.float32)
Y_train = torch.from_numpy(Y_train)
Y_test = torch.from_numpy(Y_test)

train_mean = X_train.mean()
train_std = X_train.std()
X_train -= train_mean
X_train /= train_std
X_test -= train_mean
X_test /= train_std



eye = torch.eye(10, dtype=torch.float32)
Y_train = eye[Y_train]
Y_test = eye[Y_test]

train_dataset = DigitDataset(X_train, Y_train)
test_dataset = DigitDataset(X_test, Y_test)




l = 1
d = X_train.shape[1] + 1
alpha = lambda k:  l/d * 1/k 
h = lambda k : l/d * (1/np.log(k) ** 2) if k > 1 else 1 #* 1e-1

directions = "spherical"

optimizer = SZO(directions, d, l, alpha, h) 

tr_loader = DataLoader(train_dataset, batch_size=64)
te_loader = DataLoader(test_dataset, batch_size=64)


params = np.asarray([1.0 for _ in range(d-1)] + [1e-5], dtype=np.float32)


sigma_init = torch.tensor([1.0] * X_train.shape[1], dtype=torch.float32).requires_grad_()
kernel = falkon.kernels.GaussianKernel(sigma=sigma_init, opt=flk_opt)
penalty_init = torch.tensor(1e-5, dtype=torch.float32)
centers_init = X_train[np.random.choice(X_train.shape[0], size=(300, ), replace=False)].clone()


model = HoldOut(
    kernel=kernel, penalty_init=penalty_init, centers_init=centers_init,  # The initial hp values
    opt_penalty=True, opt_centers=False, val_pct=0.3, per_iter_split = True # Whether the various hps are to be optimized
    )


#zero_th_target(params, (model, (X_train, Y_train)))

T = 1000
epoch_loss = []    
for epoch in range(T):
    params, grad, fx = optimizer.stochastic_step(zero_th_target, params, (model, (X_train, Y_train)))
    epoch_loss.append(fx.item())    
    print("[--] epoch: {}/{}\tloss: {}\ttrain err: {}".format(epoch, T, epoch_loss[-1], round(mclass_loss(Y_train, model.predict(X_train)).item()*100, 2) ))

#tr_loss, tr_err = [], []
#
#for epoch in range(50):
#
#    for Xtr, ytr in tr_loader:
#        opt_hp.zero_grad()
#        loss = model(X_train, Y_train)
#        loss.backward()
#        opt_hp.step()
#
#        tr_loss.append(loss.item())
#        tr_err.append(mclass_loss(Y_train, model.predict(X_train)))
#        print(f"Epoch {epoch} Loss {tr_loss[-1]:.3f} Error {tr_err[-1] * 100:.2f}%")
#
## Evaluate the test error:
#ts_preds = model.predict(X_test)
#print(f"Test error: {mclass_loss(Y_test, ts_preds) * 100:.2f}%")

