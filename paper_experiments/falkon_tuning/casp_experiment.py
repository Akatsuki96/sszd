import numpy as np
import torch
from datasets import CASP
from utils import build_model

from sklearn.model_selection import train_test_split

device = "cuda" if torch.cuda.is_available() else "cpu"

train_size = 0.8
rnd_state = np.random.RandomState(12)

loss = torch.nn.MSELoss().to(device)

dataset_path = "/data/mrando/CASP/CASP.csv"
#Xtr, Xte, ytr, yte = train_test_split(X, y, train_size=train_size, shuffle=True, random_state=rnd_state)

dataset = CASP(dataset_path, train_size)


Xtr, Xte, ytr, yte = dataset.build_dataset()

M = 1000

params = np.asarray([1.0 for _ in range(9)] + [1e-5])

model = build_model(params, M)
