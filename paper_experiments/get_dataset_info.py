import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn.model_selection import train_test_split

def get_housing_info():
    X, y = datasets.fetch_california_housing(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, train_size=0.8)
    print("[--] California Housing")
    print("\t[--] Training shape: {}\tTest shape: {}".format(Xtr.shape, Xte.shape))
    
def get_htru2_info():
    data = pd.read_csv("/data/mrando/HTRU2/HTRU_2.csv",dtype=np.float32).values[1:,:]

    X, y = data[:, :-1], data[:, -1]
    
    y[y==0] = -1

    Xtr, Xte, ytr, yte = train_test_split(X, y, train_size=0.8, stratify=y)
    print("[--] HTRU2")
    print("\t[--] Training shape: {}\tTest shape: {}".format(Xtr.shape, Xte.shape))
    
def get_casp_info():
    data = pd.read_csv("/data/mrando/CASP/CASP.csv",dtype=np.float32).values[1:,:]

    X, y = data[:, 1:], data[:, 0]

    Xtr, Xte, ytr, yte = train_test_split(X, y, train_size=0.8)

    print("[--] CASP")
    print("\t[--] Training shape: {}\tTest shape: {}".format(Xtr.shape, Xte.shape))
    
    
get_housing_info()
get_htru2_info()
get_casp_info()