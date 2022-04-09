import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class Dataset:
    
    def __init__(self, path, train_size):        
        self.path = path
        self.train_size = train_size
        
    def preprocess(self, Xtr, Xte, ytr, yte):
        pass

    def build_dataset(self):
        pass

class CASP(Dataset):
    
    def build_dataset(self):
        dataset = pd.read_csv(self.path, dtype=np.float32)
        feature_train = ['F{}'.format(i) for i in range(1, 10)]
        data_x = dataset[feature_train]
        
        dataset[feature_train] = (data_x - data_x.mean())/data_x.std() 
        
        print(dataset.head())
        dataset = dataset.values
        X, y = dataset[:, 1:], dataset[:, 0]
        
        return train_test_split(X, y, train_size=self.train_size)