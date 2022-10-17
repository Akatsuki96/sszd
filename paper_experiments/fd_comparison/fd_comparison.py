import numpy as np

from sszd import SSZD

d = 10

alpha = lambda k : 1/np.sqrt(k)
h = lambda k : 5/k

nesterov = SSZD('gaussian', d, 1,  alpha, h,  dtype = np.float32, seed = 12)
