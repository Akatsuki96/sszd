import os
import numpy as np

def store_result(results, out):
    os.makedirs("{}".format(out), exist_ok = True)
    mean, std, ct_mean, ct_std = results
    with open("{}/mean_value.log".format(out), "w") as f:
        for i in range(mean.shape[0]):
            f.write("{},{}\n".format(mean[i], std[i]))
    with open("{}/ctime.log".format(out), "w") as f:
        for i in range(ct_mean.shape[0]):
            f.write("{},{}\n".format(ct_mean[i], ct_std[i]))