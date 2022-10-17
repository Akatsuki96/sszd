import os
import numpy as np

class ResultHandler:
    
    def __init__(self, experiment_name, methods, T, reps, out_dir = "./"):
        self.experiment_name = experiment_name
        self.T = T
        self.reps =reps
        self.methods = methods
        self.out_dir = out_dir + experiment_name
        os.makedirs(out_dir, exist_ok=True)

    def _init_results(self):
        self.results = {}
        for method in self.methods:
            self.results[method] = {
                'iteration_time' : np.empty(self.reps, self.T),
                'values' : np.empty(self.reps, self.T)
            }
            os.makedirs(self.out_dir + "/{}".format(method), exist_ok=True)
            
    def add_result(self, method, rep, t, val, time, write_file=True):
        self.results[method]['values'][rep, t] = val
        self.results[method]['iteration_time'][rep, t] = time
        if write_file:
            with open(self.out_dir + "/{}/results.log".format(method), "a") as f:
                f.write("{},{}\n".format(val, time))
           
            