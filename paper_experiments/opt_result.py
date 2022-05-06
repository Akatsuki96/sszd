import numpy as np

class OptResult:
    
    def __init__(self, T, reps):
        self.T = T
        self.reps = reps
        self.ctime = np.zeros((reps, T))
        self.fvalues = np.zeros((reps, T))
        self.Fvalues = np.zeros((reps, T))
    
    def append_result(self, i, j, tm, fx, Fx):
        self.ctime[i,j] = tm
        self.fvalues[i,j] = fx
        self.Fvalues[i,j] = Fx
        
    def get_mean_std(self):
        ctime = np.cumsum(self.ctime, axis=1)
        avg_ctime, std_ctime = ctime.mean(axis=0), ctime.std(axis=0)
        avg_fvalues, std_fvalues = self.fvalues.mean(axis=0), self.fvalues.std(axis=0) 
        avg_Fvalues, std_Fvalues = self.Fvalues.mean(axis=0), self.Fvalues.std(axis=0) 
        return avg_ctime, std_ctime, avg_fvalues, std_fvalues, avg_Fvalues, std_Fvalues