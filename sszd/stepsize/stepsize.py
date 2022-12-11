import numpy as np

class StepSize:
    
    def __init__(self, init_value, mode, custom_fun=None):
        self.init_value = init_value
        self.mode = mode
        if mode is not 'custom':
            self.get_next = getattr(self, "_{}".format(self.mode))
        else:
            self.get_next = custom_fun    

    def _constant(self, t):
        return self.init_value
    
    def _log(self, t):
        if t == 1:
            return self.init_value
        return self.init_value / np.log(t)
        
    def _sqrt(self, t):
        return self.init_value / np.sqrt(t)
    
    def _lin(self, t):
        return self.init_value / t
    
    def __call__(self, t):
        return self.get_next(t)