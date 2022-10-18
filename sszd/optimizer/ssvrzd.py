import numpy as np

from sszd.optimizer import SSZD

class SSVRZD(SSZD):
    
    def __init__(self, num_iter, only_in_iter, *args, **kwargs):
        assert num_iter > 0
        super().__init__(*args, **kwargs)
        self.num_iter = num_iter
        self.only_in_iter = only_in_iter
        self.f_grad = np.zeros(self.d)
        self.x_old = np.zeros(self.d)
                
    def step(self, fun, x, *args):
        P_k = self.dir_strategy.build_direction_matrix()
        alpha_k = self.alpha(self.t)
        h_k = self.h(self.t)
        if (self.t - 1) % self.num_iter == 0:
#            print("[--] t: {}".format(self.t))
#            self.z_old = *args
            self.h_old = h_k
            self.Pk_old = P_k
            self.x_old = x
            self.f_grad = self.approx_gradient(self.Pk_old, self.h_old, fun, self.x_old)
        self.F_old_grad = self.approx_gradient(self.Pk_old, self.h_old, fun, self.x_old, *args)
        self.F_grad = self.approx_gradient(P_k, h_k, fun, x, *args)
        #print("[--] Old diff: {}".format(np.linalg.norm(self.f_grad - self.F_old_grad)))
        #print("FGRAD: ", self.f_grad.shape, self.F_old_grad.shape, self.F_grad.shape, x.shape)
        if self.only_in_iter and ((self.t - 1) % self.num_iter == 0):
            x_new = x - alpha_k * (self.F_grad + self.f_grad - self.F_old_grad)
        elif self.only_in_iter and ((self.t - 1) % self.num_iter != 0):
            x_new = x - alpha_k * self.F_grad 
        else:
            x_new = x - alpha_k * (self.F_grad + self.f_grad - self.F_old_grad)

        self.t += 1
        return self._clip_value(x_new)
    
    def reset(self):
        self.f_grad = np.zeros(self.d)
        self.x_old = np.zeros(self.d)        
        self.t = 1