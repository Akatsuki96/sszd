import torch
import numpy as np

class Optimizer:

    def stochastic_step(self, fun, x, batch):
        pass
    
    def reset(self):
        pass
    
    
def default_forcing_fun(alpha, d):
    return np.min([1e-5, 1e-5 * (alpha**2) * torch.linalg.norm(d).cpu().item()**2])
    
class GDSOptions:
    
    def __init__(self, d, alpha_max=10.0, exp_factor=1., 
                 cont_factor=0.5, gen_strat="compass", device='cpu',
                 sketch = None, forcing_fun = default_forcing_fun):
        self.d = d
        self.alpha_max = alpha_max
        self.exp_factor = exp_factor
        self.cont_factor = cont_factor
        self.gen_strat = gen_strat
        self.device = device
        self.forcing_fun = forcing_fun
        self.sketch = sketch # None or ("type", num_dirs)
        
        self.rnd_state = torch.Generator(device=self.device)
        self.rnd_state.manual_seed(12)
        
    def __sketched_directions(self, D):
        if self.sketch[0] == "gaussian":
            Pk = torch.randn(size=(self.d*2, self.sketch[1]), device=self.device, generator=self.rnd_state) / np.sqrt(self.sketch[1])
        elif self.sketch[0] == "orthogonal":
            Z = torch.randn(size=(self.d*2 , self.sketch[1]), device=self.device, generator=self.rnd_state)
            Q, _ = torch.linalg.qr(Z, mode="reduced")

            Pk = np.sqrt(self.sketch[1]/ self.d) * Q
        
#        print(Pk.shape, D.shape)
        
        return Pk.T @ D
        
        
    def generate_gset(self):
        if self.gen_strat == "compass":
            G = torch.vstack((torch.eye(self.d), -torch.eye(self.d)))
        elif self.gen_strat == "np1":
            G = torch.vstack((torch.eye(self.d), -torch.ones((self.d))))
        elif self.gen_strat == "random_unit":
            a = torch.randn(size=(self.d,self.d), device=self.device, generator=self.rnd_state)
            a = a / torch.linalg.norm(a)
            G = torch.vstack([a, -a])
        elif self.gen_strat == "n_half":
            a = torch.randn(size=(self.d // 2,self.d), device=self.device, generator=self.rnd_state)
            a = a / torch.linalg.norm(a)
            G = torch.vstack([a, -a])
        elif self.gen_strat == "random_orth":
            A = torch.randn(size=(self.d,self.d), device=self.device, generator=self.rnd_state)
            Q = torch.linalg.qr(A, mode='reduced')[0] 
            G = torch.vstack([Q, -Q])
        else:
            raise Exception("Unknown generation strategy!")
        if self.sketch is None:
            return G
        
        return self.__sketched_directions(G)
        # Sketch G
    
    
class STP(Optimizer):
    def __init__(self, init_alpha, options:GDSOptions):
        self.init_alpha = init_alpha
        self.alpha=init_alpha
        self.options = options
        self.rnd_state =np.random.RandomState(123)
        self.t = 1
        
    @property
    def alpha_max(self):
        return self.options.alpha_max
    
    @property
    def cont_factor(self):
        return self.options.cont_factor
    
    @property
    def exp_factor(self):
        return self.options.exp_factor
    
    @property
    def forcing_fun(self):
        return self.options.forcing_fun
    
    def stochastic_step(self, fun, x, batch):
        self.D = self.options.generate_gset()
        self.alpha = self.init_alpha/(self.t + 1)
        d_idx = self.rnd_state.choice(range(self.D.shape[0]), 1)[0]
        dirs = [x + self.alpha * self.D[d_idx], x - self.alpha * self.D[d_idx], x]
        values = [fun(x, batch).item() for x in dirs]
        #print("[--] argmin: {}\tvalues: {}".format(np.argmin(values), [round(x, 4) for x in values]))
        self.t += 1
        return dirs[np.argmin(values)], 1, values[np.argmin(values)]
        
    def reset(self):
        self.t=1
        self.alpha=self.init_alpha
    

class GDS(Optimizer):
    def __init__(self, init_alpha, options:GDSOptions):
        self.init_alpha = init_alpha
        self.alpha=init_alpha
        self.options = options
        self.t = 1
        
    @property
    def alpha_max(self):
        return self.options.alpha_max
    
    @property
    def cont_factor(self):
        return self.options.cont_factor
    
    @property
    def exp_factor(self):
        return self.options.exp_factor
    
    @property
    def forcing_fun(self):
        return self.options.forcing_fun
    
    def stochastic_step(self, fun, x, batch):
        self.D = self.options.generate_gset()
        values = torch.zeros(self.D.shape[0], device=self.options.device)
        fx = fun(x, batch)
        num_eval = 1
        while True:
            for i in range(self.D.shape[0]):
                value = fun(x + self.alpha * self.D[i], batch)
                if value < fx - self.forcing_fun(self.alpha, self.D[i]):
                    self.alpha = min(self.alpha * self.exp_factor, self.alpha_max)
                    return x + self.alpha*self.D[i], num_eval
                num_eval += 1
            if self.alpha < 1e-6:
                return x, num_eval
            self.alpha *= self.cont_factor
        
    def reset(self):
        self.t=1
        self.alpha=self.init_alpha