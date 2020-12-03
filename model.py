import torch as T
import numpy as np

class CountingGrid(T.nn.Module):

    def __init__(self, feature_dim, size, window, clamp_constant):
        super(CountingGrid, self).__init__()
        
        self.feature_dim = feature_dim
        self.size = size
        self.window = window
        self.clamp_constant = clamp_constant
        
        self.wcg = T.nn.Parameter(T.empty((feature_dim,size,size)).uniform_())
                    
    def forward(self, inputs):
        w1 = T.softmax(self.wcg, 0).unsqueeze(1)
        w1 = T.nn.functional.pad(w1, (self.window//2,)*4, 'circular')
        w1 = T.nn.functional.avg_pool2d(w1, (self.window,self.window), stride=(1,1)).squeeze(1)
        w1 = T.log(w1)
        # return log probability of input under each mixture component
        return T.einsum('bv,vij->bij', inputs, w1)
    
    def clamp(self):
        self.wcg.clamp_(min=-self.clamp_constant, max=self.clamp_constant)