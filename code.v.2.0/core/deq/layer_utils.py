import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn


class DEQWrapper:
    def __init__(self, func, z_init=list()):
        z_shape = []
        z_indexing = [0]
        for each in z_init:
            z_shape.append(each.shape)
            z_indexing.append(np.prod(each.shape[1:]))
        
        self.func = func
        self.z_shape = z_shape
        self.z_indexing = np.cumsum(z_indexing)
    
    def list2vec(self, *z_list):
        '''Convert list of tensors to a batched vector (B, ...)'''

        z_list = [each.flatten(start_dim=1) for each in z_list]
        return torch.cat(z_list, dim=1)

    def vec2list(self, z_hidden):
        '''Convert a batched vector back to a list'''

        z_list = []
        z_indexing = self.z_indexing 
        for i, shape in enumerate(self.z_shape):
            z_list.append(z_hidden[:, z_indexing[i]:z_indexing[i+1]].view(shape))
        return z_list

    def __call__(self, z_hidden):
        '''A function call to the DEQ f'''

        z_list = self.vec2list(z_hidden)
        z_list = self.func(*z_list)
        z_hidden = self.list2vec(*z_list)
        
        return z_hidden
    
    def norm_diff(self, z_new, z_old, show_list=False):
        if show_list:
            z_new, z_old = self.vec2list(z_new), self.vec2list()
            return [(z_new[i] - z_old[i]).norm().item() for i in range(len(z_new))]
        
        return (z_new - z_old).norm().item()
