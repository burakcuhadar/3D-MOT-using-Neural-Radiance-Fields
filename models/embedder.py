import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.io import device

'''
def positional_encoding(self,opt,input,L): # [B,...,N]
    input_enc = super().positional_encoding(opt,input,L=L) # [B,...,2NL]
    # coarse-to-fine: smoothly mask positional encoding for BARF
    if opt.barf_c2f is not None:
        # set weights for different frequency bands
        start,end = opt.barf_c2f
        alpha = (self.progress.data-start)/(end-start)*L
        k = torch.arange(L,dtype=torch.float32,device=opt.device)
        weight = (1-(alpha-k).clamp_(min=0,max=1).mul_(np.pi).cos_())/2
        # apply weights
        
        shape = input_enc.shape
        input_enc = (input_enc.view(-1,L)*weight).view(*shape)
    return input_enc
'''

def barf_mask(input, step, start, end, L):
    #print('input shape', input.shape)
    alpha = (step-start)/(end-start)*L
    k = torch.arange(L,dtype=torch.float32,device=device)
    weight = (1-(alpha-k).clamp_(min=0,max=1).mul_(np.pi).cos_())/2
    shape = input.shape
    input_masked = (input.contiguous().view(-1,L)*weight)
    #print('input masked shape', input_masked.shape)
    input_masked = input_masked.view(*shape)
    return input_masked

def get_embedder(multires, end_barf, i=0):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
                'end_barf': end_barf
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj, step=None : eo.embed(x, step)
    return embed, embedder_obj.out_dim



# Positional encoding (section 5.1) (with optional Barf coarse-to-fine mask)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs, step=None):
        enc =  torch.cat([fn(inputs) for fn in self.embed_fns], -1)
        
        if step is None:
            return enc
        else:
            N_freqs = self.kwargs['num_freqs']
            d = self.kwargs['input_dims']
            #start_barf = self.kwargs['start_barf'] #TODO needed?
            start_barf = 0
            end_barf = self.kwargs['end_barf'] 
            #print('input shape', inputs.shape)
            #print('enc shape', enc.shape)
            if self.kwargs['include_input']:
                masked_enc = barf_mask(enc[:,d:], step, start_barf, end_barf, N_freqs)
                return torch.cat([enc[:,:d],masked_enc], 1)
            else:
                return barf_mask(enc, step, start_barf, end_barf, N_freqs)
