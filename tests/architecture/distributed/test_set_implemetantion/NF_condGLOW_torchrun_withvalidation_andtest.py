import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import time
import os
import re
import itertools
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
         
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from sklearn.model_selection import train_test_split 
import optuna

from  NF_condGLOW import *


# class MLP(nn.Module):
#     """
#     Multi-layer perceptron (MLP) to be obtain parameters of Coupling layers
    
#     Parameters
#     ----------
#     n_input : int 
#         Number of input neurons, depend on the dimensions of the input data. 
#     n_output : int 
#         Number of output neurons, depend on the number of parameters needed for the Coupling layers.
#     n_hidden : int
#         Number of hidden neurons in each layer.
#     n_layers : int
#         Number of layers in the network.
#     neg_slope : float
#         Negative slope for the leaky ReLU activation function.
    
#     Returns
#     -------
#     None
#     """

#     def __init__(self, n_input, n_output, n_hidden, n_layers=4, neg_slope=0.2) -> None:
#         super().__init__()
#         ins = torch.ones(n_layers)*n_hidden
#         ins[0] = n_input
#         outs = torch.ones(n_layers)*n_hidden
#         outs[-1] = n_output
#         Lin_layers = list(map(nn.Linear, ins.type(torch.int), outs.type(torch.int)))
#         ReLu_layers = [nn.LeakyReLU(neg_slope) for _ in range(n_layers)]
#         self.network = nn.Sequential(*itertools.chain(*zip(Lin_layers, ReLu_layers)))
#         # self.network.apply(init_weights)
    
#     def forward(self, x):
#         return self.network(x.float())

    
# class GLOW_conv(nn.Module):
#     def __init__(self, n_dim) -> None:
#         super().__init__()
#         self.n_dim = n_dim

#         #Initialize W as orthogonal matrix and decompose it into P, L, U, the learned parameters
#         W_initialize = nn.init.orthogonal_(torch.randn(self.n_dim, self.n_dim))
#         P, L_, U_ = torch.linalg.lu(W_initialize)

#         #P not changed (no grad) but it needs to be stored in the state_dict
#         self.register_buffer("P", P)

#         # Declare as model parameters
#         #Diagonal of U sourced out to S
#         S_ = torch.diagonal(U_)
#         self.S = nn.Parameter(S_)
#         self.L = nn.Parameter(L_)
#         #Declare with diagonal 0s, without changing U_ and thus S_
#         self.U = nn.Parameter(torch.triu(U_, diagonal=1))

#     def _get_W_and_logdet(self):
#         #Make sure the pieces stay in correct shape as in GLOW
#         L = torch.tril(self.L, diagonal=-1) + torch.diag(torch.ones(self.n_dim).to(self.L.device))
#         U = torch.triu(self.U, diagonal=1)
#         S = torch.diag(self.S)
        
#         W = self.P@L@(U+S)
#         logdetW = torch.sum(torch.log(torch.abs(self.S)))

#         return W, logdetW  
    
#     # Pass condition as extra argument, that is not used in the convolution
#     #it stayes untouched, does not get permuted with values that
#     #will be transformed
#     def forward(self, x, x_condition):
#         W, logdetW = self._get_W_and_logdet()
#         y = x.float()@W
#         return y, logdetW
    
#     def backward(self, y, x_condition):
#         W, logdetW_inv = self._get_W_and_logdet()
#         #Just a minus needed
#         logdetW_inv = -logdetW_inv
#         W_inv = torch.linalg.inv(W)
#         x = y.float()@W_inv
#         return x, logdetW_inv

# class AffineCoupling(nn.Module):
#     """
#     Affine Coupling layer for conditional normalizing flows.

#     Args:
#         dim_notcond (int): Dimension of the input not conditioned part.
#         dim_cond (int): Dimension of the input conditioned part.
#         network (nn.Module): Network architecture to use for the affine coupling layer.
#         network_args (tuple): Additional arguments to pass to the network architecture.

#     Attributes:
#         dim_notcond (int): Dimension of the input not conditioned part.
#         dim_cond (int): Dimension of the input conditioned part.
#         net_notcond (nn.Module): Network for the not conditioned part.
#         net_cond (nn.Module): Network for the conditioned part.
#     """

#     def __init__(self, dim_notcond, dim_cond, network=MLP, network_args=(16, 4, 0.2)):
#         super().__init__()
#         self.dim_notcond = dim_notcond
#         self.dim_cond = dim_cond
#         self.net_notcond = network(int(self.dim_notcond / 2), int(self.dim_notcond), *network_args)
#         self.net_cond = network(self.dim_cond, int(self.dim_notcond), *network_args)

#     def forward(self, x, x_condition):
#         """
#         Forward pass of the affine coupling layer.

#         Args:
#             x (torch.Tensor): Input tensor.
#             x_condition (torch.Tensor): Condition tensor.

#         Returns:
#             torch.Tensor: Output tensor after applying the affine coupling layer.
#             torch.Tensor: Log determinant of the Jacobian.
#         """
#         x.float()
#         x_condition.float()
#         x_a, x_b = x.chunk(2, dim=1)
#         log_s, t = (self.net_notcond(x_b) * self.net_cond(x_condition)).chunk(2, dim=1)
#         log_s = 0.636*1.9*torch.atan(log_s/1.9)
#         s = torch.exp(log_s)
#         # s = F.sigmoid(log_s)
#         y_a = (s * x_a) + t
#         y_b = x_b

#         logdet = torch.sum(torch.log(s))

#         return torch.cat([y_a, y_b], dim=1), logdet

#     def backward(self, y, x_condition):
#         """
#         Backward pass of the affine coupling layer.

#         Args:
#             y (torch.Tensor): Input tensor.
#             x_condition (torch.Tensor): Condition tensor.

#         Returns:
#             torch.Tensor: Output tensor.
#             torch.Tensor: Log determinant of the Jacobian.
#         """
#         y.float()
#         x_condition.float()
#         y_a, y_b = y.chunk(2, dim=1)
#         log_s, t = (self.net_notcond(y_b) * self.net_cond(x_condition)).chunk(2, dim=1)
#         log_s = 0.636*1.9*torch.atan(log_s/1.9)
#         s = torch.exp(log_s)
#         # s = F.sigmoid(log_s)
#         x_a = (y_a - t)/s 
#         x_b = y_b

#         logdet = torch.sum(torch.log(s))

#         return torch.cat([x_a, x_b], dim=1), logdet
    
    
# def find_bin(values, bin_boarders):
#     #Make sure that a value=uppermost boarder is in last bin not last+1
#     bin_boarders[..., -1] += 10**-6
#     return torch.sum((values.unsqueeze(-1)>=bin_boarders),dim=-1)-1

# #Takes a parametrisation of RQS and points and evaluates RQS or inverse
# #Splines from [-B,B] identity else
# #Bin widths normalized for 2B interval size
# def eval_RQS(X, RQS_bin_widths, RQS_bin_heights, RQS_knot_derivs, RQS_B, inverse=False):
#     """
#     Function to evaluate points inside [-B,B] with RQS splines, given by the parameters.
#     See https://arxiv.org/abs/1906.04032
#     """
#     #Get boarders of bins as cummulative sum
#     #As they are normalized this goes up to 2B
#     #Represents upper boarders
#     bin_boardersx = torch.cumsum(RQS_bin_widths, dim=-1)

#     #We shift so that they cover actual interval [-B,B]
#     bin_boardersx -= RQS_B

#     #Now make sure we include all boarders i.e. include lower boarder B
#     #Also for bin determination make sure upper boarder is actually B and
#     #doesn't suffer from rounding (order 10**-7, searching algorithm would anyways catch this)
#     bin_boardersx = F.pad(bin_boardersx, (1,0), value=-RQS_B)
#     bin_boardersx[...,-1] = RQS_B


#     #Same for heights
#     bin_boardersy = torch.cumsum(RQS_bin_heights, dim=-1)
#     bin_boardersy -= RQS_B
#     bin_boardersy = F.pad(bin_boardersy, (1,0), value=-RQS_B)
#     bin_boardersy[...,-1] = RQS_B
    
    
#     #Now with completed parametrisation (knots+derivatives) we can evaluate the splines
#     #For this find bin positions first
#     bin_nr = find_bin(X, bin_boardersy if inverse else bin_boardersx)
    
#     #After we know bin number we need to get corresponding knot points and derivatives for each X
#     x_knot_k = bin_boardersx.gather(-1, bin_nr.unsqueeze(-1)).squeeze(-1)
#     x_knot_kplus1 = bin_boardersx.gather(-1, (bin_nr+1).unsqueeze(-1)).squeeze(-1)
    
#     y_knot_k = bin_boardersy.gather(-1, bin_nr.unsqueeze(-1)).squeeze(-1)
#     y_knot_kplus1 = bin_boardersy.gather(-1, (bin_nr+1).unsqueeze(-1)).squeeze(-1)
    
#     delta_knot_k = RQS_knot_derivs.gather(-1, bin_nr.unsqueeze(-1)).squeeze(-1)
#     delta_knot_kplus1 = RQS_knot_derivs.gather(-1, (bin_nr+1).unsqueeze(-1)).squeeze(-1)
    

#     #Evaluate splines, as shown in NSF paper
#     s_knot_k = (y_knot_kplus1-y_knot_k)/(x_knot_kplus1-x_knot_k)
#     if inverse:
#         a = (y_knot_kplus1-y_knot_k)*(s_knot_k-delta_knot_k)+(X-y_knot_k)*(delta_knot_kplus1+delta_knot_k-2*s_knot_k)
#         b = (y_knot_kplus1-y_knot_k)*delta_knot_k-(X-y_knot_k)*(delta_knot_kplus1+delta_knot_k-2*s_knot_k)
#         c = -s_knot_k*(X-y_knot_k)
        
#         Xi = 2*c/(-b-torch.sqrt(b**2-4*a*c))
#         Y = Xi*(x_knot_kplus1-x_knot_k)+x_knot_k
        
#         dY_dX = (s_knot_k**2*(delta_knot_kplus1*Xi**2+2*s_knot_k*Xi*(1-Xi)+delta_knot_k*(1-Xi)**2))/(s_knot_k+(delta_knot_kplus1+delta_knot_k-2*s_knot_k)*Xi*(1-Xi))**2
#         #No sum yet, so we can later keep track which X weren't in the intervall and need logdet 0
#         logdet = -torch.log(dY_dX)

#     else:
#         Xi = (X-x_knot_k)/(x_knot_kplus1-x_knot_k)
#         Y = y_knot_k+((y_knot_kplus1-y_knot_k)*(s_knot_k*Xi**2+delta_knot_k*Xi*(1-Xi)))/(s_knot_k+(delta_knot_kplus1+delta_knot_k-2*s_knot_k)*Xi*(1-Xi))
#         dY_dX = (s_knot_k**2*(delta_knot_kplus1*Xi**2+2*s_knot_k*Xi*(1-Xi)+delta_knot_k*(1-Xi)**2))/(s_knot_k+(delta_knot_kplus1+delta_knot_k-2*s_knot_k)*Xi*(1-Xi))**2
#         logdet = torch.log(dY_dX)

#     return Y, logdet


# def RQS_global(X, RQS_bin_widths, RQS_bin_heights, RQS_knot_derivs, RQS_B, inverse=False):
#     """Evaluates RQS spline, as given by parameters, inside [-B,B] and identity outside. Uses eval_RQS."""
#     inside_interval = (X<=RQS_B) & (X>=-RQS_B)

#     Y = torch.zeros_like(X)
#     logdet = torch.zeros_like(X)
    
#     Y[inside_interval], logdet[inside_interval] = eval_RQS(X[inside_interval], RQS_bin_widths[inside_interval,:], RQS_bin_heights[inside_interval,:], RQS_knot_derivs[inside_interval,:], RQS_B, inverse)
#     Y[~inside_interval] = X[~inside_interval]
#     logdet[~inside_interval] = 0
    
#     #Now sum the logdet, zeros will be in the right places where e.g. all X components were 0
#     logdet = torch.sum(logdet, dim=1)
    
#     return Y, logdet


# class NSF_CL(nn.Module):
#     """Neural spline flow coupling layer. See https://arxiv.org/abs/1906.04032 for details.
    
#     Implements the forward and backward transformation."""
#     def __init__(self, dim_notcond, dim_cond, split=0.5, K=8, B=3, network = MLP, network_args=(16,4,0.2)):
#         """
#         Constructs a Neural spline flow coupling layer.

#         Parameters
#         ----------

#         dim_notcond : int
#             The dimension of the input, i.e. the dimension of the data that will be transformed.
#         dim_cond : int
#             The dimension of the condition. If 0, the coupling layer is not conditioned.
#         split : float, default: 0.5
#             The fraction of the input that will be transformed. The rest will be left unchanged. The default is 0.5.
#         K : int, default: 8
#             The number of bins used for the spline.
#         B : float, default: 3
#             The interval size of the spline.
#         network : nn.Module, default: MLP
#             The neural network used to determine the parameters of the spline.
#         network_args : tuple, default: (16,4,0.2)
#             The arguments for the neural network.
#         """
#         super().__init__()
#         self.dim = dim_notcond
#         self.dim_cond = dim_cond
#         self.K = K
#         self.B = B
        
#         self.split1 = int(self.dim*split)
#         self.split2 = self.dim-self.split1
        
#         self.net = network(self.split1, (3*self.K-1)*self.split2, *network_args)
        
#         #Decide if conditioned or not
#         if self.dim_cond>0:
#             self.net_cond = network(dim_cond, (3*self.K-1)*self.split2, *network_args)
            
        
#     def forward(self, x, x_cond):
#         #Divide input into unchanged and transformed part
#         unchanged, transform = x[..., :self.split1], x[..., self.split1:]

#         #Get parameters from neural network based on unchanged part and condition
#         if self.dim_cond>0:
#             thetas = (self.net_cond(x_cond)*self.net(unchanged)).reshape(-1, self.split2, 3*self.K-1)
#         else:
#             thetas = self.net(unchanged).reshape(-1, self.split2, 3*self.K-1)
        
#         #Normalize NN outputs to get widths, heights and derivatives
#         widths, heights, derivs = torch.split(thetas, self.K, dim=-1)
#         widths = F.softmax(widths, dim=-1)*2*self.B
#         heights = F.softmax(heights, dim=-1)*2*self.B
#         derivs = F.softplus(derivs)
#         derivs = F.pad(derivs, pad=(1,1), value=1)
        
#         #Evaluate splines
#         transform, logdet = RQS_global(transform, widths, heights, derivs, self.B)

#         return torch.hstack((unchanged,transform)), logdet
    
#     def backward(self, x, x_cond):
#         unchanged, transform = x[..., :self.split1], x[..., self.split1:]
        
#         if self.dim_cond>0:
#             thetas = (self.net_cond(x_cond)*self.net(unchanged)).reshape(-1, self.split2, 3*self.K-1)
#         else:
#             thetas = self.net(unchanged).reshape(-1, self.split2, 3*self.K-1)
        
#         widths, heights, derivs = torch.split(thetas, self.K, dim=-1)
#         widths = F.softmax(widths, dim=-1)*2*self.B
#         heights = F.softmax(heights, dim=-1)*2*self.B
#         derivs = F.softplus(derivs)
#         derivs = F.pad(derivs, pad=(1,1), value=1)
        
#         transform, logdet = RQS_global(transform, widths, heights, derivs, self.B, inverse=True)
        
#         return torch.hstack((unchanged,transform)), logdet
    
    
# class NSF_CL2(nn.Module):
#     """Neural spline flow double coupling layer. First transforms the first half of the input, then the second half.
#     Works only for even dimensions.
    
#     Implements the forward and backward transformation."""
#     def __init__(self, dim_notcond, dim_cond, K=8, B=3, network = MLP, network_args=(16,4,0.2)):
#         """
#         Constructs a Neural spline flow double coupling layer.

#         Parameters
#         ----------

#         dim_notcond : int
#             The dimension of the input, i.e. the dimension of the data that will be transformed.
#         dim_cond : int
#             The dimension of the condition. If 0, the coupling layer is not conditioned.
#         K : int, default: 8
#             The number of bins used for the spline.
#         B : float, default: 3
#             The interval size of the spline.
#         network : nn.Module, default: MLP
#             The neural network used to determine the parameters of the spline.
#         network_args : tuple, default: (16,4,0.2)
#             The arguments for the neural network.
#         """
#         super().__init__()
#         self.dim = dim_notcond
#         self.dim_cond = dim_cond
#         self.K = K
#         self.B = B
        
#         self.split = self.dim//2
        
#         #Works only for even
#         self.net1 = network(self.split, (3*self.K-1)*self.split, *network_args)
#         self.net2 = network(self.split, (3*self.K-1)*self.split, *network_args)
        
#         if dim_cond>0:
#             self.net_cond1 = network(dim_cond, (3*self.K-1)*self.split, *network_args)
#             self.net_cond2 = network(dim_cond, (3*self.K-1)*self.split, *network_args)
        
#     def forward(self, x, x_cond):
#         #Divide input into first and second half
#         first, second = x[..., self.split:], x[..., :self.split]
        
#         #Get parameters from neural network based on unchanged part and condition
#         if self.dim_cond>0:
#             thetas = (self.net_cond1(x_cond)*self.net1(second)).reshape(-1, self.split, 3*self.K-1)
#         else:
#             thetas = self.net1(second).reshape(-1, self.split, 3*self.K-1)
        
#         #Normalize NN outputs to get widths, heights and derivatives
#         widths, heights, derivs = torch.split(thetas, self.K, dim=-1)
#         widths = F.softmax(widths, dim=-1)*2*self.B
#         heights = F.softmax(heights, dim=-1)*2*self.B
#         derivs = F.softplus(derivs)
#         derivs = F.pad(derivs, pad=(1,1), value=1)
        
#         #Evaluate splines
#         first, logdet = RQS_global(first, widths, heights, derivs, self.B)
        
#         #Repeat for second half
#         if self.dim_cond>0:
#             thetas = (self.net_cond2(x_cond)*self.net2(first)).reshape(-1, self.split, 3*self.K-1)
#         else:
#             thetas = self.net2(first).reshape(-1, self.split, 3*self.K-1)
#         widths, heights, derivs = torch.split(thetas, self.K, dim=-1)
#         widths = F.softmax(widths, dim=-1)*2*self.B
#         heights = F.softmax(heights, dim=-1)*2*self.B
#         derivs = F.softplus(derivs)
#         derivs = F.pad(derivs, pad=(1,1), value=1)
        
#         second, logdet_temp = RQS_global(second, widths, heights, derivs, self.B)
            
#         logdet += logdet_temp
            
#         return torch.hstack((second,first)), logdet
        
#     def backward(self, x, x_cond):
#         first, second = x[..., self.split:], x[..., :self.split]
        
#         if self.dim_cond>0:
#             thetas = (self.net_cond2(x_cond)*self.net2(first)).reshape(-1, self.split, 3*self.K-1)
#         else:
#             thetas = self.net2(first).reshape(-1, self.split, 3*self.K-1)
        
#         widths, heights, derivs = torch.split(thetas, self.K, dim=-1)
#         widths = F.softmax(widths, dim=-1)*2*self.B
#         heights = F.softmax(heights, dim=-1)*2*self.B
#         derivs = F.softplus(derivs)
#         derivs = F.pad(derivs, pad=(1,1), value=1)
            
#         second, logdet = RQS_global(second, widths, heights, derivs, self.B, inverse=True)
        
#         if self.dim_cond>0:
#             thetas = (self.net_cond1(x_cond)*self.net1(second)).reshape(-1, self.split, 3*self.K-1)
#         else:
#             thetas = self.net1(second).reshape(-1, self.split, 3*self.K-1)
        
#         widths, heights, derivs = torch.split(thetas, self.K, dim=-1)
#         widths = F.softmax(widths, dim=-1)*2*self.B
#         heights = F.softmax(heights, dim=-1)*2*self.B
#         derivs = F.softplus(derivs)
#         derivs = F.pad(derivs, pad=(1,1), value=1)
            
#         first, logdet_temp = RQS_global(first, widths, heights, derivs, self.B, inverse=True)
            
#         logdet += logdet_temp
            
#         return torch.hstack((second,first)), logdet

# class NF_condGLOW(nn.Module):
#     """Normalizing flow GLOW model with Affine coupling layers. Alternates coupling layers with GLOW convolutions Combines coupling layers and convolution layers."""

#     def __init__(self, n_layers, dim_notcond, dim_cond, CL=AffineCoupling, **kwargs_CL):
#         """
#         Constructs a Normalizing flow model.

#         Parameters
#         ----------

#         n_layers : int
#             The number of flow layers. Flow layers consist of a coupling layer and a convolution layer.
#         dim_notcond : int
#             The dimension of the input, i.e. the dimension of the data that will be transformed.
#         dim_cond : int
#             The dimension of the condition. If 0, the coupling layer is not conditioned.
#         CL : nn.Module
#             The coupling layer to use. Affine coupling layers is the only available for now
#         **kwargs_CL : dict
#             The arguments for the coupling layer.
#         """
#         super().__init__()
#         self.dim_notcond = dim_notcond
#         self.dim_cond = dim_cond

#         coupling_layers = [CL(dim_notcond, dim_cond, **kwargs_CL) for _ in range(n_layers)]
#         conv_layers = [GLOW_conv(dim_notcond) for _ in range(n_layers)]


#         self.layers = nn.ModuleList(itertools.chain(*zip(conv_layers,coupling_layers)))
        
#         self.prior = torch.distributions.MultivariateNormal(torch.zeros(2), torch.eye(2), validate_args=False)

#         #Information about hyperparameters accessible from outside
#         #The function _get_right_hypers below will then reconstruct this back to __init__ arguments, if you change the model, change both.
#         #This is needed in the background for recreating the same model in some cases like sampling with multiprocessing
#         kwargs_parsed = kwargs_CL.copy()
#         kwargs_parsed["network"] = "MLP"
#         self.give_kwargs = {"n_layers":n_layers, "dim_notcond":dim_notcond, "dim_cond":dim_cond, "CL":"AffineCoupling", **kwargs_parsed}
        
#     def forward(self, x, x_cond):
#         logdet = torch.zeros(x.shape[0]).to(x.device)
        
#         for layer in self.layers:
#             x, logdet_temp = layer.forward(x, x_cond)
#             logdet += logdet_temp
            
#         #Get p_z(f(x)) which is needed for loss function together with logdet
#         self.prior = self.prior = torch.distributions.Normal(torch.zeros(self.dim_notcond).to(x.device), torch.ones(self.dim_notcond).to(x.device))
#         prior_z_logprob = self.prior.log_prob(x).sum(-1)
        
#         return x, logdet, prior_z_logprob
    
#     def backward(self, y, x_cond):
#         logdet = torch.zeros(y.shape[0]).to(y.device)
        
#         for layer in reversed(self.layers):
#             y, logdet_temp = layer.backward(y, x_cond)
#             logdet += logdet_temp
            
#         return y, logdet
    
    # def sample_Flow(self, number, x_cond):
    #     """Samples from the prior and transforms the samples with the flow.
        
    #     Parameters
    #     ----------
        
    #     number : int
    #         The number of samples to draw. If a condition is given, the number of samples must be the same as the length of conditions.
    #     x_cond : torch.Tensor
    #         The condition for the samples. If dim_cond=0 enter torch.Tensor([]).
    #     """
    #     # return self.backward( self.prior.sample(torch.Size((number,))), torch.from_numpy(x_cond).to(self.device) )[0]
    #     samples = self.backward( self.prior.sample(torch.Size((number,))), torch.from_numpy(x_cond).to(self.device) )[0]
    #     feh_mean, feh_std = torch.from_numpy(np.load('../../data/preprocessing/mean_std_of_feh.npz')['mean']).to(self.device), torch.from_numpy(np.load('../../data/preprocessing/mean_std_of_feh.npz')['std']).to(self.device)
    #     ofe_mean, ofe_std = torch.from_numpy(np.load('../../data/preprocessing/mean_std_of_ofe.npz')['mean']).to(self.device), torch.from_numpy(np.load('../../data/preprocessing/mean_std_of_ofe.npz')['std']).to(self.device)
        
    #     samples[:, 0] = samples[:, 0]*feh_std + feh_mean
    #     samples[:, 1] = samples[:, 1]*ofe_std + ofe_mean
        
    #     return samples
    
    # def to(self, device):
    #     #Modified to also move the prior to the right device
    #     self.device = device
    #     super().to(device)
    #     self.prior = torch.distributions.Normal(torch.zeros(self.dim_notcond).to(device), torch.ones(self.dim_notcond).to(device))
    #     return self
    
    
    
# def training_flow(flow:NF_condGLOW, data:pd.DataFrame, cond_names:list,  epochs, lr=2*10**-2, batch_size=1024, loss_saver=None, checkpoint_dir=None, gamma=0.998, optimizer_obj=None):
    
#     writer = SummaryWriter()
    
#     #Device the model is on
#     device = flow.parameters().__next__().device

#     data = data[data.columns.difference(['Galaxy_name'])]

#     #Get index based masks for conditional variables
#     mask_cond = np.isin(data.columns.to_list(), cond_names)
#     mask_cond = torch.from_numpy(mask_cond).to(device)
    

#     # Convert DataFrame to tensor (index based)
#     data = torch.from_numpy(data.values).type(torch.float)
#     train_index, val_index = train_test_split(np.arange(data.shape[0]), test_size=0.1, random_state=42)

#     train_loader = torch.utils.data.DataLoader(data[train_index], batch_size=batch_size, shuffle=True)
#     val_loader = torch.utils.data.DataLoader(data[val_index], batch_size=batch_size, shuffle=True)

#     if optimizer_obj is None:
#         optimizer = optim.Adam(flow.parameters(), lr=lr)
#     else:
#         optimizer = optimizer_obj

#     lr_schedule = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=gamma)

#     #Save losses
#     if loss_saver is None:
#         losses = []
#     else:
#         losses = loss_saver

#     #Total number of steps
#     ct = 0

#     start_time = time.perf_counter()

#     best_loss = 1_000_000
#     for e in tqdm(range(epochs)):
#         running_loss = 0
#         val_running_loss = 0
#         for i, batch in enumerate(train_loader):
#             x = batch.to(device)
            
#             #Evaluate model
#             z, logdet, prior_z_logprob = flow(x[..., ~mask_cond], x[..., mask_cond])
            
#             #Get loss
#             loss = -torch.mean(logdet+prior_z_logprob) 
#             losses.append(loss.item())
            
#             #Set gradients to zero
#             optimizer.zero_grad()
#             #Compute gradients
#             loss.backward()
#             #Update parameters
#             optimizer.step()
            
#             # Gather data and report
#             running_loss += loss.item()

#             if i % 10 == 9:
#                 last_loss = running_loss / 10 # loss per batch
#                 # print('  batch {} loss: {}'.format(i + 1, last_loss))
#                 tb_x = e * len(train_loader) + i + 1
#                 writer.add_scalar('Loss/train', last_loss, tb_x)    
#                 running_loss = 0.
            
#             ct += 1
#             #Decrease learning rate every 10 steps until it is smaller than 3*10**-6, then every 120 steps
#             if lr_schedule.get_last_lr()[0] <= 3*10**-6:
#                 decrease_step = 120
#             else:
#                 decrease_step = 10

#             #Update learning rate every decrease_step steps
#             if ct % decrease_step == 0:
#                 lr_schedule.step()
                
#         for i, batch in enumerate(val_loader): 
#             x = batch.to(device)
#             z, logdet, prior_z_logprob = flow(x[..., ~mask_cond], x[..., mask_cond])
#             loss = -torch.mean(logdet+prior_z_logprob) 
#             val_running_loss += loss.item()
            
#         last_val_loss = val_running_loss / len(val_loader)
#         writer.add_scalar('Loss/val', last_val_loss, e)
#         if last_val_loss < best_loss:
#             best_loss = last_val_loss
#             torch.save(flow.state_dict(), f"{checkpoint_dir}checkpoint_best.pth")
#             # print(f'state dict saved checkpoint_best')
#             curr_time = time.perf_counter()
#             np.save(f"{checkpoint_dir}losses_best.npy", np.array(best_loss))
#         val_running_loss = 0.
            

          
def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

class Trainer:
    '''
    Class for training a model with distributed data parallelism.
    
    Args:
        model (NF_condGLOW): The model to be trained.
        train_data (torch.utils.data.DataLoader): The data loader for training data.
        val_data (torch.utils.data.DataLoader): The data loader for validation data.
        test_data (torch.utils.data.DataLoader): The data loader for test data.
        optimizer (torch.optim.Optimizer): The optimizer for model parameters.
        save_every (int): The frequency at which to save training snapshots.
        snapshot_path (str): The path to save the training snapshots.
        
    Attributes:
        gpu_id (int): The ID of the GPU being used.
        model (NF_condGLOW): The model to be trained.
        train_data (torch.utils.data.DataLoader): The data loader for training data.
        val_data (torch.utils.data.DataLoader): The data loader for validation data.
        test_data (torch.utils.data.DataLoader): The data loader for test data.
        best_loss (float): The best validation loss achieved during training.
        optimizer (torch.optim.Optimizer): The optimizer for model parameters.
        save_every (int): The frequency at which to save training snapshots.
        epochs_run (int): The number of epochs already run.
        snapshot_path (str): The path to save the training snapshots.
        logger (SummaryWriter): The logger for training progress.
    
    Methods:
        _load_snapshot(snapshot_path): Loads a training snapshot from a given path.
        _run_batch(source, train=True): Runs a batch of data through the model and computes the loss.
        _run_epoch(epoch): Runs an epoch of training and validation.
        _save_checkpoint(epoch): Saves a training snapshot at a given epoch.
        train(max_epochs): Trains the model for a maximum number of epochs.
        test(test_set): Evaluates the model on a test set.
    '''
    def __init__(
        self,
        model: NF_condGLOW,
        train_data: torch.utils.data.DataLoader,
        val_data: torch.utils.data.DataLoader,
        test_data: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,  
        save_every: int,
        snapshot_path: str,) -> None:
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = model.to(self.gpu_id)
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.best_loss = 1_000
        self.optimizer = optimizer
        self.save_every = save_every
        self.epochs_run = 0
        self.snapshot_path = snapshot_path
        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(snapshot_path)
            
        if self.gpu_id == 0:
            self.logger = SummaryWriter()
        else:
            self.logger = None
            
        self.model = DDP(self.model, device_ids=[self.gpu_id])

    
    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_batch(self, source, train=True):
        if train==True:
            mask_cond = np.ones(14).astype(bool)
            mask_cond[:2] = np.array([0, 0]).astype(bool)
            #Evaluate model
            z, logdet, prior_z_logprob = self.model(source[..., ~mask_cond], source[..., mask_cond])
            
            #Get loss
            loss = -torch.mean(logdet+prior_z_logprob) 
        
            #Set gradients to zero
            self.optimizer.zero_grad()
            #Compute gradients
            loss.backward()
            #Update parameters
            self.optimizer.step()
            return loss.item()
        else:
            with torch.no_grad():
                mask_cond = np.ones(14).astype(bool)
                mask_cond[:2] = np.array([0, 0]).astype(bool)
                #Evaluate model
                z, logdet, prior_z_logprob = self.model(source[..., ~mask_cond], source[..., mask_cond])
                
                #Get loss
                loss = -torch.mean(logdet+prior_z_logprob)
                return loss.item()

    def _run_epoch(self, epoch):
        b_sz = self.train_data.batch_size
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch) # shuffle data
        train_loss = 0.
        for source in self.train_data:
            source = source.to(self.gpu_id)
            train_loss += self._run_batch(source, train=True)/len(self.train_data)
        
        self.val_data.sampler.set_epoch(epoch)    
        dist.barrier()
        self.model.eval()
        val_running_loss = 0.
        for source in self.val_data:
            source = source.to(self.gpu_id)
            batch_loss = self._run_batch(source, train=False)
            val_running_loss += batch_loss/len(self.val_data)
        dist.barrier()
        val_running_loss = torch.tensor([val_running_loss]).to(self.gpu_id)
        dist.all_reduce(val_running_loss, op=dist.ReduceOp.SUM)
        
        train_loss =  torch.tensor([train_loss]).to(self.gpu_id)
        dist.all_reduce(train_loss, op=dist.ReduceOp.SUM)
        
        if self.gpu_id == 0:
            self.logger.add_scalar("Loss/val", val_running_loss/int(os.environ["WORLD_SIZE"]), epoch) #the WORLD_SIZE is the number of GPUs
            self.logger.add_scalar("Loss/train", train_loss/int(os.environ["WORLD_SIZE"]), epoch) #the WORLD_SIZE is the number of GPUs
        
        self.model.train()
        
        return val_running_loss/2
            
    def _save_checkpoint(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            val_running_loss = self._run_epoch(epoch)
            if val_running_loss < self.best_loss and self.gpu_id == 0 :  
                self.best_loss = val_running_loss
                self._save_checkpoint(epoch)
    
    def test(self, test_set):
        '''
        Evaluates the model on a test set.
        
        Args:
            test_set: The test set to evaluate the model on.
        
        Returns:
            float: The average test loss.
        '''
        print('start test')
        self.model.eval()
        with torch.no_grad():
            test_running_loss = 0.
            for source in test_set:
                source = source.to(self.gpu_id)
                batch_loss = self._run_batch(source, train=False)
                test_running_loss += batch_loss/len(test_set)
            test_running_loss = torch.tensor([test_running_loss]).to(self.gpu_id)
            dist.all_reduce(test_running_loss, op=dist.ReduceOp.SUM)
            return test_running_loss/int(os.environ["WORLD_SIZE"]) #the WORLD_SIZE is the number of GPUs

def get_even_space_sample(df_mass_masked):
    '''
    Given a dataframe of galaxy in a range of mass, it returns 10 equally infall time spaced samples  
    '''
    len_infall_time = len(df_mass_masked['infall_time'].unique())
    index_val_time = np.linspace(0, len_infall_time-1, 10)
    time = np.sort(df_mass_masked['infall_time'].unique())[index_val_time.astype(int)]
    df_time = pd.DataFrame(columns=df_mass_masked.columns)
    for t in time:
        temp = df_mass_masked[df_mass_masked['infall_time']==t]
        galaxy_temp = temp.sample(1)['Galaxy_name'].values[0]
        df_time = pd.concat((df_time, df_mass_masked[df_mass_masked['Galaxy_name']==galaxy_temp]) )

    return df_time
    
    
def load_train_objs():
    train_set = pd.read_parquet('/export/home/vgiusepp/MW_MH/data/preprocessing_subsample/preprocess_training_set_Galaxy_name_subsample.parquet') # load your dataset
    # Galax_name = train_set['Galaxy_name'].unique()
    # test_galaxy = np.random.choice(Galax_name, int(len(Galax_name)*0.1), replace=False)
    # test_set = train_set[train_set['Galaxy_name'].isin(test_galaxy)]
    # test_set.to_parquet('/export/home/vgiusepp/MW_MH/data/test_set.parquet')
    # train_set = train_set[~(train_set['Galaxy_name'].isin(test_galaxy))][train_set.columns.difference(['Galaxy_name'], sort=False)]
    # test_set = test_set[train_set.columns.difference(['Galaxy_name'], sort=False)]
    # test_set = torch.from_numpy(test_set.values)
    # train_set = torch.from_numpy(train_set.values)
    # train_set, val_set = train_test_split(train_set, test_size=0.2, random_state=42)
    
    low_percentile_mass, high_percentile_mass = np.percentile(train_set['star_log10mass'], 25), np.percentile(train_set['star_log10mass'], 75)
    low_mass = get_even_space_sample(train_set[train_set['star_log10mass']<=low_percentile_mass])
    intermediate_mass = get_even_space_sample(train_set[(train_set['star_log10mass']>low_percentile_mass) & (train_set['star_log10mass']<high_percentile_mass)])
    high_mass = get_even_space_sample(train_set[train_set['star_log10mass']>=high_percentile_mass])
    val_set = pd.concat([low_mass, intermediate_mass, high_mass])
    
    train_set = train_set[~train_set['Galaxy_name'].isin(val_set['Galaxy_name'])]
    
    low_percentile_mass, high_percentile_mass = np.percentile(train_set['star_log10mass'], 25), np.percentile(train_set['star_log10mass'], 75)
    low_mass = get_even_space_sample(train_set[train_set['star_log10mass']<=low_percentile_mass])
    intermediate_mass = get_even_space_sample(train_set[(train_set['star_log10mass']>low_percentile_mass) & (train_set['star_log10mass']<high_percentile_mass)])
    high_mass = get_even_space_sample(train_set[train_set['star_log10mass']>=high_percentile_mass])
    test_set = pd.concat([low_mass, intermediate_mass, high_mass])
    test_set.to_parquet(f'/export/home/vgiusepp/MW_MH/data/test_set.parquet')
    
    train_set = train_set[~train_set['Galaxy_name'].isin(test_set['Galaxy_name'])]
    print('finish prepare data')
    #remove the column Galaxy name before passing it to the model
    test_set = test_set[train_set.columns.difference(['Galaxy_name'], sort=False)]
    train_set = train_set[train_set.columns.difference(['Galaxy_name'], sort=False)]
    val_set = val_set[train_set.columns.difference(['Galaxy_name'], sort=False)]
    test_set = torch.from_numpy(test_set.values)
    val_set =torch.from_numpy(val_set.values)
    train_set = torch.from_numpy(train_set.values)
    model = NF_condGLOW(16, dim_notcond=2, dim_cond=12, CL=NSF_CL2, network_args=[512, 6, 0.2])  # load your model
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    return train_set, val_set, test_set, model, optimizer     

def prepare_dataloader(dataset, batch_size: int):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset))

def main(save_every: int, total_epochs: int, batch_size: int, snapshot_path: str = "snapshot.pt"):
    ddp_setup()
    train_set, val_set, test_set, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(train_set, batch_size)
    val_data = prepare_dataloader(val_set, batch_size)
    test_data = prepare_dataloader(test_set, batch_size)
    trainer = Trainer(model, train_data, val_data, test_data, optimizer, save_every, snapshot_path)
    trainer.train(total_epochs)
    negative_log_likelihood = trainer.test(test_data)
    np.savez('/export/home/vgiusepp/MW_MH/data/test_loss', nll=negative_log_likelihood.cpu().detach())
    destroy_process_group()
    


if __name__ == "__main__":
    print(int(os.environ["WORLD_SIZE"]))
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=1024, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()


    begin=time.time()
    main(args.save_every, args.total_epochs, args.batch_size)
    end = time.time()
    print('total time', (end-begin)/60, 'minutes')