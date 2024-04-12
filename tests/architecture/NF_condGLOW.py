import numpy as np
import matplotlib.pyplot as plt
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
from torch import autocast
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter

import optuna

class MLP(nn.Module):
    """
    Multi-layer perceptron (MLP) to be obtain parameters of Coupling layers
    
    Parameters
    ----------
    n_input : int 
        Number of input neurons, depend on the dimensions of the input data. 
    n_output : int 
        Number of output neurons, depend on the number of parameters needed for the Coupling layers.
    n_hidden : int
        Number of hidden neurons in each layer.
    n_layers : int
        Number of layers in the network.
    neg_slope : float
        Negative slope for the leaky ReLU activation function.
    
    Returns
    -------
    None
    """

    def __init__(self, n_input, n_output, n_hidden, n_layers=4, neg_slope=0.2) -> None:
        super().__init__()
        ins = torch.ones(n_layers)*n_hidden
        ins[0] = n_input
        outs = torch.ones(n_layers)*n_hidden
        outs[-1] = n_output
        Lin_layers = list(map(nn.Linear, ins.type(torch.int), outs.type(torch.int)))
        ReLu_layers = [nn.LeakyReLU(neg_slope) for _ in range(n_layers)]
        self.network = nn.Sequential(*itertools.chain(*zip(Lin_layers, ReLu_layers)))
        # self.network.apply(init_weights)
    
    def forward(self, x):
        x.float()
        return self.network(x)

    # def init_weights(m):
    #     if isinstance(m, nn.Linear):
    #         torch.nn.kaiming_uniform_(m.weight, a=0, mode='fan_out', nonlinearity='leaky_relu', generator=None)
    #         m.bias.data.fill_(0.01)
    
class GLOW_conv(nn.Module):
    def __init__(self, n_dim) -> None:
        super().__init__()
        self.n_dim = n_dim

        #Initialize W as orthogonal matrix and decompose it into P, L, U, the learned parameters
        W_initialize = nn.init.orthogonal_(torch.randn(self.n_dim, self.n_dim))
        P, L_, U_ = torch.linalg.lu(W_initialize)

        #P not changed (no grad) but it needs to be stored in the state_dict
        self.register_buffer("P", P)

        # Declare as model parameters
        #Diagonal of U sourced out to S
        S_ = torch.diagonal(U_)
        self.S = nn.Parameter(S_)
        self.L = nn.Parameter(L_)
        #Declare with diagonal 0s, without changing U_ and thus S_
        self.U = nn.Parameter(torch.triu(U_, diagonal=1))

    def _get_W_and_logdet(self):
        #Make sure the pieces stay in correct shape as in GLOW
        L = torch.tril(self.L, diagonal=-1) + torch.diag(torch.ones(self.n_dim).to(self.L.device))
        U = torch.triu(self.U, diagonal=1)
        S = torch.diag(self.S)
        
        W = self.P@L@(U+S)
        logdetW = torch.sum(torch.log(torch.abs(self.S)))

        return W, logdetW  
    
    # Pass condition as extra argument, that is not used in the convolution
    #it stayes untouched, does not get permuted with values that
    #will be transformed
    def forward(self, x, x_condition):
        W, logdetW = self._get_W_and_logdet()
        y = x.float()@W
        return y, logdetW
    
    def backward(self, y, x_condition):
        W, logdetW_inv = self._get_W_and_logdet()
        #Just a minus needed
        logdetW_inv = -logdetW_inv
        W_inv = torch.linalg.inv(W)
        x = y.float()@W_inv
        return x, logdetW_inv

class AffineCoupling(nn.Module):
    """
    Affine Coupling layer for conditional normalizing flows.

    Args:
        dim_notcond (int): Dimension of the input not conditioned part.
        dim_cond (int): Dimension of the input conditioned part.
        network (nn.Module): Network architecture to use for the affine coupling layer.
        network_args (tuple): Additional arguments to pass to the network architecture.

    Attributes:
        dim_notcond (int): Dimension of the input not conditioned part.
        dim_cond (int): Dimension of the input conditioned part.
        net_notcond (nn.Module): Network for the not conditioned part.
        net_cond (nn.Module): Network for the conditioned part.
    """

    def __init__(self, dim_notcond, dim_cond, network=MLP, network_args=(16, 4, 0.2)):
        super().__init__()
        self.dim_notcond = dim_notcond
        self.dim_cond = dim_cond
        self.net_notcond = network(int(self.dim_notcond / 2), int(self.dim_notcond), *network_args)
        self.net_cond = network(self.dim_cond, int(self.dim_notcond), *network_args)

    def forward(self, x, x_condition):
        """
        Forward pass of the affine coupling layer.

        Args:
            x (torch.Tensor): Input tensor.
            x_condition (torch.Tensor): Condition tensor.

        Returns:
            torch.Tensor: Output tensor after applying the affine coupling layer.
            torch.Tensor: Log determinant of the Jacobian.
        """
        x.float()
        x_condition.float()
        x_a, x_b = x.chunk(2, dim=1)
        log_s, t = (self.net_notcond(x_b) * self.net_cond(x_condition)).chunk(2, dim=1)
        # s = torch.exp(log_s)
        s = F.sigmoid(log_s)
        y_a = s * x_a + t
        y_b = x_b

        logdet = torch.sum(torch.log(s))

        return torch.cat([y_a, y_b], dim=1), logdet

    def backward(self, y, x_condition):
        """
        Backward pass of the affine coupling layer.

        Args:
            y (torch.Tensor): Input tensor.
            x_condition (torch.Tensor): Condition tensor.

        Returns:
            torch.Tensor: Output tensor.
            torch.Tensor: Log determinant of the Jacobian.
        """
        y.float()
        x_condition.float()
        y_a, y_b = y.chunk(2, dim=1)
        log_s, t = (self.net_notcond(y_b) * self.net_cond(x_condition)).chunk(2, dim=1)
        # s = torch.exp(log_s)
        s = F.sigmoid(log_s)
        x_a = (y_a - t) / s
        x_b = y_b

        logdet = torch.sum(torch.log(s))

        return torch.cat([x_a, x_b], dim=1), logdet

class NF_condGLOW(nn.Module):
    """Normalizing flow GLOW model with Affine coupling layers. Alternates coupling layers with GLOW convolutions Combines coupling layers and convolution layers."""

    def __init__(self, n_layers, dim_notcond, dim_cond, CL=AffineCoupling, **kwargs_CL):
        """
        Constructs a Normalizing flow model.

        Parameters
        ----------

        n_layers : int
            The number of flow layers. Flow layers consist of a coupling layer and a convolution layer.
        dim_notcond : int
            The dimension of the input, i.e. the dimension of the data that will be transformed.
        dim_cond : int
            The dimension of the condition. If 0, the coupling layer is not conditioned.
        CL : nn.Module
            The coupling layer to use. Affine coupling layers is the only available for now
        **kwargs_CL : dict
            The arguments for the coupling layer.
        """
        super().__init__()
        self.dim_notcond = dim_notcond
        self.dim_cond = dim_cond

        coupling_layers = [CL(dim_notcond, dim_cond, **kwargs_CL) for _ in range(n_layers)]
        conv_layers = [GLOW_conv(dim_notcond) for _ in range(n_layers)]


        self.layers = nn.ModuleList(itertools.chain(*zip(conv_layers,coupling_layers)))
        
        self.prior = torch.distributions.MultivariateNormal(torch.zeros(2), torch.eye(2), validate_args=False)

        #Information about hyperparameters accessible from outside
        #The function _get_right_hypers below will then reconstruct this back to __init__ arguments, if you change the model, change both.
        #This is needed in the background for recreating the same model in some cases like sampling with multiprocessing
        kwargs_parsed = kwargs_CL.copy()
        kwargs_parsed["network"] = "MLP"
        self.give_kwargs = {"n_layers":n_layers, "dim_notcond":dim_notcond, "dim_cond":dim_cond, "CL":"AffineCoupling", **kwargs_parsed}
        
    def forward(self, x, x_cond):
        logdet = torch.zeros(x.shape[0]).to(x.device)
        
        for layer in self.layers:
            x, logdet_temp = layer.forward(x, x_cond)
            logdet += logdet_temp
            
        #Get p_z(f(x)) which is needed for loss function together with logdet
        prior_z_logprob = self.prior.log_prob(x).sum(-1)
        
        return x, logdet, prior_z_logprob
    
    def backward(self, y, x_cond):
        logdet = torch.zeros(y.shape[0]).to(y.device)
        
        for layer in reversed(self.layers):
            y, logdet_temp = layer.backward(y, x_cond)
            logdet += logdet_temp
            
        return y, logdet
    
    def sample_Flow(self, number, x_cond):
        """Samples from the prior and transforms the samples with the flow.
        
        Parameters
        ----------
        
        number : int
            The number of samples to draw. If a condition is given, the number of samples must be the same as the length of conditions.
        x_cond : torch.Tensor
            The condition for the samples. If dim_cond=0 enter torch.Tensor([]).
        """
        return self.backward( self.prior.sample(torch.Size((number,))), x_cond )[0]
    
    def to(self, device):
        #Modified to also move the prior to the right device
        super().to(device)
        self.prior = torch.distributions.Normal(torch.zeros(self.dim_notcond).to(device), torch.ones(self.dim_notcond).to(device))
        return self
    
def training_flow(flow:NF_condGLOW, data:pd.DataFrame, cond_names:list,  epochs, lr=2*10**-2, batch_size=1024, loss_saver=None, checkpoint_dir=None, gamma=0.998, optimizer_obj=None):
    
    writer = SummaryWriter()
    
    #Device the model is on
    device = flow.parameters().__next__().device

    #Get index based masks for conditional variables
    mask_cond = np.isin(data.columns.to_list(), cond_names)
    mask_cond = torch.from_numpy(mask_cond).to(device)
    

    # Convert DataFrame to tensor (index based)
    data = torch.from_numpy(data.values).type(torch.float)

    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)

    if optimizer_obj is None:
        optimizer = optim.Adam(flow.parameters(), lr=lr)
    else:
        optimizer = optimizer_obj

    lr_schedule = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=gamma)

    #Save losses
    if loss_saver is None:
        losses = []
    else:
        losses = loss_saver

    #Total number of steps
    ct = 0

    start_time = time.perf_counter()

    best_loss = 1_000_000
    for e in tqdm(range(epochs)):
        running_loss = 0
        for i, batch in enumerate(data_loader):
            x = batch.to(device)
            
            #Evaluate model
            z, logdet, prior_z_logprob = flow(x[..., ~mask_cond], x[..., mask_cond])
            
            #Get loss
            loss = -torch.mean(logdet+prior_z_logprob) 
            losses.append(loss.item())
            
            #Set gradients to zero
            optimizer.zero_grad()
            #Compute gradients
            loss.backward()
            #Update parameters
            optimizer.step()
            
            # Gather data and report
            running_loss += loss.item()

            if i % 10 == 9:
                last_loss = running_loss / 10 # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = e * len(data_loader) + i + 1
                writer.add_scalar('Loss/train', last_loss, tb_x)
                
                if last_loss < best_loss:
                    best_loss = last_loss
                    torch.save(flow.state_dict(), f"{checkpoint_dir}checkpoint_best.pth")
                    print(f'state dict saved checkpoint_best')
                    curr_time = time.perf_counter()
                    np.save(f"{checkpoint_dir}losses_best.npy", np.array(best_loss))
                    
                running_loss = 0.

            ct += 1

            #Decrease learning rate every 10 steps until it is smaller than 3*10**-6, then every 120 steps
            if lr_schedule.get_last_lr()[0] <= 3*10**-6:
                decrease_step = 120
            else:
                decrease_step = 10

            #Update learning rate every decrease_step steps
            if ct % decrease_step == 0:
                lr_schedule.step()

def training_flow_MixedPrecision(flow:NF_condGLOW, data:pd.DataFrame, cond_names:list,  epochs, lr=2*10**-2, batch_size=1024, loss_saver=None, checkpoint_dir=None, gamma=0.998, optimizer_obj=None):
    
    writer = SummaryWriter()
    
    #Device the model is on
    device = flow.parameters().__next__().device

    #Get index based masks for conditional variables
    mask_cond = np.isin(data.columns.to_list(), cond_names)
    mask_cond = torch.from_numpy(mask_cond).to(device)
    
    # Convert DataFrame to tensor (index based)
    data = torch.from_numpy(data.values).type(torch.float)

    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)

    if optimizer_obj is None:
        optimizer = optim.Adam(flow.parameters(), lr=lr)
    else:
        optimizer = optimizer_obj
    
    scaler = GradScaler()

    lr_schedule = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=gamma)

    #Save losses
    if loss_saver is None:
        losses = []
    else:
        losses = loss_saver

    #Total number of steps
    ct = 0

    start_time = time.perf_counter()

    best_loss = 1_000_000
    for e in tqdm(range(epochs)):
        running_loss = 0
        for i, batch in enumerate(data_loader):
     
            #Set gradients to zero
            optimizer.zero_grad()
            
            with autocast(device_type='cuda', dtype=torch.float16):
                x = batch.to(device)
                #Evaluate model
                z, logdet, prior_z_logprob = flow(x[..., ~mask_cond], x[..., mask_cond])
                #Get loss
                loss = -torch.mean(logdet+prior_z_logprob) 
                losses.append(loss.item())
                
            
            scaler.scale(loss).backward()
            #Update parameters
            scaler.step(optimizer)
            scaler.update()
            
             # Gather data and report
            running_loss += loss.item()

            if i % 10 == 9:
                last_loss = running_loss / 10 # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = e * len(data_loader) + i + 1
                writer.add_scalar('Loss/train', last_loss, tb_x)
                
                if last_loss < best_loss:
                    best_loss = last_loss
                    torch.save(flow.state_dict(), f"{checkpoint_dir}checkpoint_best.pth")
                    print(f'state dict saved checkpoint_best')
                    curr_time = time.perf_counter()
                    np.save(f"{checkpoint_dir}losses_best.npy", np.array(best_loss))
                    
                running_loss = 0.
        
            ct += 1

            #Decrease learning rate every 10 steps until it is smaller than 3*10**-6, then every 120 steps
            if lr_schedule.get_last_lr()[0] <= 3*10**-6:
                decrease_step = 120
            else:
                decrease_step = 10

            #Update learning rate every decrease_step steps
            if ct % decrease_step == 0:
                lr_schedule.step()
                
######code to run for training the model#########
            

data = pd.read_parquet('../../data/normalized_training_set.parquet')
cond_names = list(data.keys()[2:])

device = torch.device("cuda:9" if torch.cuda.is_available() else "cpu")
Flow = NF_condGLOW(n_layers=2, dim_notcond=2, dim_cond=12).to(device=device)
losses = []
training_flow_MixedPrecision(flow = Flow, 
                data = data.head(10_000_000), 
                cond_names=cond_names, 
                epochs=10, lr=2*10**-5, batch_size=1024, 
                loss_saver=losses, checkpoint_dir='/export/home/vgiusepp/MW_MH/tests/architecture/checkpoints/checkpoint_data/', gamma=0.998, optimizer_obj=None)

# def define_model(trial):
#     #Hyperparameters
#     n_layers = trial.suggest_int("n_layers", 2, 6)
#     n_hidden = trial.suggest_int("n_hidden", 8, 32)
#     n_layers_CL = trial.suggest_int("n_layers_CL", 2, 8)
#     neg_slope = trial.suggest_float("neg_slope", 0.1, 0.5)

#     #Create model
#     Flow = NF_condGLOW(n_layers, dim_notcond=2, dim_cond=12, CL=AffineCoupling, network_args=[n_hidden, n_layers_CL, neg_slope])

#     return Flow

# def objective(trial):
#     #Hyperparameters
#     Flow = define_model(trial).to(device)
    
    
#     lr = trial.suggest_float("lr", 10**-5, 10**-2)
#     batch_size = trial.suggest_int("batch_size", 1024, 4096)
#     gamma = trial.suggest_float("gamma", 0.99, 0.999)


#     losses = []
    
#     data = pd.read_parquet('../../data/normalized_training_set.parquet')

#     #Train model
#     training_flow(Flow, data=data, cond_names=list(data.keys()[2:]), loss_saver=losses, epochs=10, lr=lr, batch_size=batch_size, gamma=gamma)

#     #Return loss
#     return np.mean(losses)

# study = optuna.create_study(direction="minimize")
# study.optimize(objective, n_trials=100, timeout=600)