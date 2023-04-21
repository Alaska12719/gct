import torch
from torch.nn.functional import one_hot as oht
import warnings
warnings.filterwarnings("ignore")

def one_hot(tensor, n_class):
    tensor = torch.tensor(tensor, dtype=torch.int64).detach()
    tensor = oht(tensor,n_class)
    return tensor

def adjust_learning_rate(optimizer, current_epoch,max_epoch,lr_min=0,lr_max=0.1,warmup=True):
    from math import pi, cos
    warmup_epoch = 10 if warmup else 0
    if current_epoch < warmup_epoch:
        lr = lr_max * current_epoch / warmup_epoch
    else:
        lr = lr_min + (lr_max-lr_min)*(1 + cos(pi * (current_epoch - warmup_epoch) / (max_epoch - warmup_epoch))) / 2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr