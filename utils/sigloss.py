import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SiLogLoss(nn.Module):
    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, pred, target):
        diff_log = torch.log(target+ 1e-6) - torch.log(pred+1e-6)
        loss = torch.sqrt(torch.pow(diff_log, 2).mean() -
                          self.lambd * torch.pow(diff_log.mean(), 2))
        return loss
    