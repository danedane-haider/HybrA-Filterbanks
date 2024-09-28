import torch
from hybra.utils import kappa_alias

class HybrALoss(torch.nn.Module):
    def __init__(self, base_loss:torch.nn.Module,decimation_factor: int, beta: float = 0.0, gamma: float = 0.0):
        super().__init__()

        self.base_loss = base_loss
        self.decimation_factor = decimation_factor
        self.beta = beta
        self.gamma = gamma

    def forward(self, prediction, target, filterbank):
        loss = self.base_loss(prediction, target)
        kappa, alias = kappa_alias(filterbank._filters.squeeze(1), self.decimation_factor)

        return loss, loss + self.beta*(kappa-1) + self.gamma * torch.sum(alias)