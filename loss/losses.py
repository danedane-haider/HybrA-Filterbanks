import torch

class MSETight(torch.nn.Module):

    def __init__(self, beta: float = 0.0):
        super().__init__()
        self.beta = beta
        self.loss = torch.nn.MSELoss()

    def forward(self, preds, target, w=None):
        loss = self.loss(preds, target)
        w_long = torch.concatenate([w, torch.zeros((w.size(-2), 16000 - w.size(-1))).to(w.device)], dim=1)
        w_hat = torch.sum(torch.abs(torch.fft.fft(w_long, dim=1)[:,:16000//2])**2, dim=0)

        kappa = w_hat.max() / w_hat.min()

        return loss, loss + self.beta * (kappa - 1), kappa.item()