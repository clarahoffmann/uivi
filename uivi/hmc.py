"""Hamiltonian Monte Carlo (HMC) for the reverse conditional."""

import torch
from torch import nn
from typing import Callable


class HMC(nn.Module):
    """Hamiltonian Monte Carlo implementation
    based on https://github.com/GavinPHR/HMC-VAE.

    The step size is not set dynamically with RMSProp
    as in Titsias, Ruiz.
    """

    def __init__(
        self,
        dim: int,
        T: int,
        L: int,
    ):
        super().__init__()
        self.dim = dim
        self.log_prob: Callable = None  # type: ignore
        self.T = T
        self.L = L
        self.step_size = torch.tensor(0.2)

    def register_log_prob(self, log_prob: Callable):
        self.log_prob = log_prob

    def grad_log_prob(self, x):
        with torch.enable_grad():
            x = x.clone().detach()
            x.requires_grad = True
            logprob = self.log_prob(x).sum()
            grad = torch.autograd.grad(logprob, x)[0]
            # clamp gradients to avoid exploding
            # epsilon values
            grad = torch.clamp(grad, min=-5, max=5)
            return grad

    def leapfrog(self, x, p):
        eps = self.step_size
        p = p + 0.5 * eps * self.grad_log_prob(x)
        for _ in range(self.L - 1):
            x = x + eps * p
            p = p + eps * self.grad_log_prob(x)
        x = x + eps * p
        p = p + 0.5 * eps * self.grad_log_prob(x)
        return x, p

    def HMC_step(self, x_old):
        def H(x, p):
            return -self.log_prob(x) + 0.5 * torch.sum(p.pow(2), dim=-1)

        p_old = torch.randn_like(x_old)
        x_new, p_new = self.leapfrog(x_old.clone(), p_old.clone())
        log_accept_prob = -(H(x_new, p_new) - H(x_old, p_old))
        log_accept_prob[log_accept_prob > 0] = 0

        accept = torch.log(torch.rand_like(log_accept_prob)) < log_accept_prob
        accept = accept.unsqueeze(dim=-1)
        ret = x_new * accept + x_old * torch.logical_not(accept), accept.sum() / accept.numel()
        return ret

    def forward(self, x):
        accept_probs = []
        samples = []
        for _ in range(self.T):
            x, accept_prob = self.HMC_step(x)
            accept_probs.append(accept_prob)
            samples.append(x.clone().detach())
        accept_prob = torch.mean(torch.tensor(accept_probs))
        # return last 5 samples of HMC chain
        return torch.stack(samples)[-5:, :, :], accept_prob