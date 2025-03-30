"""VAE class."""


import lightning as L
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal

# pylint: disable = W0632, C0115, C0116


class VAE(L.LightningModule):
    def __init__(
        self,
        dim_x: int,
        dim_eps: int,
        dim_z: list[int],
        latent_dims: list[int],
    ):
        super().__init__()

        self.dim_z = dim_z
        self.dim_eps = dim_eps
        self.pz = Normal(0, 1)

        self.encoder = self.build_model(
            [dim_x + dim_eps] + latent_dims + [dim_z * 2]
        )
        self.decoder = self.build_model(
            [dim_z] + list(reversed(latent_dims)) + [dim_x]
        )

    @staticmethod
    def build_model(dims):
        layers = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
        layers = layers[:-1]  # drop last-layer ReLU

        return nn.Sequential(*layers)

    def forward(self, x, epsilon=None):

        if epsilon is None:
            epsilon = torch.randn(x.shape[0], self.dim_eps).requires_grad_(
                False
            )

        out = self.encoder(torch.cat([x, epsilon], dim=-1))

        mu = out[:, : self.dim_z]
        sigma = torch.exp(out[:, (self.dim_z) :]) + 1e-7

        u_sample = torch.randn_like(mu)  # sample reparam variable u
        z_sample = mu + sigma * u_sample

        x_recon = self.decoder(z_sample)
        return mu, z_sample, x[:, (self.dim_eps) :], sigma, x_recon

    def decode(self, z_sample):
        return self.decoder(z_sample)

    def elbo_no_entropy(self, x, x_recon, z_sample):

        recon_loss = F.binary_cross_entropy_with_logits(
            x_recon, x, reduction="mean"
        )

        return recon_loss - self.pz.log_prob(z_sample)
