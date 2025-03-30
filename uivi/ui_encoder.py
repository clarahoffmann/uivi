"""UIVI encoder class."""
import lightning as L
import torch
from hmc import HMC
from torch import nn
from torch.distributions import MultivariateNormal, Normal


# pylint: disable = C0116, R0902
class UIEncoder(L.LightningModule):
    """Encoder with HMC step for the reverse conditional."""

    def __init__(
        self,
        mean,
        cov,
        num_eps_samples=1,
        dim_eps=3,
        dim_z=2,
        T=5,
        Ls=5,
        latent_dim=100,
    ):

        super().__init__()
        self.encoder_mu_sigma = nn.Sequential(
            nn.Linear(dim_eps, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, dim_z * 2),
        )

        self.num_eps_samples = num_eps_samples
        self.dim_eps = dim_eps
        self.dim_z = dim_z

        self.mean_batched = mean.repeat(self.num_eps_samples, 1)
        self.cov_batched = cov.repeat(self.num_eps_samples, 1, 1)
        self.dist = MultivariateNormal(
            loc=self.mean_batched, covariance_matrix=self.cov_batched
        )

        self.hmc = HMC(dim_z, T, Ls)

    def forward(self, epsilon=None):
        # 1. Sample input variable
        if epsilon is None:
            epsilon = torch.randn(
                self.num_eps_samples, self.dim_eps
            ).requires_grad_(False)

        # 2. Forward pass h_\theta(u_s; eps_s)
        out = self.encoder_mu_sigma(epsilon)
        mu = out[:, :2]
        sigma = torch.exp(out[:, 2:]) + 1e-7
        u_sample = torch.randn_like(mu)  # sample reparam variable u
        z_sample = mu + sigma * u_sample
        return mu, z_sample, epsilon, sigma

    def log_p_z(self, z):
        coord = torch.stack([z[:, 0], z[:, 1] + z[:, 0] ** 2 + 1], dim=1)
        log_p = self.dist.log_prob(coord)
        return log_p

    def register_log_prob(self, z_samples):
        def log_prob(epsilon):
            is_train = self.training
            self.eval()
            mu_sample, _, _, sigma_sample = self.forward(epsilon)

            if is_train:
                self.train()
            return self.HMC_bound(epsilon, z_samples, mu_sample, sigma_sample)

        self.hmc.register_log_prob(log_prob)

    def run_hmc(self, epsilon, z):
        self.register_log_prob(z)
        epsilon_samples, accept_prob = self.hmc(epsilon.detach().clone())
        return epsilon_samples, accept_prob

    def HMC_bound(self, epsilon, z_samples, mu_sample, sigma_sample):
        Dz = MultivariateNormal(
            loc=mu_sample,
            covariance_matrix=torch.diag_embed(sigma_sample**2),
        )
        log_qz_cond_eps_prime = Dz.log_prob(z_samples)
        p_eps_dist = Normal(0, 1)
        log_p_eps = p_eps_dist.log_prob(epsilon).sum(dim=-1)
        return log_qz_cond_eps_prime + log_p_eps
