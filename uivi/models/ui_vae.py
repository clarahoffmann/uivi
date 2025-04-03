"""UI-VI VAE."""

import torch
from torch.distributions import MultivariateNormal, Normal

from uivi.hmc.hmc import HMC
from uivi.models.vae import VAE

# pylint: disable = C0115, C0116, W0632


class UIVAE(VAE):
    """VAE with UI-VI."""

    def __init__(
        self,
        dim_x: int,
        dim_eps: int,
        dim_z: list[int],
        latent_dims: list[int],
        T: int,
        Ls=int,
        step_size: torch.Tensor = torch.tensor(0.2),
    ):

        super().__init__(dim_x, dim_eps, dim_z, latent_dims)
        self.hmc = HMC(sum(latent_dims), T, Ls, step_size)

    def register_log_prob(self, x, z_samples):
        def log_prob(epsilon):
            is_train = self.training
            self.eval()
            mu_sample, _, _, sigma_sample, _ = self.forward(
                x=x, epsilon=epsilon
            )

            if is_train:
                self.train()
            return self.HMC_bound(epsilon, z_samples, mu_sample, sigma_sample)

        self.hmc.register_log_prob(log_prob)

    def run_hmc(self, x, epsilon, z):
        self.register_log_prob(x, z)
        epsilon_samples, accept_prob = self.hmc(epsilon.detach().clone())
        return epsilon_samples, accept_prob

    def HMC_bound(self, epsilon, z_samples, mu_sample, sigma_sample):
        # check here again what the correct bound is, does VAE loss need
        # to be in here?
        Dz = MultivariateNormal(
            loc=mu_sample,
            covariance_matrix=torch.diag_embed(sigma_sample**2),
        )
        log_qz_cond_eps_prime = Dz.log_prob(z_samples)
        p_eps_dist = Normal(0, 1)
        log_p_eps = p_eps_dist.log_prob(epsilon).sum(dim=-1)
        return log_qz_cond_eps_prime + log_p_eps
