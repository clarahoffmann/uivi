"""Training loop for UIVI."""

import torch
from torch import optim
from torch.distributions import MultivariateNormal
from tqdm import tqdm


def rmsprop_scalar_step(
    step_size, grad, square_avg, lr=0.003, alpha=0.9, eps=1e-8
):
    """Root mean square propagation for the HMC step size."""
    square_avg = alpha * square_avg + (1 - alpha) * grad**2
    adjusted_grad = grad / (square_avg.sqrt() + eps)
    step_size = step_size - lr * adjusted_grad.item()
    return step_size, square_avg


# pylint: disable = R0914
def get_ent_grad(
    model, epsilon, z_sample, num_hmc_samples, mu, sigma, mode="vae"
):
    """Compute entropy term of gradient."""
    with torch.no_grad():
        # run hmc for reverse conditional
        if mode == "vae":
            x = epsilon[:, : -model.dim_eps].detach().clone()
            epsilon = epsilon[:, -model.dim_eps :].detach().clone()
            eps_prime, accept_prob = model.run_hmc(x, epsilon, z_sample)

            x_prime = x.unsqueeze(0).repeat(num_hmc_samples, 1, 1)
            mu_prime, _, _, sigma_sample_prime, _ = model.forward(
                x=x_prime.reshape(-1, model.dim_x),
                epsilon=eps_prime.reshape(-1, model.dim_eps),
            )
            bs = x.shape[0]
        elif mode == "banana":
            eps_prime, accept_prob = model.run_hmc(epsilon, z_sample)
            mu_prime, _, _, sigma_sample_prime = model.forward(
                eps_prime.reshape(-1, model.dim_eps)
            )
            bs = model.num_eps_samples

        # reshape to account for hmc sample dim
        mu_prime = mu_prime.reshape(num_hmc_samples, bs, model.dim_z)
        sigma_sample_prime = sigma_sample_prime.reshape(
            num_hmc_samples, bs, model.dim_z
        )
        z_sample_prime = z_sample.unsqueeze(0).repeat(num_hmc_samples, 1, 1)

        # compute entropy gradient w.r.t. z
        grad_z = -((z_sample_prime - mu_prime) / (sigma_sample_prime**2))

        # first mc integrate out epsilon, then average over
        # hmc samples
        grad_z = grad_z.mean(dim=0).mean(dim=0)

        # compute entropy loss for tracking
        Dz = MultivariateNormal(
            loc=mu, covariance_matrix=torch.diag_embed(sigma**2)
        )
        log_qz = Dz.log_prob(z_sample)
        log_qz = log_qz.mean()

    return grad_z, accept_prob, log_qz


def forward_model(model, mode, x_batch=None, num_hmc_samples=5):
    """Forward pass data through model and compute loss.
    The loss and inputs for the VAE and simple banana example are different."""

    if mode == "vae":
        mu, z_sample, epsilon, sigma, x_recon = model.forward(x_batch)

        # retain grad to add entropy gradient later
        z_sample.retain_grad()

        # compute mode loss + gradient, keep graph for entropy gradient
        loss = model.elbo_no_entropy(x_batch, x_recon, z_sample).mean()
        loss.backward(retain_graph=True)

        grad_z, accept_prob, log_qz = get_ent_grad(
            model,
            torch.cat([x_batch, epsilon], dim=-1),
            z_sample,
            num_hmc_samples,
            mu,
            sigma,
            "vae",
        )

    elif mode == "banana":
        mu, z_sample, epsilon, sigma = model.forward()

        # retain grad to add entropy gradient later
        z_sample.retain_grad()

        # compute mode loss + gradient, keep graph for entropy gradient
        loss = -model.log_p_z(z_sample).mean()
        loss.backward(retain_graph=True)

        grad_z, accept_prob, log_qz = get_ent_grad(
            model, epsilon, z_sample, num_hmc_samples, mu, sigma
        )
    return grad_z, accept_prob, log_qz


# pylint: disable=R0914
def train_uivi_vae(
    model, train_loader, num_epochs, lr=1e-3, num_hmc_samples=5
):
    """Train UI-VI VAE."""

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    losses_mod = []
    losses_ent = []

    square_avg = 0

    for epoch in range(num_epochs):
        with tqdm(train_loader, desc=f"Epoch {epoch + 1}") as pbar:
            for x_batch, _ in pbar:
                optimizer.zero_grad()

                mu, z_sample, epsilon, sigma, x_recon = model.forward(x_batch)

                # retain grad to add entropy gradient later
                z_sample.retain_grad()

                # compute mode loss + gradient, keep graph for entropy gradient
                loss = model.elbo_no_entropy(x_batch, x_recon, z_sample).mean()
                loss.backward(retain_graph=True)

                grad_z, accept_prob, log_qz = get_ent_grad(
                    model,
                    torch.cat([x_batch, epsilon], dim=-1),
                    z_sample,
                    num_hmc_samples,
                    mu,
                    sigma,
                    "vae",
                )
                z_sample.grad += grad_z.detach()

                # Compute new gradients w.r.t. model parameters
                # using the modified z_sample.grad
                grads = torch.autograd.grad(
                    z_sample,
                    model.parameters(),
                    grad_outputs=z_sample.grad,
                    retain_graph=False,
                    allow_unused=True,
                )

                # Assign computed gradients to model parameters
                for param, grad in zip(model.parameters(), grads):
                    if grad is not None:
                        param.grad = grad

                # backpropagate mode and entropy gradient
                optimizer.step()

                losses_mod.append(loss.item())
                losses_ent.append(log_qz.item())

                sq_grad = z_sample.grad.clone().detach().mean()
                model.hmc.step_size, square_avg = rmsprop_scalar_step(
                    model.hmc.step_size, sq_grad, square_avg
                )

            pbar.set_postfix(
                loss=f"{loss.item():05.2f}",
                log_qz=f"{log_qz.item():05.2f}",
                grad_z=f"{grad_z.mean().item():05.2f}",
                accept_prob=f"{accept_prob.mean().item():05.2f}",
                status="running",
            )

            pbar.update(1)

    return model, losses_mod, losses_ent


# pylint: disable=R0914
def train_uivi_banana(
    model,
    lr=1e-3,
    num_iter=5000,
    num_hmc_samples=5,
    rms_prop=True,
):
    """Train UI-VI model to mode a distribution p(z).
    It is possible to sample from p(z)."""

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    losses_mod = []
    losses_ent = []

    if rms_prop:
        square_avg = 0

    with tqdm(total=num_iter) as pbar:
        for _ in range(num_iter):
            optimizer.zero_grad()
            mu, z_sample, epsilon, sigma = model.forward()

            # retain grad to add entropy gradient later
            z_sample.retain_grad()

            # compute mode loss + gradient, keep graph for entropy gradient
            loss = -model.log_p_z(z_sample).mean()
            loss.backward(retain_graph=True)

            grad_z, accept_prob, log_qz = get_ent_grad(
                model, epsilon, z_sample, num_hmc_samples, mu, sigma
            )
            z_sample.grad += grad_z.detach()

            # Compute new gradients w.r.t. model parameters
            # using the modified z_sample.grad
            grads = torch.autograd.grad(
                z_sample,
                model.parameters(),
                grad_outputs=z_sample.grad,
                retain_graph=False,
            )

            # Assign computed gradients to model parameters
            for param, grad in zip(model.parameters(), grads):
                param.grad = grad

            # backpropagate mode and entropy gradient
            optimizer.step()

            losses_mod.append(loss.item())
            losses_ent.append(log_qz.item())

            if rms_prop:
                # update hmc step size with RMSprop
                sq_grad = z_sample.grad.clone().detach().mean()
                model.hmc.step_size, square_avg = rmsprop_scalar_step(
                    model.hmc.step_size, sq_grad, square_avg
                )

            pbar.set_postfix(
                loss=f"{loss.item():05.2f}",
                log_qz=f"{log_qz.item():05.2f}",
                grad_z=f"{grad_z.mean().item():05.2f}",
                accept_prob=f"{accept_prob.mean().item():05.2f}",
                status="running",
            )

            pbar.update(1)

    return model, losses_mod, losses_ent
