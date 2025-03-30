"""Training loop for UIVI."""

from tqdm import tqdm
import torch
from torch import optim
from torch.distributions import MultivariateNormal


def get_ent_grad(model, epsilon, z_sample, num_hmc_samples, mu, sigma):
    with torch.no_grad():
        # run hmc for reverse conditional
        eps_prime, accept_prob = model.run_hmc(epsilon, z_sample)
        mu_prime, _,_, sigma_sample_prime = model.forward(eps_prime.reshape(-1, model.dim_eps))

        # reshape to account for hmc sample dim
        mu_prime = mu_prime.reshape(num_hmc_samples, model.num_eps_samples, model.dim_z)
        sigma_sample_prime = sigma_sample_prime.reshape(num_hmc_samples, model.num_eps_samples, model.dim_z)
        z_sample_prime = z_sample.unsqueeze(0).repeat(num_hmc_samples, 1, 1)

        # compute entropy gradient w.r.t. z
        grad_z = - ((z_sample_prime - mu_prime)/(sigma_sample_prime**2))

        # first mc integrate out epsilon, then average over
        # hmc samples
        grad_z = grad_z.mean(dim = 0).mean(dim = 0)

        # compute entropy loss for tracking
        Dz = MultivariateNormal(loc = mu, covariance_matrix = torch.diag_embed(sigma**2))
        log_qz =  Dz.log_prob(z_sample)
        log_qz = log_qz.mean()
    
    return grad_z, accept_prob, log_qz



def train_uivi(model, lr = 1e-3,  num_iter = 5000, num_hmc_samples = 5 ):
    """Train UI-VI model."""
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    losses_mod = []
    losses_ent = []

    with tqdm(total = num_iter) as pbar:
        for i in range(num_iter):
            optimizer.zero_grad()
            mu, z_sample, epsilon, sigma = model.forward()

            # retain grad to add entropy gradient later
            z_sample.retain_grad()
            
            # compute mode loss + gradient, keep graph for entropy gradient
            loss = -  model.log_p_z(z_sample).mean()
            loss.backward(retain_graph = True) 

            grad_z, accept_prob, log_qz = get_ent_grad(model, epsilon, z_sample, num_hmc_samples, mu, sigma)
            z_sample.grad += grad_z.detach()
            
            # Compute new gradients w.r.t. model parameters using the modified z_sample.grad
            grads = torch.autograd.grad(z_sample, model.parameters(), grad_outputs=z_sample.grad, retain_graph=False)

            # Assign computed gradients to model parameters
            for param, grad in zip(model.parameters(), grads):
                param.grad = grad
            
            # backpropagate mode and entropy gradient
            optimizer.step()
            
            losses_mod.append(loss.item())
            losses_ent.append(log_qz.item())
            
            pbar.set_postfix(
                loss=f"{loss.item():05.2f}",
                log_qz=f"{log_qz.item():05.2f}",
                grad_z=f"{grad_z.mean().item():05.2f}",
                accept_prob=f"{accept_prob.mean().item():05.2f}",
                status="running")
            
            pbar.update(1)
    
    return model, losses_mod, losses_ent

