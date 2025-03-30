"""Utilities for plotting."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from tqdm import tqdm
from ui_encoder import UIEncoder


def make_meshgrid(lims_x=(-2.5, 2.5), lims_y=(-8, 2), num_points=100):
    """Create a flattened tensor meshgrid."""
    xx, yy = np.meshgrid(
        np.linspace(*lims_x, num_points), np.linspace(*lims_y, num_points)
    )
    grid = np.stack((xx, yy), axis=-1).reshape(-1, 2)

    return torch.tensor(grid), xx, yy


def get_log_probs_banana(
    mean: np.array, cov: np.array, grid: torch.tensor, m: int, n: int
):
    """Evaluate banana density over a meshgrid with dimension m,n"""
    encoder_true = UIEncoder(
        mean=mean, cov=cov, num_eps_samples=grid.shape[0], dim_eps=3, dim_z=2
    )

    log_probs = encoder_true.log_p_z(grid)
    log_probs = np.asarray(log_probs.clone().detach().reshape(m, n))

    return log_probs


def draw_uivi_samples(model, num_samples: int = 300):
    """Generate samples from fitted UIVI model."""
    z_samples = []
    mus = []
    model.eval()
    for _ in tqdm(range(num_samples)):
        with torch.no_grad():
            mu, z_sample, _, _ = model.forward()
        z_samples.append(z_sample)
        mus.append(mu)

    z_samples = torch.stack(z_samples).detach().cpu().numpy().reshape(-1, 2)
    mus = torch.stack(mus).detach().cpu().numpy().reshape(-1, 2)

    return z_samples, mus


def plot_samples(z_samples, xx, yy, log_probs):
    """Plot samples and contours."""

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

    axs[0].contour(
        xx, yy, np.exp(log_probs), levels=10, cmap="Blues", alpha=0.8
    )
    axs[0].scatter(
        z_samples[:, 0],
        z_samples[:, 1],
        color="red",
        s=3,
        alpha=1,
        label="UIVI samples",
    )
    axs[0].set_title("Samples")
    axs[0].legend()

    axs[1].contour(
        xx, yy, np.exp(log_probs), levels=10, cmap="Blues", alpha=0.8
    )
    sns.kdeplot(
        x=z_samples[:, 0],
        y=z_samples[:, 1],
        fill=False,
        cmap="Reds",
        thresh=0.01,
        levels=10,
        ax=axs[1],
    )
    axs[1].set_title("Contour")

    plt.show()

    return fig, axs
