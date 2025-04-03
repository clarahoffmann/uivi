"""Utilities for plotting."""

from typing import Tuple

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.figure import Figure
from numpy.lib.stride_tricks import sliding_window_view
from torchvision import utils
from tqdm import tqdm
from ui_encoder import UIEncoder


def make_meshgrid(
    lims_x: Tuple[float, float] = (-2.5, 2.5),
    lims_y: Tuple[float, float] = (-8, 2),
    num_points: int = 100,
) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """Create a flattened tensor meshgrid."""
    xx, yy = np.meshgrid(
        np.linspace(*lims_x, num_points), np.linspace(*lims_y, num_points)
    )
    grid = np.stack((xx, yy), axis=-1).reshape(-1, 2)

    return torch.tensor(grid), xx, yy


def get_log_probs_banana(
    mean: np.ndarray, cov: np.ndarray, grid: torch.Tensor, m: int, n: int
) -> np.ndarray:
    """Evaluate banana density over a meshgrid with dimension m,n"""
    encoder_true = UIEncoder(
        mean=mean, cov=cov, num_eps_samples=grid.shape[0], dim_eps=3, dim_z=2
    )

    log_probs = encoder_true.log_p_z(grid)
    log_probs = np.asarray(log_probs.clone().detach().reshape(m, n))

    return log_probs


def draw_uivi_samples(
    model: L.LightningModule, num_samples: int = 300
) -> Tuple[np.ndarray, np.ndarray]:
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


def plot_samples(
    z_samples: torch.Tensor,
    xx: np.ndarray,
    yy: np.ndarray,
    log_probs: np.ndarray,
) -> Tuple[Figure, np.ndarray]:
    """Plot UI-VI samples and contours."""

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


def plot_losses(losses_mod, losses_ent):
    """Plot the components of the training loss:
    mode, entropy and mode + entropy (full ELBO)."""
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10, 3))
    window_size = 100
    mod = sliding_window_view(losses_mod, window_shape=window_size)
    ent = sliding_window_view(losses_ent, window_shape=window_size)

    axs[0].plot(np.mean(mod, axis=-1))
    axs[0].set_title(r"Mode $- p(z)$ ($\downarrow$)")

    axs[1].plot(np.mean(ent, axis=-1))
    axs[1].set_title(r"Entropy $q(z)$ ($\uparrow$)")

    axs[2].plot(-np.mean(mod - ent, axis=-1))
    axs[2].set_title(r" ELBO $\uparrow$")

    fig.suptitle("Losses", fontsize=16)
    fig.subplots_adjust(top=0.80)

    plt.show()


def plot_mnist_samples(model, test_loader, img_dim=14):
    """Plot a grid of true and predicted MNIST images."""
    model.eval()

    def to_img(x):
        x = x.clamp(0, 1)
        return x

    def show_image(img):
        fig = plt.figure()
        img = to_img(img)
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.axis("off")
        return fig

    def reshape_for_plotting(img_batch):
        return img_batch.view(-1, 1, img_dim, img_dim)

    def visualise_output(images, model):

        with torch.no_grad():
            fig = plt.figure()
            _, _, _, _, images_recon = model(images)
            images_recon = torch.sigmoid(
                images_recon.view(-1, 1, img_dim, img_dim)
            )
            np_imagegrid = utils.make_grid(images_recon[1:50], 10, 5).numpy()
            plt.imshow(np.transpose(np_imagegrid, (1, 2, 0)))
            plt.axis("off")
            plt.show()
        return images_recon, fig

    images, _ = next(iter(test_loader))

    # First visualise the original images
    print("Original images")
    fig_orig = show_image(
        utils.make_grid(reshape_for_plotting(images[1:50]), 10, 5)
    )
    plt.show()

    # Reconstruct and visualise the images using the vae
    print("VAE reconstruction:")
    images_recon, fig_recon = visualise_output(images, model)

    return images_recon, fig_orig, fig_recon
