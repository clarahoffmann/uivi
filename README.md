# PyTorch Implementation of Unbiased Implicit Variational Inference (Titsias \& Ruiz, 2019)

This repository implements Unbiased-Implicit Variational Inference (UI-VI) ([Titsias \& Ruiz, 2019](https://proceedings.mlr.press/v89/titsias19a/titsias19a.pdf)) in PyTorch.
UI-VI allows to fit flexible variational distributions by training a simple neural network.
Two examples from the original paper are replicated:

### Banana-shaped distribution
Samples generated with a fitted UI-VI (red) on a banana-shaped distribution (blue):
<div align="center">
  <img src="https://raw.githubusercontent.com/clarahoffmann/uivi/main/uivi/banana.png" alt="Banana distribution" width="500"/>
</div>

### VAE trained on MNIST
Original (left) and reconstructed (left) images
<div align="center">
  <img src="https://github.com/clarahoffmann/uivi/blob/main/uivi/original_images.png" alt="Original MNIST images" width="300"/>
  <img src="https://github.com/clarahoffmann/uivi/blob/main/uivi/reconstructed_images.png" alt="Original MNIST images" width="300"/>
</div>

A large part of the code is based on the elegant HMC-VAE implementation of Haoran Peng (https://github.com/GavinPHR/HMC-VAE).

# Setup:
1. *Pre-commit hooks*: 
To set up the pre-commit hooks run ```pre-commit install```

2. *Install poetry environment*:
Install the environment via ```poetry install```  and register a kernel for the environment using

```poetry run python -m ipykernel install --user --name "uivi"```

# Caveats
*Latent Dimension:* Unlike in classic VAEs, there is no bottleneck needed for the latent z. The opposite holds, the VAE + UI-VI only produces good fits for large latent dimensions.
For example, with 14x14 MNIST images, the layer dimensions $[14*14, 64, 64, 70]$ worked well for the encoder. In the original paper, it's recommended to set the latent dimension to 200.
This resulted in pretty slow training.

*Adapting the HMC step size*: The HMC step size is adapted with RMSprop as in the original paper. Still, this often resulted in large steps for epsilon prime. When passing these values through the network, NaNs were generated. To prevent this, I added gradient clamping in the sampler. The clamping threshold is purely heuristic. This can probably be handled more elegantly :) Replacing standard HMC with a NUTS sampler would probably be the best alternative.
