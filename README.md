# PyTorch Implementation of Unbiased Implicit Variational Inference (Titsias \& Ruiz, 2019)

This repository implements Unbiased-Implicit Variational Inference (UI-VI) ([Titsias \& Ruiz, 2019](https://proceedings.mlr.press/v89/titsias19a/titsias19a.pdf)) in PyTorch.
UI-VI allows to fit flexible variational distributions by training a simple neural network.

Example of UI-VI (red) on a banana-shaped distribution (blue):
<div align="center">
  <img src="https://raw.githubusercontent.com/clarahoffmann/uivi/main/uivi/banana.png" alt="Banana distribution" width="700"/>
</div>



A large part of the code is based on the elegant HMC-VAE implementation of Haoran Peng (https://github.com/GavinPHR/HMC-VAE).

# Setup:
1. *Pre-commit hooks*: 
To set up the pre-commit hooks run ```pre-commit install```

2. *Install poetry environment*:
Install the environment via ```poetry install```  and register a kernel for the environment using

```poetry run python -m ipykernel install --user --name "uivi"```

# ToDos:
- [ ] Add VAE example
- [ ] Set step size in HMC sampler dynamically 
