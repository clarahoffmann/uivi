# PyTorch Implementation of Unbiased Implicit Variational Inference (Titsias \& Ruiz, 2019)

This repository implements Unbiased-Implicit Variational Inference (UI-VI) ([Titsias \& Ruiz, 2019](https://proceedings.mlr.press/v89/titsias19a/titsias19a.pdf)) in PyTorch.
UI-VI allows to fit flexible variational distributions by training a simple neural network.

Much of the code is based on the HMC-VAE implementation of Haoran Peng (https://github.com/GavinPHR/HMC-VAE).

# Setup:
1. **Pre-commit hooks**: To set up the pre-commit hooks run
```pre-commit install```

2. **Install poerty environemtn** Install the environment via 
```poetry install```
and register a kernel for the environment with
```poetry run python -m ipykernel install --user --name "uivi"```

# ToDos:
- [] Add VAE example
- [] Set step size in HMC sampler dynamically 
