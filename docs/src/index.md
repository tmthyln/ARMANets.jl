# ARMANets.jl Documentation
This package provides a Julia, Flux-based implementation of the paper [ARMA Nets: Expanding Receptive Field for Dense Prediction](https://arxiv.org/abs/2002.11609). (The [official implementation](https://github.com/umd-huang-lab/ARMA-Networks/) from the authors is in PyTorch.)

The layer `ARConv` is an implementation of the autoregressive component[^1] of the ARMA layer (the moving average component is just an ordinary convolution). For more details, see v1 and v2 of the paper.

[^1]: This is the reparameterized variant, as the general ARMA layer is not BIBO stable and is very difficult to train on its own.