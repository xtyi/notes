# functorch

https://github.com/pytorch/functorch

functorch is JAX-like composable function transforms for PyTorch.

It aims to provide composable vmap and grad transforms that work with PyTorch modules and PyTorch autograd with good eager-mode performance.

In addition, there is experimental functionality to trace through these transformations using FX in order to capture the results of these transforms ahead of time. This would allow us to compile the results of vmap or grad to improve performance.

