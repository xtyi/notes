# torch.library

torch.library is a collection of APIs for extending PyTorch’s core library of operators.

It contains utilities for testing custom operators, creating new custom operators, and extending operators defined with PyTorch’s C++ operator registration APIs (e.g. aten operators).

## Testing custom ops


Use torch.library.opcheck() to test custom ops for incorrect usage of the Python torch.library and/or C++ TORCH_LIBRARY APIs. Also, if your operator supports training, use torch.autograd.gradcheck() to test that the gradients are mathematically correct.

## Creating new custom ops in Python

Use torch.library.custom_op() to create new custom ops.

```
torch.library.custom_op(name, fn=None, /, *, mutates_args, device_types=None, schema=None)
```

Wraps a function into custom operator.

Reasons why you may want to create a custom op include:
- Wrapping a third-party library or custom kernel to work with PyTorch subsystems like Autograd.
- Preventing torch.compile/export/FX tracing from peeking inside your function.

This API is used as a decorator around a function (please see examples). The provided function must have type hints; these are needed to interface with PyTorch’s various subsystems.

Parameters
- name(str) - A name for the custom op that looks like “{namespace}::{name}”, e.g. “mylib::my_linear”. The name is used as the op’s stable identifier in PyTorch subsystems (e.g. torch.export, FX graphs). To avoid name collisions, please use your project name as the namespace; e.g. all custom ops in pytorch/fbgemm use “fbgemm” as the namespace.
- mutates_args (`Iterable[str]` or "unknown") – The names of args that the function mutates. This MUST be accurate, otherwise, the behavior is undefined. If “unknown”, it pessimistically assumes that all inputs to the operator are being mutated. 
- device_types (None | str | Sequence[str]) – The device type(s) the function is valid for. If no device type is provided, then the function is used as the default implementation for all device types. Examples: “cpu”, “cuda”. When registering a device-specific implementation for an operator that accepts no Tensors, we require the operator to have a “device: torch.device argument”.
- schema (None | str) – A schema string for the operator. If None (recommended) we’ll infer a schema for the operator from its type annotations. We recommend letting us infer a schema unless you have a specific reason not to. Example: “(Tensor x, int y) -> (Tensor, Tensor)”.


Return type: Callable


> We recommend not passing in a schema arg and instead letting us infer it from the type annotations.
> It is error-prone to write your own schema.
> You may wish to provide your own schema if our interpretation of the type annotation is not what you want.
> For more info on how to write a schema string, see [here](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md#func).


Examples

```py
import torch
from torch import Tensor
from torch.library import custom_op
import numpy as np

@custom_op("mylib::numpy_sin", mutates_args=())
def numpy_sin(x: Tensor) -> Tensor:
    x_np = x.cpu().numpy()
    y_np = np.sin(x_np)
    return torch.from_numpy(y_np).to(device=x.device)

x = torch.randn(3)
y = numpy_sin(x)
assert torch.allclose(y, x.sin())

# Example of a custom op that only works for one device type.
@custom_op("mylib::numpy_sin_cpu", mutates_args=(), device_types="cpu")
def numpy_sin_cpu(x: Tensor) -> Tensor:
    x_np = x.numpy()
    y_np = np.sin(x_np)
    return torch.from_numpy(y_np)

x = torch.randn(3)
y = numpy_sin_cpu(x)
assert torch.allclose(y, x.sin())

# Example of a custom op that mutates an input
@custom_op("mylib::numpy_sin_inplace", mutates_args={"x"}, device_types="cpu")
def numpy_sin_inplace(x: Tensor) -> None:
    x_np = x.numpy()
    np.sin(x_np, out=x_np)

x = torch.randn(3)
expected = x.sin()
numpy_sin_inplace(x)
assert torch.allclose(x, expected)

# Example of a factory function
@custom_op("mylib::bar", mutates_args={}, device_types="cpu")
def bar(device: torch.device) -> Tensor:
    return torch.ones(3)
bar("cpu")
```

## Extending custom ops (created from Python or C++)

Use the `register.*` methods, such as `torch.library.register_kernel()` and `func:torch.library.register_fake`, to add implementations for any operators (they may have been created using `torch.library.custom_op()` or via PyTorch’s C++ operator registration APIs).

### register_kernel

```py
torch.library.register_kernel(op, device_types, func=None, /, *, lib=None)
```

Register an implementation for a device type for this operator.

Some valid device_types are: “cpu”, “cuda”, “xla”, “mps”, “ipu”, “xpu”. This API may be used as a decorator.

Parameters
- fn (Callable) – The function to register as the implementation for the given device types.
- device_types (None | str | Sequence[str]) – The device_types to register an impl to. If None, we will register to all device types – please only use this option if your implementation is truly device-type-agnostic.

Examples

```py
import torch
from torch import Tensor
from torch.library import custom_op
import numpy as np

# Create a custom op that works on cpu
@custom_op("mylib::numpy_sin", mutates_args=(), device_types="cpu")
def numpy_sin(x: Tensor) -> Tensor:
    x_np = x.numpy()
    y_np = np.sin(x_np)
    return torch.from_numpy(y_np)

# Add implementations for the cuda device
@torch.library.register_kernel("mylib::numpy_sin", "cuda")
def _(x):
    x_np = x.cpu().numpy()
    y_np = np.sin(x_np)
    return torch.from_numpy(y_np).to(device=x.device)

x_cpu = torch.randn(3)
x_cuda = x_cpu.cuda()

assert torch.allclose(numpy_sin(x_cpu), x_cpu.sin())
assert torch.allclose(numpy_sin(x_cuda), x_cuda.sin())
```

### register_autograd

```py

```

### register_fake

```py
torch.library.register_fake(op, func=None, /, *, lib=None, _stacklevel=1)
```

Register a FakeTensor implementation (“fake impl”) for this operator.

Also sometimes known as a “meta kernel”, “abstract impl”.

An “FakeTensor implementation” specifies the behavior of this operator on Tensors that carry no data (“FakeTensor”).

Given some input Tensors with certain properties (sizes/strides/storage_offset/device), it specifies what the properties of the output Tensors are.

The FakeTensor implementation has the same signature as the operator.

It is run for both FakeTensors and meta tensors.

To write a FakeTensor implementation, assume that all Tensor inputs to the operator are regular CPU/CUDA/Meta tensors, but they do not have storage, and you are trying to return regular CPU/CUDA/Meta tensor(s) as output.

The FakeTensor implementation must consist of only PyTorch operations (and may not directly access the storage or data of any input or intermediate Tensors).

This API may be used as a decorator (see examples).

Examples

```py
import torch
import numpy as np
from torch import Tensor

# Example 1: an operator without data-dependent output shape
@torch.library.custom_op("mylib::custom_linear", mutates_args=())
def custom_linear(x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
    raise NotImplementedError("Implementation goes here")

@torch.library.register_fake("mylib::custom_linear")
def _(x, weight, bias):
    assert x.dim() == 2
    assert weight.dim() == 2
    assert bias.dim() == 1
    assert x.shape[1] == weight.shape[1]
    assert weight.shape[0] == bias.shape[0]
    assert x.device == weight.device
    return (x @ weight.t()) + bias

with torch._subclasses.fake_tensor.FakeTensorMode():
    x = torch.randn(2, 3)
    w = torch.randn(3, 3)
    b = torch.randn(3)
    y = torch.ops.mylib.custom_linear(x, w, b)
assert y.shape == (2, 3)

# Example 2: an operator with data-dependent output shape
@torch.library.custom_op("mylib::custom_nonzero", mutates_args=())
def custom_nonzero(x: Tensor) -> Tensor:
    x_np = x.numpy(force=True)
    res = np.stack(np.nonzero(x_np), axis=1)
    return torch.tensor(res, device=x.device)

@torch.library.register_fake("mylib::custom_nonzero")
def _(x):
# Number of nonzero-elements is data-dependent.
# Since we cannot peek at the data in an fake impl,
# we use the ctx object to construct a new symint that
# represents the data-dependent size.
    ctx = torch.library.get_ctx()
    nnz = ctx.new_dynamic_size()
    shape = [nnz, x.dim()]
    result = x.new_empty(shape, dtype=torch.int64)
    return result

from torch.fx.experimental.proxy_tensor import make_fx
x = torch.tensor([0, 1, 2, 3, 4, 0])
trace = make_fx(torch.ops.mylib.custom_nonzero, tracing_mode="symbolic")(x)
trace.print_readable()
assert torch.allclose(trace(x), torch.ops.mylib.custom_nonzero(x))
```



### register_vmap

```py
torch.library.register_vmap(op, func=None, /, *, lib=None)
```

Register a vmap implementation to support torch.vmap() for this custom op.

This API may be used as a decorator (see examples).

In order for an operator to work with torch.vmap(), you may need to register a vmap implementation in the following signature:

```py
vmap_func(info, in_dims: Tuple[Optional[int]], *args, **kwargs)
```

where *args and **kwargs are the arguments and kwargs for op. We do not support kwarg-only Tensor args.



### get_ctx

```
torch.library.get_ctx()
```

get_ctx() returns the current AbstractImplCtx object.

Calling get_ctx() is only valid inside of an fake impl (see torch.library.register_fake() for more usage details.

