# Python Custom Operators

PyTorch offers a large library of operators that work on Tensors (e.g. torch.add, torch.sum, etc).

However, you might wish to use a new customized operator with PyTorch, perhaps written by a third-party library.

This tutorial shows how to wrap Python functions so that they behave like PyTorch native operators.

PyTorch 提供了一个 Tensor 库，提供了大量的 op, 例如 torch.add, torch.sum

然而, 你可能想要使用一个新的自定义 op 用于 PyTorch, 这些 op 可能使用第三方库编写

这个教程展示了怎么包装一个 Python 函数，可以使它表现为一个 PyTorch 原生的 op

Reasons why you may wish to create a custom operator in PyTorch include:
- Treating an arbitrary Python function as an opaque callable with respect to torch.compile (that is, prevent torch.compile from tracing into the function).
- Adding training support to an arbitrary Python function

Please note that if your operation can be expressed as a composition of existing PyTorch operators, then there is usually no need to use the custom operator API – everything (for example torch.compile, training support) should just work.

## Example: Wrapping PIL’s crop into a custom operator

Let’s say that we are using PIL’s crop operation.

```py
import torch
from torchvision.transforms.functional import to_pil_image, pil_to_tensor
import PIL
import IPython
import matplotlib.pyplot as plt


def crop(pic, box):
    # 将 Tensor pic 移动到 cpu 上, 再转换成 PIL 对象
    img = to_pil_image(pic.cpu())
    # 调用 PIL 对象的 crop 方法, 切割图像
    cropped_img = img.crop(box)
    # 转换回 Tensor 对象, 再移动到原来的设备上
    return pil_to_tensor(cropped_img).to(pic.device) / 255.

def display(img):
    plt.imshow(img.numpy().transpose((1, 2, 0)))

img = torch.ones(3, 64, 64)
img *= torch.linspace(0, 1, steps=64) * torch.linspace(0, 1, steps=64).unsqueeze(-1)
display(img)
```

```py
cropped_img = crop(img, (10, 10, 50, 50))
display(cropped_img)
```

crop is not handled effectively out-of-the-box by torch.compile: torch.compile induces a “graph break” on functions it is unable to handle and graph breaks are bad for performance.

The following code demonstrates this by raising an error (torch.compile with `fullgraph=True` raises an error if a graph break occurs).

目前 torch.compile 无法处理 crop, 所以会导致 graph break, graph break 是会影响性能的

```py
@torch.compile(fullgraph=True)
def f(img):
    return crop(img, (10, 10, 50, 50))

# The following raises an error. Uncomment the line to see it.
# cropped_img = f(img)
```

In order to black-box crop for use with `torch.compile`, we need to do two things:
1. wrap the function into a PyTorch custom operator.
2. add a “FakeTensor kernel” (aka “meta kernel”) to the operator. Given some FakeTensors inputs (dummy Tensors that don’t have storage), this function should return dummy Tensors of your choice with the correct Tensor metadata (shape/strides/dtype/device).

```py
from typing import Sequence

# Use torch.library.custom_op to define a new custom operator.
# If your operator mutates any input Tensors, their names must be specified
# in the ``mutates_args`` argument.
@torch.library.custom_op("mylib::crop", mutates_args=())
def crop(pic: torch.Tensor, box: Sequence[int]) -> torch.Tensor:
    img = to_pil_image(pic.cpu())
    cropped_img = img.crop(box)
    return (pil_to_tensor(cropped_img) / 255.).to(pic.device, pic.dtype)

# Use register_fake to add a ``FakeTensor`` kernel for the operator
@crop.register_fake
def _(pic, box):
    channels = pic.shape[0]
    x0, y0, x1, y1 = box
    return pic.new_empty(channels, y1 - y0, x1 - x0)
```

After this, crop now works without graph breaks:


```py
@torch.compile(fullgraph=True)
def f(img):
    return crop(img, (10, 10, 50, 50))

cropped_img = f(img)
display(img)
display(cropped_img)
```


## Adding training support for crop

Use `torch.library.register_autograd` to add training support for an operator. Prefer this over directly using `torch.autograd.Function`; some compositions of `autograd.Function` with PyTorch operator registration APIs can lead to (and has led to) silent incorrectness when composed with `torch.compile`.

If you don’t need training support, there is no need to use `torch.library.register_autograd`. If you end up training with a custom_op that doesn’t have an autograd registration, we’ll raise an error message.


...


## Testing Python Custom operators

Use `torch.library.opcheck` to test that the custom operator was registered correctly. This does not test that the gradients are mathematically correct; please write separate tests for that (either manual ones or torch.autograd.gradcheck).

使用 `torch.library.opcheck` 测试自定义 op 被正确注册

这不会测试逻辑是否正确

To use opcheck, pass it a set of example inputs to test against. If your operator supports training, then the examples should include Tensors that require grad. If your operator supports multiple devices, then the examples should include Tensors from each device.

```py
examples = [
    [torch.randn(3, 64, 64), [0, 0, 10, 10]],
    [torch.randn(3, 91, 91, requires_grad=True), [10, 0, 20, 10]],
    [torch.randn(3, 60, 60, dtype=torch.double), [3, 4, 32, 20]],
    [torch.randn(3, 512, 512, requires_grad=True, dtype=torch.double), [3, 4, 32, 45]],
]

for example in examples:
    torch.library.opcheck(crop, example)
```


## Mutable Python Custom operators



## Conclusion