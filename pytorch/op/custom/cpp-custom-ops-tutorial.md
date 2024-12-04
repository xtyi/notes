# Custom C++ and CUDA Operators

PyTorch offers a large library of operators that work on Tensors (e.g. torch.add, torch.sum, etc). 

However, you may wish to bring a new custom operator to PyTorch. This tutorial demonstrates the blessed path to authoring a custom operator written in C++/CUDA.

For our tutorial, we’ll demonstrate how to author a fused multiply-add C++ and CUDA operator that composes with PyTorch subsystems. The semantics of the operation are as follows:

```py
def mymuladd(a: Tensor, b: Tensor, c: float):
    return a * b + c
```

You can find the end-to-end working example for this tutorial [here](https://github.com/pytorch/extension-cpp).

## Setting up the Build System

If you are developing custom C++/CUDA code, it must be compiled.

Note that if you’re interfacing with a Python library that already has bindings to precompiled C++/CUDA code, you might consider writing a custom Python operator instead ([Python Custom Operators](https://pytorch.org/tutorials/advanced/python_custom_ops.html)).


Use torch.utils.cpp_extension to compile custom C++/CUDA code for use with PyTorch C++ extensions may be built either “ahead of time” with setuptools, or “just in time” via load_inline; we’ll focus on the “ahead of time” flavor.

使用 `torch.utils.cpp_extension` 去编译自定义的 C++/CUDA code

Using c`pp_extension` is as simple as writing the following setup.py:

```py
from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name="extension_cpp",
      ext_modules=[
          cpp_extension.CppExtension("extension_cpp", ["muladd.cpp"])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
```

If you need to compile CUDA code (for example, .cu files), then instead use `torch.utils.cpp_extension.CUDAExtension`. Please see how extension-cpp for an example for how this is set up.


## Defining the custom op and adding backend implementations

First, let’s write a C++ function that computes mymuladd:

```cpp
at::Tensor mymuladd_cpu(at::Tensor a, const at::Tensor& b, double c) {
    // 检查
    TORCH_CHECK(a.sizes() == b.sizes());
    TORCH_CHECK(a.dtype() == at::kFloat);
    TORCH_CHECK(b.dtype() == at::kFloat);
    TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CPU);

    at::Tensor a_contig = a.contiguous();
    at::Tensor b_contig = b.contiguous();
    
    // 输出 tensor 内存分配, 包括指定 shape 和 dtype
    at::Tensor result = torch::empty(a_contig.sizes(), a_contig.options());
    
    // 使用裸指针进行计算
    const float* a_ptr = a_contig.data_ptr<float>();
    const float* b_ptr = b_contig.data_ptr<float>();
    float* result_ptr = result.data_ptr<float>();
    for (int64_t i = 0; i < result.numel(); i++) {
        result_ptr[i] = a_ptr[i] * b_ptr[i] + c;
    }
    
    // 返回包装的 Tensor
    return result;
}
```

In order to use this from PyTorch’s Python frontend, we need to register it as a PyTorch operator using the TORCH_LIBRARY API. This will automatically bind the operator to Python.

Operator registration is a two step-process:
- Defining the operator - This step ensures that PyTorch is aware of the new operator.
- Registering backend implementations - In this step, implementations for various backends, such as CPU and CUDA, are associated with the operator.

### Defining an operator


To define an operator, follow these steps:

1. select a namespace for an operator. We recommend the namespace be the name of your top-level project; we’ll use “extension_cpp” in our tutorial.
2. provide a schema string that specifies the input/output types of the operator and if an input Tensors will be mutated. We support more types in addition to Tensor and float; please see The Custom Operators Manual for more details.
    - If you are authoring an operator that can mutate its input Tensors, please see here (Creating mutable operators) for how to specify that.

```cpp
TORCH_LIBRARY(extension_cpp, m) {
   // Note that "float" in the schema corresponds to the C++ double type
   // and the Python float type.
   m.def("mymuladd(Tensor a, Tensor b, float c) -> Tensor");
 }
```

This makes the operator available from Python via torch.ops.extension_cpp.mymuladd.

### Registering backend implementations for an operator

Use TORCH_LIBRARY_IMPL to register a backend implementation for the operator.

```cpp
TORCH_LIBRARY_IMPL(extension_cpp, CPU, m) {
  m.impl("mymuladd", &mymuladd_cpu);
}
```

If you also have a CUDA implementation of myaddmul, you can register it in a separate TORCH_LIBRARY_IMPL block:

```cpp
// 实现 CUDA kernel, 接受一个 numel 和裸指针, 只关心计算, 不关心内存分配
__global__ void muladd_kernel(int numel, const float* a, const float* b, float c, float* result) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numel) result[idx] = a[idx] * b[idx] + c;
}

at::Tensor mymuladd_cuda(const at::Tensor& a, const at::Tensor& b, double c) {
    // 检查
    TORCH_CHECK(a.sizes() == b.sizes());
    TORCH_CHECK(a.dtype() == at::kFloat);
    TORCH_CHECK(b.dtype() == at::kFloat);
    TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CUDA);

    at::Tensor a_contig = a.contiguous();
    at::Tensor b_contig = b.contiguous();

    // 输出 tensor 内存分配, 包括指定 shape 和 dtype
    at::Tensor result = torch::empty(a_contig.sizes(), a_contig.options());
    
    // 使用裸指针进行计算
    const float* a_ptr = a_contig.data_ptr<float>();
    const float* b_ptr = b_contig.data_ptr<float>();
    float* result_ptr = result.data_ptr<float>();
    int numel = a_contig.numel();
    muladd_kernel<<<(numel+255)/256, 256>>>(numel, a_ptr, b_ptr, c, result_ptr);

    return result;
}
```

### Adding torch.compile support for an operator

To add `torch.compile` support for an operator, we must add a FakeTensor kernel (also known as a “meta kernel” or “abstract impl”).

FakeTensors are Tensors that have metadata (such as shape, dtype, device) but no data: the FakeTensor kernel for an operator specifies how to compute the metadata of output tensors given the metadata of input tensors.

The FakeTensor kernel should return dummy Tensors of your choice with the correct Tensor metadata (shape/strides/dtype/device).

We recommend that this be done from Python via the `torch.library.register_fake` API, though it is possible to do this from C++ as well (see The Custom Operators Manual for more details).

```py
# Important: the C++ custom operator definitions should be loaded first
# before calling ``torch.library`` APIs that add registrations for the
# C++ custom operator(s). The following import loads our
# C++ custom operator definitions.
# See the next section for more details.
from . import _C

@torch.library.register_fake("extension_cpp::mymuladd")
def _(a, b, c):
    torch._check(a.shape == b.shape)
    torch._check(a.dtype == torch.float)
    torch._check(b.dtype == torch.float)
    torch._check(a.device == b.device)
    return torch.empty_like(a)
```

### Setting up hybrid Python/C++ registration
