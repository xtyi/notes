# torch.export

> This feature is a prototype under active development and there WILL BE BREAKING CHANGES in the future.

## Overview

torch.export.export() takes an arbitrary Python callable (a torch.nn.Module, a function or a method) and produces a traced graph representing only the Tensor computation of the function in an Ahead-of-Time (AOT) fashion, which can subsequently be executed with different outputs or serialized.

```py
import torch
from torch.export import export

class Mod(torch.nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        a = torch.sin(x)
        b = torch.cos(y)
        return a + b

example_args = (torch.randn(10, 10), torch.randn(10, 10))

exported_program: torch.export.ExportedProgram = export(
    Mod(), args=example_args
)
print(exported_program)
```

打印

```
ExportedProgram:
    class GraphModule(torch.nn.Module):
        def forward(self, arg0_1: f32[10, 10], arg1_1: f32[10, 10]):
            # code: a = torch.sin(x)
            sin: f32[10, 10] = torch.ops.aten.sin.default(arg0_1);

            # code: b = torch.cos(y)
            cos: f32[10, 10] = torch.ops.aten.cos.default(arg1_1);

            # code: return a + b
            add: f32[10, 10] = torch.ops.aten.add.Tensor(sin, cos);
            return (add,)

    Graph signature: ExportGraphSignature(
        parameters=[],
        buffers=[],
        user_inputs=['arg0_1', 'arg1_1'],
        user_outputs=['add'],
        inputs_to_parameters={},
        inputs_to_buffers={},
        buffers_to_mutate={},
        backward_signature=None,
        assertion_dep_token=None,
    )
    Range constraints: {}
```

torch.export produces a clean intermediate representation (IR) with the following invariants. More specifications about the IR can be found [here](https://pytorch.org/docs/stable/export.ir_spec.html#export-ir-spec).

- Soundness: It is guaranteed to be a sound representation of the original program, and maintains the same calling conventions of the original program.
- Normalized: There are no Python semantics within the graph. Submodules from the original programs are inlined to form one fully flattened computational graph. 只有一个完全平铺的计算图
- Graph properties: The graph is purely functional, meaning it does not contain operations with side effects such as mutations or aliasing. It does not mutate any intermediate values, parameters, or buffers. 原地的 op 会被转换成函数式的
- Metadata: The graph contains metadata captured during tracing, such as a stacktrace from user’s code.

Under the hood, torch.export leverages the following latest technologies:

- TorchDynamo (torch._dynamo) is an internal API that uses a CPython feature called the Frame Evaluation API to safely trace PyTorch graphs.
    This provides a massively improved graph capturing experience, with much fewer rewrites needed in order to fully trace the PyTorch code.
- AOT Autograd provides a functionalized PyTorch graph and ensures the graph is decomposed/lowered to the ATen operator set.
- Torch FX (torch.fx) is the underlying representation of the graph, allowing flexible Python-based transformations.

### Existing frameworks

torch.export 和 torch.compile 使用相同的技术，下面是它们的对比

torch.compile() also utilizes the same PT2 stack as torch.export, but is slightly different:

- JIT vs. AOT: torch.compile() is a JIT compiler whereas which is not intended to be used to produce compiled artifacts outside of deployment.

- Partial vs. Full Graph Capture: When torch.compile() runs into an untraceable part of a model, it will “graph break” and fall back to running the program in the eager Python runtime. In comparison, torch.export aims to get a full graph representation of a PyTorch model, so it will error out when something untraceable is reached. Since torch.export produces a full graph disjoint from any Python features or runtime, this graph can then be saved, loaded, and run in different environments and languages.

- Usability tradeoff: Since torch.compile() is able to fallback to the Python runtime whenever it reaches something untraceable, it is a lot more flexible. torch.export will instead require users to provide more information or rewrite their code to make it traceable.

下面是和之前技术的对比

Compared to torch.fx.symbolic_trace(), torch.export traces using TorchDynamo which operates at the Python bytecode level, giving it the ability to trace arbitrary Python constructs not limited by what Python operator overloading supports. Additionally, torch.export keeps fine-grained track of tensor metadata, so that conditionals on things like tensor shapes do not fail tracing. In general, torch.export is expected to work on more user programs, and produce lower-level graphs (at the torch.ops.aten operator level). Note that users can still use torch.fx.symbolic_trace() as a preprocessing step before torch.export.

Compared to torch.jit.script(), torch.export does not capture Python control flow or data structures, but it supports more Python language features than TorchScript (as it is easier to have comprehensive coverage over Python bytecodes). The resulting graphs are simpler and only have straight line control flow (except for explicit control flow operators).

Compared to torch.jit.trace(), torch.export is sound: it is able to trace code that performs integer computation on sizes and records all of the side-conditions necessary to show that a particular trace is valid for other inputs.

## Exporting a PyTorch Model

### An Example

The main entrypoint is through torch.export.export(), which takes a callable (torch.nn.Module, function, or method) and sample inputs, and captures the computation graph into an torch.export.ExportedProgram. An example:

```py
import torch
from torch.export import export

# Simple module for demonstration
class M(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=3, padding=1
        )
        self.relu = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3)

    def forward(self, x: torch.Tensor, *, constant=None) -> torch.Tensor:
        a = self.conv(x)
        a.add_(constant)
        return self.maxpool(self.relu(a))

example_args = (torch.randn(1, 3, 256, 256),)
example_kwargs = {"constant": torch.ones(1, 16, 256, 256)}

exported_program: torch.export.ExportedProgram = export(
    M(), args=example_args, kwargs=example_kwargs
)
print(exported_program)
```

```
ExportedProgram:
    class GraphModule(torch.nn.Module):
        def forward(self, arg0_1: f32[16, 3, 3, 3], arg1_1: f32[16], arg2_1: f32[1, 3, 256, 256], arg3_1: f32[1, 16, 256, 256]):

            # code: a = self.conv(x)
            convolution: f32[1, 16, 256, 256] = torch.ops.aten.convolution.default(
                arg2_1, arg0_1, arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1
            );

            # code: a.add_(constant)
            add: f32[1, 16, 256, 256] = torch.ops.aten.add.Tensor(convolution, arg3_1);

            # code: return self.maxpool(self.relu(a))
            relu: f32[1, 16, 256, 256] = torch.ops.aten.relu.default(add);
            max_pool2d_with_indices = torch.ops.aten.max_pool2d_with_indices.default(
                relu, [3, 3], [3, 3]
            );
            getitem: f32[1, 16, 85, 85] = max_pool2d_with_indices[0];
            return (getitem,)

    Graph signature: ExportGraphSignature(
        parameters=['L__self___conv.weight', 'L__self___conv.bias'],
        buffers=[],
        user_inputs=['arg2_1', 'arg3_1'],
        user_outputs=['getitem'],
        inputs_to_parameters={
            'arg0_1': 'L__self___conv.weight',
            'arg1_1': 'L__self___conv.bias',
        },
        inputs_to_buffers={},
        buffers_to_mutate={},
        backward_signature=None,
        assertion_dep_token=None,
    )
    Range constraints: {}
```

Inspecting the ExportedProgram, we can note the following:
- The torch.fx.Graph contains the computation graph of the original program, along with records of the original code for easy debugging.
- The graph contains only torch.ops.aten operators found here and custom operators, and is fully functional, without any inplace operators such as torch.add_.
- The parameters (weight and bias to conv) are lifted as inputs to the graph, resulting in no `get_attr` nodes in the graph, which previously existed in the result of torch.fx.symbolic_trace().
- The torch.export.ExportGraphSignature models the input and output signature, along with specifying which inputs are parameters.
- The resulting shape and dtype of tensors produced by each node in the graph is noted. For example, the convolution node will result in a tensor of dtype torch.float32 and shape (1, 16, 256, 256). 💡

每个节点的输出的 shape 和 dtype 已经都有标注


### Non-Strict Export

In PyTorch 2.3, we introduced a new mode of tracing called non-strict mode.

It’s still going through hardening, so if you run into any issues, please file them to Github with the “oncall: export” tag.

In non-strict mode, we trace through the program using the Python interpreter.

Your code will execute exactly as it would in eager mode; the only difference is that all Tensor objects will be replaced by ProxyTensors, which will record all their operations into a graph.

In strict mode, which is currently the default, we first trace through the program using TorchDynamo, a bytecode analysis engine.

TorchDynamo does not actually execute your Python code. Instead, it symbolically analyzes it and builds a graph based on the results.

This analysis allows torch.export to provide stronger guarantees about safety, but not all Python code is supported.

An example of a case where one might want to use non-strict mode is if you run into a unsupported TorchDynamo feature that might not be easily solved, and you know the python code is not exactly needed for computation. For example:

```py
import contextlib
import torch

class ContextManager():
    def __init__(self):
        self.count = 0
    def __enter__(self):
        self.count += 1
    def __exit__(self, exc_type, exc_value, traceback):
        self.count -= 1

class M(torch.nn.Module):
    def forward(self, x):
        with ContextManager():
            return x.sin() + x.cos()

export(M(), (torch.ones(3, 3),), strict=False)  # Non-strict traces successfully
export(M(), (torch.ones(3, 3),))  # Strict mode fails with torch._dynamo.exc.Unsupported: ContextManager
```

In this example, the first call using non-strict mode (through the strict=False flag) traces successfully whereas the second call using strict mode (default) results with a failure, where TorchDynamo is unable to support context managers.

One option is to rewrite the code (see Limitations of torch.export), but seeing as the context manager does not affect the tensor computations in the model, we can go with the non-strict mode’s result.

### Expressing Dynamism



