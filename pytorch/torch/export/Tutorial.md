# torch.export Tutorial

> torch.export and its related features are in prototype status and are subject to backwards compatibility breaking changes. This tutorial provides a snapshot of torch.export usage as of PyTorch 2.3.

torch.export() is the PyTorch 2.X way to export PyTorch models into standardized model representations, intended to be run on different (i.e. Python-less) environments. The official documentation can be found here.

In this tutorial, you will learn how to use torch.export() to extract ExportedProgram’s (i.e. single-graph representations) from PyTorch programs. We also detail some considerations/modifications that you may need to make in order to make your model compatible with torch.export.

Contents
- Basic Usage
- Graph Breaks
- Non-Strict Export
- Control Flow Ops
- Constraints/Dynamic Shapes
- Custom Ops
- Decompositions
- ExportDB
- Running the Exported Program
- Conclusion

## Basic Usage

torch.export extracts single-graph representations from PyTorch programs by tracing the target function, given example inputs. torch.export.export() is the main entry point for torch.export.

In this tutorial, torch.export and torch.export.export() are practically synonymous, though torch.export generally refers to the PyTorch 2.X export process, and torch.export.export() generally refers to the actual function call.

The signature of torch.export.export() is:

```py
export(
    f: Callable,
    args: Tuple[Any, ...],
    kwargs: Optional[Dict[str, Any]] = None,
    *,
    dynamic_shapes: Optional[Dict[str, Dict[int, Dim]]] = None
) -> ExportedProgram
```

`torch.export.export()` traces the tensor computation graph from calling `f(*args, **kwargs)` and wraps it in an `ExportedProgram`, which can be serialized or executed later with different inputs.

Note that while the output `ExportedGraph` is callable and can be called in the same way as the original input callable, it is not a `torch.nn.Module`.

We will detail the `dynamic_shapes` argument later in the tutorial.

```py
import torch
from torch.export import export

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(100, 10)

    def forward(self, x, y):
        return torch.nn.functional.relu(self.lin(x + y), inplace=True)

mod = MyModule()
exported_mod = export(mod, (torch.randn(8, 100), torch.randn(8, 100)))
print(type(exported_mod))
print(exported_mod.module()(torch.randn(8, 100), torch.randn(8, 100)))
```

Let's review some attributes of ExportedProgram that are of interest.


The graph attribute is an FX graph traced from the function we exported, that is, the computation graph of all PyTorch operations.

The FX graph has some important properties:
- The operations are "ATen-level" operations.
- The graph is "functionalized", meaning that no operations are mutations.

The graph_module attribute is the GraphModule that wraps the graph attribute so that it can be ran as a torch.nn.Module.

```py
print(exported_mod)
```

打印

```
ExportedProgram:
    class GraphModule(torch.nn.Module):
        def forward(self, arg0_1: "f32[10, 100]", arg1_1: "f32[10]", arg2_1: "f32[8, 100]", arg3_1: "f32[8, 100]"):
            # File: /tmp/ipykernel_3925/3766860120.py:10 in forward, code: return torch.nn.functional.relu(self.lin(x + y), inplace=True)
            add: "f32[8, 100]" = torch.ops.aten.add.Tensor(arg2_1, arg3_1);  arg2_1 = arg3_1 = None
            t: "f32[100, 10]" = torch.ops.aten.t.default(arg0_1);  arg0_1 = None
            addmm: "f32[8, 10]" = torch.ops.aten.addmm.default(arg1_1, add, t);  arg1_1 = add = t = None
            relu: "f32[8, 10]" = torch.ops.aten.relu.default(addmm);  addmm = None
            return (relu,)
            
Graph signature: ExportGraphSignature(input_specs=[InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='arg0_1'), target='lin.weight', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='arg1_1'), target='lin.bias', persistent=None), InputSpec(kind=<InputKind.USER_INPUT: 1>, arg=TensorArgument(name='arg2_1'), target=None, persistent=None), InputSpec(kind=<InputKind.USER_INPUT: 1>, arg=TensorArgument(name='arg3_1'), target=None, persistent=None)], output_specs=[OutputSpec(kind=<OutputKind.USER_OUTPUT: 1>, arg=TensorArgument(name='relu'), target=None)])
Range constraints: {}
```

主要是 GraphModule, Graph Signature, Range Constraints 三部分


```py
type(exported_mod) # torch.export.exported_program.ExportedProgram
type(exported_mod.graph) # torch.fx.graph.Graph
type(exported_mod.graph_module) # torch.fx.graph_module.GraphModule.__new__.<locals>.GraphModuleImpl
type(exported_mod.graph_signature) # torch.export.graph_signature.ExportGraphSignature
type(exported_mod.range_constraints) # dict
exported_mod.graph is exported_mod.graph_module.graph # True
```



```py
print(exported_mod.graph_module)
```

打印

```
GraphModule()



def forward(self, arg0_1, arg1_1, arg2_1, arg3_1):
    add = torch.ops.aten.add.Tensor(arg2_1, arg3_1);  arg2_1 = arg3_1 = None
    t = torch.ops.aten.t.default(arg0_1);  arg0_1 = None
    addmm = torch.ops.aten.addmm.default(arg1_1, add, t);  arg1_1 = add = t = None
    relu = torch.ops.aten.relu.default(addmm);  addmm = None
    return (relu,)
    
# To see more debug info, please use `graph_module.print_readable()`
```

```py
pprint(exported_mod.graph_signature)
```

打印

```
ExportGraphSignature(input_specs=[InputSpec(kind=<InputKind.PARAMETER: 2>,
                                            arg=TensorArgument(name='arg0_1'),
                                            target='lin.weight',
                                            persistent=None),
                                  InputSpec(kind=<InputKind.PARAMETER: 2>,
                                            arg=TensorArgument(name='arg1_1'),
                                            target='lin.bias',
                                            persistent=None),
                                  InputSpec(kind=<InputKind.USER_INPUT: 1>,
                                            arg=TensorArgument(name='arg2_1'),
                                            target=None,
                                            persistent=None),
                                  InputSpec(kind=<InputKind.USER_INPUT: 1>,
                                            arg=TensorArgument(name='arg3_1'),
                                            target=None,
                                            persistent=None)],
                     output_specs=[OutputSpec(kind=<OutputKind.USER_OUTPUT: 1>,
                                              arg=TensorArgument(name='relu'),
                                              target=None)])
```

The printed code shows that FX graph only contains ATen-level ops (such as torch.ops.aten) and that mutations were removed.

For example, the mutating op `torch.nn.functional.relu(..., inplace=True)` is represented in the printed code by `torch.ops.aten.relu.default`, which does not mutate.

Future uses of input to the original mutating relu op are replaced by the additional new output of the replacement non-mutating relu op.

Other attributes of interest in ExportedProgram include:
- graph_signature -- the inputs, outputs, parameters, buffers, etc. of the exported graph.
- range_constraints -- constraints, covered later

# Graph Break

Although torch.export shares components with torch.compile, the key limitation of torch.export, especially when compared to torch.compile, is that it does not support graph breaks.

This is because handling graph breaks involves interpreting the unsupported operation with default Python evaluation, which is incompatible with the export use case. Therefore, in order to make your model code compatible with torch.export, you will need to modify your code to remove graph breaks.

尽管 torch.export 和 torch.compile 使用相同的组件, torch.export 的关键限制是不支持 graph break

这是因为处理 graph break 涉及到使用 Python 解释执行不支持的操作，这和 export 的使用常见是不兼容的

因此，为了使你的模型代码兼容 torch.export，你需要去修改代码移除 graph break


A graph break is necessary in cases such as:

- data-dependent control flow

```py
class Bad1(torch.nn.Module):
    def forward(self, x):
        if x.sum() > 0:
            return torch.sin(x)
        return torch.cos(x)

import traceback as tb
try:
    export(Bad1(), (torch.randn(3, 3),))
except Exception:
    tb.print_exc()
```

- accessing tensor data with .data

```py
class Bad2(torch.nn.Module):
    def forward(self, x):
        x.data[0, 0] = 3
        return x

try:
    export(Bad2(), (torch.randn(3, 3),))
except Exception:
    tb.print_exc()
```

- calling unsupported functions (such as many built-in functions)

```py
class Bad3(torch.nn.Module):
    def forward(self, x):
        x = x + 1
        return x + id(x)

try:
    export(Bad3(), (torch.randn(3, 3),))
except Exception:
    tb.print_exc()
```

- unsupported Python language features (e.g. throwing exceptions, match statements)

```py
class Bad4(torch.nn.Module):
    def forward(self, x):
        try:
            x = x + 1
            raise RuntimeError("bad")
        except:
            x = x + 2
        return x

try:
    export(Bad4(), (torch.randn(3, 3),))
except Exception:
    tb.print_exc()
```

## Non-Strict Export

To trace the program, torch.export uses TorchDynamo, a byte code analysis engine, to symbolically analyze the Python code and build a graph based on the results.

This analysis allows torch.export to provide stronger guarantees about safety, but not all Python code is supported, causing these graph breaks.

To address this issue, in PyTorch 2.3, we introduced a new mode of exporting called non-strict mode, where we trace through the program using the Python interpreter executing it exactly as it would in eager mode, allowing us to skip over unsupported Python features. This is done through adding a `strict=False` flag.

Looking at some of the previous examples which resulted in graph breaks:

- Accessing tensor data with .data now works correctly

```py
class Bad2(torch.nn.Module):
    def forward(self, x):
        x.data[0, 0] = 3
        return x

bad2_nonstrict = export(Bad2(), (torch.randn(3, 3),), strict=False)
print(bad2_nonstrict.module()(torch.ones(3, 3)))
```

- Calling unsupported functions (such as many built-in functions) traces

through, but in this case, id(x) gets specialized as a constant integer in the graph. This is because id(x) is not a tensor operation, so the operation is not recorded in the graph.

```py
class Bad3(torch.nn.Module):
    def forward(self, x):
        x = x + 1
        return x + id(x)

bad3_nonstrict = export(Bad3(), (torch.randn(3, 3),), strict=False)
print(bad3_nonstrict)
print(bad3_nonstrict.module()(torch.ones(3, 3)))
```

Unsupported Python language features (such as throwing exceptions, match statements) now also get traced through.

```py
class Bad4(torch.nn.Module):
    def forward(self, x):
        try:
            x = x + 1
            raise RuntimeError("bad")
        except:
            x = x + 2
        return x

bad4_nonstrict = export(Bad4(), (torch.randn(3, 3),), strict=False)
print(bad4_nonstrict.module()(torch.ones(3, 3)))
```

However, there are still some features that require rewrites to the original module:

## Control Flow Ops

torch.export actually does support data-dependent control flow.

But these need to be expressed using control flow ops. For example, we can fix the control flow example above using the cond op, like so:

```py
from functorch.experimental.control_flow import cond

class Bad1Fixed(torch.nn.Module):
    def forward(self, x):
        def true_fn(x):
            return torch.sin(x)
        def false_fn(x):
            return torch.cos(x)
        return cond(x.sum() > 0, true_fn, false_fn, [x])

exported_bad1_fixed = export(Bad1Fixed(), (torch.randn(3, 3),))
print(exported_bad1_fixed.module()(torch.ones(3, 3)))
print(exported_bad1_fixed.module()(-torch.ones(3, 3)))
```

There are limitations to cond that one should be aware of:
- The predicate (i.e. x.sum() > 0) must result in a boolean or a single-element tensor.
- The operands (i.e. [x]) must be tensors.
- The branch function (i.e. true_fn and false_fn) signature must match with the operands and they must both return a single tensor with the same metadata (for example, dtype, shape, etc.).
- Branch functions cannot mutate input or global variables.
- Branch functions cannot access closure variables, except for self if the function is defined in the scope of a method.

## Constraints/Dynamic Shapes

Ops can have different specializations/behaviors for different tensor shapes, so by default, `torch.export` requires inputs to ExportedProgram to have the same shape as the respective example inputs given to the initial `torch.export.export()` call.

If we try to run the ExportedProgram in the example below with a tensor with a different shape, we get an error:

我们运行 ExportedProgram 传入的 tensor 需要与 torch.export 时候的 tensor 形状相同, 否则会报错

```py
class MyModule2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(100, 10)

    def forward(self, x, y):
        return torch.nn.functional.relu(self.lin(x + y), inplace=True)

mod2 = MyModule2()
exported_mod2 = export(mod2, (torch.randn(8, 100), torch.randn(8, 100)))

try:
    exported_mod2.module()(torch.randn(10, 100), torch.randn(10, 100))
except Exception:
    tb.print_exc()
```

We can relax this constraint using the `dynamic_shapes` argument of torch.export.export(), which allows us to specify, using `torch.export.Dim`, which dimensions of the input tensors are dynamic.

我们可以使用 `dynamic_shapes` 参数放宽这个限制，该参数允许我们指定 input tensor 的哪个维度是动态的(使用 `torch.export.Dim`)

For each tensor argument of the input callable, we can specify a mapping from the dimension to a torch.export.Dim.

A torch.export.Dim is essentially a named symbolic integer with optional minimum and maximum bounds.

Then, the format of torch.export.export()'s dynamic_shapes argument is a mapping from the input callable’s tensor argument names, to dimension –> dim mappings as described above.

If there is no torch.export.Dim given to a tensor argument’s dimension, then that dimension is assumed to be static.

The first argument of torch.export.Dim is the name for the symbolic integer, used for debugging.

Then we can specify an optional minimum and maximum bound (inclusive). Below, we show a usage example.

In the example below, our input inp1 has an unconstrained first dimension, but the size of the second dimension must be in the interval [4, 18].

```py
from torch.export import Dim

inp1 = torch.randn(10, 10, 2)

class DynamicShapesExample1(torch.nn.Module):
    def forward(self, x):
        x = x[:, 2:]
        return torch.relu(x)

inp1_dim0 = Dim("inp1_dim0")
inp1_dim1 = Dim("inp1_dim1", min=4, max=18)
dynamic_shapes1 = {
    "x": {0: inp1_dim0, 1: inp1_dim1},
}

exported_dynamic_shapes_example1 = export(DynamicShapesExample1(), (inp1,), dynamic_shapes=dynamic_shapes1)

print(exported_dynamic_shapes_example1.module()(torch.randn(5, 5, 2)))

# dim1 小于指定范围
try:
    exported_dynamic_shapes_example1.module()(torch.randn(8, 1, 2))
except Exception:
    tb.print_exc()

# dim1 大于指定范围
try:
    exported_dynamic_shapes_example1.module()(torch.randn(8, 20, 2))
except Exception:
    tb.print_exc()

# dim2 不是动态的，与 export 时的维度值不同
try:
    exported_dynamic_shapes_example1.module()(torch.randn(8, 8, 3))
except Exception:
    tb.print_exc()
```

Note that if our example inputs to torch.export do not satisfy the constraints given by dynamic_shapes, then we get an error.

We can enforce that equalities between dimensions of different tensors by using the same torch.export.Dim object, for example, in matrix multiplication:

可以指定 Dim 之间的约束关系

```py
inp2 = torch.randn(4, 8)
inp3 = torch.randn(8, 2)

class DynamicShapesExample2(torch.nn.Module):
    def forward(self, x, y):
        return x @ y

inp2_dim0 = Dim("inp2_dim0")
inner_dim = Dim("inner_dim")
inp3_dim1 = Dim("inp3_dim1")

dynamic_shapes2 = {
    "x": {0: inp2_dim0, 1: inner_dim},
    "y": {0: inner_dim, 1: inp3_dim1},
}

exported_dynamic_shapes_example2 = export(DynamicShapesExample2(), (inp2, inp3), dynamic_shapes=dynamic_shapes2)

print(exported_dynamic_shapes_example2.module()(torch.randn(2, 16), torch.randn(16, 4)))

# inp2 的 dim1 不等于 inp3 的 dim0
try:
    exported_dynamic_shapes_example2.module()(torch.randn(4, 8), torch.randn(4, 2))
except Exception:
    tb.print_exc()
```

We can also describe one dimension in terms of other. There are some restrictions to how detailed we can specify one dimension in terms of another, but generally, those in the form of A * Dim + B should work.

```py
class DerivedDimExample1(torch.nn.Module):
    def forward(self, x, y):
        return x + y[1:]

foo = DerivedDimExample1()

x, y = torch.randn(5), torch.randn(6)
dimx = torch.export.Dim("dimx", min=3, max=6)
dimy = dimx + 1
derived_dynamic_shapes1 = ({0: dimx}, {0: dimy})

derived_dim_example1 = export(foo, (x, y), dynamic_shapes=derived_dynamic_shapes1)

print(derived_dim_example1.module()(torch.randn(4), torch.randn(5)))

# 不满足 dimy = dimx + 1
try:
    derived_dim_example1.module()(torch.randn(4), torch.randn(6))
except Exception:
    tb.print_exc()


class DerivedDimExample2(torch.nn.Module):
    def forward(self, z, y):
        return z[1:] + y[1::3]

foo = DerivedDimExample2()

z, y = torch.randn(4), torch.randn(10)
dx = torch.export.Dim("dx", min=3, max=6)
dz = dx + 1
dy = dx * 3 + 1
derived_dynamic_shapes2 = ({0: dz}, {0: dy})

derived_dim_example2 = export(foo, (z, y), dynamic_shapes=derived_dynamic_shapes2)
print(derived_dim_example2.module()(torch.randn(7), torch.randn(19)))
```

We can actually use torch.export to guide us as to which dynamic_shapes constraints are necessary.

We can do this by relaxing all constraints (recall that if we do not provide constraints for a dimension, the default behavior is to constrain to the exact shape value of the example input) and letting torch.export error out.

可以让 torch.export 指导我们哪些约束是必须的，我们只需要放宽所有约束，让 torch.export 报错

```py
inp4 = torch.randn(8, 16)
inp5 = torch.randn(16, 32)

class DynamicShapesExample3(torch.nn.Module):
    def forward(self, x, y):
        if x.shape[0] <= 16:
            return x @ y[:, :16]
        return y

dynamic_shapes3 = {
    "x": {i: Dim(f"inp4_dim{i}") for i in range(inp4.dim())},
    "y": {i: Dim(f"inp5_dim{i}") for i in range(inp5.dim())},
}

try:
    export(DynamicShapesExample3(), (inp4, inp5), dynamic_shapes=dynamic_shapes3)
except Exception:
    tb.print_exc()
```

We can see that the error message gives us suggested fixes to our dynamic shape constraints. Let us follow those suggestions (exact suggestions may differ slightly):

```py
def suggested_fixes():
    inp4_dim1 = Dim('shared_dim')
    # suggested fixes below
    inp4_dim0 = Dim('inp4_dim0', max=16)
    inp5_dim1 = Dim('inp5_dim1', min=17)
    inp5_dim0 = inp4_dim1
    # end of suggested fixes
    return {
        "x": {0: inp4_dim0, 1: inp4_dim1},
        "y": {0: inp5_dim0, 1: inp5_dim1},
    }

dynamic_shapes3_fixed = suggested_fixes()
exported_dynamic_shapes_example3 = export(DynamicShapesExample3(), (inp4, inp5), dynamic_shapes=dynamic_shapes3_fixed)
print(exported_dynamic_shapes_example3.module()(torch.randn(4, 32), torch.randn(32, 64)))
```

Note that in the example above, because we constrained the value of `x.shape[0]` in dynamic_shapes_example3, the exported program is sound even though there is a raw `if` statement.

指定约束后，就不会 trace 不满足约束的分支, 因此即使有 if 语句，也不会造成 graph break

If you want to see why torch.export generated these constraints, you can re-run the script with the environment variable `TORCH_LOGS=dynamic,dynamo`, or use `torch._logging.set_logs`.

```py
import logging
torch._logging.set_logs(dynamic=logging.INFO, dynamo=logging.INFO)
exported_dynamic_shapes_example3 = export(DynamicShapesExample3(), (inp4, inp5), dynamic_shapes=dynamic_shapes3_fixed)

# reset to previous values
torch._logging.set_logs(dynamic=logging.WARNING, dynamo=logging.WARNING)
```

We can view an ExportedProgram’s symbolic shape ranges using the range_constraints field.

```py
print(exported_dynamic_shapes_example3.range_constraints)
```

## Custom Ops

torch.export can export PyTorch programs with custom operators.

Currently, the steps to register a custom op for use by torch.export are:

- Define the custom op using torch.library (reference) as with any other custom op

```py
@torch.library.custom_op("my_custom_library::custom_op", mutates_args={})
def custom_op(input: torch.Tensor) -> torch.Tensor:
    print("custom_op called!")
    return torch.relu(x)
```

- Define a "Meta" implementation of the custom op that returns an empty tensor with the same shape as the expected output

```py
@custom_op.register_fake 
def custom_op_meta(x):
    return torch.empty_like(x)
```

- Call the custom op from the code you want to export using `torch.ops`





## ExportDB

torch.export will only ever export a single computation graph from a PyTorch program.

Because of this requirement, there will be Python or PyTorch features that are not compatible with torch.export, which will require users to rewrite parts of their model code.

We have seen examples of this earlier in the tutorial -- for example, rewriting if-statements using cond.

torch.export 只能从 PyTorch 程序中导出一个单独的计算图

由于这个原因，总会有一些 Python 或 PyTorch 特性与 torch.export 不兼容, 因此需要用户重写模型代码, 例如，使用 cond 重写 if 语句



ExportDB is the standard reference that documents supported and unsupported Python/PyTorch features for torch.export.

It is essentially a list a program samples, each of which represents the usage of one particular Python/PyTorch feature and its interaction with torch.export.

ExportDB 是一个标准的参考，记录了 torch.export 支持和不支持的 Python/PyTorch 特性

它本质上是一个程序示例列表，每一个示例都代表一个 Python/PyTorch 特性的使用，以及与 torch.export 的交互



Examples are also tagged by category so that they can be more easily searched.



For example, let’s use ExportDB to get a better understanding of how the predicate works in the cond operator.

We can look at the example called cond_predicate, which has a torch.cond tag. The example code looks like:

```py
def cond_predicate(x):
    """
    The conditional statement (aka predicate) passed to ``cond()`` must be one of the following:
    - ``torch.Tensor`` with a single element
    - boolean expression
    NOTE: If the `pred` is test on a dim with batch size < 2, it will be specialized.
    """
    pred = x.dim() > 2 and x.shape[2] > 10
    return cond(pred, lambda x: x.cos(), lambda y: y.sin(), [x])
```

More generally, ExportDB can be used as a reference when one of the following occurs:
- Before attempting torch.export, you know ahead of time that your model uses some tricky Python/PyTorch features and you want to know if torch.export covers that feature.
- When attempting torch.export, there is a failure and it’s unclear how to work around it.

ExportDB is not exhaustive, but is intended to cover all use cases found in typical PyTorch code.

Feel free to reach out if there is an important Python/PyTorch feature that should be added to ExportDB or supported by torch.export.


## Running the Exported Program

As torch.export is only a graph capturing mechanism, calling the artifact produced by torch.export eagerly will be equivalent to running the eager module.

`torch.export` 只是一种图捕获机制，eagerly call `torch.export` 生成的 artifact 等同于 eagerly 运行 module

To optimize the execution of the Exported Program, we can pass this exported artifact to backends such as Inductor through `torch.compile`, `AOTInductor`, or `TensorRT`.

为了优化 ExportedProgram 的执行，可以把 exported artifact 传给后端，例如 Inductor (通过 `torch.compile`)，AOTInductor，TensorRT

```py
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)

    def forward(self, x):
        x = self.linear(x)
        return x

inp = torch.randn(2, 3, device="cuda")
m = M().to(device="cuda")
ep = torch.export.export(m, (inp,))

# Run it eagerly
res = ep.module()(inp)
print(res)

# Run it with torch.compile
res = torch.compile(ep.module(), backend="inductor")(inp)
print(res)
```

打印

```
tensor([[-0.1586, -0.2832, -0.6368],
        [-0.5408, -0.6193, -0.7867]], grad_fn=<AddmmBackward0>)
tensor([[-0.1586, -0.2832, -0.6368],
        [-0.5408, -0.6193, -0.7867]], grad_fn=<CompiledFunctionBackward>)
```


```py
import torch._export
import torch._inductor

# Note: these APIs are subject to change
# Compile the exported program to a .so using ``AOTInductor``
with torch.no_grad():
    so_path = torch._inductor.aot_compile(ep.module(), [inp])

# Load and run the .so file in Python.
# To load and run it in a C++ environment, see:
# https://pytorch.org/docs/main/torch.compiler_aot_inductor.html
res = torch._export.aot_load(so_path, device="cpu")(inp)

print(so_path)
print(res)
```

打印

```
/tmp/torchinductor_xtyi/ct4aediyar3zfmqluy2dwigwfifheuxshqtexfigcznzbm72xrxy/cfwyfw77hdb3qy32uockwfcx4won7qk6os7e5n6t5yadid46ub6c.so
tensor([[-0.1586, -0.2832, -0.6368],
        [-0.5408, -0.6193, -0.7867]])
```

> 注意 compile 与 aot_compile 的区别, compile 的后端是 Inductor, aot_compile 的后端是 AOTInductor

> aot_compile 需要访问 _inductor 包调用, Linux 下默认会在 /tmp 目录下生成 .so 文件


## Conclusion

We introduced torch.export, the new PyTorch 2.X way to export single computation graphs from PyTorch programs.

我们介绍了 torch.export, PyTorch 2.X 中新的导出单一计算图的方法

In particular, we demonstrate several code modifications and considerations (control flow ops, constraints, etc.) that need to be made in order to export a graph.

我们特别展示了导出计算图时需要进行的几处代码修改和注意事项(控制流操作、约束条件等)

