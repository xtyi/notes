# Dynamo Deep Dive

TorchDynamo (or simply Dynamo) is the tracer within `torch.compile`, and it is, more often than not, the one to blame for those insane backtraces.

However, we cannot blindly blame Dynamo for these errors.

In order to provide the user with the flexibility it does, Dynamo is given the arduous task of understanding any Python program.

In particular, Dynamo has to implement a good part of the Python programming language internally!

In this post, we will go over the internal design of Dynamo from the ground up.

We will discuss the functionality it provides, and how it is implemented.

By the end of this post, you will have a better understanding of what went wrong when you torch.compiled a PyTorch program and the compilation errored out, or succeeded but the speed-up was not what you expected.

> TorchDynamo 是 `torch.compile` 内部的 tracer

## A Gentle Introduction to Dynamo

Before getting our hands dirty with all the implementation details, let’s start by discussing what it is that Dynamo does.

Dynamo is a tracer. This means, given and function and inputs to it, it executes the function and records a linear sequence of instructions (without control flow) into a graph.

> TorchDynamo 是一个 tracer, 意味着给一个函数和它的输入, Dynamo 执行函数并记录一个线性的指令序列到图中

For example, consider the following program:

```py
import torch

@torch.compile
def mse(x, y):
    z = (x - y) ** 2
    return z.sum()

x = torch.randn(200)
y = torch.randn(200)
mse(x, y)
```

If we save this program into the file `example.py` and we run

```
TORCH_LOGS=graph_code python example.py
```

we see the output that Dynamo traced

```py
def forward(l_x_: torch.Tensor, l_y_: torch.Tensor):
    # File: example.py:5, code: z = (x - y) ** 2
    sub = l_x_ - l_y_
    z = sub ** 2
    # File: example.py:6, code: return z.sum()
    sum_1 = z.sum()
    return (sum_1,)
```

We call this a graph (or trace) of the function for the given inputs.

This is represented via an FX graph. We will simply think of an FX graph as a container that stores a list of function calls.

> 这个表示被称为 FX graph, 我们可以简单地把 FX Graph 认为是一个存储一系列 function calls 的容器

The first thing we should notice is that the graph is a linear sequence of PyTorch operations.

Dynamo records all the PyTorch operations and stores them sequentially.

For example, it split z = (x - y) ** 2 into its two constituting operations, `sub = l_x_ - l_y_` and `z = sub ** 2`.

When we say that the trace is linear, we mean that there is no branching or any control flow. To see this, consider

```py
import torch

@torch.compile
def fn(x, n):
    y = x ** 2
    if n >= 0:
        return (n + 1) * y
    else:
        return y / n

x = torch.randn(200)
fn(x, 2)
```

which, when executed with TORCH_LOGS=graph_code, returns

```py
def forward(l_x_: torch.Tensor):
    # File: example.py:5, code: y = x ** 2
    y = l_x_ ** 2
    # File: example.py:7, code: return (n + 1) * y
    mul = 3 * y
    return (mul,)
```

We see that Dynamo completely removed the `if` statement from the trace and just recorded the operations that were executed with the inputs.

As such, it should be clear that the trace of a function depends on the inputs.

In particular, this means that the trace is not generated when we write `@torch.compile`, but when we execute the function `fn(x, 2)` with the actual arguments.

The other interesting thing to note here is that Dynamo removed the second argument to the function.

Instead, it treated it as a constant and recorded the result of the operation `n + 1` in the graph.

This is another feature of Dynamo: Dynamo will treat as constant any non-tensor value… other than **ints**. Let’s see now how are ints special.

The last defining property of Dynamo is that it knows how to handle dynamic shapes.

Symbolic shapes refer to Dynamo’s ability of tracing shapes, and more generally, integers, rather than leaving them as constants.

This allows for avoiding recompilations and deploying generic models that work for any size in production.

The main examples of places where dynamic shapes appear are the batch size, where we might train a model with a fixed batch size but then perform inference for an arbitrary batch size, or the variable sequence length that one encounters when processing text or audio.

We can see this by executing a few more times the example above

```py
import torch

@torch.compile
def fn(x, n):
    y = x ** 2
    if n >= 0:
        return (n + 1) * y
    else:
        return y / n

x = torch.randn(200)
fn(x, 2)
fn(x, 3)
fn(x, -2)
```

In this case, TORCH_LOGS=graph_code generates two more graphs

```py
# Graph for n==2 omitted

def forward(self, l_x_: torch.Tensor, l_n_: torch.SymInt):
    # File: a.py:5, code: y = x ** 2
    y = l_x_ ** 2

    # File: a.py:7, code: return (n + 1) * y
    add = l_n_ + 1
    mul = add * y
    return (mul,)
```

```py
def forward(self, l_x_: torch.Tensor, l_n_: torch.SymInt):
    # File: a.py:5, code: y = x ** 2
    y = l_x_ ** 2

    # File: a.py:9, code: return y / n
    truediv = y / l_n_
    return (truediv,)
```

Dynamo detected that one integer changed its value after the first call and started tracing it.

We see that these graphs are generic, and trace the variable `n` symbolically via an object of type `SymInt`.

If after these calls we call `fn(x, 4)`, Dynamo would not recompile, but rather reuse the graph that was already traced.

To summarize: 1. Dynamo is a Python tracer 2. Given some inputs, it returns an FX graph with the PyTorch functions that were executed 3. It can also trace integers if it detects that they changed between calls 4. It specializes any other value that is not a tensor or a scalar

Of course, Dynamo does many more things, like figuring out when it needs to retrace, rewriting the bytecode of the function, implementing graph breaks… To keep the introduction short, we will incrementally discuss all these in the sequel.

# PEP 523: Adding a frame evaluation API to CPython

Imagine now that we are given the task to implement Dynamo.

Where would we even start? Rather conveniently for us, PEP 523 was released with Python 3.6. 

This PEP was designed to allow third parties to create JIT compilers for Python. Let’s see how.

A note on CPython: CPython is internally implemented as a stack machine. A Python program is compiled into bytecodes that then are executed by this interpreter. To learn more about these bytecodes, see the `dis` module from the standard library.

See also the developer docs for an introduction to CPython’s interpreter. We will assume that the reader is familiar with the notion of a stack machine.

PEP 523 exposes an API where a user can add a custom per-function interpreter.

Then, CPython will use this interpreter rather than its own to execute the function.

In order to be able to execute the function, on entry, CPython provides the custom interpreter with things like - The bytecode of the function - The value of the arguments of the function (i.e., the local variables) and their names - The value of the global variables and their names - The builtin functions like abs or print

You can see all the fields [here](https://github.com/pytorch/pytorch/blob/e891a3bba9f05697d72776f6e89347231a141f03/torch/csrc/dynamo/eval_frame.c#L50-L59).

In summary, CPython provides the user’s interpreter with all the information necessary to execute the function.

With this API, we can implement a tracer by implementing an interpreter that runs the code and records in a graph all the PyTorch operations that occur during this execution. This is exactly what Dynamo does.

Dynamo uses this CPython API to parse all these objects and packs them into a Python structure. After it has done so… it goes back from C to python. Other than for this piece of code that communicates with CPython, Dynamo is fully implemented in Python.

It should be clear that it is the decorator `@torch.compile`’s job to install the necessary scaffolding that will pass the bytecode, the args, global variables and so on to Dynamo when the function is called. Again, `@torch.compile` does not actually compile anything.

## Implementing CPython in Python

So, we are back in the Python world. We have the bytecode of a function, and all the context necessary to execute it. In particular, we have landed at `_convert_frame_assert`. This is the function that the decorator `torch.compile` returns! We get to this function from `_dynamo.optimize`. The decorator `torch.compile` is just a nice API around `_dynamo.optimize`.

我们回到 Python 世界，我们有一个 function 的字节码，和所有执行它所必要的上下文

特别，我们已经登陆在 `_convert_frame_assert` 函数



Before getting into implementing a Python interpreter, we want to define an IR. In particular, we want to wrap all the local and global variables in our own internal classes. This allows us to better track these objects and group together objects that can be treated in the same way to the eyes of Dynamo.

