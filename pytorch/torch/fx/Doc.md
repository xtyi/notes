# torch.fx

## Overview

FX is a toolkit for developers to use to transform `nn.Module` instances.

FX consists of three main components: a symbolic tracer, an intermediate representation, and Python code generation.

A demonstration of these components in action:

```py
import torch
# Simple module for demonstration
class MyModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(3, 4))
        self.linear = torch.nn.Linear(4, 5)

    def forward(self, x):
        return self.linear(x + self.param).clamp(min=0.0, max=1.0)

module = MyModule()

from torch.fx import symbolic_trace

# Symbolic tracing frontend - captures the semantics of the module
symbolic_traced : torch.fx.GraphModule = symbolic_trace(module)

# High-level intermediate representation (IR) - Graph representation
print(symbolic_traced.graph)
"""
graph():
    %x : [num_users=1] = placeholder[target=x]
    %param : [num_users=1] = get_attr[target=param]
    %add : [num_users=1] = call_function[target=operator.add](args = (%x, %param), kwargs = {})
    %linear : [num_users=1] = call_module[target=linear](args = (%add,), kwargs = {})
    %clamp : [num_users=1] = call_method[target=clamp](args = (%linear,), kwargs = {min: 0.0, max: 1.0})
    return clamp
"""

# Code generation - valid Python code
print(symbolic_traced.code)
"""
def forward(self, x):
    param = self.param
    add = x + param;  x = param = None
    linear = self.linear(add);  add = None
    clamp = linear.clamp(min = 0.0, max = 1.0);  linear = None
    return clamp
"""
```

The **symbolic tracer** performs "symbolic execution" of the Python code.

It feeds fake values, called Proxies, through the code.

Operations on theses Proxies are recorded. More information about symbolic tracing can be found in the `symbolic_trace()` and Tracer documentation.


The **intermediate representation** is the container for the operations that were recorded during symbolic tracing.

It consists of a list of Nodes that represent function inputs, callsites (to functions, methods, or torch.nn.Module instances), and return values.

More information about the IR can be found in the documentation for Graph. The IR is the format on which transformations are applied.


**Python code generation** is what makes FX a Python-to-Python (or Module-to-Module) transformation toolkit.

For each Graph IR, we can create valid Python code matching the Graph’s semantics.

This functionality is wrapped up in GraphModule, which is a `torch.nn.Module` instance that holds a Graph as well as a forward method generated from the Graph.



Taken together, this pipeline of components (symbolic tracing -> intermediate representation -> transforms -> Python code generation) constitutes the Python-to-Python transformation pipeline of FX.

In addition, these components can be used separately.

For example, symbolic tracing can be used in isolation to capture a form of the code for analysis (and not transformation) purposes.

Code generation can be used for programmatically generating models, for example from a config file. There are many uses for FX!

Several example transformations can be found at the examples repository.

## Writing Transformations

What is an FX transform? Essentially, it’s a function that looks like this.

```py
import torch
import torch.fx

def transform(m: nn.Module,
              tracer_class : type = torch.fx.Tracer) -> torch.nn.Module:
    # Step 1: Acquire a Graph representing the code in `m`

    # NOTE: torch.fx.symbolic_trace is a wrapper around a call to
    # fx.Tracer.trace and constructing a GraphModule. We'll
    # split that out in our transform to allow the caller to
    # customize tracing behavior.
    graph : torch.fx.Graph = tracer_class().trace(m)

    # Step 2: Modify this Graph or create a new one
    graph = ...

    # Step 3: Construct a Module to return
    return torch.fx.GraphModule(m, graph)
```

## Limitations of Symbolic Tracing

FX uses a system of symbolic tracing (a.k.a symbolic execution) to capture the semantics of programs in a transformable/analyzable form.

The system is tracing in that it executes the program (really a torch.nn.Module or function) to record operations.

It is symbolic in that the data flowing through the program during this execution is not real data, but rather symbols (Proxy in FX parlance).

