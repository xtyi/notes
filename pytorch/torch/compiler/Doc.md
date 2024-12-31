# torch.compiler

torch.compiler is a namespace through which some of the internal compiler methods are surfaced for user consumption.

The main function and the feature in this namespace is torch.compile.

torch.compile is a PyTorch function introduced in PyTorch 2.x that aims to solve the problem of accurate graph capturing in PyTorch and ultimately enable software engineers to run their PyTorch programs faster.

torch.compile is written in Python and it marks the transition of PyTorch from C++ to Python.

torch.compile leverages the following underlying technologies:

**TorchDynamo (torch._dynamo)** is an internal API that uses a CPython feature called the **Frame Evaluation API** to safely capture PyTorch graphs. Methods that are available externally for PyTorch users are surfaced through the torch.compiler namespace.


**TorchInductor** is the default torch.compile deep learning compiler that generates fast code for multiple accelerators and backends. You need to use a backend compiler to make speedups through torch.compile possible. For NVIDIA, AMD and Intel GPUs, it leverages OpenAI Triton as the key building block.

**AOT Autograd** captures not only the user-level code, but also backpropagation, which results in capturing the backwards pass “ahead-of-time”. This enables acceleration of both forwards and backwards pass using TorchInductor.

