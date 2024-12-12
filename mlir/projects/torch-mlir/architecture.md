# Torch-MLIR Architecture

## Introduction

The Torch-MLIR project provides core infrastructure for bridging the PyTorch ecosystem and the MLIR ecosystem. For example, Torch-MLIR enables PyTorch models to be lowered to a few different MLIR dialects. Torch-MLIR does not attempt to provide a production end-to-end flow for PyTorch programs by itself, but is a useful component for constructing one.

## Overview

Torch-MLIR has two parts, which we call the "frontend" and "backend". These two halves interface at an abstraction layer that we call the "backend contract", which is a subset of the torch dialect with certain properties appealing for backends to lower from.

![](https://github.com/llvm/torch-mlir/raw/main/docs/images/architecture.png)

The frontend of Torch-MLIR is concerned with interfacing to PyTorch itself, and then normalizing the program to the "backend contract". This part involves build system complexity and exposure to PyTorch APIs to get the program into the MLIR torch dialect. When we interface with TorchScript, we additionally have a large amount of lowering and simplification to do within MLIR on the torch dialect.

The "backend" of Torch-MLIR takes IR in the "backend contract" form and lowers it to various target dialects of interest to the MLIR ecosystem (various "backends"). In particular, right now we support lowering to:

- Linalg-on-Tensors (+ arith, tensor, etc.)
- TOSA
- StableHLO

The terms "frontend" and "backend" are highly overloaded in any compiler project, but frequently in Torch-MLIR this is the meaning that they have. Sometimes "frontend" can mean something even further up the stack, such as something in PyTorch itself. When there is ambiguity we will refer to this as "at the PyTorch level". Similarly, "backend" can sometimes refer to something sitting below Linalg-on-Tensors, TOSA, or StableHLO.

