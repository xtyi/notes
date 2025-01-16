# Destination Passing Style

https://mlir.llvm.org/docs/Bufferization/#destination-passing-style

Bufferization is an algorithmically complex problem.

Given an op with a tensor result, bufferization has to choose a memref buffer in which the result can be stored. 

It is always safe to allocate a brand new buffer, but such a bufferization strategy would be unacceptable for high-performance codegen.

When choosing an already existing buffer, we must be careful not to accidentally overwrite data that is still needed later in the program.

To simplify this problem, One-Shot Bufferize was designed to take advantage of destination-passing style (DPS).

> 只要是 tensor result, 我们都要选择一个 memref buffer 去存储

> 分配一个新的 buffer 总是安全的, 但是这种 bufferization 策略对于高性能的代码生成是无法接受的

> 当选择一个已经存在的 buffer 时，我们必须注意不能意外地覆盖了还有用(后面的程序会用到)的数据

> 为了简化这个问题, One-Shot Bufferize Pass 利用了 DPS

In MLIR, DPS op should implement the DestinationStyleOpInterface.

DPS exists in itself independently of bufferization and is tied to SSA semantics: many ops are "updating" a part of their input SSA variables.

> DPS 是独立于 bufferization 的

> SSA 的语义无法表达原地修改的操作, DPS 本质上是为了表达这个语义的

For example the LLVM instruction insertelement is inserting an element inside a vector.

Since SSA values are immutable, the operation returns a copy of the input vector with the element inserted.

Another example in MLIR is linalg.generic on tensors, which always has an extra outs operand for each result, which provides the initial values to update.

`outs` operands are referred to as "destinations" in the following (quotes are important as this operand isn’t modified in place but copied) and comes into place in the context of bufferization as a possible "anchor" for the bufferization algorithm.

For every tensor result, a DPS op has a corresponding tensor operand.

If there aren’t any other conflicting uses of this tensor, the bufferization can alias it with the op result and perform the operation "in-place" by reusing the buffer allocated for this "destination" input.

As an example, consider the following op: `%r = tensor.insert %f into %t[%idx] : tensor<5xf32>`

[image]

`%t` is the "destination" in this example.

When choosing a buffer for the result `%r`, denoted as `buffer(%r)`, One-Shot Bufferize considers only two options:


1. `buffer(%r) = buffer(%t)`: store the result in the existing `buffer(%t)`. Note that this is not always possible. E.g., if the old contents of `buffer(%t)` are still needed. One-Shot Bufferize’s main task is to detect such cases and fall back to the second option when necessary.
2. `buffer(%r)` is a newly allocated buffer.

There may be other buffers in the same function that could potentially be used for `buffer(%r)`, but those are not considered by One-Shot Bufferize to keep the bufferization simple.

One-Shot Bufferize could be extended to consider such buffers in the future to achieve a better quality of bufferization.

Tensor ops that are not in destination-passing style always bufferized to a memory allocation. E.g.:

```mlir
%0 = tensor.generate %sz {
^bb0(%i : index):
  %cst = arith.constant 0.0 : f32
  tensor.yield %cst : f32
} : tensor<?xf32>
```

The result of `tensor.generate` does not have a "destination" operand, so bufferization allocates a new buffer.

This could be avoided by instead using an op such as linalg.generic, which can express the same computation with a "destination" operand, as specified behind outputs (outs):

```mlir
#map = affine_map<(i) -> (i)>
%0 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]}
                    outs(%t : tensor<?xf32>) {
  ^bb0(%arg0 : f32):
    %cst = arith.constant 0.0 : f32
    linalg.yield %cst : f32
} -> tensor<?xf32>
```

At first glance, the above linalg.generic op may not seem very useful because the output tensor %t is entirely overwritten.

Why pass the tensor %t as an operand in the first place?

As an example, this can be useful for overwriting a slice of a tensor:

```mlir
%t = tensor.extract_slice %s [%idx] [%sz] [1] : tensor<?xf32> to tensor<?xf32>
%0 = linalg.generic ... outs(%t) { ... } -> tensor<?xf32>
%1 = tensor.insert_slice %0 into %s [%idx] [%sz] [1]
    : tensor<?xf32> into tensor<?xf32>
```

The above example bufferizes to a `memref.subview`, followed by a "linalg.generic on memrefs" that overwrites the memory of the subview, assuming that the slice `%t` has no other user. The `tensor.insert_slice` then bufferizes to a no-op (in the absence of RaW conflicts such as a subsequent read of `%s`).

RaW conflicts are detected with an analysis of SSA use-def chains (details later). One-Shot Bufferize works best if there is a single SSA use-def chain, where the result of a tensor op is the operand of the next tensor ops, e.g.:

```mlir
%0 = "my_dialect.some_op"(%t) : (tensor<?xf32>) -> (tensor<?xf32>)
%1 = "my_dialect.another_op"(%0) : (tensor<?xf32>) -> (tensor<?xf32>)
%2 = "my_dialect.yet_another_op"(%1) : (tensor<?xf32>) -> (tensor<?xf32>)
```

Buffer copies are likely inserted if the SSA use-def chain splits at some point, e.g.:

```mlir
%0 = "my_dialect.some_op"(%t) : (tensor<?xf32>) -> (tensor<?xf32>)
%1 = "my_dialect.another_op"(%0) : (tensor<?xf32>) -> (tensor<?xf32>)

// "yet_another_op" likely needs to read the data of %0, so "another_op" cannot
// in-place write to buffer(%0).
%2 = "my_dialect.yet_another_op"(%0) : (tensor<?xf32>) -> (tensor<?xf32>)
```

## 思考

深度学习推理中，好像很少做原地修改

## tablegen

Ops that are in destination style have designated "init" operands, which act as initial tensor values for the results of the operation or the init buffers to which the results of the op will be written.

Init operands must be tensors or memrefs.

Input operands can have any type.

All non-init operands are DPS inputs.

The init operands of this op are specified by the `MutableOperandRange` that the `getDpsInitsMutable` interface methods returns.

This implies that the init operands must be a consecutive range of operands.

Each tensor init operand is tied to a corresponding tensor OpResult in a 1-to-1 fashion.

The op may not have any additional OpResults.

Init operands and their tied OpResults have the same type.

Dynamic dimension sizes also match at runtime.

Note: This implies that a destination style op without any tensor inits must not have any OpResults.

> inits 和 OpResults 一定是一一对应的, 类型也完全相同

An op has "pure tensor semantics" if it has at least one tensor operand and no buffer (memref) operands.

It has "pure buffer semantics" if it has at least one buffer (memref) operand and no tensor operands.

Destination-passing style abstraction makes certain transformations easier.

For example, tiling implementation can extract/insert slices from/into the destination of an op and use the resulting shaped value as an iter_arg in the surrounding loop structure.

As another example, bufferization does not have to allocate new buffers for destinations (in case of in-place bufferization) and can directly reuse the existing destination buffer.

Example of a destination style op: `%r = tensor.insert_slice %t into %d`, where `%t` is the single input and `%d` is the single init. `%d` is tied to `%r`.

Example of an op that is not in destination style: `%r = tensor.pad %t`.

This op is not in destination style because `%r` and `%t` have different shape.

```tablegen
let methods = [
    InterfaceMethod<
      /*desc=*/"Return start and end indices of the init operands range.",
      /*retTy=*/"::mlir::MutableOperandRange",
      /*methodName=*/"getDpsInitsMutable",
      /*args=*/(ins)
    >,
];

let extraSharedClassDeclaration = [{

    ::mlir::OperandRange getDpsInits() {
        ...
    }

    /// Return the number of DPS inits.
    int64_t getNumDpsInits() {
        ...
    }

    /// Return the `i`-th DPS init.
    ::mlir::OpOperand *getDpsInitOperand(int64_t i) {
        ...
    }

    /// Set the `i`-th DPS init.
    void setDpsInitOperand(int64_t i, Value value) {
        ...
    }

    /// Return the number of DPS inits.
    int64_t getNumDpsInputs() {
        ...
    }

    /// Return the DPS input operands.
    ::llvm::SmallVector<::mlir::OpOperand *> getDpsInputOperands() {
        ...
    }

    /// Return the DPS input operands.
    ::llvm::SmallVector<::mlir::Value> getDpsInputs() {
        ...
    }

    /// Return the `i`-th DPS input operand.
    ::mlir::OpOperand *getDpsInputOperand(int64_t i) {
        ...
    }

    /// Return "true" if `opOperand` is an "input".
    bool isDpsInput(::mlir::OpOperand *opOperand) {
        ...
    }

    /// Return "true" if `opOperand` is an "init".
    bool isDpsInit(::mlir::OpOperand *opOperand) {
        ...
    }

    /// Return "true" if `opOperand` is a scalar value. A sclar is defined as
    /// neither a MemRef nor a tensor value.
    bool isScalar(::mlir::OpOperand *opOperand) {
        ...
    }

    /// Return whether the op has pure buffer semantics. That is the case if the
    /// op has no tensor operands and at least one memref operand.
    bool hasPureBufferSemantics() {
        ...
    }

    /// Return whether the op has pure tensor semantics. That is the case if the
    /// op has no memref operands and at least one tensor operand.
    bool hasPureTensorSemantics() {
        ...
    }


}]


let verify = [{ return detail::verifyDestinationStyleOpInterface($_op); }];
let verifyWithRegions = 1;
```



