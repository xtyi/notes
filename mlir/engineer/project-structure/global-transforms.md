# Transforms

## TableGen

先看 td 定义, 最简单的一个 Pass, 没有声明 `constructor` 字段

```td
#ifndef XIR_TRANSFORMS_PASSES
#define XIR_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"


def PrintIRStructure : Pass<"print-ir-structure"> {
  let summary = "print ir structure";
  let description = [{

  }];
}

#endif // XIR_TRANSFORMS_PASSES
```



## include cmake

```cmake
set(LLVM_TARGET_DEFINITIONS Passes.td)
# 这里 name 只影响 register 函数的名字
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Transforms)
add_public_tablegen_target(XIRTransformsPassIncGen)
```

## Passes.h.inc

生成的 Passes.h.inc 文件

### GEN_PASS_DECL

包含全部的 GEN_PASS_DECL_XXX 宏定义

### GEN_PASS_DECL_XXX

如果有声明 `options`, 会包含 `struct XXXOptions`

如果没有声明 `constructor`, 会包含 `std::unique_ptr<::mlir::Pass> createXXX();` 定义, 没有命名空间

### GEN_PASS_DEF_XXX

如果没有声明 `constructor`, 包含下面代码

1. `createXXX()` 实现, 该实现的功能是转发到内部的 `impl::createXXX()` API

```cpp
std::unique_ptr<::mlir::Pass> createPrintDefUse() {
  return impl::createPrintDefUse();
}
```

2. `impl::createXXX()` 定义

```cpp
namespace impl {
  std::unique_ptr<::mlir::Pass> createPrintDefUse();
} // namespace impl
```

3. XXXBase 类, 

该类声明了一个友元函数, 即 `impl::createXXX()`, 同时实现也放在了这里

注意该类里面的 `createPrintDefUse` 是友元函数, 不是该类的方法

```cpp
namespace impl {

template <typename DerivedT>
class PrintDefUseBase : public ::mlir::OperationPass<> {
private:

  friend std::unique_ptr<::mlir::Pass> createPrintDefUse() {
    return std::make_unique<DerivedT>();
  }
};

} // namespace impl
```

如果声明了 `constructor`, 只包含 `XXXBase` 类, 并且类中没有 `impl::createXXX()` 友元函数的实现

此时不管是公开的 `createXXX()` 还是内部的 `impl::createXXX()` 都没有了, 但是 `register` 会调用 `createXXX()`, 所以需要自己在 `Passes.h` 里面定义, 在 Pass 的 cpp 文件中实现

### GEN_PASS_REGISTRATION

包含 `registerXXX()`, `registerXXXPass()` 函数, 两者功能一样, 都是调用外部的 `createPrintDefUse()`

```cpp
//===----------------------------------------------------------------------===//
// PrintDefUse Registration
//===----------------------------------------------------------------------===//

inline void registerPrintDefUse() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createPrintDefUse();
  });
}

// Old registration code, kept for temporary backwards compatibility.
inline void registerPrintDefUsePass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createPrintDefUse();
  });
}
```

包含一个 `registerTransformsPasses()`, 调用全部的 Transforms 注册函数

```cpp
inline void registerTransformsPasses() {
  registerPrintDefUse();
  registerPrintIRStructure();
  registerTestApplyPatterns();
}
```

### GEN_PASS_CLASSES

废弃了

## Passes.h

```cpp
#ifndef XIR_TRANSFORMS_PASSES_H
#define XIR_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

namespace mlir {

namespace xir {

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "xir/Transforms/Passes.h.inc"

} // namespace xir

} // namespace mlir

#endif // XIR_TRANSFORMS_PASSES_H
```

可以看到 `Passes.h.inc` 中的代码, 要么没有命名空间, 要么 impl 命名空间, 所以这些代码都需要放在同一个命名空间下, 一般是 `mlir`, 或者是 `mlir::{your_project_name}`

设置成 mlir 的话, 容易冲突, 比如 `registerTransformsPasses` 函数, 所以还是放到自己项目的命名空间下比较好

如果设置了 `constructor`, 就要在这里定义相关的 `createXXX()` 函数, 这里顺序比较重要, 如下

```cpp
#ifndef XIR_TRANSFORMS_PASSES_H
#define XIR_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

namespace mlir {

namespace xir {

// 先引入 Options struct, 定义 createXXX 时可能用到
#define GEN_PASS_DECL
#include "xir/Transforms/Passes.h.inc"

// 定义 createXXX, register 相关代码会用到
std::unique_ptr<Pass> createPrintDefUse();

// 最后再把 register 代码 include 进来
#define GEN_PASS_REGISTRATION
#include "xir/Transforms/Passes.h.inc"

} // namespace xir

} // namespace mlir

#endif // XIR_TRANSFORMS_PASSES_H
```

## lib cmake

```cmake
add_mlir_library(XIRTransforms
  PrintIRStructure.cpp
  PrintDefUse.cpp
  TestApplyPatterns.cpp

  DEPENDS
  XIRTransformsPassIncGen

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRSupport
  )
```

## Pass cpp file

```cpp
// 把 Passes.h include 进来
#include "xir/Transforms/Passes.h"

namespace mlir {
namespace xir {

// 这里包含了 XXXBase class
#define GEN_PASS_DEF_PRINTDEFUSE
#include "xir/Transforms/Passes.h.inc"

// 把 Pass Class 放到一个匿名 namepsace 里面, 即只有该文件可以访问到
namespace {

struct PrintDefUse : public impl::PrintDefUseBase<PrintDefUse> {
    void runOnOperation() override {}
}

} // namespace

// createXXX 的实现放这里
std::unique_ptr<Pass> createPrintDefUse() {
    return std::make_unique<PrintDefUse>();
}

} // xir
} // mlir
```

