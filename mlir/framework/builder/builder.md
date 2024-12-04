# Builder

在 mlir/include/mlir/IR/Builders.h 目录下

Builder 提供了许多 helper 函数来创建 context-global 的对象，例如 types, attributes, affine expressions

## 构造函数

```cpp
explicit Builder(MLIRContext *context) : context(context) {}
explicit Builder(Operation *op) : Builder(op->getContext()) {}
```

## Context

该 class 只保存一个 context 指针

```cpp
protected:
  MLIRContext *context;
```

访问 Context

```cpp
MLIRContext *getContext() const { return context; }
```

## Location

Builder 提供了两个获取 Location 的 helper 方法


```cpp
Location getUnknownLoc() { return UnknownLoc::get(context); }

Location getFusedLoc(ArrayRef<Location> locs,
                     Attribute metadata = Attribute()) {
    return FusedLoc::get(locs, metadata, context);
}
```

## Type

Builder 提供了许多获取各种 Type 的 helper 方法

通用的 getType 函数(不常用), 主要还是调用相应 Type 的 get static 方法，把参数透传一下

```cpp
/// Get or construct an instance of the type `Ty` with provided arguments.
template <typename Ty, typename... Args>
Ty getType(Args &&...args) {
  return Ty::get(context, std::forward<Args>(args)...);
}
```

FloatType

```cpp
FloatType getFloat8E5M2Type() { return FloatType::getFloat8E5M2(context); }
FloatType getFloat8E4M3FNType() { return FloatType::getFloat8E4M3FN(context); }
FloatType getFloat8E5M2FNUZType() { return FloatType::getFloat8E5M2FNUZ(context); }
FloatType getFloat8E4M3FNUZType() { return FloatType::getFloat8E4M3FNUZ(context); }
FloatType getFloat8E4M3B11FNUZType() { return FloatType::getFloat8E4M3B11FNUZ(context); }
FloatType getBF16Type() { return FloatType::getBF16(context); }
FloatType getF16Type() { return FloatType::getF16(context); }
FloatType getTF32Type() { return FloatType::getTF32(context); }
FloatType getF32Type() { return FloatType::getF32(context); }
FloatType getF64Type() { return FloatType::getF64(context); }
FloatType getF80Type() { return FloatType::getF80(context); }
FloatType getF128Type() { return FloatType::getF128(context); }
```

IndexType

```cpp
IndexType getIndexType() { return IndexType::get(context); }
```

IntegerType

```cpp
IntegerType getI1Type() { return IntegerType::get(context, 1); }
IntegerType getI2Type() { return IntegerType::get(context, 2); }
IntegerType getI4Type() { return IntegerType::get(context, 4); }
IntegerType getI8Type() { return IntegerType::get(context, 8); }
IntegerType getI16Type() { return IntegerType::get(context, 16); }
IntegerType getI32Type() { return IntegerType::get(context, 32); }
IntegerType getI64Type() { return IntegerType::get(context, 64); }
```

```cpp
IntegerType getIntegerType(unsigned width) {
  return IntegerType::get(context, width);
}

IntegerType getIntegerType(unsigned width, bool isSigned) {
  return IntegerType::get(
      context, width, isSigned ? IntegerType::Signed : IntegerType::Unsigned);
}
```

FunctionType

```cpp
FunctionType getFunctionType(TypeRange inputs, TypeRange results) {
  return FunctionType::get(context, inputs, results);
}
```

TupleType

```cpp
TupleType getTupleType(TypeRange elementTypes) {
  return TupleType::get(context, elementTypes);
}
```

NoneType

```cpp
NoneType getNoneType() { return NoneType::get(context); }
```

## Affine

```cpp
// Affine expressions and affine maps.
AffineExpr getAffineDimExpr(unsigned position);
AffineExpr getAffineSymbolExpr(unsigned position);
AffineExpr getAffineConstantExpr(int64_t constant);

// Special cases of affine maps and integer sets
/// Returns a zero result affine map with no dimensions or symbols: () -> ().
AffineMap getEmptyAffineMap();
/// Returns a single constant result affine map with 0 dimensions and 0
/// symbols.  One constant result: () -> (val).
AffineMap getConstantAffineMap(int64_t val);
// One dimension id identity map: (i) -> (i).
AffineMap getDimIdentityMap();
// Multi-dimensional identity map: (d0, d1, d2) -> (d0, d1, d2).
AffineMap getMultiDimIdentityMap(unsigned rank);
// One symbol identity map: ()[s] -> (s).
AffineMap getSymbolIdentityMap();

/// Returns a map that shifts its (single) input dimension by 'shift'.
/// (d0) -> (d0 + shift)
AffineMap getSingleDimShiftAffineMap(int64_t shift);

/// Returns an affine map that is a translation (shift) of all result
/// expressions in 'map' by 'shift'.
/// Eg: input: (d0, d1)[s0] -> (d0, d1 + s0), shift = 2
///   returns:    (d0, d1)[s0] -> (d0 + 2, d1 + s0 + 2)
AffineMap getShiftedAffineMap(AffineMap map, int64_t shift);
```