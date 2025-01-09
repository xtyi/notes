# DenseElementsAttr

## DenseIntOrFPElementsAttr

An Attribute containing a dense multi-dimensional array of integer or floating-point values.

A dense int-or-float elements attribute is an elements attribute containing a densely packed vector or tensor of integer or floating-point values.

The element type of this attribute is required to be either an IntegerType or a FloatType.

```cpp
// A splat tensor of integer values.
dense<10> : tensor<2xi32>
// A tensor of 2 float32 elements.
dense<[10.0, 11.0]> : tensor<2xf32>
```

| Parameter | C++ type         | Description |
| --------- | ---------------- | ----------- |
| type      | `ShapedType`     |             |
| rawData   | `ArrayRef<char>` |             |


```cpp
ShapedType type;
```

- 存储属性的形状和元素类型信息
- 包含维度信息（例如：[2,3] 表示 2x3 的矩阵）
- 指定元素的具体类型（如 i32, i64 等）

### DenseIntElementsAttr


```cpp
/// An attribute that represents a reference to a dense integer vector or tensor
/// object.
class DenseIntElementsAttr : public DenseIntOrFPElementsAttr {
public:
  /// DenseIntElementsAttr iterates on APInt, so we can use the raw element
  /// iterator directly.
  using iterator = DenseElementsAttr::IntElementIterator;

  using DenseIntOrFPElementsAttr::DenseIntOrFPElementsAttr;

  /// Get an instance of a DenseIntElementsAttr with the given arguments. This
  /// simply wraps the DenseElementsAttr::get calls.
  template <typename Arg>
  static DenseIntElementsAttr get(const ShapedType &type, Arg &&arg) {
    return llvm::cast<DenseIntElementsAttr>(
        DenseElementsAttr::get(type, llvm::ArrayRef(arg)));
  }
  template <typename T>
  static DenseIntElementsAttr get(const ShapedType &type,
                                  const std::initializer_list<T> &list) {
    return llvm::cast<DenseIntElementsAttr>(DenseElementsAttr::get(type, list));
  }

  /// Generates a new DenseElementsAttr by mapping each value attribute, and
  /// constructing the DenseElementsAttr given the new element type.
  DenseElementsAttr mapValues(Type newElementType,
                              function_ref<APInt(const APInt &)> mapping) const;

  /// Iterator access to the integer element values.
  iterator begin() const { return raw_int_begin(); }
  iterator end() const { return raw_int_end(); }

  /// Method for supporting type inquiry through isa, cast and dyn_cast.
  static bool classof(Attribute attr);
};
```



```cpp
// 创建一个 DenseIntElementsAttr
std::vector<int32_t> values = {1, 2, 3, 4, 5, 6};
auto type = RankedTensorType::get({2, 3}, builder.getIntegerType(32));
auto attr = DenseIntElementsAttr::get(type, values);

// 访问数据
for (auto value : attr.getValues<int32_t>()) {
  // 处理每个元素
}

// 获取形状信息
ArrayRef<int64_t> shape = attr.getType().getShape();

// 获取元素类型
Type elementType = attr.getType().getElementType();

// 获取原始数据
llvm::ArrayRef<char> rawData = attr.getRawData();
```



