# broadcast_in_dim



```
// 示例 1: 将 1D 张量广播到 2D
%0 = stablehlo.broadcast_in_dim %input, dims = [1] : (tensor<5xf32>) -> tensor<3x5xf32>

// 示例 2: 将标量广播到数组
%1 = stablehlo.broadcast_in_dim %scalar, dims = [] : (tensor<f32>) -> tensor<10x10xf32>

// 示例 3: 保持某些维度不变，扩展其他维度
%2 = stablehlo.broadcast_in_dim %input, dims = [0, 2] : (tensor<2x3xf32>) -> tensor<2x4x3xf32>
```



```
%result = stablehlo.broadcast_in_dim %operand, dims = [2, 1] : (tensor<4x3xi32>) -> tensor<2x3x4xi32>
```





%input: 输入张量

dims: 指定输入维度如何映射到输出维度的属性

dims 的长度等于 inputs 的维度

dims 元素值不能超过 outputs 的维度

输入和输出形状在类型签名中静态指定

dims 指定的输入维度, 要么等于输出维度, 要么等于 1 (扩展)






静态广播: 在编译时就确定输出形状

维度映射: 通过 dims 属性指定输入维度如何映射到输出维度

形状扩展: 可以在指定维度上扩展张量大小

## to linalg

```
%0 = "stablehlo.broadcast_in_dim"(%operand)
         {broadcast_dimensions = array<i64: 4, 0, 2>}
         : (tensor<5x7x1xf32>) -> tensor<7x10x6x4x5xf32>
```

```
tensor.empty() : tensor<7x10x6x4x5xf32>
tensor.collapse_shape
stablehlo.transpose permutation = [1, 0]
linalg.broadcast dimensions = [1, 2, 3]
```

