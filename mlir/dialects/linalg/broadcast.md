# broadcast

```
// 示例 1：将 1x3 广播到 4x3
%input = ... : tensor<1x3xf32>
%init = ... : tensor<4x3xf32>
%result = linalg.broadcast 
  ins(%input : tensor<1x3xf32>) 
  outs(%init : tensor<4x3xf32>) 
  dimensions = [0]

// 示例 2：将 1x1x3 广播到 2x4x3
%input = ... : tensor<1x1x3xf32>
%init = ... : tensor<2x4x3xf32>
%result = linalg.broadcast 
  ins(%input : tensor<1x1x3xf32>) 
  outs(%init : tensor<2x4x3xf32>) 
  dimensions = [0, 1]

// 示例 3：将标量广播到向量
%input = ... : tensor<1xf32>
%init = ... : tensor<4xf32>
%result = linalg.broadcast 
  ins(%input : tensor<1xf32>) 
  outs(%init : tensor<4xf32>) 
  dimensions = [0]
```

输出张量必须预先分配（通过 outs 操作数提供）

dimensions 属性指定要广播的维度

dimensions 属性指定的是输出张量的维度

dimensions 属性中的维度索引必须是严格递增的

不在 dimensions 中的维度必须在输入和输出之间匹配

输入秩 + 广播维度数 = 输出秩