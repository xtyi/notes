# broadcast

Broadcast the input into the given shape by adding dimensions.

```
%bcast = linalg.broadcast
    ins(%input:tensor<16xf32>)
    outs(%init:tensor<16x64xf32>)
    dimensions = [1]
```

```
%broadcast = linalg.broadcast
      ins(%input : tensor<2xf32>)
      outs(%init1 : tensor<2x4xf32>)
      dimensions = [1]
```


```
%broadcast = linalg.broadcast
      ins(%input : tensor<2x4x5xf32>)
      outs(%init1 : tensor<1x2x3x4x5x6xf32>)
      dimensions = [0, 2, 5]
```

```
linalg.broadcast ins(%input: tensor<2x3xf32>) outs(%init: tensor<2x3xf32>) dimensions = []
```

```
%broadcast = linalg.broadcast
      ins(%input : tensor<?x?x5xf32>)
      outs(%init1 : tensor<1x?x3x?x5x6xf32>)
      dimensions = [0, 2, 5]
```



输出张量必须预先分配（通过 outs 操作数提供）

dimensions 属性指定要广播的维度

dimensions 属性指定的是输出张量的维度

dimensions 属性中的维度索引必须是严格递增的

不在 dimensions 中的维度必须在输入和输出之间匹配

输入秩 + 广播维度数 = 输出秩