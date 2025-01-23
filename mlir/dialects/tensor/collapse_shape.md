# collapse_shape

```mlir
// Dimension collapse (i, j) -> i' and k -> k'
%b = tensor.collapse_shape %a [[0, 1], [2]]
    : tensor<?x?x?xf32> into tensor<?x?xf32>
```

