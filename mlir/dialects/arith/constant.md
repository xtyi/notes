




```mlir
%true = arith.constant true
```

很奇怪, 当创建一个 bool ssa 时不能加类型, 比如下面的代码是错误的

```mlir
%true = arith.constant true: i1
```