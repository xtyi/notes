# real_dynamic_slice

```
%result = stablehlo.real_dynamic_slice %input, %start_indices, %limit_indices, %strides
```


```
// 从2D张量中切片
%input = ... : tensor<4x6xf32>
%start = stablehlo.constant dense<[1, 2]> : tensor<2xi64>
%limit = stablehlo.constant dense<[3, 5]> : tensor<2xi64>
%strides = stablehlo.constant dense<[1, 1]> : tensor<2xi64>

// 提取 input[1:3, 2:5]
%result = stablehlo.real_dynamic_slice %input, %start, %limit, %strides : (tensor<4x6xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<2x3xf32>
```




```
// 输入有动态维度
%input = ... : tensor<?x4xf32>
%start = ... : tensor<2xi64>  // 运行时确定
%limit = ... : tensor<2xi64>  // 运行时确定
%strides = stablehlo.constant dense<[1, 1]> : tensor<2xi64>

// 结果维度也是动态的
%result = stablehlo.real_dynamic_slice %input, %start, %limit, %strides : (tensor<?x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<?x?xf32>
```

