# dynamic_broadcast_in_dim


```
%result = stablehlo.dynamic_broadcast_in_dim %input, %output_dimensions, dims = [dim1, dim2, ...] 
```

%input: 输入张量

%output_dimensions: 目标输出形状（动态）

dims: 指定输入维度如何映射到输出维度的属性

```
%6 = stablehlo.dynamic_broadcast_in_dim %5, %from_elements_206, dims = [0, 1] : (tensor<?x4096xf32>, tensor<2xindex>) -> tensor<?x4096xf32>
```

将输入张量 %5 (类型为 tensor<?x4096xf32>)

按照 %from_elements_206 指定的目标形状进行广播

dims = [0, 1] 表示输入的第0维和第1维分别对应输出的第0维和第1维

输出仍然是一个 tensor<?x4096xf32> 类型的张量





1. broadcast_dimensions

这是一个必需的属性，用于指定输入张量的维度如何映射到输出张量的维度
它是一个整数数组，表示输入维度对应输出维度的索引位置
例如在示例中 broadcast_dimensions = array<i64: 2, 1> 表示:
输入的第 0 维映射到输出的第 2 维
输入的第 1 维映射到输出的第 1 维


2. known_expanding_dimensions
这是一个可选属性，用于指定哪些输入维度是确定会扩展的
扩展意味着这个维度在输出中的大小会大于输入维度的大小
在示例中 known_expanding_dimensions = array<i64: 0> 表示:
输入的第 0 维(值为 1)会被扩展到输出的对应维度(值为 2)

3. known_nonexpanding_dimensions
这是一个可选属性，用于指定哪些输入维度确定不会扩展
非扩展意味着这个维度在输出中的大小与输入维度的大小相同
在示例中 known_nonexpanding_dimensions = array<i64: 1> 表示:
输入的第 1 维(值为 3)在输出中保持不变(值为 3)

重要说明：
这些静态知识可以帮助编译器进行优化
expanding 和 nonexpanding 维度集合必须是不相交的
如果不指定这两个可选属性，所有维度都被认为是可能会扩展的
这些维度索引必须是输入张量维度的有效子集