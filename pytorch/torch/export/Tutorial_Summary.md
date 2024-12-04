
ExportedProgram 有几个重要的属性
- graph_module
  - graph: fx 计算图
- graph_signature: 计算图的签名, 包含输入，输出，参数，buffer 等
- range_constraints: 输入 Dim 的约束


ExportedProgram 可以直接 eagerly 执行, 和 eager 模式执行一个 nn.Module 一样

也可以使用编译, 使用不同的后端
- Inductor(compile)
- AOTInductor(aot_compile): 生成 .so 文件
- TensorRT


torch.export
- 使用 Dynamo, 和 torch.compile 相同
- 不支持 graph break
  - 使用 non-strict 模式: eager 模式执行模型, 使用 Python 解释执行不支持的代码，跟踪执行的路径，这样最后的计算图只有执行路径上的 op
  - 修改模型: 使用 cond op 代替 if statement
- 默认将约束限制在 input tensor 的形状
  - 使用符号放宽约束，以支持动态shape

符号
- 创建符号需要指定一个名字(用于调试), 范围(约束)
- 支持符号间的约束, 支持 affine 表达式

