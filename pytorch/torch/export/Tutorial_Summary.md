
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

