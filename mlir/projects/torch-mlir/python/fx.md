# torch_mlir.fx

fx 模块定义了 3 个函数

```
export_and_import
stateless_fx_import
_module_lowering
```


## export_and_import


torch-mlir 导出主要使用的方法是 fx 模块下的 `export_and_import`

第一个位置参数传入 nn.Module 或者 ExportedProgram，如果使用 nn.Module，还会先调用 torch.export 导出为 ExportedProgram，如果传入 ExportedProrgam，则直接使用
变长参数传入模型的参数

func_name，导出 mlir 函数的名称

```py
def export_and_import(
    f: Union[nn.Module, ExportedProgram],
    *args,
    output_type: Union[str, OutputType] = OutputType.RAW,
    fx_importer: Optional[FxImporter] = None,
    dynamic_shapes: Optional[Union[Dict[str, Any], Tuple[Any]]] = None,
    experimental_support_mutation: bool = False,
    import_symbolic_shape_expressions: bool = False,
    hooks: Optional[FxImporterHooks] = None,
    decomposition_table: Optional[Dict[torch._ops.OperatorBase, Callable]] = None,
    func_name: str = "main",
    enable_graph_printing: bool = False,
    enable_ir_printing: bool = False,
    **kwargs,
):
    # 创建一个 MLIR Context
    context = ir.Context()
    # 注册 torch dialect
    torch_d.register_dialect(context)

    # 如果没有指定 fx_importer，则创建一个
    if fx_importer is None:
        fx_importer = FxImporter(context=context, hooks=hooks)
    # 如果传入的是 ExportedProgram，则直接使用
    if isinstance(f, ExportedProgram):
        prog = f
    else: # 果传入的是 nn.Module，则先导出为 ExportedProgram
        # pytorch 2.1 or lower doesn't have `dyanmic_shapes` keyword argument in torch.export
        if version.Version(torch.__version__) >= version.Version("2.2.0"):
            prog = torch.export.export(f, args, kwargs, dynamic_shapes=dynamic_shapes)
        else:
            prog = torch.export.export(f, args, kwargs)
    # 如果没有指定 decomposition, 使用 torch-mlir 默认的
    if decomposition_table is None:
        decomposition_table = get_decomposition_table()
    
    # 执行 decomposition
    if decomposition_table:
        prog = prog.run_decompositions(decomposition_table)
    # 打印调试信息
    if enable_graph_printing:
        prog.graph_module.print_readable()
    if experimental_support_mutation:
        if torch.__version__ < "2.3.0.dev20240207":
            warnings.warn("Mutable program import only supported on PyTorch 2.3+")
        fx_importer.import_program(
            prog,
            func_name=func_name,
            import_symbolic_shape_expressions=import_symbolic_shape_expressions,
        )
    else:
        fx_importer.import_frozen_program(
            prog,
            func_name=func_name,
            import_symbolic_shape_expressions=import_symbolic_shape_expressions,
        )

    return _module_lowering(
        enable_ir_printing, OutputType.get(output_type), fx_importer.module
    )
```

这里不好的是, 没法传 _module_lowering 的参数 extra_library_file_name


## _module_lowering

torch_mod 类型为 torch-mlir 中的 ir.Module, 表示一个 MLIR Module，可以通过 FXImporter 的 module 属性获取

```py
def _module_lowering(
    verbose,
    output_type,
    torch_mod,
    extra_library_file_name=None,
):
    # 如果 output ir 为 RAW, 直接返回即可
    if output_type == OutputType.RAW:
        if verbose:
            print(torch_mod)
        return torch_mod
    
    
    # TODO: pass extra_library_file_name by caller
    if extra_library_file_name is None:
        extra_library_file_name = ""
    option_string = "{extra-library=" + extra_library_file_name + "}"

    # 只要是其他 output ir 类型, 都先执行 torch-match-quantized-custom-ops 和 torchdynamo-export-to-torch-backend-pipeline,
    # 并且 torchdynamo-export-to-torch-backend-pipeline 要传入 extra-library
    run_pipeline_with_repro_report(
        torch_mod,
        f"builtin.module(func.func(torch-match-quantized-custom-ops), torchdynamo-export-to-torch-backend-pipeline{option_string})",
        "Lowering TorchFX IR -> Torch Backend IR",
        enable_ir_printing=verbose,
    )
    # 然后才继续向下 lowering
    return lower_mlir_module(verbose, output_type, torch_mod)
```

