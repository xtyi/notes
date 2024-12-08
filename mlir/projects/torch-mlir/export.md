


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
    context = ir.Context()
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


```py
class OutputType(Enum):

    # Output torch dialect in backend form. When converting from TorchDynamo,
    # this comes after some decomposition and reduce op variants passes are
    # applied to the raw torch dialect. When converting from TorchScript, this
    # comes after some cleanup passes which attempt to de-alias, decompose and infer shapes.
    # These should be roughly the same level of abstraction since those
    # steps are done within PyTorch itself when coming directly from Dynamo/FX.
    TORCH = "torch"

    # The output type contains a mix of `linalg`-on-tensors ops, `scf`, and
    # `arith` ops (and also `math` and `tm_tensor`). It can be thought of
    # as taking the `TORCH` output type and lowering it so that tensor
    # computations are done with `linalg`-on-tensors ops.
    LINALG_ON_TENSORS = "linalg-on-tensors"

    # This output type consists of `tosa` dialect ops. It can be thought of
    # as taking the `TORCH` output type and lowering it to TOSA.
    TOSA = "tosa"

    # This output type consists of `stablehlo` dialect ops. It can be thought of
    # as taking the `TORCH` output type and lowering it to StableHLO.
    STABLEHLO = "stablehlo"

    # Raw output of the JIT IR importer in the TorchScript frontend or that of
    # the FX IR importer in the TorchDynamo frontend. This is not expected to be useful
    # for end-users, but can be convenient for development or reporting bugs.
    RAW = "raw"

    @staticmethod
    def get(spec: Union[str, "OutputType"]) -> "OutputType":
        """Gets an OutputType from allowed way to specify one.

        Args:
          spec: An OutputType instance or the case-insensitive name of one of the
            enum values.
        Returns:
          An OutputType instance.
        """
        if isinstance(spec, OutputType):
            return spec
        spec = spec.upper().replace("-", "_")
        if spec not in OutputType.__members__:
            raise ValueError(
                f"For output_type= argument, expected one of: "
                f"{', '.join(OutputType.__members__.keys())}"
            )
        return OutputType[spec]
```

```py

def _module_lowering(
    verbose,
    output_type,
    torch_mod,
    extra_library_file_name=None,
):
    # RAW 则什么都不做
    if output_type == OutputType.RAW:
        if verbose:
            print(torch_mod)
        return torch_mod
    # TODO: pass extra_library_file_name by caller
    if extra_library_file_name is None:
        extra_library_file_name = ""
    option_string = "{extra-library=" + extra_library_file_name + "}"
    run_pipeline_with_repro_report(
        torch_mod,
        f"builtin.module(func.func(torch-match-quantized-custom-ops), torchdynamo-export-to-torch-backend-pipeline{option_string})",
        "Lowering TorchFX IR -> Torch Backend IR",
        enable_ir_printing=verbose,
    )
    # 其他 output type 则执行 lower_mlir_module
    return lower_mlir_module(verbose, output_type, torch_mod)
```

torch_mod 类型为 torch-mlir 中的 ir.Module, 表示一个 MLIR Module，可以通过 FXImporter 的 module 属性获取


```py

def lower_mlir_module(verbose, output_type, module):
    if verbose:
        print("\n====================")
        print("Torch Backend IR")
        print(module)

    if output_type == OutputType.TORCH:
        return module

    if output_type == OutputType.TOSA:
        run_pipeline_with_repro_report(
            module,
            "builtin.module(torch-backend-to-tosa-backend-pipeline)",
            "Lowering Torch Backend IR -> TOSA Backend IR",
        )
        if verbose:
            print("\n====================")
            print("TOSA Backend IR")
            print(module)
        return module

    if output_type == OutputType.LINALG_ON_TENSORS:
        run_pipeline_with_repro_report(
            module,
            "builtin.module(torch-backend-to-linalg-on-tensors-backend-pipeline)",
            "Lowering Torch Backend IR -> Linalg-on-Tensors Backend IR",
        )
        if verbose:
            print("\n====================")
            print("LINALG Backend IR")
            print(module)
        return module

    elif output_type == OutputType.STABLEHLO:
        run_pipeline_with_repro_report(
            module,
            "builtin.module(torch-backend-to-stablehlo-backend-pipeline)",
            "Lowering Torch Backend IR -> StableHLO Backend IR",
        )
        if verbose:
            print("\n====================")
            print("StableHLO Backend IR")
            print(module)
        return module
    raise Exception(f"Unknown OutputType: {output_type}")
```

torch.operator 用于表示 registry 中没有使用 torch_ods_gen.py 生成的 op

emit_operation 的函数

```py
def _emit_operation(
    mlir_op_name: str, result_types: List[IrType], operands: List[Value], loc: Location
) -> Operation:
    # Support unregistered torch ops using torch.operator.
    # torch.operator is used to represent ops from registry
    # which haven't been generated by torch_ods_gen.py.
    context = loc.context
    if not context.is_registered_operation(mlir_op_name):
        operation = Operation.create(
            "torch.operator",
            attributes={"name": StringAttr.get(mlir_op_name)},
            results=result_types,
            operands=operands,
            loc=loc,
        )
    else:
        operation = Operation.create(
            mlir_op_name,
            results=result_types,
            operands=operands,
            loc=loc,
        )
    return operation
```

tablegen 定义

```
def Torch_OperatorOp : Torch_Op<"operator", [
    AllowsTypeRefinement
  ]> {
  let summary = "Opaque torch operator";
  let description = [{
    Represents an invocation of a `torch::jit::Operator` for which we don't
    have a registered MLIR operation.

    The `name` attribute contains the name that the MLIR op would have
    (excluding `torch.`) if we did have it registered, which allows easy
    cross referencing with `JITOperatorRegistryDump.txt`.
  }];

  let arguments = (ins StrAttr:$name, Variadic<AnyTorchType>:$operands);
  let results = (outs Variadic<AnyTorchType>:$results);
  let regions = (region VariadicRegion<AnyRegion>:$regions);

  let assemblyFormat = [{
    $name `(` $operands `)` attr-dict `:` functional-type($operands, $results) $regions
  }];
}
```