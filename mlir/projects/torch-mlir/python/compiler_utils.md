
一个枚举类型, 表示 IR 的类型

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

根据 output_type 的值, 即期望输出的 IR, 运行不同的 pipeline 去 lower mlir

```py
def lower_mlir_module(verbose, output_type, module):
    # 先打印当前 module ir
    if verbose:
        print("\n====================")
        print("Torch Backend IR")
        print(module)

    # 如果 output ir 为 Torch, 直接返回
    if output_type == OutputType.TORCH:
        return module

    # 如果 output ir 为 TOSA, 运行 torch-backend-to-tosa-backend-pipeline
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

    # 如果 output ir 为 linalg, 运行 torch-backend-to-linalg-on-tensors-backend-pipeline
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

    # 如果 output ir 为 stablehlo, 运行 torch-backend-to-stablehlo-backend-pipeline
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
