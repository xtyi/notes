# torch-backend-to-stablehlo-backend-pipeline


```cpp
mlir::PassPipelineRegistration<
      TorchConversion::StablehloBackendPipelineOptions>(
      "torch-backend-to-stablehlo-backend-pipeline",
      "Pipeline lowering torch backend contract to StableHLO backend "
      "contract.",
      TorchConversion::createTorchBackendToStablehloBackendPipeline);
```


```cpp
void TorchConversion::createTorchBackendToStablehloBackendPipeline(
    OpPassManager &pm,
    const TorchConversion::StablehloBackendPipelineOptions &options) {
  // Generate Stablehlo & Chlo ops.
  pm.addNestedPass<func::FuncOp>(createConvertTorchToStablehloPass(
      options.enableStaticShape, options.enableI32Index));
  // Lowering Chlo ops to Stablehlo
  pm.addNestedPass<func::FuncOp>(
      stablehlo::createChloLegalizeToStablehloPass());
  // Lowering remained ops to Arith
  pm.addNestedPass<func::FuncOp>(createConvertTorchToArithPass());

  // Clean up any non-canonical code introduced above..
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  // The resolution of `dim` ops tends to create identical ops. CSE them.
  pm.addNestedPass<func::FuncOp>(createCSEPass());

  // Finish the type conversion from `torch` types to the types of the
  // StableHLO backend contract.
  pm.addPass(
      TorchConversion::createFuncBackendTypeConversionForStablehloPass());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(
      TorchConversion::createFinalizingBackendTypeConversionForStablehloPass());

  // Verify that we have lowered to Stablehlo ops.
  pm.addPass(TorchConversion::createVerifyStablehloBackendContractPass());

  // Canonicalize Stablehlo dynamic ops to static ops
  pm.addNestedPass<func::FuncOp>(
      stablehlo::createStablehloCanonicalizeDynamismPass());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addPass(stablehlo::createStablehloRefineShapesPass());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(
      stablehlo::createStablehloCanonicalizeDynamismPass());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());

  // Legalize deprecated ops to Stablehlo ops
  stablehlo::StablehloLegalizeDeprecatedOpsPassOptions stablehloOptions;
  stablehloOptions.failOnUnusedOps = false;
  pm.addNestedPass<func::FuncOp>(
      stablehlo::createStablehloLegalizeDeprecatedOpsPass(stablehloOptions));
  pm.addPass(createCanonicalizerPass());
}
```

torch-mlir: torch to stablehlo

```cpp
createConvertTorchToStablehloPass(
      options.enableStaticShape, options.enableI32Index)
```

stablehlo: chlo to stablehlo

```cpp
stablehlo::createChloLegalizeToStablehloPass()
```

torch-mlir: torch to arith

```cpp
createConvertTorchToArithPass();
```



```cpp
TorchConversion::createFuncBackendTypeConversionForStablehloPass()
TorchConversion::createFinalizingBackendTypeConversionForStablehloPass()
```


```cpp
TorchConversion::createVerifyStablehloBackendContractPass()
```