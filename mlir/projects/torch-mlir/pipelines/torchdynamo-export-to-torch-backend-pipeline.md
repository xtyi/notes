# torchdynamo-export-to-torch-backend-pipeline

lib/Dialect/Torch/Transforms/Passes.cpp

```cpp
mlir::PassPipelineRegistration<Torch::TorchLoweringPipelineOptions>(
      "torchdynamo-export-to-torch-backend-pipeline",
      "Pipeline lowering TorchDynamo exported graph IR to Torch backend form.",
      mlir::torch::Torch::createTorchDynamoExportToTorchBackendPipeline);
```


```cpp
void mlir::torch::Torch::createTorchDynamoExportToTorchBackendPipeline(
    OpPassManager &pm, const TorchLoweringPipelineOptions &options) {
  pm.addNestedPass<func::FuncOp>(
      createReduceOpVariantsPass(options.extraLibrary));
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  if (options.decompose) {
    pm.addNestedPass<func::FuncOp>(
        Torch::createDecomposeComplexOpsPass(options.backendLegalOps));
    pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  }
}
```
