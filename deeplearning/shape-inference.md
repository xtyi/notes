# Shape Inference

PyTorch 等框架实现 Shape Inference 是因为 CUDA 算子只关心计算，不关心类型

因此 PyTorch 就要负责管理 Tensor 的 Shape 和类型

这里可以动态执行，但是导出图是静态的

然后动态shape场景下，导出图也变成动态的了

因此推理引擎也需要引入动态Shape


 