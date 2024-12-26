# Dev

## Build From Source

```
sudo apt install python3-dev
```

```
git clone https://github.com/llvm/torch-mlir
cd torch-mlir
git submodule update --init --progress --depth=1
```

创建一个虚拟环境

```
conda create -n torch-customop-dev python=3.11
conda activate torch-customop-dev
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
pip install tabulate expecttest
```

out-of-tree 构建

先构建 LLVM, commit 要和 external 目录下的一致，其实直接用 external 下的就好

```
cmake -G Ninja ../llvm
    -DLLVM_ENABLE_PROJECTS=mlir
    -DLLVM_TARGETS_TO_BUILD=host
    -DCMAKE_BUILD_TYPE=RelWithDebInfo
    -DLLVM_ENABLE_ASSERTIONS=ON
    -DCMAKE_C_COMPILER=clang-18
    -DCMAKE_CXX_COMPILER=clang++-18
    -DLLVM_ENABLE_LLD=ON
    -DLLVM_CCACHE_BUILD=ON
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON
    -DPython3_EXECUTABLE="$CONDA_PREFIX/bin/python"
```

再构建 torch-mlir

```
cmake -GNinja -Bbuild .
    -DLLVM_TARGETS_TO_BUILD=host
    -DCMAKE_BUILD_TYPE=RelWithDebInfo
    -DLLVM_ENABLE_ASSERTIONS=ON
    -DCMAKE_C_COMPILER=clang-18
    -DCMAKE_CXX_COMPILER=clang++-18
    -DPython3_FIND_VIRTUALENV=ONLY
    -DMLIR_DIR="$LLVM_INSTALL_DIR/lib/cmake/mlir/" 
    -DLLVM_DIR="$LLVM_INSTALL_DIR/lib/cmake/llvm/" 
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON
    -DPython3_EXECUTABLE="$CONDA_PREFIX/bin/python" 
```