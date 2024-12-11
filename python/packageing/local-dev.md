# 本地构建开发包

构建包, 传统方式

```
python setup.py wheel bdist
```

新的方式, 使用 build 构建, 自动下载 build-backend, 构建包

```
python -m build
```

构建并安装当前路径下的包

```
pip install .
```

环境中的包会连接到该路径下, 适用于开发模式, 修改会直接生效

```
pip install -e .
```

安装 wheel

```
pip install example.whl
```


对于构建好的包，还可以指定 PYTHONPATH

```
export PYTHONPATH="torch-mlir/build/python_packages/torch_mlir"
```

这样的话 vscode 找不到, 最好写在一个 .env 文件里

```
PYTHONPATH=/home/xtyi/proj/aicompiler/torch-mlir/build/python_packages/torch_mlir
```

然后在 vscode workspace setting.json 指定

```
{
    "python.envFile": "${workspaceFolder}/.env"
}
```

还可以创建一个 setenv.sh, 用于命令行 source

```
export $(grep -v '^#' .env | xargs)
```

