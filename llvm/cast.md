# type cast

```
llvm::cast<T>(arg)


llvm::dyn_cast<T>(arg)
```

dyn_cast 如果转换不合法, 则返回 nullptr

cast 分 debug 和 release 模式

debug 模式下，如果转换不合法，则断言失败

release 模式下，如果转换不合法，则是未定义行为

所以，如果确定程序没 bug 的情况下，转换一定成功，则使用 cast

否则使用 dyn_cast