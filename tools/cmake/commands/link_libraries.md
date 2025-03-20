# link_libraries

CMake 会正确处理库的依赖关系

会自动在系统的标准库路径中查找 gcov 库

确保库的链接顺序正确（CMake 会自动调整链接顺序）

会根据不同平台自动处理库名称（比如在 Windows 上可能需要不同的命名）

使用 link_libraries 而不要使用 add_link_options

例如使用

```
link_libraries(gcov)
```

代替

```
add_link_options(-lgcov)
```