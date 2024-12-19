# DenseMap

try_emplace 是 LLVM 中 DenseMap 容器提供的一个插入方法，其行为类似于 C++ 标准库中 std::map 的 try_emplace。这个方法有以下几个重要特点：

1. 返回值：
- 返回一个 pair，包含：
  - first: 指向插入位置的迭代器
  - second: 一个布尔值，表示是否进行了新的插入
    - true 表示执行了新插入
    - false 表示键已存在，没有执行插入

