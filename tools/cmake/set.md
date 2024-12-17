# cmake set

CMake 中 set 命令的作用范围取决于变量的类型和在哪里定义。主要有以下几种情况：

1. 普通变量（Normal Variables）

- 默认只在当前 CMakeLists.txt 文件及其子目录的 CMakeLists.txt 中生效
- 子目录可以修改从父目录继承的变量值，但不会影响父目录的值


2. 缓存变量（Cache Variables）

- 使用 set(VAR VALUE CACHE TYPE "DESCRIPTION" FORCE) 语法定义
- 全局生效，会被存储在 CMakeCache.txt 中
- 可以跨 CMakeLists.txt 文件访问

3. 环境变量（Environment Variables）

- 使用 set(ENV{VAR} VALUE) 语法定义
- 修改的是 CMake 进程的环境变量
- 全局生效，但仅限于 CMake 构建过程

