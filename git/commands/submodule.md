# submodule

git submodule 是 Git 提供的一种机制，用于在一个 Git 仓库中包含另一个 Git 仓库。这在处理大型项目时非常有用，因为它允许你将项目拆分成多个独立的子项目，每个子项目都可以有自己独立的版本控制历史。以下是 git submodule 最常用的功能和命令。


常用场景

代码复用：可以将一些通用的代码库或者组件作为子模块包含到多个项目中，方便代码的复用和维护。
独立开发：子模块有自己独立的仓库和版本历史，不同的开发团队可以独立开发和维护子模块，主项目只需要引用特定版本的子模块。
项目拆分：对于大型项目，可以将其拆分成多个小的子项目，每个子项目都可以独立进行版本控制，使得项目结构更加清晰。

常用命令
1. 添加子模块

将指定的远程仓库作为子模块添加到当前项目的指定本地路径下。

```
git submodule add <仓库地址> <本地路径>
```


```
git submodule add https://github.com/example/submodule.git submodule_dir
```

这会在当前项目下创建一个名为 submodule_dir 的目录，并将 `https://github.com/example/submodule.git` 仓库克隆到该目录下。同时，会在当前项目的 `.gitmodules` 文件中记录子模块的信息。



2. 初始化子模块

当你克隆一个包含子模块的项目时，默认情况下，子模块目录只是一个空文件夹，并没有实际的代码内容。git submodule init 的作用就是读取项目根目录下的 .gitmodules 文件，并将其中记录的子模块配置信息复制到本地仓库的 .git/config 文件中，为后续拉取子模块代码做准备。

```
git submodule init
```

初始化指定的子模块

```
git submodule init submodule_dir
```



3. 更新子模块

将子模块更新到当前记录的版本。

```
git submodule update
```


将子模块更新到其远程仓库的最新版本。

```
git submodule update --remote
```

更新指定的子模块, 而非所有子模块

```
git submodule update submodule_dir

git submodule update --remote submodule_dir
```

4. 查看子模块状态

```
git submodule status
```

显示所有子模块的当前状态，包括子模块的提交哈希、是否有修改等信息。

## submodule info

.gitmodules 文件主要记录子模块的配置信息，包括子模块的路径和远程仓库地址，但不直接记录提交信息。

```
[submodule "submodule_dir"]
    path = submodule_dir
    url = https://github.com/example/submodule.git
```

主项目的提交历史中会记录子模块在每次提交时所引用的具体提交哈希。你可以在主项目的根目录下使用 git log 命令查看

```
git log
```

当你在主项目中提交对子模块的更改（例如更新子模块到新的版本）时，提交信息中会包含子模块引用的新提交哈希。在 git log 的输出中，子模块的更改会以类似以下的形式显示：

```
commit 123456789abcdef
Author: Your Name <your.email@example.com>
Date:   Thu Feb 6 10:00:00 2025 +0800

    Update submodule to latest version

 Submodule submodule_dir 987654321abcdef..123456789abcdef
```

这里的 987654321abcdef 是子模块之前引用的提交哈希，123456789abcdef 是更新后子模块引用的提交哈希。

