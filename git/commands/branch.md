# branch

查看本地分支

```
git branch
```

示例输出

```
* master
  feature-branch
  bugfix-branch
```

显示每个分支的最后一次提交的哈希值和提交信息

```
git branch -v
```


查看远程分支

```
git branch -r
```

示例输出

```
origin/master
origin/feature-branch
```

查看本地和远程分支

```
git branch -a
```

创建分支: 基于当前所在分支创建一个新的本地分支，但不会自动切换到新分支。

```
git branch <分支名>
```

删除指定的本地分支

```
git branch -d <分支名>
```

强制删除本地分支

```
git branch -D <分支名>
```


重命名分支

```
git branch -m <旧分支名> <新分支名>
```

