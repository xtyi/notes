# worktree

通常情况下，一个 Git 仓库只有一个工作目录，当你切换分支时，工作目录中的文件会被更新以反映新分支的状态

git worktree 打破了这种限制，允许你在不同的工作目录中同时处理多个分支，每个工作目录都是独立的，互不干扰

添加新的工作树

```
git worktree add <路径> <分支名>
```


```
git worktree add ../feature-branch-worktree feature-branch
```

列出所有工作树

```
git worktree list
```

删除工作树

```
git worktree remove <路径>
```

移动工作树

```
git worktree move <旧路径> <新路径>
```

```
git worktree move ../feature-branch-worktree ../new-feature-branch-worktree
```

