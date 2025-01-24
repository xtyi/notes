# tips


## squash

当前有两个分支 master 和 dev

如何将 dev 上比 master 多的 commit 都合并成一个

```
git switch dev
git reset --soft $(git merge-base master HEAD)
git commit -m "one commit on yourBranch"
git push -f
```



```
git merge-base master HEAD
```

找到 master 和 HEAD 分叉的 commit

```
git reset --soft $commit
```

相比于该 commit 的修改全部撤销, 放到暂存区