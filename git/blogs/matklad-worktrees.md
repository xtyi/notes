# How I Use Git Worktrees

There are a bunch of posts on the internet about using git worktree command. As far as I can tell, most of them are primarily about using worktrees as a replacement of, or a supplement to git branches.

Instead of switching branches, you just change directories. This is also how I originally had used worktrees, but that didn’t stick, and I abandoned them. But recently worktrees grew on me, though my new use-case is unlike branching.

## When a Branch is Enough

If you use worktrees as a replacement for branching, that’s great, no need to change anything! But let me start with explaining why that workflow isn’t for me.

## Worktree Per Concurrent Activity

It’s a bit hard to describe, but:

- I have a fixed number of worktrees (5, to be exact)
- worktrees are mostly uncorrelated to branches
- but instead correspond to my concurrent activities during coding.

Specifically:

