# A walk through the Magit interface

This article demonstrates some of Magit’s most essential features in order to give you an impression of how the interface works.

It also hints at some of the design principles behind that interface.

But this is only the tip of the iceberg.

Magit is a complete interface to Git, which does not limit itself to the “most essential features everyone needs”.

I hope that this article succeeds in demonstrating how Magit’s focus on work-flows allows its users to become more effective Git users.

Here we concentrate on some essential work-flows, but note that more advanced features and work-flows have been optimized with the same attention to detail.

If you would rather concentrate on the big picture, then read the article Magit the magical Git interface instead (or afterwards).

## In Magit everything is actionable

Almost everything you see in a Magit buffer can be acted on by pressing some key, but that’s not obvious from just seeing how Magit looks. The screenshots below are accompanied by text explaining how what you see can be used to perform a variety of actions.

Regardless of where in a Magit buffer you are, you can always show more details about (or an alternative view of) the thing at point without having to type or copy-paste any information, as you often would have to do on the command line.

## The status buffer

The status buffer, which can be shown while inside a Git repository by typing C-x g (Control+x followed by g 1), is Magit’s equivalent of typing git status in a shell. It shows a quick overview of the current repository.

![alt text](https://magit.vc/screenshots/status.png)

As you can see, Magit shows more information than Git. On the command line, you would have to also use git diff, git diff --cached, git log --oneline origin/master.., git log --oneline ..origin/master, and a few other commands to get the same information.

## Other Emacs themes

Many themes exist for Emacs. Throughout this guide we use the Solarized light theme, but here is how the status buffer looks when using some other popular themes:

## Hiding details

