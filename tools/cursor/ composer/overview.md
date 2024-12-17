# Overview

Composer is your AI coding assistant that lives in your editor.

It helps you explore code, write new features, and modify existing code while staying in your workflow.

Access it with: ⌘I to open, and ⌘N to create a new composer.

总结:
- composer 会直接帮你修改代码
- composer 有 normal 模式和 agent 模式
  - agent 模式会使用工具
    - 创建和修改文件
    - 自动拉取相关的上下文, 即 @recommended 功能
    - 运行命令行
- 使用 @ 调出 context 选项
  - 使用 # 可以快速选择文件
- checkpoint 支持恢复到该点的修改
- 支持自动修复 lint 错误


- composer 支持两种 layout
  - 一种就是在侧边栏
  - 一种是作为一个编辑器面板打开
- cmd+i 打开 composer, cmd+alt+i 打开 history, cmd+n 新的 composer


## Agent

Enable Agent with ⌘. to get a coding partner that can work proactively with your codebase:

- Pull relevant context automatically (try @Recommended)
- Run terminal commands
- Create and modify files
- Search code semantically
- Execute file operations

Agent has the ability to execute up to 25 tool calls before it will stop. If you need more, let us know at hi@cursor.com!

For now, Agent only supports Claude models.

> Each tool operation counts as a separate request in your quota.

## Normal

Normal mode gives you core features for code exploration and generation:

- Search through your codebase and documentation
- Use web search
- Create and write files
- Access expanded @-symbol commands

## Working with Context

Type @ to see context options based on your current work.

Navigate with arrow keys, select with Enter, and filter by typing after @.

Use Ctrl/⌘ M to switch file reading methods. @Recommended in Agent automatically pulls relevant context.

![alt text](https://mintlify.s3.us-west-1.amazonaws.com/cursor/images/context/@-symbols-basics.png)

`#` File Selection

Use # followed by a filename to focus on specific files. Combine this with @ symbols for precise context control.

**Context Pills**

Pills at the top of your chat show active context. Add or remove pills to adjust what Composer sees. Use # to select files, which then appear as pills.

## Generating & Applying Changes

When Composer suggests changes:

- Review them in the diff view
- Accept or reject changes with the buttons provided
- Use checkpoints to undo if needed

## Checkpoints

Every time you generate code, Composer creates a checkpoint. You can return to any previous version by clicking on checkout near that checkpoint. This is handy if you don’t like the current changes and want to revert to an earlier state.

![alt text](https://mintlify.s3.us-west-1.amazonaws.com/cursor/images/composer/checkpoints.png)

## History

Access previous Composer sessions and chats through the history. Open it from the history icon to the right of Cursor Tab. You’ll see a list of past composers and chats, which you can revisit, rename, or remove.

Open with ⌘+⌥+L or Ctrl+Alt+L when Composer is focused.

![alt text](https://mintlify.s3.us-west-1.amazonaws.com/cursor/images/composer/history.png)

## Layout

Composer offers two layout modes:

- Pane: A sidebar with chat on the left and your code editor on the right.
- Editor: A single editor window, similar to viewing code normally. You can move it around, split it, or even place it in a separate window.

## Beta Features

### Iterate on lints

Composer attempts to fix linting issues in generated code for most programming languages. If Composer detects lint errors, it will try to fix them automatically when this feature is enabled. Currently, only one iteration is supported.

![alt text](https://mintlify.s3.us-west-1.amazonaws.com/cursor/images/composer/iterate-on-lint.png)

> Some languages (like Rust) require files to be saved before lint errors appear, which may limit this feature’s effectiveness in all languages.

## FAQ

### What’s the difference between Chat and Composer?

Cursor Chat helps you search and understand your code. Use it to explore your codebase, ask questions, and get explanations. You can search your code with ⌘⏎.

Composer helps you write and edit code. It provides a workspace where you can generate new code and apply changes directly to your files.

