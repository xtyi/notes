# Codebase Indexing

For better and more accurate codebase answers using @codebase or Ctrl/⌘ Enter, you can index your codebase. Behind the scenes, Cursor computes embeddings for each file in your codebase, and will use these to improve the accuracy of your codebase answers.

Your codebase index will automatically synchronize with your latest codebase changes.

The status of your codebase indexing is under Cursor Settings > Features > Codebase Indexing.

![alt text](https://mintlify.s3.us-west-1.amazonaws.com/cursor/images/chat/codebase-indexing.png)

给代码库建立索引有助于提高 AI 的性能

`Cursor Settings > Features > Codebase Indexing` 代码库的索引相关设置


## Advanced Settings

By default, Cursor will index all files in your codebase.

You can also expand the Show Settings section to access more advanced options. Here, you can decide whether you want to enable automatic indexing for new repositories and configure the files that Cursor will ignore during repository indexing, in addition to your .gitignore settings.

If you have any large content files in your project that the AI definitely doesn’t need to read, ignoring those files could improve the accuracy of the answers.

可以设置
- 是否为新的目录自动创建索引
- 忽略的目录/文件
  - .gitignore 里面的会被忽略
  - 如果有的文件不能加到 .gitignore, 但是想忽略, 会创建一个 .cursorignore


