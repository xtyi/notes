# Overview

Cursor Chat lets you ask questions or solve problems in your codebase with the most capable language models, all in your editor.

![alt text](https://mintlify.s3.us-west-1.amazonaws.com/cursor/images/chat/chat.png)

For language models to give good answers, they need to know specific things that are relevant to your codebase — context.

要让语言模型给出好的答案，它们需要了解与您的代码库相关的具体内容--上下文。

Cursor has several built in features to provide context in chat, such as automatically including context across your entire codebase, searching the web, indexing documentation, and user-specified references to code blocks. They are built to eliminate the tedious copy-pasting otherwise necessary for working with language models on code.

Cursor 具有多种内置功能，可在聊天中提供上下文，例如自动包含整个代码库的上下文、搜索网络、索引文档以及用户指定的代码块引用。这些内置功能可消除在代码中使用语言模型时所必需的繁琐复制粘贴工作。

By default, Cursor Chat is in the AI pane, which is on the opposite side of your primary sidebar. You can toggle the AI pane by pressing Ctrl/⌘ + L, which focuses onto the chat when opened. To submit your query, press Enter.

## User and AI Messages

User messages contain the text you type, along with the context you’ve referenced. You can go back to any previous user messages to edit and rerun your queries. This will overwrite any messages after that and regenerate new ones.

AI messages are the responses generated from the AI model you’ve picked. They are paired with the user message before them. AI messages may contain parsed code blocks which can be added to your codebase with instant apply.

All user/AI messages together in the same thread are called a chat thread, and each chat thread is saved in your chat history.

引用的上下文和键入的文本都是 user message

ai 的回答是 ai message

所有一起的用户/AI 消息都称为一个 chat thread，每个 chat thread 都会保存在 chat history 中。

## Chat History

By pressing on the “Previous Chats” button on the top right of the AI pane, or by pressing Ctrl/⌘ + Alt/Option + L, you can see the chat history. You can click on any chat thread to go back and see the messages that make up that thread, and you can also modify the title of the thread by clicking the pen icon, or delete the thread by clicking the garbage can icon upon hovering over the thread in the history.

The title of a Cursor thread is just the first few words of the first user message.

