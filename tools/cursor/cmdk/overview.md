# Cmd K Overview

https://docs.cursor.com/cmdk/overview

cmdk 调出 prompt bar

可以选中代码然后调用 cmdk, 这样会根据提示修改选中的代码

也可以不选中代码调用 cmdk, 这样会根据提示插入代码

prompt bar 里面输完提示按 enter, ai 就会修改代码, 之后按 cmd+enter 确认修改, 按 cmd+backspace 取消修改

如果不满意修改, 可以继续在 prompt bar 里面输入指令, 即支持 follow up instructions

输入提示可以不按 enter, 而是按 alt+enter, 这样不会修改代码, 而是会回答问题, 这是一个快速提问的功能

cursor 会找各种上下文信息, 包括相关文件, 最近查看的文件等等, 加上我们给它的上下文提示 `@`



## Terminal

在终端中可以使用 cmdk 打开终端底部的 prompt, 可以描述所需要的操作, 让 ai 自动生成命令

这里会把最近的终端历史记录作为上下文