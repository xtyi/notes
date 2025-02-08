LANG
- 默认的语言环境变量
- 作为其他 LC_ 变量的后备值
- 影响所有未被其他 LC_ 变量设置的本地化类别

```
export LANG=en_US.UTF-8
```

LC_ALL
- 最高优先级的语言环境变量
- 会覆盖所有其他 LC_ 变量和 LANG
- 通常在需要强制统一设置时使用

```
export LC_ALL=en_US.UTF-8  # 覆盖所有本地化设置
```

LANGUAGE
- 用于指定语言优先级列表
- 可以设置多个语言，按优先级排序
- 主要用于消息显示

```
export LANGUAGE="zh_CN:en_US:en"  # 优先中文，其次英文
```

```
LC_CTYPE        # 字符分类和转换
LC_NUMERIC      # 数字格式
LC_TIME         # 时间日期格式
LC_COLLATE      # 字符串排序规则
LC_MONETARY     # 货币格式
LC_MESSAGES     # 系统消息语言
LC_PAPER        # 纸张大小
LC_NAME         # 姓名格式
LC_ADDRESS      # 地址格式
LC_TELEPHONE    # 电话号码格式
LC_MEASUREMENT  # 度量衡单位
LC_IDENTIFICATION # locale信息
```

C.UTF-8
- 最小化的 UTF-8 locale
- 不包含任何本地化设置
- 系统默认行为
- 适合服务器环境

en_US.UTF-8
- 完整的美式英语本地化设置
- 包含日期、时间、数字等格式化规则
- 更适合日常使用
- 占用更多资源


服务器最小化设置

```
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
```

桌面环境完整设置：

```
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export LANGUAGE=en_US:en
```

## locale-gen

- 生成指定的 locale
- 在 /usr/share/locale 创建必要文件

```
# 生成单个 locale
sudo locale-gen en_US.UTF-8

# 生成多个 locale
sudo locale-gen en_US.UTF-8 zh_CN.UTF-8

# 显示当前所有 locale 设置
locale

# 显示可用的 locale
locale -a
```



## update-locale

- 更新系统级别的 locale 设置
- 修改 /etc/default/locale 文件

```
sudo update-locale LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8
```

## dpkg-reconfigure

Debian 系统的重新配置工具
用于重新配置已安装的包

```
# 重新配置 locales 包
sudo dpkg-reconfigure locales

# 交互式选择要生成的 locale
# 设置系统默认 locale
```


## files

/usr/share/locale

/etc/default/locale

/etc/locale.gen