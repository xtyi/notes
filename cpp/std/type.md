# type

```cpp
using T = typename std::decay_t<decltype(vec)>::value_type;
```

decltype(vec) 获取 vec 变量的精确类型
std::decay_t 移除引用、const 等修饰符，得到基础类型
::value_type 获取容器(vector)中元素的类型



## decltype()

