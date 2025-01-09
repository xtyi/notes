# variant


## visit

基础用法

```cpp
#include <variant>
#include <iostream>

int main() {
    // 定义一个可以存储 int 或 string 的 variant
    std::variant<int, std::string> v = "hello";
    
    // 使用 visit 访问 variant
    std::visit([](const auto& value) {
        std::cout << value << std::endl;
    }, v);
    
    v = 42;  // 切换到 int
    std::visit([](const auto& value) {
        std::cout << value << std::endl;
    }, v);
}
```

这里使用了 auto 关键字和 lambda 表达式，编译器会为每种可能的类型生成对应的函数调用。实际上，编译器会将这个 lambda 展开为类似这样的代码

```cpp
struct Lambda {
    void operator()(const int& value) const {
        std::cout << value << std::endl;
    }
    
    void operator()(const std::string& value) const {
        std::cout << value << std::endl;
    }
};
```

处理多个 variant

```cpp
#include <variant>
#include <iostream>

int main() {
    std::variant<int, std::string> v1 = 42;
    std::variant<int, std::string> v2 = "hello";
    
    // 同时访问两个 variant
    std::visit([](const auto& a, const auto& b) {
        std::cout << "v1: " << a << ", v2: " << b << std::endl;
    }, v1, v2);
}
```


定义 Visitor 类

```cpp
#include <variant>
#include <iostream>

// 定义访问器
struct Visitor {
    void operator()(int i) const {
        std::cout << "整数: " << i << std::endl;
    }
    
    void operator()(const std::string& s) const {
        std::cout << "字符串: " << s << std::endl;
    }
};

int main() {
    std::variant<int, std::string> v = "hello";
    std::visit(Visitor{}, v);  // 输出: 字符串: hello
    
    v = 42;
    std::visit(Visitor{}, v);  // 输出: 整数: 42
}
```


返回值处理

```cpp
#include <variant>
#include <iostream>

int main() {
    std::variant<int, std::string> v = 42;
    
    // visit 可以返回值
    auto result = std::visit([](const auto& value) -> int {
        if constexpr (std::is_same_v<decltype(value), const int&>) {
            return value * 2;
        } else {
            return value.length();
        }
    }, v);
    
    std::cout << "结果: " << result << std::endl;
}
```