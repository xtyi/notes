# RTTI

RTTI 是 C++ 中的一种机制，它允许程序在运行时获取和使用对象的类型信息

C++ 中的 RTTI 主要通过以下方式提供：
- typeid 运算符
- dynamic_cast 运算符

```cpp
class Base {
public:
    virtual ~Base() {} // 需要至少一个虚函数才能启用 RTTI
};

class Derived : public Base {
};

void example() {
    Base* ptr = new Derived();
    
    // 使用 typeid 获取类型信息
    const std::type_info& info = typeid(*ptr);
    std::cout << info.name() << std::endl;
    
    // 使用 dynamic_cast 进行类型转换
    if (Derived* derived = dynamic_cast<Derived*>(ptr)) {
        // 转换成功
    }
}
```

