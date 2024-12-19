1. dynamic_cast

主要用于处理多态类型的转换

只能用于含有虚函数的类层次结构中

在运行时进行类型检查，相对较安全但性能较低

如果转换失败，对指针返回 nullptr，对引用抛出 bad_cast 异常

```cpp
class Base {
    virtual void foo() {} // 必须有虚函数
};
class Derived : public Base { };

Base* b = new Derived();
Derived* d = dynamic_cast<Derived*>(b); // 安全的向下转换
```

2. static_cast

编译时进行类型检查

用于非多态类型的转换

可以进行内置数据类型之间的转换

可以进行具有继承关系的指针或引用之间的转换

性能好但相对不太安全

```cpp
float f = 3.14;
int i = static_cast<int>(f);    // 浮点数转整数

Base* b = new Base();
Derived* d = static_cast<Derived*>(b); // 不安全，但编译通过
```

3. const_cast

用于移除或添加 const 属性

是唯一能够操作常量性的 C++ 类型转换操作符

```cpp
const int constant = 21;
const int* const_p = &constant;
int* modifiable = const_cast<int*>(const_p); // 移除 const
```

4. reinterpret_cast


最不安全的转换方式


可以在任何指针/引用类型之间进行转换

可以在指针和足够大的整数类型之间进行转换

结果与编译器实现相关，不具可移植性

```cpp
float f = 3.14;
int i = (int)f;    // C 风格转换，不推荐
```