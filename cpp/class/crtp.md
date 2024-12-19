# CRTP

## Why CRTP

思考一下, C++ 中, 基类如何访问到派生类, 这里有三个层次

1. 能访问派生类的部分函数
2. 能访问派生类的实例
3. 能访问派生类

**使用虚函数**

```cpp
#include <iostream>

class Base {
public:
    virtual void print() = 0;  // 纯虚函数
    void doSomething()
    {
        print();  // 会调用派生类的实现
    }
};

class Derived : public Base {
public:
    void print() override
    {
        std::cout << "Derived" << std::endl;
    }
};

int main()
{
    Derived d;
    d.doSomething(); // Derived
}
```

特点
- 只能访问派生类对虚函数的实现
- 运行时多态

缺点
- 有虚函数表带来的性能开销

但是这里只能访问派生类的虚函数, 访问不到派生类实例

但是对于其他某些语言, 例如 Python, 访问派生类的实例太简单了

```py
class Base:
    def do_something(self):
        self.print()

class Derived(Base):
    def print(self):
        print("Derived")

derived = Derived()
derived.do_something()
```

在 Python 中, 得益于鸭子类型, 使用 `derived.do_something()` 的方式调用, 就会把 `derived` 传给 `do_something()`, 所以此时 `do_something` 的 `self` 参数就是 `derived`

甚至可以使用一个与 `Base` 完全无关的实例去调用 `do_something()`

```py
class Whatever:
    def print(self):
        print("Whatever")

whatever = Whatever()
Base.do_something(whatever)
```

Python 也可以通过 `__class__` 轻易访问到派生类

Java 如何实现的呢? 通过反射

```java
import java.util.*;
import java.lang.reflect.Method;
import java.lang.reflect.InvocationTargetException;

public class Main {
    public static void main(String[] args) {
      Derived derived = new Derived();
      derived.doSomething();
  }
}

public class Base {
    public void doSomething() {
        // 获取实际运行时类型的实例
        Object derived = this.getClass().cast(this);
        
        try {
            Method printMethod = this.getClass().getMethod("print");
            printMethod.invoke(derived);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}

public class Derived extends Base {
    public void print() {
        System.out.println("Derived");
    }
}
```

Java 支持反射, 可以通过 `this.getClass()` 直接获得对象实际的类型, 即派生类, 然后做一下类型转换即可

为什么 C++ 不行呢? 

首先 C++ 没有 Python 那样的动态性, C++ 的 this 虽然在内存中也指向派生类实例, 但是在基类中, this 类型是基类, 所以无法访问派生类的成员, 而且基类对派生类一无所知, 无法转换成派生类

所以关键是要访问到派生类, 知道了派生类, 就可以把 this 转换成派生类

如何直接访问到派生类(Derived), 由于 C++ 没有 Java 那样的反射机制, 所以只能使用模板, 把派生类作为模板参数传递给基类, 即 **CRTP**

## What CRTP

```cpp
#include <iostream>

template <typename T>
class Base {
public:
    void doSomething()
    {
        T& derived = static_cast<T&>(*this);
        derived.print();
    }
};

class Derived : public Base<Derived> {
public:
    void print()
    {
        std::cout << "Derived" << std::endl;
    }
};

int main()
{
    Derived d;
    d.doSomething();
}
```

特点
- 编译时多态，没有运行时开销
- 直接访问到派生类, 而不仅仅是实例

缺点
- 模板代码膨胀

