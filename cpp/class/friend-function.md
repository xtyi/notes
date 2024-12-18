# Friend Function

```cpp
class MyClass {
private:
    int data;
    
    // 声明友元函数
    friend void printData(const MyClass& obj);
};

// 实现友元函数
void printData(const MyClass& obj) {
    std::cout << obj.data;  // 可以访问私有成员
}
```

## 友元函数的三种定义方式

在类外声明和定义：

```cpp
class MyClass {
private:
    int data;
    friend void foo();  // 只声明
};

void foo() {  // 在类外定义
    MyClass obj;
    obj.data = 10;  // 可以访问私有成员
}
```

在类内定义：

```cpp
class MyClass {
private:
    int data;
    
    friend void foo() {  // 直接在类内定义
        MyClass obj;
        obj.data = 10;
    }
};
```

在命名空间中声明并在类内定义：

```cpp
namespace NS {
    void foo();  // 命名空间中声明
    
    class MyClass {
    private:
        int data;
        
        friend void foo() {  // 在类内定义，成为命名空间中声明的实现
            MyClass obj;
            obj.data = 10;
        }
    };
}
```

## 友元函数的特性

访问权限：

```cpp
class MyClass {
private:
    int private_data;
protected:
    int protected_data;
    
    friend void access_all() {
        MyClass obj;
        obj.private_data;    // 可以访问
        obj.protected_data;  // 可以访问
    }
};
```

不是成员函数：

```cpp
class MyClass {
    friend void foo() {
        // this->data;  // 错误：友元函数不是成员函数，没有 this 指针
        MyClass obj;
        obj.data;      // 正确：通过对象访问
    }
};
```

作用域：

```cpp
namespace NS {
    
    class MyClass {
        friend void foo() { }  // 定义
    };
    
    // foo() 现在在 NS 命名空间中可用
    void bar() {
        foo();  // 可以直接调用
    }
}
```

## 模板类中的友元函数

普通友元函数：

```cpp
template<typename T>
class Container {
    T data;
    friend void print(const Container& c) {
        std::cout << c.data;  // 每个模板实例化都会生成一个新的友元函数
    }
};
```

模板友元函数

```cpp
template<typename T>
class MyClass {
    T data;
    
    template<typename U>
    friend void process(MyClass<U>& obj) {
        // 可以访问任何类型的 MyClass 的私有成员
    }
};
```

##  实际应用示例

工厂模式：

```cpp
class Widget {
private:
    Widget() = default;  // 私有构造函数
    
    friend std::unique_ptr<Widget> createWidget() {
        return std::unique_ptr<Widget>(new Widget());
    }
};
```

运算符重载：

```cpp
class Complex {
private:
    double real, imag;
    
    friend Complex operator+(const Complex& a, const Complex& b) {
        return Complex(a.real + b.real, a.imag + b.imag);
    }
};
```

序列化：

```cpp
class Data {
private:
    int secret;
    
    friend void serialize(const Data& d, std::ostream& os) {
        os << d.secret;  // 可以访问私有成员进行序列化
    }
};
```

