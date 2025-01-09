# APFloat

APFloat 是 LLVM 中用于处理浮点数运算的核心类，它支持任意精度的浮点数计算，主要用于编译器的常量折叠和浮点数分析



APFloat 内部使用以下组件来表示浮点数：

```cpp
class APFloat {
    // 语义信息，定义了浮点数的格式（如 IEEE-754）
    const fltSemantics *semantics;
    
    // 符号位
    bool sign;
    
    // 指数
    exponent_t exponent;
    
    // 尾数，使用一个特殊的位数组存储
    integerPart *significand;

    // Constructors specify semantics:
    explicit APFloat(const fltSemantics &); // empty
    APFloat(const fltSemantics &, StringRef);
    APFloat(const fltSemantics &, integerPart);
};
```


```cpp
// 从双精度浮点数创建
APFloat float1(3.14159);

// 使用特定语义创建
APFloat float2(APFloat::IEEEdouble(), "3.14159");
```


```cpp
APFloat a(1.0), b(2.0);
APFloat::opStatus status;

// 加法
status = a.add(b, APFloat::rmNearestTiesToEven);

// 乘法
status = a.multiply(b, APFloat::rmNearestTiesToEven);

// 除法
status = a.divide(b, APFloat::rmNearestTiesToEven);
```


```cpp
const fltSemantics &IEEEsingle();    // f32
const fltSemantics &IEEEdouble();    // f64
const fltSemantics &IEEEquad();      // f128
```

```cpp
// 创建 f32 的 APFloat
APFloat f32_val(APFloat::IEEEsingle(), "1.0");

// 创建 f64 的 APFloat
APFloat f64_val(APFloat::IEEEdouble(), "1.0");

// 可以通过 getSemantics() 获取精度信息
const fltSemantics &sem = f32_val.getSemantics();
if (&sem == &APFloat::IEEEsingle()) {
    // 这是 f32
}

// 转换精度
bool losesInfo;
f32_val.convert(APFloat::IEEEdouble(), 
                APFloat::rmNearestTiesToEven,
                &losesInfo);
```
