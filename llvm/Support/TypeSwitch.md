# TypeSwitch

位于 `llvm/include/llvm/Support/TypeSwitch.h`

简化版 TypeSwitch 实现

```cpp
template <typename T, typename ResultT>
class TypeSwitch {
    ResultT result;
    bool matched = false;

public:
    TypeSwitch(T value) : value(value) {}

    template <typename CaseT>
    TypeSwitch& Case(std::function<ResultT(CaseT)> fn) {
        if (!matched && value == CaseT()) {
            matched = true;
            result = fn(CaseT());
        }
        return *this;
    }

    TypeSwitch& Default(std::function<ResultT(T)> fn) {
        if (!matched) {
            result = fn(value);
            matched = true;
        }
        return *this;
    }

    operator ResultT() { return result; }

private:
    T value;
};
```

使用示例

```cpp
// 使用示例
enum VecValue {
    VecValue_Int64VecValue,
    VecValue_Int32VecValue,
    VecValue_Float64VecValue,
    VecValue_Float32VecValue
};

void example() {
    VecValue type = VecValue_Int64VecValue;
    
    auto result = TypeSwitch<VecValue, std::string>(type)
        .Case<VecValue_Int64VecValue>([](auto) {
            return "Int64";
        })
        .Case<VecValue_Int32VecValue>([](auto) {
            return "Int32";
        })
        .Default([](auto) {
            return "Unknown";
        });
}
```

