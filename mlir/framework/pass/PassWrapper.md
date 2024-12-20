# PassWrapper


```cpp
/// This class provides a CRTP wrapper around a base pass class to define
/// several necessary utility methods. This should only be used for passes that
/// are not suitably represented using the declarative pass specification(i.e.
/// tablegen backend).
template <typename PassT, typename BaseT>
class PassWrapper : public BaseT {
public:
  /// Support isa/dyn_cast functionality for the derived pass class.
  static bool classof(const Pass *pass) {
    return pass->getTypeID() == TypeID::get<PassT>();
  }

protected:
  PassWrapper() : BaseT(TypeID::get<PassT>()) {}
  PassWrapper(const PassWrapper &) = default;

  /// Returns the derived pass name.
  StringRef getName() const override { return llvm::getTypeName<PassT>(); }

  /// A clone method to create a copy of this pass.
  std::unique_ptr<Pass> clonePass() const override {
    return std::make_unique<PassT>(*static_cast<const PassT *>(this));
  }
};
```

可以看到 `PassWrapper` 提供了 getName 和 clonePass 方法, 如果没有使用 tablegen 的 codegen, 可以使用该 class 为 Pass 提供这两个方法

