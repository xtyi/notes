# CodeGen

一个最简单的 tablegen 的 Pass

```td
def PrintIRStructure : Pass<"print-ir-structure"> {
  let summary = "print ir structure";
  let description = [{

  }];
}
```

会生成下面的代码

```cpp
template <typename DerivedT> // 具体的 Pass 类将作为模板传入
class PrintIRStructureBase : public ::mlir::OperationPass<> { // 继承 mlir::OperationPass
public:
  using Base = PrintIRStructureBase;

  PrintIRStructureBase() : ::mlir::OperationPass<>(::mlir::TypeID::get<DerivedT>()) {} // 默认构造函数
  PrintIRStructureBase(const PrintIRStructureBase &other) : ::mlir::OperationPass<>(other) {} // 拷贝构造函数

  /// Returns the command-line argument attached to this pass.
  static constexpr ::llvm::StringLiteral getArgumentName() {
    return ::llvm::StringLiteral("print-ir-structure");
  }
  ::llvm::StringRef getArgument() const override { return "print-ir-structure"; }

  ::llvm::StringRef getDescription() const override { return "print ir structure"; }

  /// Returns the derived pass name.
  static constexpr ::llvm::StringLiteral getPassName() {
    return ::llvm::StringLiteral("PrintIRStructure");
  }
  ::llvm::StringRef getName() const override { return "PrintIRStructure"; }

  /// Support isa/dyn_cast functionality for the derived pass class.
  static bool classof(const ::mlir::Pass *pass) {
    return pass->getTypeID() == ::mlir::TypeID::get<DerivedT>();
  }

  /// A clone method to create a copy of this pass.
  std::unique_ptr<::mlir::Pass> clonePass() const override {
    return std::make_unique<DerivedT>(*static_cast<const DerivedT *>(this));
  }

  /// Return the dialect that must be loaded in the context before this pass.
  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    
  }

  /// Explicitly declare the TypeID for this class. We declare an explicit private
  /// instantiation because Pass classes should only be visible by the current
  /// library.
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrintIRStructureBase<DerivedT>)

protected:
private:

  friend std::unique_ptr<::mlir::Pass> createPrintIRStructure() {
    return std::make_unique<DerivedT>();
  }
};
```

