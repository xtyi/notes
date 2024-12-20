# OperationPass

```cpp
/// Pass to transform an operation.
///
/// Operation passes must not:
///   - modify any other operations within the parent region, as other threads
///     may be manipulating them concurrently.
///   - modify any state within the parent operation, this includes adding
///     additional operations.
///
/// Derived operation passes are expected to provide the following:
///   - A 'void runOnOperation()' method.
///   - A 'StringRef getName() const' method.
///   - A 'std::unique_ptr<Pass> clonePass() const' method.
template <>
class OperationPass<void> : public Pass {
protected:
  OperationPass(TypeID passID) : Pass(passID) {}
  OperationPass(const OperationPass &) = default;

  /// Indicate if the current pass can be scheduled on the given operation type.
  /// By default, generic operation passes can be scheduled on any operation.
  bool canScheduleOn(RegisteredOperationName opName) const override {
    return true;
  }
};
```

```cpp
/// Pass to transform an operation of a specific type.
///
/// Operation passes must not:
///   - modify any other operations within the parent region, as other threads
///     may be manipulating them concurrently.
///   - modify any state within the parent operation, this includes adding
///     additional operations.
///
/// Derived operation passes are expected to provide the following:
///   - A 'void runOnOperation()' method.
///   - A 'StringRef getName() const' method.
///   - A 'std::unique_ptr<Pass> clonePass() const' method.
template <typename OpT = void>
class OperationPass : public Pass {
protected:
  OperationPass(TypeID passID) : Pass(passID, OpT::getOperationName()) {}
  OperationPass(const OperationPass &) = default;

  /// Support isa/dyn_cast functionality.
  static bool classof(const Pass *pass) {
    return pass->getOpName() == OpT::getOperationName();
  }

  /// Indicate if the current pass can be scheduled on the given operation type.
  bool canScheduleOn(RegisteredOperationName opName) const final {
    return opName.getStringRef() == getOpName();
  }

  /// Return the current operation being transformed.
  OpT getOperation() { return cast<OpT>(Pass::getOperation()); }

  /// Query an analysis for the current operation of the specific derived
  /// operation type.
  template <typename AnalysisT>
  AnalysisT &getAnalysis() {
    return Pass::getAnalysis<AnalysisT, OpT>();
  }
};

```