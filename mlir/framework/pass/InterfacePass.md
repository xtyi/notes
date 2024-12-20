# InterfacePass

```cpp
/// Pass to transform an operation that implements the given interface.
///
/// Interface passes must not:
///   - modify any other operations within the parent region, as other threads
///     may be manipulating them concurrently.
///   - modify any state within the parent operation, this includes adding
///     additional operations.
///
/// Derived interface passes are expected to provide the following:
///   - A 'void runOnOperation()' method.
///   - A 'StringRef getName() const' method.
///   - A 'std::unique_ptr<Pass> clonePass() const' method.
template <typename InterfaceT>
class InterfacePass : public OperationPass<> {
protected:
  using OperationPass::OperationPass;

  /// Indicate if the current pass can be scheduled on the given operation type.
  /// For an InterfacePass, this checks if the operation implements the given
  /// interface.
  bool canScheduleOn(RegisteredOperationName opName) const final {
    return opName.hasInterface<InterfaceT>();
  }

  /// Return the current operation being transformed.
  InterfaceT getOperation() { return cast<InterfaceT>(Pass::getOperation()); }

  /// Query an analysis for the current operation.
  template <typename AnalysisT>
  AnalysisT &getAnalysis() {
    return Pass::getAnalysis<AnalysisT, InterfaceT>();
  }
};
```

