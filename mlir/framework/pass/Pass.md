# Pass

```cpp

/// The abstract base pass class. This class contains information describing the
/// derived pass object, e.g its kind and abstract TypeID.
class Pass {
public:
  virtual ~Pass() = default;

  /// Returns the pass info for this pass, or null if unknown.
  const PassInfo *lookupPassInfo() const {
    return PassInfo::lookup(getArgument());
  }

  //===--------------------------------------------------------------------===//
  // Options
  //===--------------------------------------------------------------------===//

  /// This class represents a specific pass option, with a provided data type.
  template <typename DataType,
            typename OptionParser = detail::PassOptions::OptionParser<DataType>>
  struct Option : public detail::PassOptions::Option<DataType, OptionParser> {
    template <typename... Args>
    Option(Pass &parent, StringRef arg, Args &&...args)
        : detail::PassOptions::Option<DataType, OptionParser>(
              parent.passOptions, arg, std::forward<Args>(args)...) {}
    using detail::PassOptions::Option<DataType, OptionParser>::operator=;
  };
  /// This class represents a specific pass option that contains a list of
  /// values of the provided data type.
  template <typename DataType,
            typename OptionParser = detail::PassOptions::OptionParser<DataType>>
  struct ListOption
      : public detail::PassOptions::ListOption<DataType, OptionParser> {
    template <typename... Args>
    ListOption(Pass &parent, StringRef arg, Args &&...args)
        : detail::PassOptions::ListOption<DataType, OptionParser>(
              parent.passOptions, arg, std::forward<Args>(args)...) {}
    using detail::PassOptions::ListOption<DataType, OptionParser>::operator=;
  };

  /// Attempt to initialize the options of this pass from the given string.
  /// Derived classes may override this method to hook into the point at which
  /// options are initialized, but should generally always invoke this base
  /// class variant.
  virtual LogicalResult initializeOptions(StringRef options);

  /// Prints out the pass in the textual representation of pipelines. If this is
  /// an adaptor pass, print its pass managers.
  void printAsTextualPipeline(raw_ostream &os);

  //===--------------------------------------------------------------------===//
  // Statistics
  //===--------------------------------------------------------------------===//

  /// This class represents a single pass statistic. This statistic functions
  /// similarly to an unsigned integer value, and may be updated and incremented
  /// accordingly. This class can be used to provide additional information
  /// about the transformations and analyses performed by a pass.
  class Statistic : public llvm::Statistic {
  public:
    /// The statistic is initialized by the pass owner, a name, and a
    /// description.
    Statistic(Pass *owner, const char *name, const char *description);

    /// Assign the statistic to the given value.
    Statistic &operator=(unsigned value);
  };

  /// Returns the main statistics for this pass instance.
  ArrayRef<Statistic *> getStatistics() const { return statistics; }
  MutableArrayRef<Statistic *> getStatistics() { return statistics; }

  /// Returns the thread sibling of this pass.
  ///
  /// If this pass was cloned by the pass manager for the sake of
  /// multi-threading, this function returns the original pass it was cloned
  /// from. This is useful for diagnostic purposes to distinguish passes that
  /// were replicated for threading purposes from passes instantiated by the
  /// user. Used to collapse passes in timing statistics.
  const Pass *getThreadingSibling() const { return threadingSibling; }

  /// Returns the thread sibling of this pass, or the pass itself it has no
  /// sibling. See `getThreadingSibling()` for details.
  const Pass *getThreadingSiblingOrThis() const {
    return threadingSibling ? threadingSibling : this;
  }

protected:







  /// Schedule an arbitrary pass pipeline on the provided operation.
  /// This can be invoke any time in a pass to dynamic schedule more passes.
  /// The provided operation must be the current one or one nested below.
  LogicalResult runPipeline(OpPassManager &pipeline, Operation *op) {
    return passState->pipelineExecutor(pipeline, op);
  }

  /// A clone method to create a copy of this pass.
  std::unique_ptr<Pass> clone() const {
    auto newInst = clonePass();
    newInst->copyOptionValuesFrom(this);
    return newInst;
  }





  /// Copy the option values from 'other', which is another instance of this
  /// pass.
  void copyOptionValuesFrom(const Pass *other);

private:
  /// Out of line virtual method to ensure vtables and metadata are emitted to a
  /// single .o file.
  virtual void anchor();



  /// The set of statistics held by this pass.
  std::vector<Statistic *> statistics;

  /// The pass options registered to this pass instance.
  detail::PassOptions passOptions;

  /// A pointer to the pass this pass was cloned from, if the clone was made by
  /// the pass manager for the sake of multi-threading.
  const Pass *threadingSibling = nullptr;

  /// Allow access to 'clone'.
  friend class OpPassManager;

  /// Allow access to 'canScheduleOn'.
  friend detail::OpPassManagerImpl;

  /// Allow access to 'passState'.
  friend detail::OpToOpPassAdaptor;

  /// Allow access to 'passOptions'.
  friend class PassInfo;
};
```


## passID

```cpp
private:
    /// Represents a unique identifier for the pass.
    TypeID passID;

public:
    /// Returns the unique identifier that corresponds to this pass.
    TypeID getTypeID() const { return passID; }

protected:
    explicit Pass(TypeID passID, std::optional<StringRef> opName = std::nullopt)
      : passID(passID), opName(opName) {}
    Pass(const Pass &other) : Pass(other.passID, other.opName) {}
```


## opName

```cpp
private:
  /// The name of the operation that this pass operates on, or std::nullopt if
  /// this is a generic OperationPass.
  std::optional<StringRef> opName;

public:
  /// Returns the name of the operation that this pass operates on, or
  /// std::nullopt if this is a generic OperationPass.
  std::optional<StringRef> getOpName() const { return opName; }

protected:
  /// Indicate if the current pass can be scheduled on the given operation type.
  /// This is useful for generic operation passes to add restrictions on the
  /// operations they operate on.
  virtual bool canScheduleOn(RegisteredOperationName opName) const = 0;
```


## derived pass

```cpp
public:

    /// Returns the derived pass name.
    // PassWrapper 或者 tablegen 生成 PassBase 实现
    virtual StringRef getName() const = 0;

    /// The polymorphic API that runs the pass over the currently held operation.
    // 用户编写 Pass 实现
    virtual void runOnOperation() = 0;

    /// Register dependent dialects for the current pass.
    /// A pass is expected to register the dialects it will create entities for
    /// (Operations, Types, Attributes), other than dialect that exists in the
    /// input. For example, a pass that converts from Linalg to Affine would
    /// register the Affine dialect but does not need to register Linalg.
    // tablegen 生成 PassBase 实现
    virtual void getDependentDialects(DialectRegistry &registry) const {}

    /// Return the command line argument used when registering this pass. Return
    /// an empty string if one does not exist.
    // tablegen 生成 PassBase 实现
    virtual StringRef getArgument() const { return ""; }

    /// Return the command line description used when registering this pass.
    /// Return an empty string if one does not exist.
    virtual StringRef getDescription() const { return ""; }

protected:


    /// Initialize any complex state necessary for running this pass. This hook
    /// should not rely on any state accessible during the execution of a pass.
    /// For example, `getContext`/`getOperation`/`getAnalysis`/etc. should not be
    /// invoked within this hook.
    /// This method is invoked after all dependent dialects for the pipeline are
    /// loaded, and is not allowed to load any further dialects (override the
    /// `getDependentDialects()` for this purpose instead). Returns a LogicalResult
    /// to indicate failure, in which case the pass pipeline won't execute.
    virtual LogicalResult initialize(MLIRContext *context) { return success(); }


    /// Create a copy of this pass, ignoring statistics and options.
    virtual std::unique_ptr<Pass> clonePass() const = 0;
```



## Pass State

```cpp
/// The state for a single execution of a pass. This provides a unified
/// interface for accessing and initializing necessary state for pass execution.
struct PassExecutionState {
  PassExecutionState(Operation *ir, AnalysisManager analysisManager,
                     function_ref<LogicalResult(OpPassManager &, Operation *)>
                         pipelineExecutor)
      : irAndPassFailed(ir, false), analysisManager(analysisManager),
        pipelineExecutor(pipelineExecutor) {}

  /// The current operation being transformed and a bool for if the pass
  /// signaled a failure.
  llvm::PointerIntPair<Operation *, 1, bool> irAndPassFailed;

  /// The analysis manager for the operation.
  AnalysisManager analysisManager;

  /// The set of preserved analyses for the current execution.
  detail::PreservedAnalyses preservedAnalyses;

  /// This is a callback in the PassManager that allows to schedule dynamic
  /// pipelines that will be rooted at the provided operation.
  function_ref<LogicalResult(OpPassManager &, Operation *)> pipelineExecutor;
};
```

```cpp
private:
    /// The current execution state for the pass.
    std::optional<detail::PassExecutionState> passState;

protected:
    /// Returns the current pass state.
    detail::PassExecutionState &getPassState() {
        assert(passState && "pass state was never initialized");
        return *passState;
    }

    /// Return the current operation being transformed.
    Operation *getOperation() {
        return getPassState().irAndPassFailed.getPointer();
    }

    /// Return the MLIR context for the current operation being transformed.
    MLIRContext &getContext() { return *getOperation()->getContext(); }

    /// Signal that some invariant was broken when running. The IR is allowed to
    /// be in an invalid state.
    void signalPassFailure() { getPassState().irAndPassFailed.setInt(true); }
```

## Analysis

```cpp

protected:

  /// Returns the current analysis manager.
  AnalysisManager getAnalysisManager() {
    return getPassState().analysisManager;
  }


    /// Query an analysis for the current ir unit.
  template <typename AnalysisT>
  AnalysisT &getAnalysis() {
    return getAnalysisManager().getAnalysis<AnalysisT>();
  }

  /// Query an analysis for the current ir unit of a specific derived operation
  /// type.
  template <typename AnalysisT, typename OpT>
  AnalysisT &getAnalysis() {
    return getAnalysisManager().getAnalysis<AnalysisT, OpT>();
  }

  /// Query a cached instance of an analysis for the current ir unit if one
  /// exists.
  template <typename AnalysisT>
  std::optional<std::reference_wrapper<AnalysisT>> getCachedAnalysis() {
    return getAnalysisManager().getCachedAnalysis<AnalysisT>();
  }

  /// Mark all analyses as preserved.
  void markAllAnalysesPreserved() {
    getPassState().preservedAnalyses.preserveAll();
  }

  /// Mark the provided analyses as preserved.
  template <typename... AnalysesT>
  void markAnalysesPreserved() {
    getPassState().preservedAnalyses.preserve<AnalysesT...>();
  }
  void markAnalysesPreserved(TypeID id) {
    getPassState().preservedAnalyses.preserve(id);
  }

  /// Returns the analysis for the given parent operation if it exists.
  template <typename AnalysisT>
  std::optional<std::reference_wrapper<AnalysisT>>
  getCachedParentAnalysis(Operation *parent) {
    return getAnalysisManager().getCachedParentAnalysis<AnalysisT>(parent);
  }

  /// Returns the analysis for the parent operation if it exists.
  template <typename AnalysisT>
  std::optional<std::reference_wrapper<AnalysisT>> getCachedParentAnalysis() {
    return getAnalysisManager().getCachedParentAnalysis<AnalysisT>(
        getOperation()->getParentOp());
  }

  /// Returns the analysis for the given child operation if it exists.
  template <typename AnalysisT>
  std::optional<std::reference_wrapper<AnalysisT>>
  getCachedChildAnalysis(Operation *child) {
    return getAnalysisManager().getCachedChildAnalysis<AnalysisT>(child);
  }

  /// Returns the analysis for the given child operation, or creates it if it
  /// doesn't exist.
  template <typename AnalysisT>
  AnalysisT &getChildAnalysis(Operation *child) {
    return getAnalysisManager().getChildAnalysis<AnalysisT>(child);
  }

  /// Returns the analysis for the given child operation of specific derived
  /// operation type, or creates it if it doesn't exist.
  template <typename AnalysisT, typename OpTy>
  AnalysisT &getChildAnalysis(OpTy child) {
    return getAnalysisManager().getChildAnalysis<AnalysisT>(child);
  }

```
