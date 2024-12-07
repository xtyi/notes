# Canonicalization

Canonicalization is an important part of compiler IR design: it makes it easier to implement reliable compiler transformations and to reason about what is better or worse in the code, and it forces interesting discussions about the goals of a particular level of IR.

Dan Gohman wrote an [article](https://sunfishcode.github.io/blog/2018/10/22/Canonicalization.html) exploring these issues; it is worth reading if you're not familiar with these concepts.

Most compilers have canonicalization passes, and sometimes they have many different ones (e.g. instcombine, dag combine, etc in LLVM).

Because MLIR is a multi-level IR, we can provide a single canonicalization infrastructure and reuse it across many different IRs that it represents.

This document describes the general approach, global canonicalizations performed, and provides sections to capture IR-specific rules for reference.

## General Design

```
def Canonicalizer : Pass<"canonicalize"> {
  let summary = "Canonicalize operations";
  let description = [{
    This pass performs various types of canonicalizations over a set of
    operations by iteratively applying the canonicalization patterns of all
    loaded dialects until either a fixpoint is reached or the maximum number of
    iterations/rewrites is exhausted. Canonicalization is best-effort and does
    not guarantee that the entire IR is in a canonical form after running this
    pass. See [Operation Canonicalization](Canonicalization.md) for more
    details.
  }];
  let constructor = "mlir::createCanonicalizerPass()";
  let options = [
    Option<"topDownProcessingEnabled", "top-down", "bool",
           /*default=*/"true",
           "Seed the worklist in general top-down order">,
    Option<"enableRegionSimplification", "region-simplify", "mlir::GreedySimplifyRegionLevel",
           /*default=*/"mlir::GreedySimplifyRegionLevel::Normal",
           "Perform control flow optimizations to the region tree",
             [{::llvm::cl::values(
               clEnumValN(mlir::GreedySimplifyRegionLevel::Disabled, "disabled",
                "Don't run any control-flow simplification."),
               clEnumValN(mlir::GreedySimplifyRegionLevel::Normal, "normal",
                "Perform simple control-flow simplifications (e.g. dead args elimination)."),
               clEnumValN(mlir::GreedySimplifyRegionLevel::Aggressive, "aggressive",
                "Perform aggressive control-flow simplification (e.g. block merging).")
              )}]>,
    Option<"maxIterations", "max-iterations", "int64_t",
           /*default=*/"10",
           "Max. iterations between applying patterns / simplifying regions">,
    Option<"maxNumRewrites", "max-num-rewrites", "int64_t", /*default=*/"-1",
           "Max. number of pattern rewrites within an iteration">,
    Option<"testConvergence", "test-convergence", "bool", /*default=*/"false",
           "Test only: Fail pass on non-convergence to detect cyclic pattern">
  ] # RewritePassUtils.options;
}
```


```cpp
/// Creates an instance of the Canonicalizer pass with the specified config.
std::unique_ptr<Pass>
mlir::createCanonicalizerPass(const GreedyRewriteConfig &config,
                              ArrayRef<std::string> disabledPatterns,
                              ArrayRef<std::string> enabledPatterns) {
  return std::make_unique<Canonicalizer>(config, disabledPatterns,
                                         enabledPatterns);
}

/// Create a Canonicalizer pass.
std::unique_ptr<Pass> mlir::createCanonicalizerPass() {
  return std::make_unique<Canonicalizer>();
}
```

```cpp
/// Canonicalize operations in nested regions.
struct Canonicalizer : public impl::CanonicalizerBase<Canonicalizer> {
  Canonicalizer() = default;
  Canonicalizer(const GreedyRewriteConfig &config,
                ArrayRef<std::string> disabledPatterns,
                ArrayRef<std::string> enabledPatterns)
      : config(config) {
    this->topDownProcessingEnabled = config.useTopDownTraversal;
    this->enableRegionSimplification = config.enableRegionSimplification;
    this->maxIterations = config.maxIterations;
    this->maxNumRewrites = config.maxNumRewrites;
    this->disabledPatterns = disabledPatterns;
    this->enabledPatterns = enabledPatterns;
  }

  /// Initialize the canonicalizer by building the set of patterns used during
  /// execution.
  LogicalResult initialize(MLIRContext *context) override {
    // Set the config from possible pass options set in the meantime.
    config.useTopDownTraversal = topDownProcessingEnabled;
    config.enableRegionSimplification = enableRegionSimplification;
    config.maxIterations = maxIterations;
    config.maxNumRewrites = maxNumRewrites;

    // 把 context 中注册的 dialect 和 op 的 CanonicalizationPatterns 都注册到 PatternSet 中
    RewritePatternSet owningPatterns(context);
    for (auto *dialect : context->getLoadedDialects())
      dialect->getCanonicalizationPatterns(owningPatterns);
    for (RegisteredOperationName op : context->getRegisteredOperations())
      op.getCanonicalizationPatterns(owningPatterns, context);

    patterns = std::make_shared<FrozenRewritePatternSet>(
        std::move(owningPatterns), disabledPatterns, enabledPatterns);
    return success();
  }
  void runOnOperation() override {
    LogicalResult converged =
        applyPatternsAndFoldGreedily(getOperation(), *patterns, config);
    // Canonicalization is best-effort. Non-convergence is not a pass failure.
    if (testConvergence && failed(converged))
      signalPassFailure();
  }
  GreedyRewriteConfig config;
  std::shared_ptr<const FrozenRewritePatternSet> patterns;
};
```

各个 op 都有一个或多个 pattern 实现 canonicalization

canonicalize Pass 只是把当前 context 下注册的 op 的 canonicalization pattern 都 apply 一下



MLIR has a single canonicalization pass, which iteratively applies the canonicalization patterns of all loaded dialects in a greedy way.

Canonicalization is best-effort and not guaranteed to bring the entire IR in a canonical form.

It applies patterns until either fixpoint is reached or the maximum number of iterations/rewrites (as specified via pass options) is exhausted.

This is for efficiency reasons and to ensure that faulty patterns cannot cause infinite looping.

Canonicalization patterns are registered with the operations themselves, which allows each dialect to define its own set of operations and canonicalizations together.

Some important things to think about w.r.t. canonicalization patterns:
- The goal of canonicalization is to make subsequent analyses and optimizations more effective. Therefore, performance improvements are not necessary for canonicalization.
- Pass pipelines should not rely on the canonicalizer pass for correctness. They should work correctly with all instances of the canonicalization pass removed.
  - 实际上很多 pipeline，后面的 pass 都依赖前面的 canonicalization
- Repeated applications of patterns should converge. Unstable or cyclic rewrites are considered a bug: they can make the canonicalizer pass less predictable and less effective (i.e., some patterns may not be applied) and prevent it from converging.
- It is generally better to canonicalize towards operations that have fewer uses of a value when the operands are duplicated, because some patterns only match when a value has a single user. For example, it is generally good to canonicalize “x + x” into “x * 2”, because this reduces the number of uses of x by one.
- It is always good to eliminate operations entirely when possible, e.g. by folding known identities (like “x + 0 = x”).
- Pattens with expensive running time (i.e. have O(n) complexity) or complicated cost models don't belong to canonicalization: since the algorithm is executed iteratively until fixed-point we want patterns that execute quickly (in particular their matching phase).
- Canonicalize shouldn't lose the semantic of original operation: the original information should always be recoverable from the transformed IR.


canonicalization 的目标是让后续的分析和优化更高效，因此性能提升不是 canonicalization 的目标

pattern 的重复应用应该收敛

通常，当操作数重复时，最好将其规范化为使用较少的值的操作

可能的话，完全消除操作总是好的

具有昂贵的运行时间 (即具有 O(n) 复杂度) 的模式或复杂的成本模型不属于规范化

Canonicalize 不应该失去原始操作的语义：原始信息应该总是可以从转换后的IR中恢复的


For example, a pattern that transform

```mlir
%transpose = linalg.transpose
    ins(%input : tensor<1x2x3xf32>)
    outs(%init1 : tensor<2x1x3xf32>)
    dimensions = [1, 0, 2]
%out = linalg.transpose
    ins(%tranpose: tensor<2x1x3xf32>)
    outs(%init2 : tensor<3x1x2xf32>)
    permutation = [2, 1, 0]
```

```mlir
%out= linalg.transpose
    ins(%input : tensor<1x2x3xf32>)
    outs(%init2: tensor<3x1x2xf32>)
    permutation = [2, 0, 1]
```

is a good canonicalization pattern because it removes a redundant operation, making other analysis optimizations and more efficient.

## Globally Applied Rules

These transformations are applied to all levels of IR:
- Elimination of operations that have no side effects and have no uses.
- Constant folding - e.g. “(addi 1, 2)” to “3”. Constant folding hooks are specified by operations.
- Move constant operands to commutative operators to the right side - e.g. “(addi 4, x)” to “(addi x, 4)”.
- constant-like operations are uniqued and hoisted into the entry block of the first parent barrier region. This is a region that is either isolated from above, e.g. the entry block of a function, or one marked as a barrier via the shouldMaterializeInto method on the DialectFoldInterface.

## Defining Canonicalizations

Two mechanisms are available with which to define canonicalizations; general RewritePatterns and the fold method.

This mechanism allows for providing canonicalizations as a set of RewritePatterns, either imperatively defined in C++ or declaratively as Declarative Rewrite Rules.

The pattern rewrite infrastructure allows for expressing many different types of canonicalizations.

These transformations may be as simple as replacing a multiplication with a shift, or even replacing a conditional branch with an unconditional one.

In ODS, an operation can set the `hasCanonicalizer` bit or the `hasCanonicalizeMethod` bit to generate a declaration for the `getCanonicalizationPatterns` method:

```tablegen
def MyOp : ... {
  // I want to define a fully general set of patterns for this op.
  let hasCanonicalizer = 1;
}

def OtherOp : ... {
  // A single "matchAndRewrite" style RewritePattern implemented as a method
  // is good enough for me.
  let hasCanonicalizeMethod = 1;
}
```

Canonicalization patterns can then be provided in the source file:

```cpp
void MyOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                       MLIRContext *context) {
  patterns.add<...>(...);
}

LogicalResult OtherOp::canonicalize(OtherOp op, PatternRewriter &rewriter) {
  // patterns and rewrites go here.
  return failure();
}
```