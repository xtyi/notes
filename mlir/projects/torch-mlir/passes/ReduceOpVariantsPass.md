# ReduceOpVariantsPass

Conversion Pass

```cpp
std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::Torch::createReduceOpVariantsPass(StringRef extraLibrary) {
  return std::make_unique<ReduceOpVariantsPass>(extraLibrary);
}
```

```
def ReduceOpVariants : Pass<"torch-reduce-op-variants", "func::FuncOp"> {
  let summary = "Reduces variants of ops to a smaller set of ops.";
  let constructor = [{
    mlir::torch::Torch::createReduceOpVariantsPass(/*extraLibrary=*/"")
  }];
  let options = [
    Option<"extraLibrary", "extra-library", "std::string", /*default=*/"",
           "MLIR module for verifying custom op value semantics">,
  ];
  let description = [{
    Replaces ops with other ops to reduce the number of variants that
    need to be handled elsewhere in the code.

    Examples of the transformations done in this pass are:
    - Convert operations with value semantics to operate on immutable tensors
    - Convert operations with in-place semantics (e.g. `add_`) or inherently
      mutable semantics (e.g. `add.out`) to their value-semantic equivalent.
    - Convert operations that involve a scalar promotion to the tensor
      variant plus a scalar promotion op.
  }];
}
```


```cpp
struct ReduceOpVariantsPass
    : public ReduceOpVariantsBase<ReduceOpVariantsPass> {
  
  ReduceOpVariantsPass() = default;

  ReduceOpVariantsPass(StringRef extraLibrary) {
    this->extraLibrary = extraLibrary.str();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    // 创建一个 moduleOp 和 symboltable
    OwningOpRef<ModuleOp> extraLibraryModule =
        ModuleOp::create(UnknownLoc::get(context));
    std::optional<SymbolTable> extraLibraryModuleSymTable = std::nullopt;
    
    if (!extraLibrary.empty()) {
      if (failed(loadExtraLibrary(extraLibrary, extraLibraryModule))) {
        emitError(getOperation()->getLoc(),
                  "Failed to load extra-library file at " + extraLibrary);
        return signalPassFailure();
      }

      extraLibraryModuleSymTable =
          SymbolTable(extraLibraryModule->getOperation());
    }
    
    // 添加以下 pattern
    patterns.add<ConvertHasValueSemanticsOpsToValueTensors>(
        context, extraLibraryModuleSymTable);
    patterns.add<ReduceTrailingUnderscoreInplaceVariant>(context);
    patterns.add(reduceNonValueTensorLiteralOpToValueTensorLiteralOp);
    patterns.add<ReduceNonValueSemanticOps>(context);

    // 该 pass 处理注册的 opname, 目前就注册了 torch.aten._scaled_dot_product_flash_attention_for_cpu
    // Create specialized matcher:
    auto specialized =
        TorchMatchSpecializedBackendOp::getPopulatedMatcher(context);
    DenseSet<StringAttr> specializedNames;
    specialized->populateLegalizedNames(specializedNames);

    patterns.insert(std::move(specialized));

    // 设置非法 op
    ConversionTarget target(*context);
    target.addIllegalOp<NonValueTensorLiteralOp>();
    target.addIllegalOp<AtenBernoulli_FloatOp>();
    target.addIllegalOp<AtenArangeStartOutOp>();
    
    // 如果是 unknown 的 op, 根据回调函数动态地设置是否合法
    target.markUnknownOpDynamicallyLegal([&extraLibraryModuleSymTable,
                                          &specializedNames](Operation *op) {
      // 如果是 torch.operator
      if (isa<OperatorOp>(op)) {
        // 获取 name, 如果 specializedNames 包含 op name, 则该 op 是非法的, 因为 specializedNames 中的 opname 会被 handlerFn 处理
        if (specializedNames.contains(cast<OperatorOp>(op).getNameAttr())) {
          return false;
        }
      }
      //
      if (op->hasTrait<Torch::OpTrait::HasValueSemantics>() ||
          (isa<OperatorOp>(op) &&
           operatorOpHasValueSemantics(cast<OperatorOp>(op),
                                       extraLibraryModuleSymTable))) {
        auto hasValueSemantics = [](Type t) {
          // TODO: Make this an allowlist based on a closed torch dialect
          // type system.
          if (auto tensorType = dyn_cast<NonValueTensorType>(t)) {
            return false;
          }
          return true;
        };
        return llvm::all_of(op->getOperandTypes(), hasValueSemantics) &&
               llvm::all_of(op->getResultTypes(), hasValueSemantics);
      }

      
      if (op->hasTrait<Torch::OpTrait::IsTrailingUnderscoreInplaceVariant>()) {
        return false;
      }

      if (isa<OperatorOp>(op) && isSpecializedOperation(cast<OperatorOp>(op))) {
        return false;
      }

      return true;
    });

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
```



## TorchMatchSpecializedBackendOp

```cpp

class TorchMatchSpecializedBackendOp
    : public OpConversionPattern<Torch::OperatorOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  // handlerFn 用于处理 operator op, 不同的 opname 会映射到不同的 handlerFn
  using HandlerFn = LogicalResult (*)(OperatorOp op,
                                      ConversionPatternRewriter &rewriter);

  LogicalResult
  matchAndRewrite(Torch::OperatorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // 该 pass 会看 namedHandlers 有没有注册当前 opname
    // 如果有的话, 调用 handlerFn
    if (namedHandlers.contains(op.getNameAttr())) {
      return namedHandlers.lookup(op.getNameAttr()).front()(op, rewriter);
    }

    return failure();
  }

  static void
  populateSpecializedConversions(TorchMatchSpecializedBackendOp &matcher);

  // 一个静态工厂方法, 创建 TorchMatchSpecializedBackendOp pattern, 调用 populateSpecializedConversions
  static std::unique_ptr<TorchMatchSpecializedBackendOp>
  getPopulatedMatcher(MLIRContext *context) {
    auto matcher = std::make_unique<TorchMatchSpecializedBackendOp>(context);
    populateSpecializedConversions(*matcher);
    return matcher;
  };

  // 注册到 namedHandlers
  void populate(StringRef name, HandlerFn fn) {
    namedHandlers[StringAttr::get(getContext(), name)].push_back(fn);
  }

    // 把 namedHandlers 中注册的 opname 都放到集合中返回
  void populateLegalizedNames(llvm::DenseSet<StringAttr> &set) {
    for (auto handle : namedHandlers) {
      set.insert(handle.first);
    }
  }

private:
  // opname 到 handlerFn 的映射
  DenseMap<StringAttr, SmallVector<HandlerFn, 1>> namedHandlers;
};
```

```cpp
void TorchMatchSpecializedBackendOp::populateSpecializedConversions(
    TorchMatchSpecializedBackendOp &matcher) {
  matcher.populate(
      "torch.aten._scaled_dot_product_flash_attention_for_cpu",
      [](Torch::OperatorOp op,
         ConversionPatternRewriter &rewriter) -> LogicalResult {
        auto uses = op.getResult(1).getUses();
        if (uses.end() == uses.begin()) {
          auto oldOperands = op->getOperands();
          llvm::SmallVector<Value> newOperands{
              oldOperands[0], oldOperands[1], oldOperands[2], oldOperands[5],
              oldOperands[3], oldOperands[4], oldOperands[6]};
          Value enableGQA =
              rewriter.create<ConstantBoolOp>(op->getLoc(), false);
          newOperands.push_back(enableGQA);

          auto newOp = rewriter.create<Torch::AtenScaledDotProductAttentionOp>(
              op.getLoc(), op->getResultTypes()[0], newOperands,
              op->getAttrs());
          rewriter.replaceAllUsesWith(op.getResult(0), newOp.getResult());
          rewriter.eraseOp(op);
          return success();
        }
        return failure();
      });
}
```

