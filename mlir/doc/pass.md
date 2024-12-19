# Pass Infrastructure

## Pass Manager

The above sections introduced the different types of passes and their invariants. This section introduces the concept of a PassManager, and how it can be used to configure and schedule a pass pipeline. There are two main classes related to pass management, the PassManager and the OpPassManager. The PassManager class acts as the top-level entry point, and contains various configurations used for the entire pass pipeline. The OpPassManager class is used to schedule passes to run at a specific level of nesting. The top-level PassManager also functions as an OpPassManager.

## OpPassManager

An OpPassManager is essentially a collection of passes anchored to execute on operations at a given level of nesting. A pass manager may be op-specific (anchored on a specific operation type), or op-agnostic (not restricted to any specific operation, and executed on any viable operation type). Operation types that anchor pass managers must adhere to the following requirement:

TODO

## Dynamic Pass Pipelines

In some situations it may be useful to run a pass pipeline within another pass, to allow configuring or filtering based on some invariants of the current operation being operated on. For example, the Inliner Pass may want to run intraprocedural simplification passes while it is inlining to produce a better cost model, and provide more optimal inlining. To enable this, passes may run an arbitrary OpPassManager on the current operation being operated on or any operation nested within the current operation via the LogicalResult Pass::runPipeline(OpPassManager &, Operation *) method. This method returns whether the dynamic pipeline succeeded or failed, similarly to the result of the top-level PassManager::run method. A simple example is shown below:

```cpp
void MyModulePass::runOnOperation() {
  ModuleOp module = getOperation();
  if (hasSomeSpecificProperty(module)) {
    OpPassManager dynamicPM("builtin.module");
    ...; // Build the dynamic pipeline.
    if (failed(runPipeline(dynamicPM, module)))
      return signalPassFailure();
  }
}
```

TODO



## Pass Registration

Briefly shown in the example definitions of the various pass types is the PassRegistration class. This mechanism allows for registering pass classes so that they may be created within a textual pass pipeline description. An example registration is shown below:

```cpp
void registerMyPass() {
  PassRegistration<MyPass>();
}
```

MyPass is the name of the derived pass class.

The pass getArgument() method is used to get the identifier that will be used to refer to the pass.

The pass getDescription() method provides a short summary describing the pass.

For passes that cannot be default-constructed, `PassRegistration` accepts an optional argument that takes a callback to create the pass:


```cpp
void registerMyPass() {
  PassRegistration<MyParametricPass>(
    []() -> std::unique_ptr<Pass> {
      std::unique_ptr<Pass> p = std::make_unique<MyParametricPass>(/*options*/);
      /*... non-trivial-logic to configure the pass ...*/;
      return p;
    });
}
```

This variant of registration can be used, for example, to accept the configuration of a pass from command-line arguments and pass it to the pass constructor.

> Note: Make sure that the pass is copy-constructible in a way that does not share data as the pass manager may create copies of the pass to run in parallel.


## Pass Pipeline Registration

Described above is the mechanism used for registering a specific derived pass class. On top of that, MLIR allows for registering custom pass pipelines in a similar fashion. This allows for custom pipelines to be available to tools like mlir-opt in the same way that passes are, which is useful for encapsulating common pipelines like the “-O1” series of passes. Pipelines are registered via a similar mechanism to passes in the form of PassPipelineRegistration. Compared to PassRegistration, this class takes an additional parameter in the form of a pipeline builder that modifies a provided OpPassManager.

```cpp
void pipelineBuilder(OpPassManager &pm) {
  pm.addPass(std::make_unique<MyPass>());
  pm.addPass(std::make_unique<MyOtherPass>());
}

void registerMyPasses() {
  // Register an existing pipeline builder function.
  PassPipelineRegistration<>(
    "argument", "description", pipelineBuilder);

  // Register an inline pipeline builder.
  PassPipelineRegistration<>(
    "argument", "description", [](OpPassManager &pm) {
      pm.addPass(std::make_unique<MyPass>());
      pm.addPass(std::make_unique<MyOtherPass>());
    });
}
```

## Textual Pass Pipeline Specification

The previous sections detailed how to register passes and pass pipelines with a specific argument and description. Once registered, these can be used to configure a pass manager from a string description. This is especially useful for tools like mlir-opt, that configure pass managers from the command line, or as options to passes that utilize dynamic pass pipelines.



## Declarative Pass Specification

Some aspects of a Pass may be specified declaratively, in a form similar to operations. This specification simplifies several mechanisms used when defining passes. It can be used for generating pass registration calls, defining boilerplate pass utilities, and generating pass documentation.

Consider the following pass specified in C++:

```td
def MyPass : Pass<"my-pass", "ModuleOp"> {
  let summary = "My Pass Summary";
  let description = [{
    Here we can now give a much larger description of `MyPass`, including all of
    its various constraints and behavior.
  }];

  // A constructor must be provided to specify how to create a default instance
  // of MyPass. It can be skipped for this specific example, because both the
  // constructor and the registration methods live in the same namespace.
  let constructor = "foo::createMyPass()";

  // Specify any options.
  let options = [
    Option<"option", "example-option", "bool", /*default=*/"true",
           "An example option">,
    ListOption<"listOption", "example-list", "int64_t",
               "An example list option">
  ];

  // Specify any statistics.
  let statistics = [
    Statistic<"statistic", "example-statistic", "An example statistic">
  ];
}
```

Using the gen-pass-decls generator, we can generate most of the boilerplate above automatically. This generator takes as an input a -name parameter, that provides a tag for the group of passes that are being generated. This generator produces code with multiple purposes:

The first is to register the declared passes with the global registry. For each pass, the generator produces a registerPassName where PassName is the name of the definition specified in tablegen. It also generates a registerGroupPasses, where Group is the tag provided via the -name input parameter, that registers all of the passes present.

```
// Tablegen options: -gen-pass-decls -name="Example"

// Passes.h

namespace foo {
#define GEN_PASS_REGISTRATION
#include "Passes.h.inc"
} // namespace foo

void registerMyPasses() {
  // Register all of the passes.
  foo::registerExamplePasses();
  
  // Or

  // Register `MyPass` specifically.
  foo::registerMyPass();
}
```

## Tablegen Specification

The Pass class is used to begin a new pass definition. This class takes as an argument the registry argument to attribute to the pass, as well as an optional string corresponding to the operation type that the pass operates on. The class contains the following fields:


- summary

A short one-line summary of the pass, used as the description when registering the pass.

- description

A longer, more detailed description of the pass. This is used when generating pass documentation.

- dependentDialects

A list of strings representing the Dialect classes this pass may introduce entities, Attributes/Operations/Types/etc., of.

- constructor(*)

A code block used to create a default instance of the pass.

- options(*)

A list of pass options used by the pass.

- statistics(*)

A list of pass statistics used by the pass.


