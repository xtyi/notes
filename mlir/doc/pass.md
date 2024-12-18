# Pass Infrastructure

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

## Declarative Pass Specification
