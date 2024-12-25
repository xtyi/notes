# FindPython

Interpreter: search for Python interpreter.
Compiler: search for Python compiler. Only offered by IronPython.
Development: search for development artifacts (include directories and libraries).

> If no COMPONENTS is specified, Interpreter is assumed.

To ensure consistent versions between components Interpreter, Compiler and Development, specify all components at the same time:

```
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
```

这里一定要这么写, 这样可以自动找当前环境下的 Python

## Artifacts Specification

To solve special cases, it is possible to specify directly the artifacts by setting the following variables:


Python_EXECUTABLE
The path to the interpreter.

