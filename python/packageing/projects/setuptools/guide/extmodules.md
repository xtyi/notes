# Building Extension Modules

Setuptools can build C/C++ extension modules. The keyword argument ext_modules of setup() should be a list of instances of the `setuptools.Extension` class.

For example, let’s consider a simple project with only one extension module:

```
<project_folder>
├── pyproject.toml
└── foo.c
```

and all project metadata configuration in the pyproject.toml file:

```toml
# pyproject.toml
[build-system]
requires = ["setuptools"]
build-backend = "setuptools.build_meta"

[project]
name = "mylib-foo"  # as it would appear on PyPI
version = "0.42"
```

To instruct setuptools to compile the `foo.c` file into the extension module `mylib.foo`, we need to define an appropriate configuration in either pyproject.toml [1] or setup.py file , similar to the following:

```py
from setuptools import Extension, setup

setup(
    ext_modules=[
        Extension(
            name="mylib.foo",
            sources=["foo.c"],
        ),
    ]
)
```

