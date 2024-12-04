# Quickstart

## Installation

You can install the latest version of setuptools using pip:

```
pip install --upgrade setuptools[core]
```

Most of the times, however, you don’t have to…

Instead, when creating new Python packages, it is recommended to use a command line tool called `build`.

This tool will automatically download `setuptools` and any other build-time dependencies that your project might have.

You just need to specify them in a `pyproject.toml` file at the root of your package, as indicated in the following section.


You can also install build using pip:

```
pip install --upgrade build
```

This will allow you to run the command: `python -m build.`

Every python package must provide a pyproject.toml and specify the backend (build system) it wants to use.

The distribution can then be generated with whatever tool that provides a build sdist-like functionality.

## Basic Use

When creating a Python package, you must provide a pyproject.toml file containing a build-system section similar to the example below:

```toml
[build-system]
requires = ["setuptools"]
build-backend = "setuptools.build_meta"
```

This section declares what are your build system dependencies, and which library will be used to actually do the packaging.

In addition to specifying a build system, you also will need to add some package information such as metadata, contents, dependencies, etc.

This can be done in the same pyproject.toml file, or in a separated one: setup.cfg or setup.py [1].

The following example demonstrates a minimum configuration (which assumes the project depends on requests and importlib-metadata to be able to run):

```toml
[project]
name = "mypackage"
version = "0.0.1"
dependencies = [
    "requests",
    'importlib-metadata; python_version<"3.10"',
]
```

```cfg
[metadata]
name = mypackage
version = 0.0.1

[options]
install_requires =
    requests
    importlib-metadata; python_version<"3.10"
```

```py
from setuptools import setup

setup(
    name='mypackage',
    version='0.0.1',
    install_requires=[
        'requests',
        'importlib-metadata; python_version<"3.10"',
    ],
)
```

Finally, you will need to organize your Python code to make it ready for distributing into something that looks like the following (optional files marked with #):

```
mypackage
├── pyproject.toml  # and/or setup.cfg/setup.py (depending on the configuration method)
|   # README.rst or README.md (a nice description of your package)
|   # LICENCE (properly chosen license information, e.g. MIT, BSD-3, GPL-3, MPL-2, etc...)
└── mypackage
    ├── __init__.py
    └── ... (other Python files)
```

With build installed in your system, you can then run:

```
python -m build
```

You now have your distribution ready (e.g. a tar.gz file and a .whl file in the dist directory), which you can upload to PyPI!

Of course, before you release your project to PyPI, you’ll want to add a bit more information to help people find or learn about your project. And maybe your project will have grown by then to include a few dependencies, and perhaps some data files and scripts. In the next few sections, we will walk through the additional but essential information you need to specify to properly package your project.

## Overview

### Package Discovery

For projects that follow a simple directory structure, setuptools should be able to automatically detect all packages and namespaces.

However, complex projects might include additional folders and supporting files that not necessarily should be distributed (or that can confuse setuptools auto discovery algorithm).

Therefore, setuptools provides a convenient way to customize which packages should be distributed and in which directory they should be found, as shown in the example below:

```py
from setuptools import setup, find_packages  # or find_namespace_packages

setup(
    # ...
    packages=find_packages(
        # All keyword arguments below are optional:
        where='src',  # '.' by default
        include=['mypackage*'],  # ['*'] by default
        exclude=['mypackage.tests'],  # empty by default
    ),
    # ...
)
```

When you pass the above information, alongside other necessary information, setuptools walks through the directory specified in where (defaults to .) and filters the packages it can find following the include patterns (defaults to *), then it removes those that match the exclude patterns (defaults to empty) and returns a list of Python packages.

