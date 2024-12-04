# Quickstart

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

