# Writing your pyproject.toml

pyproject.toml is a configuration file used by packaging tools, as well as other tools such as linters, type checkers, etc.

There are three possible TOML tables in this file.

- The [build-system] table is strongly recommended. It allows you to declare which build backend you use and which other dependencies are needed to build your project.
- The [project] table is the format that most build backends use to specify your project’s basic metadata, such as the dependencies, your name, etc.
- The [tool] table has tool-specific subtables, e.g., [tool.hatch], [tool.black], [tool.mypy]. We only touch upon this table here because its contents are defined by each tool. Consult the particular tool’s documentation to know what it can contain.

> The [build-system] table should always be present, regardless of which build backend you use ([build-system] defines the build tool you use).
> On the other hand, the [project] table is understood by most build backends, but some build backends use a different format.
> As of August 2024, Poetry is a notable build backend that does not use the [project] table, it uses the [tool.poetry] table instead. Also, the setuptools build backend supports both the [project] table, and the older format in setup.cfg or setup.py.
> For new projects, use the [project] table, and keep setup.py only if some programmatic configuration is needed (such as building C extensions), but the setup.cfg and setup.py formats are still valid. See Is setup.py deprecated?.



```py
[build-system]
requires = ["setuptools >= 61.0"]
build-backend = "setuptools.build_meta"
```

