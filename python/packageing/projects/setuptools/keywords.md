# Keywords

The following are keywords setuptools.setup() accepts.

They allow configuring the build process for a Python distribution or adding metadata via a `setup.py` script placed at the root of your project.

All of them are optional; you do not have to supply them unless you need the associated setuptools feature.


Metadata and configuration supplied via setup() is complementary to (and may be overwritten by) the information present in setup.cfg and pyproject.toml.

Some important metadata, such as name and version, may assume a default degenerate value if not specified.

Users are strongly encouraged to use a declarative config either via setup.cfg or pyproject.toml and only rely on setup.py if they need to tap into special behaviour that requires scripting (such as building C extensions).

建议使用 setup.cfg 或 pyproject.toml 声明式指定配置

只有必须依赖 Python 执行能力的时候，才使用 setup.py

setup.py 是以上两者的补充，如果重复，setup.py 会覆盖以上两者

name
A string specifying the name of the package.

version
A string specifying the version number of the package.

description
A string describing the package in a single line.

long_description
A string providing a longer description of the package.

long_description_content_type
A string specifying the content type is used for the long_description (e.g. text/markdown)

author
A string specifying the author of the package.

author_email
A string specifying the email address of the package author.

maintainer
A string specifying the name of the current maintainer, if different from the author. Note that if the maintainer is provided, setuptools will use it as the author in PKG-INFO.

maintainer_email
A string specifying the email address of the current maintainer, if different from the author.

url
A string specifying the URL for the package homepage.

download_url
A string specifying the URL to download the package.

license
A string specifying the license of the package.

license_files
A list of glob patterns for license related files that should be included. If neither license_file nor license_files is specified, this option defaults to LICEN[CS]E*, COPYING*, NOTICE*, and AUTHORS*.

keywords
A list of strings or a comma-separated string providing descriptive meta-data. See: Core Metadata Specifications.

platforms
A list of strings or comma-separated string.


关键

packages
A list of strings specifying the packages that setuptools will manipulate.

ext_modules
A list of instances of setuptools.Extension providing the list of Python extensions to be built.

cmdclass
A dictionary providing a mapping of command names to Command subclasses.