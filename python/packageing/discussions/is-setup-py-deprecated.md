# Is setup.py deprecated?

No, setup.py and Setuptools are not deprecated.

Setuptools is perfectly usable as a build backend for packaging Python projects. And setup.py is a valid configuration file for Setuptools that happens to be written in Python, instead of in TOML for example (a similar practice is used by other tools like nox and its noxfile.py configuration file, or pytest and conftest.py).

