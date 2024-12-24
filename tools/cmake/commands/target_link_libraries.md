# target_link_libraries


```
target_link_libraries(<target> ... <item>... ...)
```

Each `<item>` may be:

A library target name: The generated link line will have the full path to the linkable library file associated with the target. The buildsystem will have a dependency to re-link `<target>` if the library file changes.

The named target must be created by `add_library()` within the project or as an IMPORTED library. If it is created within the project an ordering dependency will automatically be added in the build system to make sure the named library target is up-to-date before the `<target>` links.


A full path to a library file: The generated link line will normally preserve the full path to the file. The buildsystem will have a dependency to re-link <target> if the library file changes.