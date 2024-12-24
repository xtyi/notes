# link_directories

推荐使用 `target_link_directories` 替代

Add directories in which the linker will look for libraries.

```
link_directories([AFTER|BEFORE] directory1 [directory2 ...])
```

Adds the paths in which the linker should search for libraries. Relative paths given to this command are interpreted as relative to the current source directory, see CMP0015.

The command will apply only to targets created after it is called.

The directories are added to the LINK_DIRECTORIES directory property for the current CMakeLists.txt file, converting relative paths to absolute as needed.

This command is rarely necessary and should be avoided where there are other choices. Prefer to pass full absolute paths to libraries where possible, since this ensures the correct library will always be linked. The find_library() command provides the full path, which can generally be used directly in calls to target_link_libraries(). 

