# Ignore files

The .cursorignore file lets you exclude files and directories from Cursor’s codebase indexing

## About .cursorignore

To ignore files from being included in codebase indexing, you can use a `.cursorignore` file in the root of your project. It works the same way as .gitignore works for git.

.cursorignore respects .gitignore. If you already have .gitignore, the files will be ignored by default. If you want to ignore additional files, you can add them to the .cursorignore file.

You can read more about how this works on our [security page](https://www.cursor.com/security).

## Chat and Composer Context

Currently, Cursor Chat and Composer have access to all files in their context, regardless of .cursorignore settings.

More info on how we handle AI request can be found on our security page.

## Example .cursorignore files

### Ignore specific files

```
# Ignore all files in the `dist` directory
dist/

# Ignore all `.log` files
*.log

# Ignore specific file `config.json`
config.json
```

### 