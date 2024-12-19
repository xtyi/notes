# SmartScopedReader

llvm::sys::SmartScopedReader 是 LLVM 中的一个读写锁(RWMutex)的读锁包装器, 实现了 RAII (Resource Acquisition Is Initialization) 模式，用于自动管理读锁的获取和释放

SmartScopedReader 的构造函数会自动获取读锁，析构函数会自动释放读锁

模板参数 `<true>` 表示这是一个可递归的锁，意味着同一个线程可以多次获取这个锁而不会死锁。这在递归调用或复杂的调用链中很有用。
