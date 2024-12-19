# SpecificBumpPtrAllocator

SpecificBumpPtrAllocator 是 LLVM 提供的一个特殊的内存分配器，它是一个 bump pointer allocator（也叫线性分配器）的特化版本。其主要特点是：

1. 快速分配：分配内存时只需要移动指针，不需要复杂的内存管理
2. 批量释放：所有分配的内存会在分配器析构时一起释放
3. 固定大小：专门用于分配特定类型（模板参数指定）的对象

