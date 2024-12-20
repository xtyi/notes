# TypeID

TypeID 是 MLIR 中用于实现类型识别和运行时类型信息(RTTI)的一个重要机制，它提供了一种高效的方式来为 C++ 类型生成唯一标识符

主要作用

类型唯一标识

1. 为每个 C++ 类型提供一个唯一的运行时标识符

2. 支持类型比较、哈希和在不透明上下文中存储

替代标准 RTTI

1. 提供了一种比 C++ 标准 RTTI (如 typeid、dynamic_cast) 更高效的替代方案

2. 避免了共享库环境下 RTTI 可能带来的问题

支持类型层次结构

1. 可用于实现 LLVM 风格的 isa/dyn_cast 功能

2. 便于进行类型检查和转换



优势
性能优化
比基于字符串比较的类型识别更高效
避免了运行时字符串比较的开销
安全性
提供编译时类型安全
在共享库环境中更可靠
3. 灵活性
支持多种注册方式
可用于任何 C++ 类型


## TypeID class


```cpp
/// This class represents the storage of a type info object.
/// Note: We specify an explicit alignment here to allow use with
/// PointerIntPair and other utilities/data structures that require a known
/// pointer alignment.
struct alignas(8) Storage {};
```

```cpp
private:
/// The storage of this type info object.
const Storage *storage;
```

```cpp
private:
  TypeID(const Storage *storage) : storage(storage) {}
```


```cpp
public:
TypeID() : TypeID(get<void>()) {}
```

默认构造函数, 这里会委托编译器自动生成的拷贝构造函数, 传入一个 void 的 TypeID 对象


```cpp
public:

/// Comparison operations.
inline bool operator==(const TypeID &other) const {
    return storage == other.storage;
}
inline bool operator!=(const TypeID &other) const {
    return !(*this == other);
}
```

实现比较操作

```cpp
public:

/// Construct a type info object for the given type T.
template <typename T>
static TypeID get();

template <template <typename> class Trait>
static TypeID get();
```

实现

```cpp
template <typename T>
TypeID TypeID::get() {
  return detail::TypeIDResolver<T>::resolveTypeID();
}

template <template <typename> class Trait>
TypeID TypeID::get() {
  // An empty class used to simplify the use of Trait types.
  struct Empty {};
  return TypeID::get<Trait<Empty>>();
}
```


```cpp
/// Methods for supporting PointerLikeTypeTraits.
const void *getAsOpaquePointer() const {
  return static_cast<const void *>(storage);
}
static TypeID getFromOpaquePointer(const void *pointer) {
  return TypeID(reinterpret_cast<const Storage *>(pointer));
}
```

```cpp
/// Enable hashing TypeID.
friend ::llvm::hash_code hash_value(TypeID id);

/// Enable hashing TypeID.
inline ::llvm::hash_code hash_value(TypeID id) {
  return DenseMapInfo<const TypeID::Storage *>::getHashValue(id.storage);
}
```





## TypeIDAllocator

```cpp
/// This class provides a way to define new TypeIDs at runtime.
/// When the allocator is destructed, all allocated TypeIDs become invalid and
/// therefore should not be used.
class TypeIDAllocator {
public:
  /// Allocate a new TypeID, that is ensured to be unique for the lifetime
  /// of the TypeIDAllocator.
  TypeID allocate() { return TypeID(ids.Allocate()); }

private:
  /// The TypeIDs allocated are the addresses of the different storages.
  /// Keeping those in memory ensure uniqueness of the TypeIDs.
  llvm::SpecificBumpPtrAllocator<TypeID::Storage> ids;
};
```

allocate 的实现如下

`ids.Allocate()` 会分配一个 `TypeID::Storage` 对象, 返回指向其的指针

使用 `Storage*` 构造一个 TypeID 对象


## FallbackTypeIDResolver

```cpp
/// This class provides a fallback for resolving TypeIDs. It uses the string
/// name of the type to perform the resolution, and as such does not allow the
/// use of classes defined in "anonymous" contexts.
class FallbackTypeIDResolver {
protected:
  /// Register an implicit type ID for the given type name.
  static TypeID registerImplicitTypeID(StringRef name);
};
```

实现如下

```cpp
TypeID detail::FallbackTypeIDResolver::registerImplicitTypeID(StringRef name) {
  static ImplicitTypeIDRegistry registry;
  return registry.lookupOrInsert(name);
}
```

这里创建一个静态的 ImplicitTypeIDRegistry 实例, 直接调用其 lookupOrInsert 方法, 把实现细节隐藏在一个内部类中

```cpp
struct ImplicitTypeIDRegistry {
  /// Lookup or insert a TypeID for the given type name.
  TypeID lookupOrInsert(StringRef typeName) {

    // Perform a heuristic check to see if this type is in an anonymous
    // namespace. String equality is not valid for anonymous types, so we try to
    // abort whenever we see them.
#ifndef NDEBUG
#if defined(_MSC_VER)
    // 在 MSVC 编译器中，匿名命名空间在类型名中显示为 "anonymous-namespace"
    if (typeName.contains("anonymous-namespace")) {
#else
    // 在其他编译器（如 GCC、Clang）中，匿名命名空间显示为 "anonymous namespace"
    if (typeName.contains("anonymous namespace")) {
#endif
    // 首先检查类型是否在匿名空间下, 由于匿名空间下的类型无法使用字符串比较是否相同, 这里直接报错
      std::string errorStr;
      {
        llvm::raw_string_ostream errorOS(errorStr);
        errorOS << "TypeID::get<" << typeName
                << ">(): Using TypeID on a class with an anonymous "
                   "namespace requires an explicit TypeID definition. The "
                   "implicit fallback uses string name, which does not "
                   "guarantee uniqueness in anonymous contexts. Define an "
                   "explicit TypeID instantiation for this type using "
                   "`MLIR_DECLARE_EXPLICIT_TYPE_ID`/"
                   "`MLIR_DEFINE_EXPLICIT_TYPE_ID` or "
                   "`MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID`.\n";
      }
      llvm::report_fatal_error(errorStr);
    }
#endif

    // 先尝试找一下是否已经存在
    { // Try a read-only lookup first.
      llvm::sys::SmartScopedReader<true> guard(mutex); // 获取读锁
      auto it = typeNameToID.find(typeName);
      if (it != typeNameToID.end())
        return it->second;
    } // 释放读锁

    // 其他线程可能在此时插入

    llvm::sys::SmartScopedWriter<true> guard(mutex); // 获取写锁
    // 上面只是尝试一下, 由于支持多线程, 即使上面没有找到, 这里还是有可能已经存在其他线程已经插入的 TypeID
    // 所以还是要使用 try_emplace, 而且这里使用一个空的 TypeID() 实例, 该实例没有任何作用
    // 这里不能直接使用 typeIDAllocator.allocate(), 因为如果没有成功插入, 就会出现 Storage 已经出现但是没有 TypeID 指向
    auto it = typeNameToID.try_emplace(typeName, TypeID());
    if (it.second)
      it.first->second = typeIDAllocator.allocate();
    return it.first->second;
  }

  /// A mutex that guards access to the registry.
  llvm::sys::SmartRWMutex<true> mutex;

  /// An allocator used for TypeID objects.
  TypeIDAllocator typeIDAllocator;

  /// A map type name to TypeID.
  DenseMap<StringRef, TypeID> typeNameToID;
};
```

## TypeIDResolver

```cpp
/// This class provides a resolver for getting the ID for a given class T. This
/// allows for the derived type to specialize its resolution behavior. The
/// default implementation uses the string name of the type to resolve the ID.
/// This provides a strong definition, but at the cost of performance (we need
/// to do an initial lookup) and is not usable by classes defined in anonymous
/// contexts.
///
/// TODO: The use of the type name is only necessary when building in the
/// presence of shared libraries. We could add a build flag that guarantees
/// "static"-like environments and switch this to a more optimal implementation
/// when that is enabled.
template <typename T, typename Enable = void>
class TypeIDResolver : public FallbackTypeIDResolver {
public:
  /// Trait to check if `U` is fully resolved. We use this to verify that `T` is
  /// fully resolved when trying to resolve a TypeID. We don't technically need
  /// to have the full definition of `T` for the fallback, but it does help
  /// prevent situations where a forward declared type uses this fallback even
  /// though there is a strong definition for the TypeID in the location where
  /// `T` is defined.
  template <typename U>
  using is_fully_resolved_trait = decltype(sizeof(U));
  template <typename U>
  using is_fully_resolved = llvm::is_detected<is_fully_resolved_trait, U>;

  static TypeID resolveTypeID() {
    // 断言  T is_fully_resolved
    static_assert(is_fully_resolved<T>::value,
                  "TypeID::get<> requires the complete definition of `T`");
    // 获取 T 的类型名, 调用 FallbackTypeIDResolver 的 registerImplicitTypeID 方法
    static TypeID id = registerImplicitTypeID(llvm::getTypeName<T>());
    return id;
  }
};
```

