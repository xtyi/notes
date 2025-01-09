# NamedAttribute

```cpp
/// NamedAttribute represents a combination of a name and an Attribute value.
class NamedAttribute {
public:
  NamedAttribute(StringAttr name, Attribute value);

  /// Return the name of the attribute.
  StringAttr getName() const;

  /// Return the dialect of the name of this attribute, if the name is prefixed
  /// by a dialect namespace. For example, `llvm.fast_math` would return the
  /// LLVM dialect (if it is loaded). Returns nullptr if the dialect isn't
  /// loaded, or if the name is not prefixed by a dialect namespace.
  Dialect *getNameDialect() const;

  /// Return the value of the attribute.
  Attribute getValue() const { return value; }

  /// Set the name of this attribute.
  void setName(StringAttr newName);

  /// Set the value of this attribute.
  void setValue(Attribute newValue) {
    assert(value && "expected valid attribute value");
    value = newValue;
  }

  /// Compare this attribute to the provided attribute, ordering by name.
  bool operator<(const NamedAttribute &rhs) const;
  /// Compare this attribute to the provided string, ordering by name.
  bool operator<(StringRef rhs) const;

  bool operator==(const NamedAttribute &rhs) const {
    return name == rhs.name && value == rhs.value;
  }
  bool operator!=(const NamedAttribute &rhs) const { return !(*this == rhs); }

private:
  NamedAttribute(Attribute name, Attribute value) : name(name), value(value) {}

  /// Allow access to internals to enable hashing.
  friend ::llvm::hash_code hash_value(const NamedAttribute &arg);
  friend DenseMapInfo<NamedAttribute>;

  /// The name of the attribute. This is represented as a StringAttr, but
  /// type-erased to Attribute in the field.
  Attribute name;
  /// The value of the attribute.
  Attribute value;
};
```