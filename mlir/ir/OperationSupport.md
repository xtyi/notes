

## OperationName

```cpp
class OperationName {
public:
  using FoldHookFn = llvm::unique_function<LogicalResult(
      Operation *, ArrayRef<Attribute>, SmallVectorImpl<OpFoldResult> &) const>;
  using HasTraitFn = llvm::unique_function<bool(TypeID) const>;
  using ParseAssemblyFn =
      llvm::unique_function<ParseResult(OpAsmParser &, OperationState &)>;
  // Note: RegisteredOperationName is passed as reference here as the derived
  // class is defined below.
  using PopulateDefaultAttrsFn =
      llvm::unique_function<void(const OperationName &, NamedAttrList &) const>;
  using PrintAssemblyFn =
      llvm::unique_function<void(Operation *, OpAsmPrinter &, StringRef) const>;
  using VerifyInvariantsFn =
      llvm::unique_function<LogicalResult(Operation *) const>;
  using VerifyRegionInvariantsFn =
      llvm::unique_function<LogicalResult(Operation *) const>;

  /// This class represents a type erased version of an operation. It contains
  /// all of the components necessary for opaquely interacting with an
  /// operation. If the operation is not registered, some of these components
  /// may not be populated.
  struct InterfaceConcept {
    virtual ~InterfaceConcept() = default;
    virtual LogicalResult foldHook(Operation *, ArrayRef<Attribute>,
                                   SmallVectorImpl<OpFoldResult> &) = 0;
    virtual void getCanonicalizationPatterns(RewritePatternSet &,
                                             MLIRContext *) = 0;
    virtual bool hasTrait(TypeID) = 0;
    virtual OperationName::ParseAssemblyFn getParseAssemblyFn() = 0;
    virtual void populateDefaultAttrs(const OperationName &,
                                      NamedAttrList &) = 0;
    virtual void printAssembly(Operation *, OpAsmPrinter &, StringRef) = 0;
    virtual LogicalResult verifyInvariants(Operation *) = 0;
    virtual LogicalResult verifyRegionInvariants(Operation *) = 0;
    /// Implementation for properties
    virtual std::optional<Attribute> getInherentAttr(Operation *,
                                                     StringRef name) = 0;
    virtual void setInherentAttr(Operation *op, StringAttr name,
                                 Attribute value) = 0;
    virtual void populateInherentAttrs(Operation *op, NamedAttrList &attrs) = 0;
    virtual LogicalResult
    verifyInherentAttrs(OperationName opName, NamedAttrList &attributes,
                        function_ref<InFlightDiagnostic()> emitError) = 0;
    virtual int getOpPropertyByteSize() = 0;
    virtual void initProperties(OperationName opName, OpaqueProperties storage,
                                OpaqueProperties init) = 0;
    virtual void deleteProperties(OpaqueProperties) = 0;
    virtual void populateDefaultProperties(OperationName opName,
                                           OpaqueProperties properties) = 0;
    virtual LogicalResult
    setPropertiesFromAttr(OperationName, OpaqueProperties, Attribute,
                          function_ref<InFlightDiagnostic()> emitError) = 0;
    virtual Attribute getPropertiesAsAttr(Operation *) = 0;
    virtual void copyProperties(OpaqueProperties, OpaqueProperties) = 0;
    virtual bool compareProperties(OpaqueProperties, OpaqueProperties) = 0;
    virtual llvm::hash_code hashProperties(OpaqueProperties) = 0;
  };

public:
  class Impl : public InterfaceConcept {
  public:
    Impl(StringRef, Dialect *dialect, TypeID typeID,
         detail::InterfaceMap interfaceMap);
    Impl(StringAttr name, Dialect *dialect, TypeID typeID,
         detail::InterfaceMap interfaceMap)
        : name(name), typeID(typeID), dialect(dialect),
          interfaceMap(std::move(interfaceMap)) {}

    /// Returns true if this is a registered operation.
    bool isRegistered() const { return typeID != TypeID::get<void>(); }
    detail::InterfaceMap &getInterfaceMap() { return interfaceMap; }
    Dialect *getDialect() const { return dialect; }
    StringAttr getName() const { return name; }
    TypeID getTypeID() const { return typeID; }
    ArrayRef<StringAttr> getAttributeNames() const { return attributeNames; }

  protected:
    //===------------------------------------------------------------------===//
    // Registered Operation Info

    /// The name of the operation.
    StringAttr name;

    /// The unique identifier of the derived Op class.
    TypeID typeID;

    /// The following fields are only populated when the operation is
    /// registered.

    /// This is the dialect that this operation belongs to.
    Dialect *dialect;

    /// A map of interfaces that were registered to this operation.
    detail::InterfaceMap interfaceMap;

    /// A list of attribute names registered to this operation in StringAttr
    /// form. This allows for operation classes to use StringAttr for attribute
    /// lookup/creation/etc., as opposed to raw strings.
    ArrayRef<StringAttr> attributeNames;

    friend class RegisteredOperationName;
  };

protected:
  /// Default implementation for unregistered operations.
  struct UnregisteredOpModel : public Impl {
    using Impl::Impl;
    LogicalResult foldHook(Operation *, ArrayRef<Attribute>,
                           SmallVectorImpl<OpFoldResult> &) final;
    void getCanonicalizationPatterns(RewritePatternSet &, MLIRContext *) final;
    bool hasTrait(TypeID) final;
    OperationName::ParseAssemblyFn getParseAssemblyFn() final;
    void populateDefaultAttrs(const OperationName &, NamedAttrList &) final;
    void printAssembly(Operation *, OpAsmPrinter &, StringRef) final;
    LogicalResult verifyInvariants(Operation *) final;
    LogicalResult verifyRegionInvariants(Operation *) final;
    /// Implementation for properties
    std::optional<Attribute> getInherentAttr(Operation *op,
                                             StringRef name) final;
    void setInherentAttr(Operation *op, StringAttr name, Attribute value) final;
    void populateInherentAttrs(Operation *op, NamedAttrList &attrs) final;
    LogicalResult
    verifyInherentAttrs(OperationName opName, NamedAttrList &attributes,
                        function_ref<InFlightDiagnostic()> emitError) final;
    int getOpPropertyByteSize() final;
    void initProperties(OperationName opName, OpaqueProperties storage,
                        OpaqueProperties init) final;
    void deleteProperties(OpaqueProperties) final;
    void populateDefaultProperties(OperationName opName,
                                   OpaqueProperties properties) final;
    LogicalResult
    setPropertiesFromAttr(OperationName, OpaqueProperties, Attribute,
                          function_ref<InFlightDiagnostic()> emitError) final;
    Attribute getPropertiesAsAttr(Operation *) final;
    void copyProperties(OpaqueProperties, OpaqueProperties) final;
    bool compareProperties(OpaqueProperties, OpaqueProperties) final;
    llvm::hash_code hashProperties(OpaqueProperties) final;
  };

public:
  OperationName(StringRef name, MLIRContext *context);

  /// Return if this operation is registered.
  bool isRegistered() const { return getImpl()->isRegistered(); }

  /// Return the unique identifier of the derived Op class, or null if not
  /// registered.
  TypeID getTypeID() const { return getImpl()->getTypeID(); }

  /// If this operation is registered, returns the registered information,
  /// std::nullopt otherwise.
  std::optional<RegisteredOperationName> getRegisteredInfo() const;

  /// This hook implements a generalized folder for this operation. Operations
  /// can implement this to provide simplifications rules that are applied by
  /// the Builder::createOrFold API and the canonicalization pass.
  ///
  /// This is an intentionally limited interface - implementations of this
  /// hook can only perform the following changes to the operation:
  ///
  ///  1. They can leave the operation alone and without changing the IR, and
  ///     return failure.
  ///  2. They can mutate the operation in place, without changing anything
  ///  else
  ///     in the IR.  In this case, return success.
  ///  3. They can return a list of existing values that can be used instead
  ///  of
  ///     the operation.  In this case, fill in the results list and return
  ///     success.  The caller will remove the operation and use those results
  ///     instead.
  ///
  /// This allows expression of some simple in-place canonicalizations (e.g.
  /// "x+0 -> x", "min(x,y,x,z) -> min(x,y,z)", "x+y-x -> y", etc), as well as
  /// generalized constant folding.
  LogicalResult foldHook(Operation *op, ArrayRef<Attribute> operands,
                         SmallVectorImpl<OpFoldResult> &results) const {
    return getImpl()->foldHook(op, operands, results);
  }

  /// This hook returns any canonicalization pattern rewrites that the
  /// operation supports, for use by the canonicalization pass.
  void getCanonicalizationPatterns(RewritePatternSet &results,
                                   MLIRContext *context) const {
    return getImpl()->getCanonicalizationPatterns(results, context);
  }

  /// Returns true if the operation was registered with a particular trait, e.g.
  /// hasTrait<OperandsAreSignlessIntegerLike>(). Returns false if the operation
  /// is unregistered.
  template <template <typename T> class Trait>
  bool hasTrait() const {
    return hasTrait(TypeID::get<Trait>());
  }
  bool hasTrait(TypeID traitID) const { return getImpl()->hasTrait(traitID); }

  /// Returns true if the operation *might* have the provided trait. This
  /// means that either the operation is unregistered, or it was registered with
  /// the provide trait.
  template <template <typename T> class Trait>
  bool mightHaveTrait() const {
    return mightHaveTrait(TypeID::get<Trait>());
  }
  bool mightHaveTrait(TypeID traitID) const {
    return !isRegistered() || getImpl()->hasTrait(traitID);
  }

  /// Return the static hook for parsing this operation assembly.
  ParseAssemblyFn getParseAssemblyFn() const {
    return getImpl()->getParseAssemblyFn();
  }

  /// This hook implements the method to populate defaults attributes that are
  /// unset.
  void populateDefaultAttrs(NamedAttrList &attrs) const {
    getImpl()->populateDefaultAttrs(*this, attrs);
  }

  /// This hook implements the AsmPrinter for this operation.
  void printAssembly(Operation *op, OpAsmPrinter &p,
                     StringRef defaultDialect) const {
    return getImpl()->printAssembly(op, p, defaultDialect);
  }

  /// These hooks implement the verifiers for this operation.  It should emits
  /// an error message and returns failure if a problem is detected, or
  /// returns success if everything is ok.
  LogicalResult verifyInvariants(Operation *op) const {
    return getImpl()->verifyInvariants(op);
  }
  LogicalResult verifyRegionInvariants(Operation *op) const {
    return getImpl()->verifyRegionInvariants(op);
  }

  /// Return the list of cached attribute names registered to this operation.
  /// The order of attributes cached here is unique to each type of operation,
  /// and the interpretation of this attribute list should generally be driven
  /// by the respective operation. In many cases, this caching removes the
  /// need to use the raw string name of a known attribute.
  ///
  /// For example the ODS generator, with an op defining the following
  /// attributes:
  ///
  ///   let arguments = (ins I32Attr:$attr1, I32Attr:$attr2);
  ///
  /// ... may produce an order here of ["attr1", "attr2"]. This allows for the
  /// ODS generator to directly access the cached name for a known attribute,
  /// greatly simplifying the cost and complexity of attribute usage produced
  /// by the generator.
  ///
  ArrayRef<StringAttr> getAttributeNames() const {
    return getImpl()->getAttributeNames();
  }

  /// Returns an instance of the concept object for the given interface if it
  /// was registered to this operation, null otherwise. This should not be used
  /// directly.
  template <typename T>
  typename T::Concept *getInterface() const {
    return getImpl()->getInterfaceMap().lookup<T>();
  }

  /// Attach the given models as implementations of the corresponding
  /// interfaces for the concrete operation.
  template <typename... Models>
  void attachInterface() {
    // Handle the case where the models resolve a promised interface.
    (dialect_extension_detail::handleAdditionOfUndefinedPromisedInterface(
         *getDialect(), getTypeID(), Models::Interface::getInterfaceID()),
     ...);

    getImpl()->getInterfaceMap().insertModels<Models...>();
  }

  /// Returns true if `InterfaceT` has been promised by the dialect or
  /// implemented.
  template <typename InterfaceT>
  bool hasPromiseOrImplementsInterface() const {
    return dialect_extension_detail::hasPromisedInterface(
               getDialect(), getTypeID(), InterfaceT::getInterfaceID()) ||
           hasInterface<InterfaceT>();
  }

  /// Returns true if this operation has the given interface registered to it.
  template <typename T>
  bool hasInterface() const {
    return hasInterface(TypeID::get<T>());
  }
  bool hasInterface(TypeID interfaceID) const {
    return getImpl()->getInterfaceMap().contains(interfaceID);
  }

  /// Returns true if the operation *might* have the provided interface. This
  /// means that either the operation is unregistered, or it was registered with
  /// the provide interface.
  template <typename T>
  bool mightHaveInterface() const {
    return mightHaveInterface(TypeID::get<T>());
  }
  bool mightHaveInterface(TypeID interfaceID) const {
    return !isRegistered() || hasInterface(interfaceID);
  }

  /// Lookup an inherent attribute by name, this method isn't recommended
  /// and may be removed in the future.
  std::optional<Attribute> getInherentAttr(Operation *op,
                                           StringRef name) const {
    return getImpl()->getInherentAttr(op, name);
  }

  void setInherentAttr(Operation *op, StringAttr name, Attribute value) const {
    return getImpl()->setInherentAttr(op, name, value);
  }

  void populateInherentAttrs(Operation *op, NamedAttrList &attrs) const {
    return getImpl()->populateInherentAttrs(op, attrs);
  }
  /// This method exists for backward compatibility purpose when using
  /// properties to store inherent attributes, it enables validating the
  /// attributes when parsed from the older generic syntax pre-Properties.
  LogicalResult
  verifyInherentAttrs(NamedAttrList &attributes,
                      function_ref<InFlightDiagnostic()> emitError) const {
    return getImpl()->verifyInherentAttrs(*this, attributes, emitError);
  }
  /// This hooks return the number of bytes to allocate for the op properties.
  int getOpPropertyByteSize() const {
    return getImpl()->getOpPropertyByteSize();
  }

  /// This hooks destroy the op properties.
  void destroyOpProperties(OpaqueProperties properties) const {
    getImpl()->deleteProperties(properties);
  }

  /// Initialize the op properties.
  void initOpProperties(OpaqueProperties storage, OpaqueProperties init) const {
    getImpl()->initProperties(*this, storage, init);
  }

  /// Set the default values on the ODS attribute in the properties.
  void populateDefaultProperties(OpaqueProperties properties) const {
    getImpl()->populateDefaultProperties(*this, properties);
  }

  /// Return the op properties converted to an Attribute.
  Attribute getOpPropertiesAsAttribute(Operation *op) const {
    return getImpl()->getPropertiesAsAttr(op);
  }

  /// Define the op properties from the provided Attribute.
  LogicalResult setOpPropertiesFromAttribute(
      OperationName opName, OpaqueProperties properties, Attribute attr,
      function_ref<InFlightDiagnostic()> emitError) const {
    return getImpl()->setPropertiesFromAttr(opName, properties, attr,
                                            emitError);
  }

  void copyOpProperties(OpaqueProperties lhs, OpaqueProperties rhs) const {
    return getImpl()->copyProperties(lhs, rhs);
  }

  bool compareOpProperties(OpaqueProperties lhs, OpaqueProperties rhs) const {
    return getImpl()->compareProperties(lhs, rhs);
  }

  llvm::hash_code hashOpProperties(OpaqueProperties properties) const {
    return getImpl()->hashProperties(properties);
  }

  /// Return the dialect this operation is registered to if the dialect is
  /// loaded in the context, or nullptr if the dialect isn't loaded.
  Dialect *getDialect() const {
    return isRegistered() ? getImpl()->getDialect()
                          : getImpl()->getName().getReferencedDialect();
  }

  /// Return the name of the dialect this operation is registered to.
  StringRef getDialectNamespace() const;

  /// Return the operation name with dialect name stripped, if it has one.
  StringRef stripDialect() const { return getStringRef().split('.').second; }

  /// Return the context this operation is associated with.
  MLIRContext *getContext() { return getIdentifier().getContext(); }

  /// Return the name of this operation. This always succeeds.
  StringRef getStringRef() const { return getIdentifier(); }

  /// Return the name of this operation as a StringAttr.
  StringAttr getIdentifier() const { return getImpl()->getName(); }

  void print(raw_ostream &os) const;
  void dump() const;

  /// Represent the operation name as an opaque pointer. (Used to support
  /// PointerLikeTypeTraits).
  void *getAsOpaquePointer() const { return const_cast<Impl *>(impl); }
  static OperationName getFromOpaquePointer(const void *pointer) {
    return OperationName(
        const_cast<Impl *>(reinterpret_cast<const Impl *>(pointer)));
  }

  bool operator==(const OperationName &rhs) const { return impl == rhs.impl; }
  bool operator!=(const OperationName &rhs) const { return !(*this == rhs); }

protected:
  OperationName(Impl *impl) : impl(impl) {}
  Impl *getImpl() const { return impl; }
  void setImpl(Impl *rhs) { impl = rhs; }

private:
  /// The internal implementation of the operation name.
  Impl *impl = nullptr;

  /// Allow access to the Impl struct.
  friend MLIRContextImpl;
  friend DenseMapInfo<mlir::OperationName>;
  friend DenseMapInfo<mlir::RegisteredOperationName>;
};

```