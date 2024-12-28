# Extending torch Python API

You can create custom types that emulate Tensor by defining a custom class with methods that match Tensor.

But what if you want to be able to pass these types to functions like torch.add() in the top-level torch namespace that accept Tensor operands?

If your custom Python type defines a method named `__torch_function__`, PyTorch will invoke your `__torch_function__` implementation when an instance of your custom class is passed to a function in the torch namespace.

This makes it possible to define custom implementations for any of the functions in the torch namespace which your `__torch_function__` implementation can call, allowing your users to make use of your custom type with existing PyTorch workflows that they have already written for Tensor.

This works with "duck" types that are unrelated to Tensor as well as user-defined subclasses of Tensor.

你可以创建一个自定义类型模拟 Tensor, 但是如果你想把你的自定义 Tensor 传给类似于 torch.add() 这些 torch namespace 下的函数, 该怎么做呢

如果自定义 Python 类型定义了一个 `__torch_function__` 方法, PyTorch 会在实例传给 torch 下的函数的时候调用

## Extending torch with a Tensor-like type

> This functionality is inspired by the NumPy `__array_function__` protocol. See the NumPy documentation and NEP-0018 for more details.

To make this concrete, let’s begin with a simple example that illustrates the API dispatch mechanism. We’ll create a custom type that represents a 2D scalar tensor, parametrized by the order N and value along the diagonal entries, value:

```py
class ScalarTensor(object):
   def __init__(self, N, value):
       self._N = N
       self._value = value

   def __repr__(self):
       return "ScalarTensor(N={}, value={})".format(self._N, self._value)

   def tensor(self):
       return self._value * torch.eye(self._N)
```

This first iteration of the design isn’t very useful. The main functionality of ScalarTensor is to provide a more compact string representation of a scalar tensor than in the base tensor class:

```
>>> d = ScalarTensor(5, 2)
>>> d
ScalarTensor(N=5, value=2)
>>> d.tensor()
tensor([[2., 0., 0., 0., 0.],
        [0., 2., 0., 0., 0.],
        [0., 0., 2., 0., 0.],
        [0., 0., 0., 2., 0.],
        [0., 0., 0., 0., 2.]])
```

If we try to use this object with the torch API, we will run into issues:

```
>>> import torch
>>> torch.mean(d)
TypeError: mean(): argument 'input' (position 1) must be Tensor, not ScalarTensor
```

Adding a `__torch_function__` implementation to ScalarTensor makes it possible for the above operation to succeed. Let’s re-do our implementation, this time adding a `__torch_function__` implementation:

```py
HANDLED_FUNCTIONS = {}
class ScalarTensor(object):
    def __init__(self, N, value):
        self._N = N
        self._value = value

    def __repr__(self):
        return "ScalarTensor(N={}, value={})".format(self._N, self._value)

    def tensor(self):
        return self._value * torch.eye(self._N)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func not in HANDLED_FUNCTIONS or not all(
            issubclass(t, (torch.Tensor, ScalarTensor))
            for t in types
        ):
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)
```

The `__torch_function__` method takes four arguments: func, a reference to the torch API function that is being overridden, types, the list of types of Tensor-likes that implement `__torch_function__`, args, the tuple of arguments passed to the function, and kwargs, the dict of keyword arguments passed to the function.

It uses a global dispatch table named HANDLED_FUNCTIONS to store custom implementations. The keys of this dictionary are functions in the torch namespace and the values are implementations for ScalarTensor.

> Using a global dispatch table is not a mandated part of the `__torch_function__` API, it is just a useful design pattern for structuring your override implementations.

This class definition isn’t quite enough to make `torch.mean` do the right thing when we pass it a ScalarTensor – we also need to define an implementation for torch.mean for ScalarTensor operands and add the implementation to the HANDLED_FUNCTIONS dispatch table dictionary. One way of doing this is to define a decorator:

```py
import functools
def implements(torch_function):
    """Register a torch function override for ScalarTensor"""
    def decorator(func):
        functools.update_wrapper(func, torch_function)
        HANDLED_FUNCTIONS[torch_function] = func
        return func
    return decorator
```

which can be applied to the implementation of our override:

```py
@implements(torch.mean)
def mean(input):
    return float(input._value) / input._N
```

With this change we can now use torch.mean with ScalarTensor:

```py
d = ScalarTensor(5, 2)
torch.mean(d)
```

Of course torch.mean is an example of the simplest kind of function to override since it only takes one operand. We can use the same machinery to override a function that takes more than one operand, any one of which might be a tensor or tensor-like that defines `__torch_function__`, for example for torch.add():

```py
def ensure_tensor(data):
    if isinstance(data, ScalarTensor):
        return data.tensor()
    return torch.as_tensor(data)

@implements(torch.add)
def add(input, other):
   try:
       if input._N == other._N:
           return ScalarTensor(input._N, input._value + other._value)
       else:
           raise ValueError("Shape mismatch!")
   except AttributeError:
       return torch.add(ensure_tensor(input), ensure_tensor(other))
```

This version has a fast path for when both operands are ScalarTensor instances and also a slower path which degrades to converting the data to tensors when either operand is not a ScalarTensor. That makes the override function correctly when either operand is a ScalarTensor or a regular Tensor:

```
>>> s = ScalarTensor(2, 2)
>>> torch.add(s, s)
ScalarTensor(N=2, value=4)
>>> t = torch.tensor([[1, 1,], [1, 1]])
>>> torch.add(s, t)
tensor([[3., 1.],
        [1., 3.]])
```

Note that our implementation of add does not take alpha or out as keyword arguments like torch.add() does:

```
>>> torch.add(s, s, alpha=2)
TypeError: add() got an unexpected keyword argument 'alpha'
```

For speed and flexibility the `__torch_function__` dispatch mechanism does not check that the signature of an override function matches the signature of the function being overridden in the torch API. For some applications ignoring optional arguments would be fine but to ensure full compatibility with Tensor, user implementations of torch API functions should take care to exactly emulate the API of the function that is being overridden.


Functions in the torch API that do not have explicit overrides will return NotImplemented from `__torch_function__`.

If all operands with `__torch_function__` defined on them return `NotImplemented`, PyTorch will raise a TypeError.

This means that most of the time operations that do not have explicit overrides for a type will raise a TypeError when an instance of such a type is passed:

```
>>> torch.mul(s, 3)
TypeError: no implementation found for 'torch.mul' on types that
implement __torch_function__: [ScalarTensor]
```

In practice this means that if you would like to implement your overrides using a `__torch_function__` implementation along these lines, you will need to explicitly implement the full torch API or the entire subset of the API that you care about for your use case. This may be a tall order as the full torch API is quite extensive.

Another option is to not return NotImplemented for operations that are not handled but to instead pass a Tensor to the original torch function when no override is available. For example, if we change our implementation of `__torch_function__` for ScalarTensor to the one below:

```py
@classmethod
def __torch_function__(cls, func, types, args=(), kwargs=None):
    if kwargs is None:
        kwargs = {}
    if func not in HANDLED_FUNCTIONS or not all(
            issubclass(t, (torch.Tensor, ScalarTensor))
            for t in types
        ):
        args = [a.tensor() if hasattr(a, 'tensor') else a for a in args]
        return func(*args, **kwargs)
    return HANDLED_FUNCTIONS[func](*args, **kwargs)
```

Then torch.mul() will work correctly, although the return type will always be a Tensor rather than a ScalarTensor, even if both operands are ScalarTensor instances:

```py
>>> s = ScalarTensor(2, 2)
>>> torch.mul(s, s)
tensor([[4., 0.],
        [0., 4.]])
```

Also see the MetadataTensor example below for another variation on this pattern but instead always returns a MetadataTensor to propagate metadata through operations in the torch API.



The `__torch_function__` protocol is designed for full coverage of the API, partial coverage may lead to undesirable results, in particular, certain functions raising a TypeError.

This is especially true for subclasses, where all three of `torch.add`, `torch.Tensor.__add__` and `torch.Tensor.add` must be covered, even if they return exactly the same result. Failing to do this may also lead to infinite recursion.

If one requires the implementation of a function from torch.Tensor subclasses, they must use `super().__torch_function__` inside their implementation.

## Subclassing `torch.Tensor`

As of version 1.7.0, methods on torch.Tensor and functions in public torch.* namespaces applied on torch.Tensor subclasses will return subclass instances instead of torch.Tensor instances:

```py
>>> class SubTensor(torch.Tensor):
...     pass
>>> type(torch.add(SubTensor([0]), SubTensor([1]))).__name__
'SubTensor'
>>> type(torch.add(SubTensor([0]), torch.tensor([1]))).__name__
'SubTensor'
```

If multiple subclasses exist, the lowest one in the hierarchy will be chosen by default. If there is no unique way to determine such a case, then a TypeError is raised:

```py
>>> type(torch.add(SubTensor2([0]), SubTensor([1]))).__name__
'SubTensor2'
>>> type(torch.add(SubTensor2([0]), torch.tensor([1]))).__name__
'SubTensor2'
>>> torch.add(SubTensor([0]), OtherSubTensor([1]))
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: no implementation found for 'torch.add' on types that implement __torch_function__: [SubTensor, OtherSubTensor]
```

If one wishes to have a global override for all tensor methods, one can use `__torch_function__`. Here is an example that logs all function/method calls:

```py
class LoggingTensor(torch.Tensor):
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        # NOTE: Logging calls Tensor.__repr__, so we can't log __repr__ without infinite recursion
        if func is not torch.Tensor.__repr__:
            logging.info(f"func: {func.__name__}, args: {args!r}, kwargs: {kwargs!r}")
        if kwargs is None:
            kwargs = {}
        return super().__torch_function__(func, types, args, kwargs)
```

However, if one instead wishes to override a method on the Tensor subclass, there one can do so either by directly overriding the method (by defining it for a subclass), or by using `__torch_function__` and matching with func.

One should be careful within `__torch_function__` for subclasses to always call `super().__torch_function__(func, ...)` instead of func directly, as was the case before version 1.7.0.

Failing to do this may cause func to recurse back into `__torch_function__` and therefore cause infinite recursion.

## Extending `torch` with a `Tensor` wrapper type

Another useful case is a type that wraps a Tensor, either as an attribute or via subclassing. Below we implement a special case of this sort of type, a MetadataTensor that attaches a dictionary of metadata to a Tensor that is propagated through torch operations. Since this is a generic sort of wrapping for the full torch API, we do not need to individually implement each override so we can make the __torch_function__ implementation more permissive about what operations are allowed:

```py
class MetadataTensor(object):
    def __init__(self, data, metadata=None, **kwargs):
        self._t = torch.as_tensor(data, **kwargs)
        self._metadata = metadata

    def __repr__(self):
        return "Metadata:\n{}\n\ndata:\n{}".format(self._metadata, self._t)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        metadatas = tuple(a._metadata for a in args if hasattr(a, '_metadata'))
        args = [getattr(a, '_t', a) for a in args]
        assert len(metadatas) > 0
        ret = func(*args, **kwargs)
        return MetadataTensor(ret, metadata=metadatas[0])
```

This simple implementation won’t necessarily work with every function in the torch API but it is good enough to capture most common operations:

```
>>> metadata = {'owner': 'Ministry of Silly Walks'}
>>> m = MetadataTensor([[1, 2], [3, 4]], metadata=metadata)
>>> t = torch.tensor([[1, 2], [1, 2]])
>>> torch.add(t, m)
Metadata:
{'owner': 'Ministry of Silly Walks'}

data:
tensor([[2, 4],
        [4, 6]])
>>> torch.mul(t, m)
Metadata:
{'owner': 'Ministry of Silly Walks'}

data:
tensor([[1, 4],
        [3, 8]])
```

## Operations on multiple types that define `__torch_function__`

It is possible to use the torch API with multiple distinct types that each have a `__torch_function__` implementation, but special care must be taken. In such a case the rules are:

The dispatch operation gathers all distinct implementations of __torch_function__ for each operand and calls them in order: subclasses before superclasses, and otherwise left to right in the operator expression.

If any value other than NotImplemented is returned, that value is returned as the result. Implementations can register that they do not implement an operation by returning NotImplemented.

If all of the __torch_function__ implementations return NotImplemented, PyTorch raises a TypeError.

