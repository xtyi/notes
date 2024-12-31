
# Extending torch native API

While `__torch_function__` allows one to effectively extend PyTorch’s pure Python components’ behavior, it does not allow one to extend the parts of PyTorch implemented in C++.

To that end, a Tensor subclass can also define `__torch_dispatch__` which will be able to override the behavior at the C++ level.

To effectively use this feature, it is important to know how the native part of PyTorch is implemented.

The most important component there is what we call the "dispatcher" (the best description can be found in this blog post even though it is slightly outdated).

As hinted by its name, it is responsible for calling the right backend function for a specific call of a function.

For example, when calling torch.add(a, b), the dispatcher will inspect both arguments, figure out which "feature" (autograd, autocast, functionalization, etc) and which "backend" (CPU, CUDA, MPS, etc) should be used for this specific call and finally call all the right kernels.

A very common thing done by a kernel is to "redispatch". For example, when running your neural network on GPU with autocast, the first call will be the autocast kernel that will handle any potential autocast logic and redispatch down.

The next feature in line will be autograd that will properly create the autograd graph and then redispatch down.

Finally, we reach the backend kernel for CUDA which will launch the right CUDA kernel and return the final result. On the way out, autograd will attach the graph to the output and, finally, autocast will have a chance to do any update it needs on exit.


One configuration of the dispatcher is the order in which all these feature and backend keys are called. The latest list and their order can be found in `DispatchKey.h` inside the `DispatchKey` enum. For the purpose of extending torch, the important subset of the ordering for this discussion is:

```
vmap -> Autocast -> Autograd -> ZeroTensor -> Neg/Conj -> Functionalize -> Python -> Backends
```

The most important key for the purpose of this discussion is Python as every Tensor subclass with the `__torch_dispatch__` method defined will call into this feature.

It is from there that the user-defined method is called and where the behavior can be overwritten arbitrarily. From there, calling the provided func again will perform a “redispatch”.

Some important implications of this implementation are:

- This code runs "below all features". It is thus only responsible, like a regular backend, for generating the output value of each Tensor (and can, and should, ignore all advanced features like autograd, autocast, etc).

- If any high level feature implements a given function without redispatching, it will never reach the Python key and so the `__torch_dispatch__` callback will never be triggered. This happens in particular for CompositeImplicitAutograd functions which are evaluated at the Autograd level without redispatching. This is because a CompositeImplicitAutograd function specifies its autograd formula by implicitly calling other native ops, so at the Autograd level, the function is decomposed into its native ops and those are evaluated instead.

- When calling back to Python and when wrapping the results, the same conversions are used as the regular PyTorch Python/C++ binding. In particular, some objects cannot be represented in Python and need special handling (undefined Tensors for example become None).

- Our native functions are lazily populated as torch.ops.{namespace}.{func_name}.{overload_name} as callable Python objects to enable easily interacting with them from Python. The func object given to __torch_dispatch__ is always an entry from this namespace. This namespace can be used to directly call native ops and bypass the usual Python API and binding code.

- In a similar way where `__torch_function__` is able to interpose on all of torch’s Python API and Tensor methods, `__torch_dispatch__` is able intercepting all calls into the aten native API. Note that all methods on Tensors are converted into function calls before entering the dispatcher and thus will appear as function calls here: torch.add(a, 2) and a + 2 will lead to exactly the same aten call. Most of these functions are defined in native_functions.yaml which specifies the properties of these functions as well as their backend implementation. Their implementation alongside specified features are then automatically registered via codegen. Some more exotic functions or features are also registered in other places in the C++ codebase or in user-defined C++ extensions.

It is also possible to add new native functions using torch.library. This Python feature allows defining and/or adding new implementations to native functions. This can be used to add missing kernels, replace existing ones or define brand new native functions.

You can find many examples of __torch_dispatch__-based subclasses in the subclass zoo repo.

# Torch Dispatch Mode

A TorchDispatchMode allows you to override the meaning of all `__torch_dispatch__` overrideable functions within a dynamic scope,
without having to actually create a tensor subclass or manually monkey-patch functions in the PyTorch API.

Some common situations where you should use a mode:
- You want to override the meaning of factory functions, or other functions that do not otherwise take a tensor as a argument (these cannot be overridden with tensor subclasses).
- You want to override the behavior of all functions without needing to wrap your inputs in tensor subclasses; e.g. if you are just interested in logging intermediate computations.
- You want to control the order of execution of various tensor subclasses explicitly, rather than implicitly via the return of NotImplemented.

Independent subclasses of TorchDispatchMode are compositional:

```py
class TorchDispatchMode:

    """
    Independent subclasses of :class:`TorchDispatchMode` are compositional:
    modes can be pushed onto a stack using ``with MyMode():``.
    When you call functions in the PyTorch API inside your
    ``__torch_dispatch__`` implementation, by default, they will forward on to
    the next mode on the mode stack.  If you want recursively call back into
    your current ``__torch_dispatch__`` implementation, either explicitly
    invoke ``self.__torch_dispatch__(...)``, or use the context manager
    ``__torch_dispatch__(self)`` to make PyTorch
    API self-referential (beware of infinite loops, in this case!)
    """

    def __init__(self, _dispatch_key=None):
        if _dispatch_key is not None:
            assert isinstance(_dispatch_key, torch._C.DispatchKey)
            self.__dict__["_dispatch_key"] = _dispatch_key

        self.old_dispatch_mode_flags: Deque[bool] = deque()
        self.old_non_infra_dispatch_mode_flags: Deque[bool] = deque()

    def _lazy_init_old_dispatch_mode_flags(self):
        if not hasattr(self, "old_dispatch_mode_flags"):
            self.old_dispatch_mode_flags: Deque[bool] = deque()  # type: ignore[no-redef]

        if not hasattr(self, "old_non_infra_dispatch_mode_flags"):
            self.old_non_infra_dispatch_mode_flags: Deque[bool] = deque()  # type: ignore[no-redef]

    # 最核心的方法，需要在子类中实现
    # func: 被拦截的 PyTorch 操作
    # types: 操作涉及的类型信息
    # args: 位置参数
    # kwargs: 关键字参数
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        raise NotImplementedError

    # 上下文管理器实现
    # 保存当前的调度模式状态
    # 设置新的调度模式状态
    # 将当前模式推入模式栈
    def __enter__(self):
        global _is_in_torch_dispatch_mode
        global _is_in_non_infra_torch_dispatch_mode
        # Previously, there wasn't any state in this class' constructor
        # super calls were added to existing modes, but for any new modes
        # this will replicate the previous behavior of not strictly needing
        # to call super().__init__()
        self._lazy_init_old_dispatch_mode_flags()
        self.old_dispatch_mode_flags.append(_is_in_torch_dispatch_mode)
        _is_in_torch_dispatch_mode = True
        self.old_non_infra_dispatch_mode_flags.append(_is_in_non_infra_torch_dispatch_mode)
        _is_in_non_infra_torch_dispatch_mode = _is_in_non_infra_torch_dispatch_mode or not self.is_infra_mode()
        _push_mode(self)
        return self

    # 上下文管理器实现
    def __exit__(self, exc_type, exc_val, exc_tb):
        mb_dk_or_mode_key = self.__dict__.get("_dispatch_key", None)
        if mb_dk_or_mode_key is None:
            # Today, mode keys are not used at all in the per-dispatch-key-mode logic (for pre-dispatch)
            # We should probably revisit this.
            mb_dk_or_mode_key = self.__dict__.get("_mode_key", None)
        global _is_in_torch_dispatch_mode
        _is_in_torch_dispatch_mode = self.old_dispatch_mode_flags.pop()
        global _is_in_non_infra_torch_dispatch_mode
        _is_in_non_infra_torch_dispatch_mode = self.old_non_infra_dispatch_mode_flags.pop()
        _pop_mode(mb_dk_or_mode_key)

    @classmethod
    def is_infra_mode(cls):
        return False
```