import collections.abc as cabc
import functools
import inspect
import keyword
import numbers
import pathlib as plib
import typing as typ


def check(**kwargs: cabc.Mapping[str, cabc.Callable[[typ.Any], bool]]) -> cabc.Callable:
    """
    Validate function parameters using boolean tests.

    It is common to check parameters for correctness before executing the function/class to which
    they are bound using boolean tests. This function is a decorator that intercepts the output of
    boolean functions and raises :py:exc:`ValueError` when the result is :py:obj:`False`.

    Parameters
    ----------
    **kwargs
        key: name of decorated function's parameter to test;
        value: boolean function to apply to the parameter value.

    Raises
    ------
    :py:exc:`ValueError`
        If any of the boolean functions return :py:obj:`False`.
    """
    key_error = lambda k: f"Key[{k}] must be a valid string identifier."
    value_error = lambda k: f"Value[Key[{k}]] must be a boolean function."

    for k, v in kwargs.items():
        if not isinstance(k, str):
            raise TypeError(key_error(k))
        if not (k.isidentifier() and (not keyword.iskeyword(k))):
            raise ValueError(key_error(k))
        if not inspect.isfunction(v):
            raise TypeError(value_error(k))

    def decorator(func: cabc.Callable) -> cabc.Callable:
        @functools.wraps(func)
        def wrapper(*ARGS, **KWARGS):
            func_args = inspect.getcallargs(func, *ARGS, **KWARGS)
            for k, fn in kwargs.items():
                if k not in func_args:
                    error_msg = f"Parameter[{k}] not part of {func.__qualname__}() parameter list."
                    raise ValueError(error_msg)
                if not fn(func_args[k]):
                    error_msg = " ".join(
                        [
                            f"Parameter[{k}] of {func.__qualname__}()",
                            f"does not satisfy {fn.__name__}().",
                        ]
                    )
                    raise ValueError(error_msg)
            return func(*ARGS, **KWARGS)

        return wrapper

    return decorator


def allow_None(func: cabc.Callable[[typ.Any], bool]) -> cabc.Callable[[typ.Any], bool]:
    if not inspect.isfunction(func):
        raise TypeError("Parameter[func] must be a boolean function.")

    @functools.wraps(func)
    def wrapper(x: typ.Any) -> bool:
        return True if (x is None) else func(x)

    wrapper.__name__ = f"allow_None({func.__name__})"
    return wrapper


# This function does lazy evaluation.
def accept_any(*funcs: tuple[cabc.Callable[[typ.Any], bool]]) -> cabc.Callable[[typ.Any], bool]:
    if not all(inspect.isfunction(_) for _ in funcs):
        raise TypeError("Parameter[*funcs] must contain boolean functions.")

    def union(x: typ.Any) -> bool:
        for fn in funcs:
            if fn(x):
                return True
        else:
            return False

    union.__name__ = f"accept_any({[fn.__name__ for fn in funcs]})"
    return union


# This function does lazy evaluation.
def require_all(*funcs: tuple[cabc.Callable[[typ.Any], bool]]) -> cabc.Callable[[typ.Any], bool]:
    if not all(inspect.isfunction(_) for _ in funcs):
        raise TypeError("Parameter[*funcs] must contain boolean functions.")

    def intersection(x: typ.Any) -> bool:
        for fn in funcs:
            if not fn(x):
                return False
        else:
            return True

    intersection.__name__ = f"require_all({[fn.__name__ for fn in funcs]})"
    return intersection


def is_instance(*klass: tuple[type]) -> cabc.Callable[[typ.Any], bool]:
    if not all(inspect.isclass(_) for _ in klass):
        raise TypeError("Parameter[*klass] must contain types.")

    def _is_instance(x: typ.Any) -> bool:
        return True if isinstance(x, klass) else False

    _is_instance.__name__ = f"is_instance({[cl.__name__ for cl in klass]})"
    return _is_instance


def is_container(
    ct: type,
    of: type,
    where: typ.Optional[cabc.Callable[[typ.Any], bool]] = None,
) -> cabc.Callable[[typ.Any], bool]:
    if not isinstance(ct, type):
        raise TypeError("Parameter[ct]: expected a type.")
    if not isinstance(of, type):
        raise TypeError("Parameter[of]: expected a type.")
    if where is not None:
        if not inspect.isfunction(where):
            raise TypeError("Parameter[where] must be a boolean function.")
    else:
        where = lambda _: True

    fc = is_instance(ct)
    fo = is_instance(of)

    def _func(x: typ.Any) -> bool:
        return fc(x) and all(fo(_) and where(_) for _ in x)

    _func.__name__ = f"is_container({ct.__name__}, {of.__name__}, {where.__name__})"
    return _func


def is_mapping(
    kt: type,
    vt: type,
    where_k: typ.Optional[cabc.Callable[[typ.Any], bool]] = None,
    where_v: typ.Optional[cabc.Callable[[typ.Any], bool]] = None,
) -> cabc.Callable[[typ.Any], bool]:
    if not isinstance(kt, type):
        raise TypeError("Parameter[kt]: expected a type.")
    if not isinstance(vt, type):
        raise TypeError("Parameter[kt]: expected a type.")
    if where_k is not None:
        if not inspect.isfunction(where_k):
            raise TypeError("Parameter[where_k] must be a boolean function.")
    else:
        where_k = lambda _: True
    if where_v is not None:
        if not inspect.isfunction(where_v):
            raise TypeError("Parameter[where_v] must be a boolean function.")
    else:
        where_v = lambda _: True

    fc = is_instance(cabc.Mapping)
    fk = is_instance(kt)
    fv = is_instance(vt)

    def _func(x: typ.Any) -> bool:
        return fc(x) and all(fk(k) and where_k(k) and fv(v) and where_v(v) for k, v in x.items())

    _func.__name__ = (
        f"is_mapping({kt.__name__}, {vt.__name__}, {where_k.__name__}, {where_v.__name__})"
    )
    return _func


def is_integer(x: typ.Any) -> bool:
    f = is_instance(numbers.Integral)
    return f(x)


def is_boolean(x: typ.Any) -> bool:
    f = is_instance(bool)
    return f(x)


def is_string(x: typ.Any) -> bool:
    f = is_instance(str)
    return f(x)


def is_path(x: typ.Any) -> bool:
    f = is_instance(plib.Path)
    return f(x)


def is_file(x: plib.Path) -> bool:
    x = x.expanduser().resolve()
    return x.is_file()


def is_dir(x: plib.Path) -> bool:
    x = x.expanduser().resolve()
    return x.is_dir()
