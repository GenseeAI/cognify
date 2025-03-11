import functools


def deprecate_func(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        raise DeprecationWarning(f"{func.__name__} is deprecated")

    return wrapper
