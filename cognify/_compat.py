import functools

try:
    from typing import override
except ImportError:

    def override(func):
        return func
