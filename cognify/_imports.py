import types
import importlib
from typing import Any

class _LazyModule(types.ModuleType):
    def __init__(self, mname: str) -> None:
        super().__init__(mname)
        self._name = mname

    def _load(self) -> types.ModuleType:
        module = importlib.import_module(self._name)
        self.__dict__.update(module.__dict__)
        return module

    def __getattr__(self, item: str) -> Any:
        return getattr(self._load(), item)
    