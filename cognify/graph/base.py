from dataclasses import dataclass
from collections import defaultdict, deque
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import List, Iterable, Callable, Type, TypeVar
import warnings
import time
import logging
import copy
from graphviz import Digraph

from cognify.graph.utils import get_function_kwargs

import logging


logger = logging.getLogger(__name__)


class State:
    def __init__(self, version_id, data, is_static) -> None:
        self.version_id = version_id  # currently not used
        self.data = data
        self.is_static = is_static  # if True, the state will not be updated anymore


class StatePool:
    def __init__(self, reducer_dict={}, debug=True):
        self.states: dict[str, list[State]] = defaultdict(list)
        self.reducer_dict = reducer_dict
        self.debug = debug

    def news(self, key: str, default=None):
        if key not in self.states or not self.states[key]:
            if default is None:
                raise ValueError(f"Key {key} not found in state")
            return default
        return self.states[key][-1].data

    def init(self, kvs: dict):
        self.publish(kvs, is_static=True, version_id=0)

    def publish(self, kvs, is_static, version_id):
        keys = list(kvs.keys())
        for key in keys:
            if key in self.reducer_dict:
                new_value = self.reducer_dict[key](self.news(key), new_value)
                self.states[key].append(State(version_id, new_value, is_static))
                kvs.pop(key)
        for key, value in kvs.items():
            self.states[key].append(State(version_id, value, is_static))

    def all_news(self, fields: Iterable = None, excludes: Iterable = None):
        report = {}
        for key in self.states:
            if fields is not None and key not in fields:
                continue
            if excludes is not None and key in excludes:
                continue
            report[key] = self.news(key)
        return report

    def history(self, key: str):
        if key not in self.states:
            raise ValueError(f"Key {key} not found in state")
        hist = []
        for state in self.states[key]:
            hist.append(state.data)
        return hist

    def all_history(self, fields=None, excludes=None):
        report = {}
        for key in self.states:
            if fields is not None and key not in fields:
                continue
            if excludes is not None and key in excludes:
                continue
            report[key] = self.history(key)
        return report

    def dump(self, path: str):
        raise NotImplementedError

    def load(self, path: str):
        raise NotImplementedError


@dataclass
class Context:
    calling_module: "Module"
    predecessor: str
    invoke_time: int  # start from 1


def hint_possible_destinations(dests: list[str]):
    if not isinstance(dests, list):
        raise ValueError("dests should be a list of strings")

    def hinter(func):
        func._possible_destinations = dests
        return func

    return hinter


class ModuleStatus(Enum):
    PENDING = auto()
    RUNNING = auto()
    SUCCESS = auto()
    FAILED = auto()
    SKIPPED = auto()


class ModuleInterface(ABC):
    @abstractmethod
    def invoke(self, statep: StatePool): ...

    @abstractmethod
    def reset(self):
        """clear metadata for new run"""
        ...


MT = TypeVar("MT", bound="Module")


class Module(ModuleInterface):
    def __init__(self, name, kernel, opt_register: bool = False) -> None:
        self.name = name
        self.kernel = kernel
        self.outputs = []
        self.exec_times = []
        self.status = None
        self.is_static = False
        self.version_id = 0
        self.enclosing_module: ComposibleModuleInterface = None
        self.prepare_input_env()

        self.opt_target = opt_register
        if opt_register:
            from cognify.optimizer import register_opt_module

            register_opt_module(self)

    def prepare_input_env(self):
        if self.kernel is not None:
            self.input_fields, self.defaults = get_function_kwargs(self.kernel)
        else:
            self.input_fields, self.defaults = [], {}
        self.on_signature_generation()
        logger.debug(f"Module {self.name} kernel has input fields {self.input_fields}")

    def reset(self):
        self.outputs = []
        self.exec_times = []
        self.status = None
        self.version_id = 0

    def forward(self, **kwargs):
        raise NotImplementedError

    def get_immediate_enclosing_module(self):
        """return the immediate enclosing module of this module, None if this module is not in any module"""
        return self.enclosing_module

    def on_signature_generation(self):
        """
        allows each child class to modify the input fields and defaults
        """
        pass

    def invoke(self, statep: StatePool):
        logger.debug(f"Invoking {self}")
        for field in self.input_fields:
            if field not in self.defaults and field not in statep.states:
                raise ValueError(
                    f"Missing field {field} in state when calling {self.name}, available fields: {statep.states.keys()}"
                )
        kargs = {
            field: statep.news(field)
            for field in statep.states
            if field in self.input_fields
        }
        # time the execution
        start = time.perf_counter()
        result = self.forward(**kargs)
        dur = time.perf_counter() - start
        result_snapshot = copy.deepcopy(result)
        statep.publish(result_snapshot, self.version_id, self.is_static)
        self.outputs.append(result_snapshot)
        # update metadata
        self.exec_times.append(dur)
        self.version_id += 1

    @staticmethod
    def all_with_predicate(
        modules: Iterable["Module"], predicate: Callable[["Module"], bool]
    ) -> list["Module"]:
        """Find all modules that satisfy the predicate

        If not match and is composible module, will search recursively into all composible modules
        """
        targets = []
        for m in modules:
            if predicate(m):
                targets.append(m)
            elif isinstance(m, ComposibleModuleInterface):
                targets.extend(m.get_all_modules(predicate))
            else:
                continue
        return targets

    @staticmethod
    def all_of_type(modules: Iterable["Module"], T: Type[MT]) -> list[MT]:
        """Find all modules of type T in the given modules

        will search recursively into all composible modules if not match
        """
        return Module.all_with_predicate(modules, lambda x: isinstance(x, T))

    def chameleon(self, other: "Module") -> bool:
        if self is not other:
            if self.enclosing_module is not None:
                # NOTE: in-place replacement
                logger.debug(f"Replacing {self.name} with {other.name}")
                if not self.enclosing_module.replace_node(self, other, other):
                    logger.warning(f"Failed to replace {self.name} with {other.name}")
                    logger.warning("option apply failed, continue")
                    return False
        return True


class ComposibleModuleInterface(Module, ABC):
    @abstractmethod
    def immediate_submodules(self) -> List[Module]:
        pass

    def get_all_modules(self, predicate=None) -> list[Module]:
        """get all modules that satisfy the predicate

        will search recursively into all composible modules
        """
        module_queue = deque(self.immediate_submodules())
        result = []

        while module_queue:
            module = module_queue.popleft()
            if predicate is None or predicate(module):
                result.append(module)
            if isinstance(module, ComposibleModuleInterface):
                module_queue.extend(module.immediate_submodules())
        return result

    @abstractmethod
    def _visualize(self, dot: Digraph):
        pass

    def sub_module_validation(self, module: Module, reset_parent: bool):
        """
        Example:
        ```python
        a = Module('a', None)
        b = Workflow('b'); b.add_module(a)
        c = Workflow('c')

        # YES
        c.add_module(b)
        # NO
        c.add_module(a)
        ```
        """
        if not reset_parent and (parent := module.enclosing_module) is not None:
            if parent is not self:
                raise ValueError(
                    f"Source module {module.name} already registered in {parent.name}, please avoid adding the same module to multiple modules"
                )

    @abstractmethod
    def replace_node_handler(
        self, old_node: Module, new_node_in: Module, new_node_out: Module
    ) -> bool:
        pass

    def replace_check(
        self, old_node: Module, new_node_in: Module, new_node_out: Module
    ):
        if old_node not in self.immediate_submodules():
            warnings.warn(f"Node {old_node.name} not found in {self.name}")
            return False
        if (parent := new_node_in.enclosing_module) is not None:
            if parent is not self:
                warnings.warn(
                    f"Please avoid replacing a node with a module that is already registered in another module: {new_node_in.name} in {parent.name}"
                )
                return False
        if (parent := new_node_out.enclosing_module) is not None:
            if parent is not self:
                warnings.warn(
                    f"Please avoid replacing a node with a module that is already registered in another module: {new_node_out.name} in {parent.name}"
                )
                return False
        return True

    def replace_node(
        self, old_node: Module, new_node_in: Module, new_node_out: Module
    ) -> bool:
        """Replace the old node with the new node

        Will register the new node to the same parent module as the old node, if new node is already registered in another module, will raise an error

        Incoming dataflow will be redirected to the new_node_in, and outgoing dataflow will be redirected to the new_node_out
        new_node_in and new_node_out can be the same module, equivalent to replacing the old node with another node
        """
        is_ok = self.replace_check(old_node, new_node_in, new_node_out)
        if not is_ok:
            return False
        return self.replace_node_handler(old_node, new_node_in, new_node_out)
