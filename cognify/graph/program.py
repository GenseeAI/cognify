from typing import List, Optional, Tuple, Callable, Hashable, Union
from collections import defaultdict
from graphviz import Digraph
from functools import partial
import functools
import traceback

from cognify.llm import Model
from cognify.llm.response import ResponseMetadata
from cognify.graph.modules import InputModule, Output, Branch, Identity
from cognify.graph.base import *
from cognify.graph.rewriter.utils import replace_branch_destination
from cognify.graph.utils import simple_cycles
import json
import concurrent.futures
import time
from itertools import chain


logger = logging.getLogger(__name__)


class Trigger:
    """synchronization barrier
    This decides when synchronous barrier is satisfied
    thus decide what next module can be invoked
    """

    def __init__(
        self,
        immediate_deps: set[str],
        potential_deps: set[str],
        next_step: str,
    ):
        self.immediate_deps = immediate_deps
        self.potential_deps = potential_deps
        self.next_step = next_step
        self.notified: dict[str, bool] = {}
        self.active = False

    def notify(self, src: str, is_static: bool):
        if src in self.notified:
            raise ValueError(
                f"Trigger {self} has already been notified by {src} and not consumed yet"
            )
        self.active = True
        self.notified[src] = is_static

    def can_fire(self, scheduled_modules: set[str]) -> set[str]:
        """determine if this trigger can invoke the module

        1. If all immediate deps are notified, fire
        2. If any potential deps are scheduled, wait, otherwise fire
        """
        if self.immediate_deps == set(self.notified.keys()):
            self.active = False
            return set(self.notified.keys())
        if not self.potential_deps.intersection(scheduled_modules):
            self.active = False
            return set(self.notified.keys())
        return None

    def consume(self, notified: set[str]):
        for notif in notified:
            if notif in self.notified and not self.notified[notif]:
                del self.notified[notif]


class SyncBarrierManager:
    dependency = set[str]

    def __init__(self) -> None:
        # sync barriers: dest_name -> [dependencies]
        self._synchronizations: dict[str, list[SyncBarrierManager.dependency]] = (
            defaultdict(list)
        )
        self.branches: dict[str, Branch] = {}

        # edges are not used for preparing the next module to run
        self.edges: dict[str, list[str]] = defaultdict(list)
        # the previous module will notify the trigger when it's done
        self.triggers: list[Trigger] = []
        self.publish_channels: dict[str, list[Trigger]] = defaultdict(list)

    def add_dependency(self, srcs: set[str], dests: set[str], enhance_existing: bool):
        for dest in dests:
            if enhance_existing and dest in self._synchronizations:
                for dep in self._synchronizations[dest]:
                    dep.update(srcs)
            else:
                self._synchronizations[dest].append(copy.deepcopy(srcs))

    def add_branch_dependency(
        self,
        branch: Branch,
        enhance_existing: bool,
    ):
        self.branches[branch.name] = branch
        self.add_dependency(set(branch.src), set([branch.name]), False)
        self.add_dependency(
            set([branch.name]), set(branch.destinations), enhance_existing
        )

    def notify_completion(self, src_module: Module, statep: StatePool):
        src_name = src_module.name
        if src_name in self.publish_channels:
            if src_name in self.branches:
                # depend on the branch result, notify all triggers involved
                next_ms = statep.news(src_name + "#branch_result")
                for trigger in self.publish_channels[src_name]:
                    if trigger.next_step in next_ms:
                        trigger.notify(src_name, src_module.is_static)
            else:
                for trigger in self.publish_channels[src_name]:
                    trigger.notify(src_name, src_module.is_static)

    def fire_next_round(self, scheduled: set[str] = None):
        candidates = set()
        triggered_notifs = set()
        fired_triggers: list[Trigger] = []
        # check immediate deps
        for trigger in self.triggers:
            if trigger.active and (notifs := trigger.can_fire(scheduled)):
                candidates.add(trigger.next_step)
                triggered_notifs.update(notifs)
                fired_triggers.append(trigger)
        # after scheduling, check potential deps
        # if current step does not scheduled any potential deps, fire
        for trigger in self.triggers:
            if trigger.active and (notifs := trigger.can_fire(candidates)):
                candidates.add(trigger.next_step)
                triggered_notifs.update(notifs)
                fired_triggers.append(trigger)
        # invalidate triggered notifications
        for trigger in fired_triggers:
            trigger.consume(triggered_notifs)
        return candidates

    def clean(self):
        self.edges.clear()
        self.triggers.clear()
        self.publish_channels.clear()

    def reset_triggers(self):
        for trigger in self.triggers:
            trigger.active = False
            trigger.notified.clear()

    def get_dependency(self, module_names: Iterable[str]):
        """build hyperthetical dependency graph

        get all dependent nodes given destination node consider cycles
        """
        re_edges: dict[str, set[str]] = defaultdict(set)

        for dest_name, deps in self._synchronizations.items():
            re_edges[dest_name] = functools.reduce(lambda x, y: x.union(y), deps)

        def dfs(dest, visited: set[str]):
            if dest in visited:
                return visited
            visited.add(dest)
            reachable = {dest}
            for src in re_edges[dest]:
                reachable.update(dfs(src, visited))
            visited.remove(dest)
            return reachable

        dependency_graph: dict[str, set[str]] = {}
        for module in module_names:
            dependency_graph[module] = dfs(module, set())
        return dependency_graph

    def compile_dependency_runtime(self, module_names: Iterable[str]):
        self.clean()
        dep_graph: dict[str, set[str]] = self.get_dependency(module_names)

        # For each module, register their triggers and corresponding dependencies
        for dest_name, dep_list in self._synchronizations.items():
            for immediate_dep in dep_list:
                potential_dep = set().union(
                    chain.from_iterable(dep_graph[src] for src in immediate_dep)
                )
                trigger = Trigger(immediate_dep, potential_dep, dest_name)
                self.triggers.append(trigger)
                for src in immediate_dep:
                    self.publish_channels[src].append(trigger)

        # build src -> [dests] edges
        for dest_name, dep_list in self._synchronizations.items():
            for immediate_dep in dep_list:
                for src in immediate_dep:
                    self.edges[src].append(dest_name)
        # remove duplication
        for src, dests in self.edges.items():
            self.edges[src] = list(set(dests))
        for name in module_names:
            if name not in self.edges:
                self.edges[name] = []

    def replace_dependency(
        self, old_node: Module, new_node_in: Module, new_node_out: Module
    ):
        # update edges
        dest_names = list(self._synchronizations.keys())
        for dest_name in dest_names:
            # replace outgoing dataflows
            for dep in self._synchronizations[dest_name]:
                if old_node.name in dep:
                    dep.remove(old_node.name)
                    dep.add(new_node_out.name)
            # replace incoming dataflows
            if old_node.name == dest_name:
                self._synchronizations[new_node_in.name] = self._synchronizations.pop(
                    dest_name
                )

        # update branches
        for name, branch in self.branches.items():
            if old_node.name in (dests := branch.destinations):
                # replace destination hint
                dests.remove(old_node.name)
                dests.add(new_node_in.name)
                # replace return value
                new_multiplexer, new_code_str = replace_branch_destination(
                    branch.multiplexer,
                    old_node.name,
                    new_node_in.name,
                    branch.multiplexer_str,
                )
                branch.multiplexer = new_multiplexer
                branch.multiplexer_str = new_code_str
                branch.kernel = new_multiplexer
            if old_node.name in branch.src:
                branch.src.remove(old_node.name)
                branch.src.add(new_node_out.name)
        return True


class Workflow(ComposibleModuleInterface):
    def __init__(self, name) -> None:
        self.modules: dict[str, Module] = {}
        self.sync_barrier_manager = SyncBarrierManager()

        """
        Following will be (re)populated during (re)compilation
        """
        self.start: InputModule = None
        self.end: Output = None

        """
        some runtime meta
        """
        self.token_usage_buffer = {"total": {}}
        self.current_step = 0
        super().__init__(name, None)

    def add_module(self, module: Module, reset_parent=True) -> None:
        self.sub_module_validation(module, reset_parent)
        self.modules[module.name] = module
        if reset_parent:
            module.enclosing_module = self

    def _edge_validation(self, src: Iterable[str], dest: Iterable[str]):
        for s in src:
            if s not in self.modules:
                raise ValueError(
                    f"Source module {s} not found, please add the module first"
                )
            assert self.modules[s].enclosing_module is self
        for d in dest:
            if d not in self.modules:
                raise ValueError(
                    f"Destination module {d} not found, please add the module first"
                )
            assert self.modules[d].enclosing_module is self

    def add_edge_old(
        self, src: Union[str, list[str]], dest: Union[str, list[str]]
    ) -> None:
        """add static dataflow

        Args:
            src (Union[str, list[str]]): source module(s)
            dest (Union[str, list[str]]): destination module(s)

        NOTE:
            src added in one add_edge call will be treated as a synchronization group
            i.e. the dest module will only run after all src modules are done

            If you prefer individual triggers for each src module, call add_edge multiple times
        """
        if isinstance(src, str):
            src = [src]
        if isinstance(dest, str):
            dest = [dest]
        self._edge_validation(src, dest)
        self.static_dependencies[tuple(src)].extend(set(dest))

    def add_edge(
        self,
        src: Union[str, list[str]],
        dest: Union[str, list[str]],
        enhance_existing=False,
    ) -> None:
        """add static dataflow

        Args:
            src (Union[str, list[str]]): source module(s)
            dest (Union[str, list[str]]): destination module(s)
            enhance (bool): whether to enhance the existing edge

        src added in one add_edge call will be treated as a synchronization group
        i.e. the dest module will only run after all src modules are done

        If you prefer independent triggers for each src module, call add_edge multiple times with each src separately
        For example,
        >>> add_edge('a', 'c') # add a trigger that a -> c
        >>> add_edge('b', 'c') # add another trigger that b -> c
        >>> add_edge(['a', 'b'], 'd') # add a more strict trigger that a, b -> d

        If enhance is True, the existing edge will be augmented with new srcs. For example, if you have exising edges as:
        - a -> b
        - [c, d] -> [b, x, y]

        call add_edge(['m', 'n'], ['b'], True) will result in:

        - [a, m, n] -> b
        - [c, d, m, n] -> [b]
        - [c, d] -> [x, y]
        """
        if isinstance(src, str):
            src = [src]
        if isinstance(dest, str):
            dest = [dest]

        src, dest = set(src), set(dest)
        self._edge_validation(src, dest)
        self.sync_barrier_manager.add_dependency(src, dest, enhance_existing)

    def add_branch_old(
        self,
        name: str,
        src: Union[str, list[str]],
        multiplexer: Callable[..., Union[Hashable, list[Hashable]]],
        multiplexer_str: Optional[str] = None,
    ) -> None:
        """add control flow

        If need complex synchronization, use add_edge to add a preceding module

        Args:
            src (Union[str, list[str]]):
                the module that the control flow need to synchronize with

            multiplexer (callable):
                signature should be (ctx, arg1, arg2, ...) -> Hashable
                ctx contains some useful information for the multiplexer to make decision

        Examples:
        NOTE: please hint all possible destinations for the multiplexer
            ```python
            from cognify.graph.program import hint_possible_destinations

            @hint_possible_destinations(['a', 'b'])
            def multiplexer(ctx, smth):
            ... if f(smth):
            ...    return ['a', 'b]
            ... else:
            ...    return 'b'
            ```
        """
        src = sorted(list(set([src])))
        self._edge_validation(src, multiplexer._possible_destinations)

        branch = Branch(
            name, src, multiplexer, multiplexer._possible_destinations, multiplexer_str
        )
        self.add_module(branch)
        self.branches[name] = branch
        self.add_edge(src, branch.name)

    def add_branch(
        self,
        name: str,
        src: Union[str, list[str]],
        multiplexer: Callable[..., Union[str, list[str]]],
        multiplexer_str: Optional[str] = None,
        enhance_existing: bool = False,
    ):
        """add control flow

        Args:
            src (Union[str, list[str]]):
                the module that the control flow need to synchronize with

            multiplexer (callable):
                signature should be (ctx, arg1, arg2, ...) -> Hashable
                ctx contains some useful information for the multiplexer to make decision

            enhance_existing (bool): whether to enhance the existing edge

        Examples:
        NOTE: please hint all possible destinations for the multiplexer
            ```python
            from cognify.graph.program import hint_possible_destinations

            @hint_possible_destinations(['a', 'b'])
            def multiplexer(ctx, smth):
            ... if f(smth):
            ...    return ['a', 'b]
            ... else:
            ...    return 'b'
            ```

        Enhance_existing is useful when you want to add conditional dependency to an existing synchronization point. This will have the same effect as the add_edge with enhance=True

        For example, if you have exising edges as:
        - [x, y] -> [a, m]

        then add the above multiplexer with enhance_existing=True will result in:
        - [x, y, branch_name] -> [a] # only after x, y, and branch gives 'a' as the result will a be activated
        - [x, y] -> [m]
        """
        if isinstance(src, str):
            src = [src]
        src, dest = set(src), set(multiplexer._possible_destinations)
        self._edge_validation(
            src, []
        )  # avoid circular dependency or dependency on other unadded branches
        branch = Branch(name, src, multiplexer, dest, multiplexer_str)
        self.add_module(branch)
        self.sync_barrier_manager.add_branch_dependency(branch, enhance_existing)

    def validate(self):
        # Clear previous compilation
        self.start = None
        self.end = None

        # TODO: add more validation check
        for name, module in self.modules.items():
            if isinstance(module, InputModule):
                if self.start is not None:
                    raise ValueError("Multiple start points detected")
                self.start = module
            if isinstance(module, Output):
                if self.end is not None:
                    raise ValueError("Multiple end points detected")
                self.end = module
        if self.start is None:
            raise ValueError("No start point detected")
        if self.end is None:
            raise ValueError("No end point detected")

    def compile_old(self):
        """config each module with graph analysis"""
        self.validate()
        # Compile all subgraphs
        all_sub_graphs = self.get_all_modules(lambda x: isinstance(x, Workflow))
        for sub_graph in all_sub_graphs:
            sub_graph.compile()
        # Derive the dependency graph
        dep_graph: dict[str, set[str]] = self.get_dependency()
        # For each module, register their triggers and corresponding dependencies
        for srcs, dests in self.static_dependencies.items():
            immediate_deps = set(srcs)
            potential_deps = set().union(
                chain.from_iterable(dep_graph[src] for src in srcs)
            )

            def make_foo(dests):
                return lambda statep: dests

            trigger = Trigger(immediate_deps, potential_deps, make_foo(dests))
            self.triggers.append(trigger)
            for src in srcs:
                self.publish_channels[src].append(trigger)
        # same for dynamic branches
        for src, branch in self.branches.items():
            immediate_deps = {src}
            potential_deps = dep_graph[src]

            def make_dfoo(src, statep):
                next_m = statep.news(src + "#branch_result")
                logger.debug(f"Branch {src} result: {next_m}")
                return next_m

            trigger = Trigger(immediate_deps, potential_deps, partial(make_dfoo, src))
            self.triggers.append(trigger)
            self.publish_channels[src].append(trigger)
        # Identify dynamic modules, i.e. steps will be invoked multiple times
        for srcs, dests in self.static_dependencies.items():
            for src in srcs:
                self.edges[src].extend(dests)
        for src, branch in self.branches.items():
            self.edges[src].extend(branch.destinations)
        # remove duplication
        for src, dests in self.edges.items():
            self.edges[src] = list(set(dests))
        for name in self.modules:
            if name not in self.edges:
                self.edges[name] = []
        nodes_in_cycles = set(chain.from_iterable(simple_cycles(self.edges)))
        for name in nodes_in_cycles:
            self.modules[name].is_static = False
        # TODO: populate history states

    def compile(self):
        """config each module with graph analysis"""
        self.validate()
        # Compile all subgraphs
        all_sub_graphs = self.get_all_modules(lambda x: isinstance(x, Workflow))
        for sub_graph in all_sub_graphs:
            sub_graph.compile()

        self.sync_barrier_manager.compile_dependency_runtime(self.modules.keys())

        nodes_in_cycles = set(
            chain.from_iterable(simple_cycles(self.sync_barrier_manager.edges))
        )
        for name in nodes_in_cycles:
            self.modules[name].is_static = False
        # TODO: populate history states

    def reset(self) -> None:
        # clear sub-llms history
        self.update_token_usage_summary()
        self.token_usage_buffer = {"total": {}}
        self.sync_barrier_manager.reset_triggers()
        self.current_step = 0

        for module in self.immediate_submodules():
            module.reset()

    def exec_module(self, module: Module, statep: StatePool):
        try:
            module.invoke(statep)
            module.status = ModuleStatus.SUCCESS
        except Exception as e:
            logger.error(f"Error in {module.name}: {e}")
            module.status = ModuleStatus.FAILED
            traceback.print_exc()
            raise e

        if module.status == ModuleStatus.SUCCESS:
            self.sync_barrier_manager.notify_completion(module, statep)

    def pregel_run(
        self,
        statep,
        start_from: Optional[str] = None,
        stop_before: Optional[str] = None,
    ):
        scheduled = set()
        if start_from is None:
            start_from = self.start.name
        scheduled.add(start_from)

        while True:
            num_tasks = len(scheduled)
            if num_tasks == 0:
                break
            logger.debug(f"Graph {self.name} - Step {self.current_step}: {scheduled}")
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=num_tasks
            ) as executor:
                futures = [
                    executor.submit(self.exec_module, self.modules[name], statep)
                    for name in scheduled
                ]
                concurrent.futures.wait(futures)
            # for name in scheduled:
            #     self.exec_module(self.modules[name], statep)
            scheduled = self.sync_barrier_manager.fire_next_round(scheduled)
            if stop_before is not None:
                scheduled.discard(stop_before)
            self.current_step += 1

    def visualize(self, dir: str):
        dot = Digraph()
        dot.attr(compound="true")
        self._visualize(dot)
        dot.render(directory=dir)

    def _visualize(self, dot: Digraph):
        dot.node(
            f"_{self.name}_cluster_ancor",
            style="invis",
            fixedsize="true",
            width="0",
            height="0",
        )
        for src, dests in self.sync_barrier_manager.edges.items():
            attrs = {}
            if src in self.sync_barrier_manager.branches:
                attrs["style"] = "dashed"
            if isinstance(self.modules[src], ComposibleModuleInterface):
                attrs["ltail"] = f"cluster_{src}"
                src = f"_{src}_cluster_ancor"
            for dest in dests:
                cattrs = {**attrs}
                if isinstance(self.modules[dest], ComposibleModuleInterface):
                    cattrs["lhead"] = f"cluster_{dest}"
                    dest = f"_{dest}_cluster_ancor"
                dot.edge(src, dest, **cattrs)
        for name, m in self.modules.items():
            if isinstance(m, ComposibleModuleInterface):
                with dot.subgraph(name=f"cluster_{name}") as s:
                    m._visualize(s)
                    s.attr(label=name)

    def update_token_usage_summary(self):
        """get token usage summary for all LLM modules recursively"""
        for lm in (m for m in self.get_all_modules(lambda x: isinstance(x, Model))):
            lm: Model
            if lm.name not in self.token_usage_buffer:
                self.token_usage_buffer[lm.name] = {}
            for meta in lm.response_metadata_history:
                meta: ResponseMetadata
                model = meta.model
                if model not in self.token_usage_buffer[lm.name]:
                    self.token_usage_buffer[lm.name][model] = defaultdict(int)
                self.token_usage_buffer[lm.name][model]["prompt_tokens"] += (
                    meta.usage.prompt_tokens
                )
                self.token_usage_buffer[lm.name][model]["completion_tokens"] += (
                    meta.usage.completion_tokens
                )
                if model not in self.token_usage_buffer["total"]:
                    self.token_usage_buffer["total"][model] = defaultdict(int)
                self.token_usage_buffer["total"][model]["prompt_tokens"] += (
                    meta.usage.prompt_tokens
                )
                self.token_usage_buffer["total"][model]["completion_tokens"] += (
                    meta.usage.completion_tokens
                )
            # NOTE: clear incase of double counting
            lm.response_metadata_history = []
        return self.token_usage_buffer

    def log_module_time(self, path):
        import numpy as np

        times = {}
        for module in self.modules.values():
            times[module.name] = np.mean(module.exec_times)
        with open(path, "w+") as f:
            json.dump(times, f, indent=4)

    def log_token_usage(self, path):
        self.update_token_usage_summary()
        with open(path, "w+") as f:
            json.dump(self.token_usage_buffer, f, indent=4)

    def invoke(self, statep: StatePool):
        start = time.perf_counter()
        self.pregel_run(statep)
        dur = time.perf_counter() - start
        # update metadata
        self.exec_times.append(dur)
        self.status = ModuleStatus.SUCCESS
        self.version_id += 1

    def immediate_submodules(self) -> List[Module]:
        return list(self.modules.values())

    def bypass_node(self, node_name):
        node_in_deps: list[Union[Branch, Tuple[str, ...]]] = []
        node_in_dests: list[Union[Branch, Tuple[str, ...]]] = []
        for deps, dests in self.static_dependencies.items():
            if node_name in deps:
                node_in_deps.append(deps)
            if node_name in dests:
                node_in_dests.append(deps)
        for branch in self.branches.values():
            if node_name in branch.src:
                node_in_deps.append(branch)
            if node_name in branch.destinations:
                node_in_dests.append(branch)

        if node_in_deps:
            assert (
                len(node_in_deps) > 0
            ), f"Node {node_name} is in dependency but no one activates it"

        counter = 0
        new_dynamic_dests = []
        for deps_to_replace in node_in_deps:
            for replace_candidate in node_in_dests:
                if isinstance(deps_to_replace, Branch):
                    if isinstance(replace_candidate, Branch):
                        buffer_id = f"sync_buffer_{replace_candidate.name}_to_{deps_to_replace.name}"
                        self.add_module(Identity(buffer_id))

                        deps_to_replace.src[deps_to_replace.src.index(node_name)] = (
                            buffer_id
                        )
                        new_dynamic_dests.append(replace_candidate.name)
                    else:
                        pass
                else:
                    pass

    def replace_node_handler(
        self, old_node: Module, new_node_in: Module, new_node_out: Module
    ) -> bool:
        if isinstance(old_node, Branch):
            NotImplementedError("Branch replacement is not supported yet")

        if old_node.name not in self.modules:
            logger.warning(f"Node {old_node.name} not found in {self.name}")
            return False
        del self.modules[old_node.name]

        self.add_module(new_node_in, True)
        self.add_module(new_node_out, True)

        return self.sync_barrier_manager.replace_dependency(
            old_node, new_node_in, new_node_out
        )
