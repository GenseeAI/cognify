import os
import sys
from typing import Optional, Tuple, Type, Iterable
import logging
from collections import defaultdict
import uuid
from dataclasses import dataclass, field, asdict, fields

from cognify.graph.program import Module
from cognify.hub.cogs.common import CogBase
from cognify.hub.cogs.utils import build_param
from cognify.optimizer.plugin import capture_module_from_fs
from cognify.optimizer.registry import get_registered_opt_modules
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class ModuleTransformTrace:
    """
    This represents the morphing of the workflow
    Example:
        orignal workflow: [a, b, c]
        valid module_name_paths:
            a -> [a1, a2]
            a1 -> [a3, a4]
            c -> [c1]

        meaning original module A is now replaced by A2, A3, A4, upon which optimization will be performed
    """

    def __init__(self, ori_module_dict) -> None:
        self.ori_module_dict: dict[str, Type[Module]] = ori_module_dict
        # old_name to new_name
        self.module_name_paths: dict[str, str] = {}
        # level_name -> module_name -> [(param_name, option_name)]
        self.aggregated_proposals: dict[str, dict[str, list[tuple[str, str]]]] = {}
        self.flattened_name_paths: dict[str, str] = {}

    def add_mapping(self, ori_name: str, new_name: str):
        if ori_name in self.module_name_paths:
            raise ValueError(f"{ori_name} already been changed")
        self.module_name_paths[ori_name] = new_name

    def register_proposal(self, level_name: str, proposal: list[tuple[str, str, str]]):
        if level_name not in self.aggregated_proposals:
            self.aggregated_proposals[level_name] = defaultdict(list)
        for module_name, param_name, option_name in proposal:
            self.aggregated_proposals[level_name][module_name].append(
                (param_name, option_name)
            )

    def mflatten(self):
        """flatten the mapping

        Example:
            original:
                a -> a1
                a1 -> a2
            after:
                a -> a2
        """

        def is_cyclic_util(graph, v, visited, rec_stack):
            visited[v] = True
            rec_stack[v] = True

            if v in graph:
                neighbor = graph[v]
                if not visited[neighbor]:
                    if is_cyclic_util(graph, neighbor, visited, rec_stack):
                        return True
                elif rec_stack[neighbor]:
                    return True

            rec_stack[v] = False
            return False

        visited = defaultdict(False.__bool__)
        rec_stack = defaultdict(False.__bool__)
        for key in self.module_name_paths:
            if not visited[key]:
                if is_cyclic_util(self.module_name_paths, key, visited, rec_stack):
                    raise ValueError("Cyclic mapping")

        result: dict[str, str] = {}
        # use self.ori_module_dict to exclude sub-module mappings
        for ori in self.ori_module_dict:
            new_m_name = ori
            while new_m_name in self.module_name_paths:
                new_m_name = self.module_name_paths[new_m_name]
            result[ori] = new_m_name
        self.flattened_name_paths = result

    def get_derivatives_of_same_type(
        self, new_module: Module
    ) -> Tuple[str, list[Module]]:
        """
        NOTE: call mflatten before this
        find the parent module of the given new module, will search recursively in the new_module

        return old module name and names of new derivatives of the same type as the old module
        """
        if new_module.name in self.ori_module_dict:
            return (new_module.name, [new_module])
        for ori_name, new_name in self.flattened_name_paths.items():
            if new_module.name == new_name:
                derivatives = Module.all_of_type(
                    [new_module], self.ori_module_dict[ori_name]
                )
                return (ori_name, derivatives)
        return (new_module.name, [new_module])

    def eq_transform_path(self, other: dict[str, str]) -> bool:
        if self.module_name_paths.keys() != other.keys():
            return False
        for old_name, new_name in self.module_name_paths.items():
            if new_name != other[old_name]:
                return False
        return True

@dataclass
class PatienceConfig:
    quality_min_delta: float
    cost_min_delta: float
    exec_time_min_delta: float
    n_iterations: int

    def __post_init__(self):
        if self.quality_min_delta < 0 or self.cost_min_delta < 0 or self.exec_time_min_delta < 0 or self.n_iterations < 0:
            raise ValueError("patience values should be non-negative")

@dataclass
class OptConfig:
    """Configuration for optimization of each layer

    Attributes:
        n_trials (int): number of iterations of search.

        throughput (int, optional): number of trials to run in parallel. Defaults to 2.

        log_dir (str): directory to save logs.

        evolve_interval (int): interval to evolve the dynamic cogs (e.g., how frequently should few-shot examples be updated)

        opt_log_path (str): path to save optimization logs.

        param_save_path (str): path to save optimized parameters.

        frugal_eval_cost (bool): whether to favor cheaper evaluations in early stage.

        patience (Patience, optional): dataclass of (quality_min_delta, cost_min_delta, exec_time_min_delta, n_iteration) to set the early stop threshold.

        frac (float): fraction of the optimization from the last layer.
        
        use_HB_allocation (bool): whether to use Hyperband strategy.
        use_SH_allocation (bool): whether to use Successive Halving strategy.
        prune_rate (int): rate to prune the trials in HB or SH.
        
        initial_step_budget (int): initial budget R for HB.
    """
    n_trials: int
    throughput: int = field(default=2)
    log_dir: str = field(default=None)
    evolve_interval: int = field(default=2)
    opt_log_path: str = field(default=None)
    param_save_path: str = field(default=None)
    frugal_eval_cost: bool = field(default=True)
    use_SH_allocation: bool = field(default=False)
    patience: Optional[PatienceConfig] = field(default_factory=lambda: PatienceConfig(0.01,0.01,0.01,5))
    frac: float = field(default=1.0)
    use_HB_allocation: bool = field(default=False)
    use_SH_allocation: bool = field(default=False)
    prune_rate: int = field(default=2)
    initial_step_budget: int = field(default=8)

    def finalize(self):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)
        if self.opt_log_path is None:
            self.opt_log_path = os.path.join(self.log_dir, "opt_logs.json")
        if self.param_save_path is None:
            self.param_save_path = os.path.join(self.log_dir, "opt_params.json")

    def update(self, other: "OptConfig"):
        # for all None fields in self, update from other
        for f in fields(other):
            if getattr(self, f.name) is None:
                setattr(self, f.name, getattr(other, f.name))
                
    
    @classmethod
    def _set_log_dir_for_next(cls, log_dir: str):
        # set other fields to None so that the next layer can populate
        return cls(
            n_trials=None,
            throughput=None,
            log_dir=log_dir,
            evolve_interval=None,
            opt_log_path=None,
            param_save_path=None,
            frugal_eval_cost=None,
            use_SH_allocation=None,
            patience=None,
            frac=None,
        )

@dataclass
class TopDownInformation:
    """
    Information that is passed from the top level to the lower level
    """

    # optimization config for current level
    opt_config: OptConfig

    # optimization meta inherit from the previous level
    all_params: Optional[dict[str, CogBase]]
    module_ttrace: Optional[ModuleTransformTrace]
    current_module_pool: Optional[dict[str, Module]]

    # optimization configs
    script_path: str
    script_args: Optional[list[str]]
    other_python_paths: Optional[list[str]]

    # some optimization history
    trace_back: list[str] = field(
        default_factory=list
    )  # series of [layer_name_trial_id]

    def initialize(self, ori_opt_config: OptConfig):
        if self.opt_config is None:
            self.opt_config = ori_opt_config
        else:
            self.opt_config.update(ori_opt_config)
        
        self.opt_config.finalize()
        self.all_params = self.all_params or {}
        self.script_args = self.script_args or []
        self.other_python_paths = self.other_python_paths or []

        if self.current_module_pool is None:
            dir = os.path.dirname(self.script_path)
            if dir not in sys.path:
                sys.path.insert(0, dir)
            sys.argv = [self.script_path] + self.script_args
            capture_module_from_fs(self.script_path)
            self.current_module_pool = {m.name: m for m in get_registered_opt_modules()}

        if self.module_ttrace is None:
            name_2_type = {m.name: type(m) for m in self.current_module_pool.values()}
            self.module_ttrace = ModuleTransformTrace(name_2_type)
        self.module_ttrace.mflatten()


class LayerConfig:
    def __init__(
        self,
        layer_name: str,
        dedicate_params: list[CogBase] = [],
        universal_params: list[CogBase] = [],
        target_modules: Iterable[str] = None,
        save_ckpt_interval: int = 1,
        opt_config: Optional[OptConfig] = None,
    ):
        """Config for each optimization layer

        Args:
            layer_name (str): name of the layer

            dedicate_params (list[ParamBase], optional): dedicated params for this layer. Defaults to [].

            universal_params (list[ParamBase], optional): universal params for this layer. Defaults to [].

            target_modules (Iterable[str], optional): target modules for this layer. Defaults to None.

            save_ckpt_interval (int, optional): save checkpoint interval. Defaults to 0.

            opt_config (OptConfig, optional): optimization config. Defaults to None.
                all fields not set here will be decided by the upper layer

        """
        self.layer_name = layer_name
        self.dedicate_params = dedicate_params
        self.universal_params = universal_params
        self.target_modules = target_modules
        self.save_ckpt_interval = save_ckpt_interval
        self.opt_config = opt_config

        if len(self.dedicate_params) + len(self.universal_params) == 0:
            raise ValueError(f"No params provided for optimization layer {layer_name}")

        if self.opt_config is None:
            self.opt_config = OptConfig(n_trials=5)

    def to_dict(self):
        return {
            "layer_name": self.layer_name,
            "dedicate_params": [p.to_dict() for p in self.dedicate_params],
            "universal_params": [p.to_dict() for p in self.universal_params],
            "target_modules": self.target_modules,
            "save_ckpt_interval": self.save_ckpt_interval,
            "opt_config": asdict(self.opt_config),
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            layer_name=d["layer_name"],
            dedicate_params=[build_param(p) for p in d["dedicate_params"]],
            universal_params=[build_param(p) for p in d["universal_params"]],
            target_modules=d["target_modules"],
            save_ckpt_interval=d["save_ckpt_interval"],
            opt_config=OptConfig(**d["opt_config"]),
        )
