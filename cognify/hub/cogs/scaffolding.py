from typing import Optional
import os
import sys
import json
import logging
import copy


logger = logging.getLogger(__name__)

from cognify.graph.base import Module
from cognify.graph.program import Workflow
from cognify.llm.model import Model
from cognify.hub.cogs.common import (
    CogBase,
    CogLayerLevel,
    OptionBase,
    NoChange,
    AddNewModuleImportInterface,
)
from cognify.hub.cogs.decompose import LMTaskDecompose, StructuredAgentSystem
from cognify.optimizer.plugin import OptimizerSchema
from cognify.optimizer import clear_registry


class LMScaffolding(CogBase, AddNewModuleImportInterface):
    level = CogLayerLevel.GRAPH

    def __init__(
        self,
        name: str,
        log_dir: str,
        module_name: str = None,
        new_agent_systems: list[StructuredAgentSystem] = [],
        default_identity: bool = True,
    ):
        self.log_dir = log_dir
        if default_identity:
            options = [NoChange()]
        else:
            options = []
        for i, new_sys in enumerate(new_agent_systems):
            options.append(
                DecomposeOption(f"Decompose_{module_name}_option_{i}", new_sys, log_dir)
            )
        super().__init__(name, options, 0, module_name)

    @classmethod
    def bootstrap(
        cls,
        workflow: Optional[Workflow] = None,
        lm_modules: Optional[list[Model]] = None,
        decompose_threshold: int = 4,
        default_identity: bool = True,
        target_modules: Optional[list[str]] = None,
        log_dir: str = "task_decompose_logs",
    ):
        decomposer = LMTaskDecompose(workflow=workflow, lm_modules=lm_modules)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        decomposer.prepare_decompose_metadata(
            target_modules, decompose_threshold, log_dir
        )
        decomposer.finalize_decomposition(target_modules, log_dir)

        params = []
        for module_name, new_system in decomposer.lm_2_final_system.items():
            params.append(
                cls(
                    name=f"Scaffold_{module_name}",
                    log_dir=log_dir,
                    module_name=module_name,
                    new_agent_systems=[new_system],
                    default_identity=default_identity,
                )
            )
        return params

    @classmethod
    def bootstrap_from_source(
        cls,
        script_path: str,
        script_args: list[str] = [],
        decompose_threshold: int = 4,
        target_modules: Optional[list[str]] = None,
        default_identity: bool = True,
        log_dir: str = "task_decompose_logs",
    ):
        dir = os.path.dirname(script_path)
        if dir not in sys.path:
            sys.path.insert(0, dir)
        sys.argv = [script_path] + script_args
        schema = OptimizerSchema.capture(script_path)
        clear_registry()

        lm_modules = copy.deepcopy(schema.opt_target_modules)

        return cls.bootstrap(
            lm_modules=lm_modules,
            decompose_threshold=decompose_threshold,
            target_modules=target_modules,
            log_dir=log_dir,
            default_identity=default_identity,
        )

    @classmethod
    def from_dict(cls, data: dict):
        name, module_name, log_dir = data["name"], data["module_name"], data["log_dir"]
        param = cls(
            name=name,
            log_dir=log_dir,
            module_name=module_name,
        )

        loaded_options = data["options"]
        loaded_options.pop("Identity", None)
        loaded_options = {
            name: DecomposeOption.from_dict(option)
            for name, option in loaded_options.items()
        }
        param.options.update(loaded_options)

        return param

    def to_dict(self):
        base = super().to_dict()
        base["log_dir"] = self.log_dir
        return base

    def get_python_paths(self) -> list[str]:
        return [self.log_dir]


class DecomposeOption(OptionBase):
    def __init__(self, name: str, new_system: StructuredAgentSystem, log_dir: str):
        super().__init__(name)
        self.new_system = new_system
        self.log_dir = log_dir

    def describe(self):
        new_agents_prmopt = {
            name: meta.agent_prompt for name, meta in self.new_system.agents.items()
        }
        new_agents_prmopt = json.dumps(new_agents_prmopt, indent=4)
        desc = f"""
        - Agent Scaffolding -
        Decomposed agents:
        {new_agents_prmopt}
        """

    def _get_cost_indicator(self):
        return len(self.new_system.agents)

    def apply(self, module: Model) -> Module:
        new_agent = LMTaskDecompose.materialize_decomposition(
            lm=module,
            new_agents=self.new_system,
            default_lm_config=None,
            log_dir=self.log_dir,
        )
        return new_agent

    def to_dict(self):
        base = super().to_dict()
        base["new_system"] = self.new_system.model_dump()
        base["log_dir"] = self.log_dir
        return base

    @classmethod
    def from_dict(cls, data: dict):
        name = data["name"]
        new_system = StructuredAgentSystem.model_validate(data["new_system"])
        log_dir = data["log_dir"]
        return cls(name, new_system, log_dir)
