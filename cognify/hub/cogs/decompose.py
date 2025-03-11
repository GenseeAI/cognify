import os
import json
from typing import Optional, Dict, List
import logging
from collections import defaultdict
import concurrent.futures
from devtools import pprint

from cognify.graph.program import Workflow, InputModule, Output, hint_possible_destinations
from cognify.graph.rewriter.utils import add_argument_to_position
from pydantic import BaseModel
from cognify.graph.modules import CodeBox
from cognify.llm import Model, StructuredModel, OutputFormat, OutputLabel, Input
from cognify.hub.cogs.decompose_agents import *
from cognify.hub.cogs.decompose_agents.estimate_complexity import ComplexityEstimation
from cognify.optimizer.utils import aggregator_factory, json_schema_to_pydantic_model
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DecomposeCandidate:
    lm: Model
    score: float
    rationale: str


class LMTaskDecompose:
    def __init__(
        self,
        workflow: Optional[Workflow] = None,
        lm_modules: Optional[list[Model]] = None,
    ):
        """provide either workflow or lm_modules to initialize the decomposer

        If both are provided, workflow will be used

        The provided workflow or lm_modules will be changed in-place
        """
        self.workflow = workflow
        if self.workflow is not None:
            self.lm_modules: list[Model] = workflow.get_all_modules(
                lambda m: isinstance(m, Model)
            )
        else:
            if lm_modules is None:
                raise ValueError("Either workflow or lm_modules should be provided")
            self.lm_modules = lm_modules
        self.decompose_target_lms: list[Model] = []

        # lm name -> structured agent system
        self.lm_to_final_system: dict[str, StructuredAgentSystem] = {}

    def _load_existing_decomposition(
        self, log_path, system: BaseModel, target_modules
    ) -> dict:
        with open(log_path) as f:
            json_lm_to_system = json.load(f)
            lm_to_system = {
                k: system.model_validate(v) for k, v in json_lm_to_system.items()
            }
            self.decompose_target_lms = [
                m
                for m in self.lm_modules
                if m.name in lm_to_system
                and (target_modules is None or m.name in target_modules)
            ]
            return lm_to_system

    def _get_decomposition_candidates(
        self, target_modules, log_dir: str
    ) -> List[DecomposeCandidate]:
        logger.info("Estimating complexity of agents")

        agent_prompts = [
            m.get_agent_role()
            for m in self.lm_modules
            if target_modules is None or m.name in target_modules
        ]
        complexity_list: List[ComplexityEstimation] = estimate_complexity_kernel(
            agent_prompts
        )

        decompose_candidates: List[DecomposeCandidate] = [
            DecomposeCandidate(lm, complexity.score, complexity.rationale)
            for lm, complexity in zip(self.lm_modules, complexity_list)
        ]
        decompose_candidates = sorted(
            decompose_candidates, key=lambda x: x[1], reverse=True
        )

        json_decompose_candidates = {}
        for candidate in decompose_candidates:
            json_decompose_candidates[candidate.lm.name] = {
                "score": candidate.score,
                "rationale": candidate.rationale,
            }
            logger.info(
                f"Complexity of {candidate.lm.name}: {candidate.score}\nrationale: {candidate.rationale}\n\n"
            )
        with open(os.path.join(log_dir, "task_decompose_candidates.json"), "w+") as f:
            json.dump(json_decompose_candidates, f, indent=4)
        return decompose_candidates

    def _high_level_decomposition(
        self, decompose_candidates: List[DecomposeCandidate], threshold: float
    ) -> Dict[str, HighLevelDecompose]:
        logger.info("Performing high-level agent decomposition")
        high_level_result = {}

        def _hd(candidate: DecomposeCandidate):
            if int(candidate.score) >= threshold:
                new_agents: HighLevelDecompose = high_level_decompose_kernel(
                    candidate.lm.get_agent_role()
                )
                high_level_result[candidate.lm.name] = new_agents
                self.decompose_target_lms.append(candidate.lm)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(lambda x: _hd(*x), decompose_candidates)
        return high_level_result

    def _mid_level_decomposition(
        self, high_level_result: Dict[str, HighLevelDecompose], log_path: str
    ) -> Dict[str, NewAgentSystem]:
        logger.info("Adding concrete dependencies to decomposed system")
        new_agent_system = {}
        json_new_agent_system = {}

        def _ld(lm: Model):
            agent_metadata = {
                name: high_level_decompose.model_dump()
                for name, high_level_decompose in high_level_result[
                    lm.name
                ].agents.items()
            }
            new_system: NewAgentSystem = decompose_refine_kernel(
                agent_metadata, lm.get_high_level_info()
            )
            new_agent_system[lm.name] = NewAgentSystem.model_validate(new_system)
            json_new_agent_system[lm.name] = json.loads(new_system.model_dump_json())

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(_ld, lm) for lm in self.decompose_target_lms]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Failed to decompose: {e}")

        with open(log_path, "w+") as f:
            json.dump(json_new_agent_system, f, indent=4)
        return new_agent_system

    def _final_decomposition(
        self, log_path: str, mid_level_agent_system: Dict[str, NewAgentSystem]
    ):
        logger.info("Finalizing new agent system")
        final_agent_system: Dict[str, StructuredAgentSystem] = {}

        def _fd(lm: Model):
            mid_level_desc: NewAgentSystem = mid_level_agent_system[lm.name]
            final_agents = finalize_new_agents_kernel(
                lm.get_formatted_info(), mid_level_desc
            )
            final_agent_system[lm.name] = final_agents

        for lm in self.decompose_target_lms:
            _fd(lm)
        with open(log_path, "w+") as f:
            json_lm_to_final_system = {
                k: json.loads(v.model_dump_json())
                for k, v in final_agent_system.items()
            }
            json.dump(json_lm_to_final_system, f, indent=4)

    def prepare_decompose_metadata(
        self,
        target_modules,
        threshold,
        log_dir,
    ):
        # Get mid-level decomposition meta
        log_path = os.path.join(log_dir, "task_decompose_mid_level.json")
        new_agent_system: Dict[str, NewAgentSystem] = {}
        if os.path.exists(log_path):
            logger.info(
                "mid-level decomposition already exists, read and skip sampling"
            )
            new_agent_system = self._load_existing_decomposition(
                log_path, NewAgentSystem, target_modules
            )
        else:
            decompose_candidates: List[DecomposeCandidate] = (
                self._get_decomposition_candidates(target_modules, log_dir)
            )
            high_level_result: Dict[str, HighLevelDecompose] = (
                self._high_level_decomposition(decompose_candidates, threshold)
            )
            logger.info("High-level decomposition results:\n")
            pprint(high_level_result)

            new_agent_system = self._mid_level_decomposition(
                high_level_result, log_path
            )
            logger.info("Mid-level decomposition results:\n")
            pprint(new_agent_system)

        # Get final decomposition meta
        log_path = os.path.join(log_dir, "task_decompose_final.json")

        if os.path.exists(log_path):
            logger.info("final decomposition already exists, read and skip sampling")
            self.lm_to_final_system = self._load_existing_decomposition(
                log_path, StructuredAgentSystem, target_modules
            )
        else:
            self.lm_to_final_system = self._final_decomposition(new_agent_system)
            logger.info("Final decomposition results:\n")
            pprint(self.lm_to_final_system)

    @staticmethod
    def materialize_decomposition(
        lm: Model,
        new_agents: StructuredAgentSystem,
        default_lm_config: dict,
        log_dir: str,
    ) -> Workflow:
        """Actually transform the graph to apply the decomposition

        1. First create a sub-graph to represent the new agent system
        2. Then replace the original agent with the new agent system
            - If the encolsing module is a graph, flatten the sub-graph to avoid hierarchy
            - otherwise, replace the agent directly
        """
        # TODO: add more validation checks
        sub_graph = Workflow(f"{lm.name}_sub_graph")
        input_name, output_name = (
            f"{lm.name}_sub_graph_input",
            f"{lm.name}_sub_graph_output",
        )
        sub_graph.add_module(InputModule(input_name))
        sub_graph.add_module(Output(output_name))

        logical_end_name = output_name
        # Add final aggregator
        if new_agents.final_output_aggregator_code != "None":
            code_kernel = aggregator_factory(
                lm, new_agents.final_output_aggregator_code
            )
            aggregator = CodeBox(f"{lm.name}_final_aggregator", code_kernel)
            sub_graph.add_module(aggregator)
            sub_graph.add_edge(aggregator.name, output_name)
            logical_end_name = aggregator.name

        # Materialize each agent
        name_to_new_lm: dict[str, Model] = {}
        for agent_name, agent_meta in new_agents.agents.items():
            valid_file_name = (
                agent_name.replace(" ", "").replace("\n", "").replace("\t", "") + ".py"
            )
            module_fpath = os.path.join(log_dir, valid_file_name)
            if isinstance(agent_meta.output_json_schema, str):
                output_label = OutputLabel(
                    label=agent_meta.output_json_schema,
                    custom_output_format_instructions=lm.get_custom_format_instructions_if_any(),
                )
                cog_agent = Model(
                    agent_name,
                    agent_meta.agent_prompt,
                    [
                        Input(name=input_name)
                        for input_name in agent_meta.input_variables
                    ],
                    output_label,
                )
            else:
                output_model = json_schema_to_pydantic_model(
                    agent_meta.output_json_schema, module_fpath
                )
                # if there's only one str output field for non-ending agents, remove output format
                if (
                    "END" not in agent_meta.next_action
                    and len(output_model.model_fields) == 1
                ):
                    output_model = list(output_model.model_fields.keys())[0]
                    output = OutputLabel(
                        label=agent_meta.output_json_schema,
                        custom_output_format_instructions=lm.get_custom_format_instructions_if_any(),
                    )
                    cog_agent = Model(
                        agent_name,
                        agent_meta.agent_prompt,
                        [
                            Input(name=input_name)
                            for input_name in agent_meta.input_variables
                        ],
                        output,
                    )
                else:
                    structured_lm: StructuredModel = lm
                    new_output_format: OutputFormat = OutputFormat(
                        schema=output_model,
                        should_hint_format_in_prompt=structured_lm.output_format.should_hint_format_in_prompt,
                        custom_output_format_instructions=structured_lm.output_format.custom_output_format_instructions,
                    )
                    cog_agent = StructuredModel(
                        agent_name,
                        agent_meta.agent_prompt,
                        [
                            Input(name=input_name)
                            for input_name in agent_meta.input_variables
                        ],
                        output_format=new_output_format,
                    )

            cog_agent.lm_config = (
                default_lm_config if default_lm_config else lm.lm_config
            )
            name_to_new_lm[agent_name] = cog_agent
            sub_graph.add_module(cog_agent)

        # Get static dependency edges
        agent_to_srcs = defaultdict(
            list, {agent_name: [input_name] for agent_name in new_agents.agents}
        )  # default to input node
        for agent_name, agent_meta in new_agents.agents.items():
            next_action = agent_meta.next_action
            is_static_edge = agent_meta.dynamic_action_decision == "None"
            if is_static_edge:
                for next_agent in next_action:
                    if next_agent == "END":
                        agent_to_srcs[logical_end_name].append(agent_name)
                    else:
                        agent_to_srcs[name_to_new_lm[next_agent].name].append(
                            agent_name
                        )  # for check name existance
        for agent_name, srcs in agent_to_srcs.items():
            sub_graph.add_edge(srcs, agent_name)

        # Add dynamic dependency edges
        def get_branch_function(dynamic_action_code: str, next_actions: List[str]):
            next_actions_str = ", ".join([f'"{na}"' for na in next_actions])
            new_func, func_name = add_argument_to_position(
                dynamic_action_code, "ctx", 0
            )
            template = f"""\
@hint_possible_destinations([{next_actions_str}])
{new_func}
"""
            clean_code = template.replace("END", f"{logical_end_name}")
            return clean_code, func_name

        for agent_name, agent_meta in new_agents.agents.items():
            next_action = agent_meta.next_action
            is_static_edge = agent_meta.dynamic_action_decision == "None"
            if not is_static_edge:
                # dynamic edge
                decision_code = agent_meta.dynamic_action_decision
                clean_decision_code, func_name = get_branch_function(
                    decision_code, next_action
                )
                func_obj = compile(clean_decision_code, "<string>", "exec")
                local_name_space = {}
                exec(
                    func_obj,
                    {"hint_possible_destinations": hint_possible_destinations},
                    local_name_space,
                )
                callable_code = local_name_space[func_name]
                sub_graph.add_branch(
                    f"condition_flow_after_{agent_name}",
                    agent_name,
                    callable_code,
                    clean_decision_code,
                    enhance_existing=True,
                )
        return sub_graph
