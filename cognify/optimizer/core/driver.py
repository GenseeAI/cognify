import os
import json
from typing import Union, Optional, Callable
import logging
import re

from cognify.optimizer.evaluator import (
    EvaluationResult,
    EvaluatorPlugin,
    EvalTask,
    GeneralEvaluatorInterface,
)
from cognify.optimizer.core.flow import LayerConfig
from cognify.optimizer.core.opt_layer import OptLayer
from cognify.optimizer.control_param import SelectedObjectives
from cognify.optimizer.trace.checkpoint import LogManager, TrialLog, dump_config_result
from cognify.optimizer.trace.progress_info import pbar
from cognify.optimizer.utils import _report_cost_reduction, _report_quality_impv, _report_latency_reduction
from cognify._tracing import trace_quality_improvement, trace_cost_improvement, trace_latency_improvement
from cognify.optimizer.core.common_stats import CommonStats

logger = logging.getLogger(__name__)

def get_layer_evaluator_factory(
    next_layer_factory: Callable[[], GeneralEvaluatorInterface], 
    layer_config: LayerConfig,
    level: int,
    is_leaf: bool
):
    """
    Factory for creating the "evaluator"s for each optimization layer
    
    Any upper-layer will use the next-layer's optimizer as the evaluator
    The leaf layer will use the evaluator plugin to run the actual evaluation
    """
    def _factory():
        return OptLayer(
            name=layer_config.layer_name,
            opt_config=layer_config.opt_config,
            hierarchy_level=level,
            is_leaf=is_leaf,
            next_layer_factory=next_layer_factory,
            dedicate_params=layer_config.dedicate_params,
            universal_params=layer_config.universal_params,
            target_modules=layer_config.target_modules,
        )
    return _factory

class MultiLayerOptimizationDriver:
    def __init__(
        self,
        layer_configs: list[LayerConfig],
        opt_log_dir: str,
        objectives: SelectedObjectives,
        quality_constraint: float = None,
        base_quality: float = None,
        base_cost: float = None,
        base_exec_time: float = None,
    ):
        """Driver for multi-layer optimization

        Args:
            layer_configs (Sequence[LayerConfig]): configs for each optimization layer

        NOTE: the order of the layers is from top to bottom, i.e., the last layer will run program evaluation directly while others will run layer evaluation
        """
        self.layer_configs = layer_configs
        CommonStats.objectives = objectives
        CommonStats.quality_constraint = quality_constraint
        CommonStats.base_quality = base_quality
        CommonStats.base_cost = base_cost
        CommonStats.base_exec_time = base_exec_time
        _log_mng = LogManager(opt_log_dir)

        # initialize optimization layers
        # NOTE: also included the evaluator in this list
        self.opt_layer_factories: list[callable] = [None] * (len(layer_configs) + 1)

        self.opt_log_dir = opt_log_dir

        # config log dir for layer opts
        # NOTE: only the top layer will be set, others are decided at runtime
        self.layer_configs[0].opt_config.log_dir = os.path.join(
            opt_log_dir, self.layer_configs[0].layer_name
        )
        for layer_config in self.layer_configs[1:]:
            layer_config.opt_config.log_dir = None
            layer_config.opt_config.opt_log_path = None
            layer_config.opt_config.param_save_path = None

    def build_tiered_optimization(self, evaluator: EvaluatorPlugin):
        """Build tiered optimization from bottom to top"""
        
        # Add the evaluator as the special next-level for the innermost optimization layer
        self.opt_layer_factories[-1] = lambda: evaluator
        
        # Build all optimization layers
        for ri, layer_config in enumerate(reversed(self.layer_configs)):
            idx = len(self.layer_configs) - ri - 1
            self.check_update_layer_config(idx)
            next_layer_factory = self.opt_layer_factories[idx + 1]
            current_layer_factory = get_layer_evaluator_factory(next_layer_factory, layer_config, idx, ri == 0)
            self.opt_layer_factories[idx] = current_layer_factory
    
    def check_update_layer_config(self, idx: int):
        """Finalize settings for advanced allocation strategies
        
        1. Reset mode to base for leaf layer
        2. Set budget parameter according to the next layer's budget
        """
        opt_config = self.layer_configs[idx].opt_config
        if idx == len(self.layer_configs) - 1:
            if opt_config.alloc_strategy.mode != "base":
                logger.info(f"You are using {opt_config.alloc_strategy.mode} allocation strategy for the last layer, which will be ignored")
                opt_config.alloc_strategy.mode = "base"
        else:
            # calculate the initial step budget for SSH strategy if set
            opt_config.alloc_strategy.set_initial_step_budget(self.layer_configs[idx + 1].opt_config.n_trials)

    def run(
        self,
        evaluator: EvaluatorPlugin,
        script_path: str,
        script_args: Optional[list[str]] = None,
        other_python_paths: Optional[list[str]] = None,
    ) -> tuple[float, list[tuple[TrialLog, str]], dict[str, TrialLog]]:
        self.build_tiered_optimization(evaluator)
        logger.info("----------------- Start Optimization -----------------")
        top_layer = self.opt_layer_factories[0]()
        _, pareto_frontier, finished_trials = top_layer.optimization_entry(
            script_path=script_path,
            script_args=script_args,
            other_python_paths=other_python_paths,
        )
        logger.info("----------------- Optimization Finished -----------------")
        self.dump_frontier_details(pareto_frontier, finished_trials, trace_impv=True)
        return pareto_frontier

    def _extract_trial_id(self, config_id: str) -> str:
        param_log_dir = os.path.join(self.opt_log_dir, "optimized_workflow_details")
        if not os.path.exists(param_log_dir):
            raise ValueError(
                f"Cannot find the optimization log directory at {param_log_dir}"
            )

        with open(os.path.join(param_log_dir, f"{config_id}.cog"), "r") as f:
            first_line = f.readline().strip()
        match = re.search(r"Trial - (.+)", first_line)
        if match:
            trial_id = match.group(1)
            return trial_id
        else:
            raise ValueError(
                f"Cannot extract trial id from the log file {config_id}.cog"
            )

    def evaluate(
        self,
        evaluator: EvaluatorPlugin,
        config_id: str,
    ) -> EvaluationResult:
        self._load_from_file()
        trial_id = self._extract_trial_id(config_id)
        trial_log = LogManager().get_log_by_id(trial_id)
        
        # apply selected trial
        print(f"----- Testing select trial {trial_id} -----")
        # print("  Training Quality: {:.3f}, Cost per 1K invocation: ${:.2f}\n".format(trial_log.score, trial_log.price * 1000))
        print("  Params: {}".format(trial_log.params))
        
        eval_task = EvalTask.from_dict(trial_log.eval_task_dict)
        # run evaluation
        eval_result = evaluator.get_score(mode='test', task=eval_task, show_progress_bar=True)

        print(f"=========== Evaluation Results ===========")
        print(dump_config_result(eval_result))
        print("===========================================")

        return eval_result


    def load(
        self,
        config_id: str,
    ):
        self._load_from_file()
        trial_id = self._extract_trial_id(config_id)
        trial_log = LogManager().get_log_by_id(trial_id)

        eval_task = EvalTask.from_dict(trial_log.eval_task_dict)
        schema, _ = eval_task.load_and_transform()
        return schema

    def inspect(self, dump_details: bool = False):
        self._load_from_file()

        _, pareto_frontier, finished_trials = LogManager().get_global_summary(verbose=True)
        # dump frontier details to file
        if dump_details:
            self.dump_frontier_details(pareto_frontier, finished_trials, trace_impv=False)
        return

    def dump_frontier_details(self, frontier: list[TrialLog], finished_trials: dict[str, tuple[TrialLog, str]], trace_impv: bool):
        param_log_dir = os.path.join(self.opt_log_dir, "optimized_workflow_details")
        if not os.path.exists(param_log_dir):
            os.makedirs(param_log_dir, exist_ok=True)
        for i, trial_log in enumerate(frontier):
            trial_log: TrialLog
            dump_path = os.path.join(param_log_dir, f"Optimization_{i+1}.cog")
            cog_transformations = EvalTask.from_dict(trial_log.eval_task_dict).show_opt_trace()
            details = f"Trial - {trial_log.id}\n"
            log_path = finished_trials[trial_log.id][1]
            details += f"Log at: {log_path}\n"
            details += f"Optimized for {str(CommonStats.objectives)}\n"
            details += dump_config_result(trial_log.result, trace_impv)
            details += cog_transformations
            with open(dump_path, "w") as f:
                f.write(details)
    
    def _load_from_file(self):
        """Recursively load all optimization logs
        """
        root_log = self.layer_configs[0].opt_config
        _log_dir_stack = [root_log.log_dir]
        leaf_layer_name = self.layer_configs[-1].layer_name
        
        while _log_dir_stack:
            log_dir = _log_dir_stack.pop()
            opt_log_path = os.path.join(log_dir, "opt_logs.json")
            
            if not os.path.exists(opt_log_path):
                continue
            with open(opt_log_path, "r") as f:
                opt_trace = json.load(f)
                
                for log_id, log in opt_trace.items():
                    layer_instance = log_id.rsplit("_", 1)[0]
                    layer_name = log["layer_name"]
                    trial_number = log_id.rsplit("_", 1)[-1]
                    sub_layer_log_dir = os.path.join(log_dir, f"{layer_name}_trial_{trial_number}")
                    _log_dir_stack.append(sub_layer_log_dir)
                    
                LogManager().register_layer(
                    layer_name=layer_name,
                    layer_instance=layer_instance,
                    opt_log_path=opt_log_path,
                    is_leaf=layer_name == leaf_layer_name,
                )
                LogManager().load_existing_logs(layer_instance)