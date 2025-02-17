import os
import json
from typing import Union, Optional
import logging
import re

from cognify.optimizer.evaluator import (
    EvaluationResult,
    EvaluatorPlugin,
    EvalTask,
)
from cognify.optimizer.core.flow import TrialLog, TopDownInformation, LayerConfig
from cognify.optimizer.core.opt_layer import OptLayer, get_pareto_front, _log_optimization_results
from cognify.optimizer.control_param import SelectedObjectives
from cognify.optimizer.core.upper_layer import UpperLevelOptimization, LayerEvaluator
from cognify.optimizer.utils import _report_cost_reduction, _report_quality_impv, _report_latency_reduction
from cognify._tracing import trace_quality_improvement, trace_cost_improvement, trace_latency_improvement
from cognify.optimizer.core.common_stats import CommonStats

logger = logging.getLogger(__name__)

def get_layer_evaluator_factory(
    next_layer_factory, 
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
        CommonStats.base_price = base_cost
        CommonStats.base_exec_time = base_exec_time

        # initialize optimization layers
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
        self.opt_layer_factories[-1] = lambda: evaluator
        
        for ri, layer_config in enumerate(reversed(self.layer_configs)):
            idx = len(self.layer_configs) - ri - 1
            next_layer_factory = self.opt_layer_factories[idx + 1]
            current_layer_factory = get_layer_evaluator_factory(next_layer_factory, layer_config, idx, ri == 0)
            self.opt_layer_factories[idx] = current_layer_factory

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
        pareto_frontier = top_layer.easy_optimize(
            script_path=script_path,
            script_args=script_args,
            other_python_paths=other_python_paths,
        )
        logger.info("----------------- Optimization Finished -----------------")
        self.dump_frontier_details(pareto_frontier)
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

    def _find_config_log_path(self, trial_id: str) -> str:
        opt_config = self.layer_configs[0].opt_config
        opt_config.finalize()
        tdi = TopDownInformation(
            opt_config=opt_config,
            all_params=None,
            module_ttrace=None,
            current_module_pool=None,
            script_path=None,
            script_args=None,
            other_python_paths=None,
        )

        top_layer = self.opt_layers[0]
        top_layer.load_opt_log(opt_config.opt_log_path)
        top_layer.top_down_info = tdi
        all_configs = top_layer.get_all_candidates()
        config_path = None

        for opt_log, path in all_configs:
            if opt_log.id == trial_id:
                config_path = path
                break
        else:
            raise ValueError(f"Config {trial_id} not found in the optimization log.")
        return config_path

    def evaluate(
        self,
        evaluator: EvaluatorPlugin,
        config_id: str,
    ) -> EvaluationResult:
        opt_logs = self._load_from_file()
        trial_id = self._extract_trial_id(config_id)
        trial_log = opt_logs[trial_id]
        
        # apply selected trial
        print(f"----- Testing select trial {trial_id} -----")
        # print("  Training Quality: {:.3f}, Cost per 1K invocation: ${:.2f}\n".format(trial_log.score, trial_log.price * 1000))
        print("  Params: {}".format(trial_log.params))
        
        eval_task = EvalTask.from_dict(trial_log.eval_task_dict)
        # run evaluation
        eval_result = evaluator.get_score(mode='test', task=eval_task, show_progress_bar=True)

        print(f"=========== Evaluation Results ===========")
        if CommonStats.base_quality is not None:
            print("  Quality improvement: {:.0f}%".format(_report_quality_impv(eval_result.reduced_score, CommonStats.base_quality)))
        if CommonStats.base_price is not None:
            print("  Cost is {:.2f}x original".format(_report_cost_reduction(eval_result.reduced_price, CommonStats.base_price)))
        if CommonStats.base_exec_time is not None:
            print("  Execution time is {:.2f}x original".format(_report_latency_reduction(eval_result.reduced_exec_time, CommonStats.base_exec_time)))
        print("  Quality: {:.2f}, Cost per 1K invocation: ${:.2f}, Execution time: {:.2f}s".format(eval_result.reduced_score, eval_result.reduced_price * 1000, eval_result.reduced_exec_time))
        print("===========================================")

        return eval_result


    def load(
        self,
        config_id: str,
    ):
        opt_logs = self._load_from_file()
        trial_id = self._extract_trial_id(config_id)
        trial_log = opt_logs[trial_id]

        eval_task = EvalTask.from_dict(trial_log.eval_task_dict)
        schema, old_name_2_new_module = eval_task.load_and_transform()
        return schema, old_name_2_new_module

    def inspect(self, dump_details: bool = False):
        opt_logs = self._load_from_file()

        cancidates = []
        for log_id, log in opt_logs.items():
            if not log.finished:
                continue
            # if not meet the quality constraint, skip
            if (
                CommonStats.quality_constraint is not None
                and log.score < CommonStats.quality_constraint
            ):
                continue
            cancidates.append(log)
        pareto_frontier = get_pareto_front(cancidates)
        _log_optimization_results(pareto_frontier)

        # dump frontier details to file
        if dump_details:
            self.dump_frontier_details(pareto_frontier)
        return

    def dump_frontier_details(self, frontier):
        param_log_dir = os.path.join(self.opt_log_dir, "optimized_workflow_details")
        if not os.path.exists(param_log_dir):
            os.makedirs(param_log_dir, exist_ok=True)
        for i, (trial_log, opt_path) in enumerate(frontier):
            trial_log: TrialLog
            dump_path = os.path.join(param_log_dir, f"Optimization_{i+1}.cog")
            trans = EvalTask.from_dict(trial_log.eval_task_dict).show_opt_trace()
            details = f"Trial - {trial_log.id}\n"
            details += f"Optimized for {str(CommonStats.objectives)}\n"
            details += f"Log at: {opt_path}\n"
            if CommonStats.base_quality is not None:
                quality_improvement = _report_quality_impv(trial_log.score, CommonStats.base_quality)
                details += ("  Quality improves by {:.0f}%\n".format(quality_improvement))
                trace_quality_improvement(quality_improvement)
            if CommonStats.base_price is not None:
                cost_improvement = _report_cost_reduction(trial_log.price, CommonStats.base_price)
                details += ("  Cost is {:.2f}x original".format(cost_improvement))
                trace_cost_improvement(cost_improvement)
            if CommonStats.base_exec_time is not None:
                exec_time_improvement = _report_latency_reduction(trial_log.exec_time, CommonStats.base_exec_time)
                details += ("  Execution time is {:.2f}x original".format(exec_time_improvement))
                trace_latency_improvement(exec_time_improvement)
            details += f"Quality: {trial_log.score:.3f}, Cost per 1K invocation: ${trial_log.price * 1000:.2f}, Execution time: {trial_log.exec_time:.2f}s \n"
            details += trans
            with open(dump_path, "w") as f:
                f.write(details)

    
    def _load_from_file(self) -> dict[str, TrialLog]:
        """Recursively load all optimization logs
        """
        root_log = self.layer_configs[0].opt_config
        root_log.finalize()
        _log_dir_stack = [root_log.log_dir]
        leaf_layer_name = self.layer_configs[-1].layer_name
        opt_logs = {}
        
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
                    
                    # load the log
                    opt_logs[log_id] = TrialLog.from_dict(log)
        return opt_logs
                