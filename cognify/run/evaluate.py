import os
import json
from typing import Optional, Union, Callable
import logging

from cognify.optimizer.control_param import ControlParameter
from cognify.optimizer.core import driver
from cognify.optimizer.evaluator import (
    EvaluatorPlugin,
    EvaluationResult,
    EvalTask,
)

from cognify._signal import _should_exit, _init_exit_gracefully

_init_exit_gracefully(msg="Stopping main", verbose=True)

logger = logging.getLogger(__name__)

def evaluate(
    *,
    config_id: str,
    test_set,
    opt_result_path: Optional[str] = None,
    workflow: Optional[str] = None,
    control_param: Optional[ControlParameter] = None,
    n_parallel: int = 10,
    eval_fn: Callable = None,
    eval_path: str = None,
    save_to: str = None,
) -> EvaluationResult:
    evaluator = EvaluatorPlugin(
        trainset=None,
        evalset=None,
        testset=test_set,
        evaluator_path=eval_path,
        evaluator_fn=eval_fn,
        n_parallel=n_parallel,
    )
    if config_id == "NoChange":
        if workflow is None:
            raise ValueError(
                "If evaluating the original workflow, path to the script should be provided"
            )
        eval_task = EvalTask(
            script_path=workflow,
            args=[],
            other_python_paths=[],
            all_params={},
            module_name_paths={},
            aggregated_proposals={},
            trace_back=["evaluate_raw"],
        )
        result = evaluator.get_score(mode="test", task=eval_task, show_process=True)
        print(f"----- Testing Raw Program -----")
        print(f"=========== Evaluation Results ===========")
        print(
            "  Quality: {:.3f}, Cost per 1K invocation ($): {:.2f} $".format(
                result.reduced_score, result.reduced_price * 1000
            )
        )
        print("===========================================")
        return result

    if not control_param and not opt_result_path:
        # If both are provided, control_param will be used
        raise ValueError("Either control_param or opt_result_path should be provided")

    if control_param is None:
        control_param_save_path = os.path.join(opt_result_path, "control_param.json")
        control_param = ControlParameter.from_json_profile(control_param_save_path)
        
    # get dry run result on train set
    dry_run_log_path = os.path.join(
        control_param.opt_history_log_dir, "dry_run_train.json"
    )

    quality_constraint = None
    base_quality = None
    base_cost = None
    base_exec_time = None
    if os.path.exists(dry_run_log_path):
        with open(dry_run_log_path, "r") as f:
            dry_run_result = EvaluationResult.from_dict(json.load(f))
        logger.info(f"Loading existing dry run result at {dry_run_log_path}")
        if control_param.quality_constraint:
            quality_constraint = (
                control_param.quality_constraint * dry_run_result.reduced_score
            )
            base_quality = dry_run_result.reduced_score
        base_cost = dry_run_result.reduced_price
        base_exec_time = dry_run_result.reduced_exec_time
    else:
        logger.warning(
            f"Quality constraint is set but no dry run result found at {dry_run_log_path}, will ignore constraint"
        )

    opt_driver = driver.MultiLayerOptimizationDriver(
        layer_configs=control_param.opt_layer_configs,
        opt_log_dir=control_param.opt_history_log_dir,
        quality_constraint=quality_constraint,
        base_quality=base_quality,
        base_cost=base_cost,
        base_exec_time=base_exec_time
    )
    result = opt_driver.evaluate(
        evaluator=evaluator,
        config_id=config_id,
    )

    if save_to is not None:
        with open(save_to, "w") as f:
            json.dump(result.to_dict(), f, indent=4)
    return result


def load_workflow(
    *,
    config_id: str,
    opt_result_path: Optional[str] = None,
    control_param: Optional[ControlParameter] = None,
) -> Callable:
    assert (
        control_param or opt_result_path
    ), "Either control_param or opt_result_path should be provided"
    # If both are provided, control_param will be used

    if control_param is None:
        control_param_save_path = os.path.join(opt_result_path, "control_param.json")
        control_param = ControlParameter.from_json_profile(control_param_save_path)

    opt_driver = driver.MultiLayerOptimizationDriver(
        layer_configs=control_param.opt_layer_configs,
        opt_log_dir=control_param.opt_history_log_dir,
    )
    schema, _ = opt_driver.load(config_id)
    return schema.program
