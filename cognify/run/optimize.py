import os
import json
from typing import Union, Callable
import logging
import shutil

from cognify.optimizer.trace.checkpoint import TrialLog
from cognify._signal import _should_exit, _init_exit_gracefully
from cognify.optimizer.control_param import ControlParameter
from cognify.optimizer.core import driver
from cognify.optimizer.evaluator import (
    EvaluatorPlugin,
    EvalTask,
    EvaluationResult,
)


logger = logging.getLogger(__name__)

_init_exit_gracefully(msg="Stopping main", verbose=True)


def dry_run(script_path, evaluator: EvaluatorPlugin, log_dir):
    eval_task = EvalTask(
        script_path=script_path,
        args=[],
        other_python_paths=[],
        all_params={},
        module_name_paths={},
        aggregated_proposals={},
        trace_back=["dry_run"],
    )
    print("Dry run with the original workflow...")
    logger.info(
        f"Dry run on train set: {len(evaluator.dataset['train'])} samples for optimizer analysis"
    )
    dry_run_log_path = os.path.join(log_dir, "dry_run_train.json")

    if os.path.exists(dry_run_log_path):
        with open(dry_run_log_path, "r") as f:
            dry_run_result = EvaluationResult.from_dict(json.load(f))
        logger.info(f"Loading existing dry run result at {dry_run_log_path}")
        return dry_run_result

    result = evaluator.get_score("train", eval_task, show_progress_bar=True, is_dry_run=True)
    if result.complete:
        with open(dry_run_log_path, "w+") as f:
            json.dump(result.to_dict(), f, indent=4)
        logger.info(f"Dry run result saved to {dry_run_log_path}")
    else:
        raise ValueError("Dry run not completed, result will be discarded")
    return result


def downsample_data(script_path, source: EvaluatorPlugin, mode, sample_size, log_dir):
    plain_task = EvalTask(
        script_path=script_path,
        args=[],
        other_python_paths=[],
        all_params={},
        module_name_paths={},
        aggregated_proposals={},
    )
    source.down_sample(
        sample_size=sample_size,
        mode=mode,
        task=plain_task,
        sample_mode="difficulty",
        log_dir=log_dir,
    )


def optimize(
    script_path: str,
    control_param: ControlParameter,
    train_set,
    *,
    eval_fn: Callable = None,
    eval_path: str = None,
    val_set=None,
    resume: bool = False,
    force: bool = False,
) -> list[TrialLog]:
    # Validate and prepare the pipeline
    assert (
        eval_fn is not None or eval_path is not None
    ), "Either eval_fn or eval_path should be provided"

    # if both provided, use eval_path
    if eval_path is not None:
        eval_fn = None

    # create directory for logging
    if not os.path.exists(control_param.opt_history_log_dir):
        os.makedirs(control_param.opt_history_log_dir, exist_ok=True)

    # if exist opt logs, check if reuse or force
    top_layer_opt_dir = os.path.join(
        control_param.opt_history_log_dir, control_param.opt_layer_configs[0].layer_name
    )
    if os.path.isdir(top_layer_opt_dir) and len(os.listdir(top_layer_opt_dir)) > 0:
        if not resume and not force:
            raise ValueError(
                f"Directory {control_param.opt_history_log_dir} is not empty, if you want to resume/overwrite previous checkpoint, please set -r/-f flag in cli or pass resume=True/force=True in function call"
            )
    if force:
        # clear the directory
        for f in os.listdir(control_param.opt_history_log_dir):
            file_path = os.path.join(control_param.opt_history_log_dir, f)
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

    # dump control params
    param_log_path = os.path.join(
        control_param.opt_history_log_dir, "control_param.json"
    )
    with open(param_log_path, "w") as f:
        json.dump(control_param.to_dict(), f, indent=4)

    # create evaluator
    evaluator = EvaluatorPlugin(
        trainset=train_set,
        evalset=val_set,
        testset=None,
        evaluator_path=eval_path,
        evaluator_fn=eval_fn,
        n_parallel=control_param.evaluator_batch_size,
    )

    # dry run on train set
    raw_result = dry_run(
        script_path=script_path,
        evaluator=evaluator,
        log_dir=control_param.opt_history_log_dir,
    )

    if _should_exit():
        return None, None, None

    # downsample data
    if control_param.train_down_sample > 0:
        downsample_data(
            script_path=script_path,
            source=evaluator,
            mode="train",
            sample_size=control_param.train_down_sample,
            log_dir=control_param.opt_history_log_dir,
        )
    if _should_exit():
        return None, None, None

    if control_param.val_down_sample > 0:
        downsample_data(
            script_path=script_path,
            source=evaluator,
            mode="eval",
            sample_size=control_param.val_down_sample,
            log_dir=control_param.opt_history_log_dir,
        )
    if _should_exit():
        return None, None, None

    # build optimizer from parameters
    opt_driver = driver.MultiLayerOptimizationDriver(
        layer_configs=control_param.opt_layer_configs,
        objectives=control_param.objectives,
        opt_log_dir=control_param.opt_history_log_dir,
        quality_constraint=control_param.quality_constraint * raw_result.reduced_score,
        base_quality=raw_result.reduced_score,
        base_cost=raw_result.reduced_price,
        base_exec_time=raw_result.reduced_exec_time
    )

    pareto_frontier = opt_driver.run(
        evaluator=evaluator,
        script_path=script_path,
    )
    return pareto_frontier
