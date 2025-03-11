import os
from typing import Optional, Callable
import logging

from cognify.optimizer.control_param import ControlParameter
from cognify.optimizer.core import driver
from cognify.optimizer.plugin import OptimizerSchema

from cognify._signal import _should_exit, _init_exit_gracefully

_init_exit_gracefully(msg="Stopping main", verbose=True)

logger = logging.getLogger(__name__)

def run(
    *,
    config_id: str,
    workflow: str,
    input: str,
    control_param: Optional[ControlParameter] = None,
):
    if config_id == 'Original':
        program = OptimizerSchema.capture(workflow).program
        print("Running the provided input on original workflow")
    else:
        program = load_workflow(config_id=config_id, control_param=control_param)
        print("Running the provided input on optimized workflow")
    result = program(input)
    print(f"Output: {list(result.values())[0]}")


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
        objectives=control_param.objectives,
    )
    schema = opt_driver.load(config_id)
    return schema.program