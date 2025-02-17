from cognify.optimizer.control_param import SelectedObjectives

class CommonStats:
    """Global statistics for optimizers in all layers
    """
    eval_time_out: int = 180
    quality_constraint: float = None
    base_quality: float = None
    base_price: float = None
    base_exec_time: float = None
    objectives: SelectedObjectives = None   