from cognify.optimizer.control_param import SelectedObjectives

class CommonStats:
    """Global statistics for optimizers in all layers
    """
    quality_constraint: float = None
    base_quality: float = None
    base_cost: float = None
    base_exec_time: float = None
    objectives: SelectedObjectives = None 