from typing import Callable
import logging
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)
from concurrent.futures import Future

from cognify.optimizer.evaluator import (
    EvaluationResult,
    GeneralEvaluatorInterface,
)
from cognify.optimizer.core.common_stats import CommonStats
from cognify.optimizer.core.flow import TopDownInformation

logger = logging.getLogger(__name__)

class SuccessiveHalving:
    """Policy for successive halving resource allocation
    
    If a layer (A) adopts this policy, it will allocate resources for the next layer (B)
    
    layer A will first propose `n` trials, then
    
    **SH routine**:
    1. each remaining trial at layer B will have a fixed step_budget = `r_i`
    2. after all trials are evaluated, unpromising trials will be pruned (top `n = n/prune_rate` will be kept)
    3. repeat from step 1 until no trials left or iteration limit reached
    """
    def __init__(
        self,
        prune_rate: int,
        num_SH_iter: int,
        initial_step_budget: int,
        hierarchy_level: int,
        next_layer_factory: Callable[[], GeneralEvaluatorInterface],
        selected_runs: list[TopDownInformation],
    ):
        self.prune_rate = prune_rate
        self.num_SH_iter = num_SH_iter
        self.initial_step_budget = initial_step_budget
        self.hierarchy_level = hierarchy_level
        self.selected_runs: list[TopDownInformation] = selected_runs
        self.can_continue = [i for i in range(len(selected_runs))]
        self.num_inner_trials = [0] * len(selected_runs)
        self._next_layer_factory = next_layer_factory
    
    def _evaluate_one_config(self, i) -> EvaluationResult:
        _next_layer = self._next_layer_factory()
        # NOTE: frac not needed bc next layer will never be evaluator
        result = _next_layer.evaluate(self.selected_runs[i])
        return result

    def _halving_jointly(self, indicators, not_converged, n_left):
        """Round-Robin in each dimension
        """
        # NOTE: indicators all smaller the better
        groups_by_metric = list(map(list, zip(*indicators)))
        sorted_indices = [sorted(not_converged, key=lambda x: group[x]) for group in groups_by_metric]
        # round robin select from top of each list
        # avoid duplicates
        left_to_run = set()
        from_group = 0
        while len(left_to_run) < n_left:
            # if all indices are empty, break
            if not any(len(sorted_indices[i]) > 0 for i in range(len(groups_by_metric))):
                break
            if len(sorted_indices[from_group]) > 0:
                left_to_run.add(sorted_indices[from_group].pop(0))
            from_group = (from_group + 1) % len(groups_by_metric)
        self.can_continue = [self.can_continue[i] for i in left_to_run]
    
    def run_and_prune(self):
        for sh_iter in range(self.num_SH_iter):
            if len(self.can_continue) == 0:
                break
            
            # print(f"SH next with {self.can_continue}, budget: {self.initial_step_budget * self.prune_rate ** sh_iter}")
            for j in self.can_continue:
                self.num_inner_trials[j] += int(self.initial_step_budget * self.prune_rate ** sh_iter)
                self.selected_runs[j].opt_config.n_trials = int(self.initial_step_budget * self.prune_rate ** sh_iter)
                
            futures: list[Future] = []
            with ThreadPoolExecutor(max_workers=len(self.can_continue)) as executor:
                for j in self.can_continue:
                    futures.append(executor.submit(
                        self._evaluate_one_config, j
                    ))
                    
                try:
                    outer_indicators = []
                    not_converged = []
                    for idx, f in enumerate(futures):
                        eval_result: EvaluationResult = f.result()
                        # If next layer converged, even for promising ones, we will not continue them
                        if not eval_result.converged:
                            not_converged.append(idx)
                        outer_indicators.append(CommonStats.objectives.select_from(-eval_result.reduced_score, eval_result.reduced_price, eval_result.reduced_exec_time))
                except Exception as e:
                    logger.error(f"Error in SH: {e}")
                    raise
            
            # not consider converged ones
            n =  int(len(self.selected_runs) * self.prune_rate ** -(sh_iter+1))
            self._halving_jointly(outer_indicators, not_converged, n)
    
    def execute(self) -> tuple[list[EvaluationResult], list[int]]:
        self.run_and_prune()
        # Collect inner loop performance
        outer_run_evals = []
        # get next layer results without any trials
        for i in range(len(self.selected_runs)):
            self.selected_runs[i].opt_config.n_trials = 0
            _next_layer = self._next_layer_factory()
            outer_run_evals.append(_next_layer.evaluate(self.selected_runs[i]))
        return outer_run_evals, self.num_inner_trials