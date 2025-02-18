import os
import uuid
import copy
import json
from typing import Any
import logging
import numpy as np
import threading

from cognify.optimizer.utils import _report_cost_reduction, _report_quality_impv, _report_latency_reduction
from cognify.optimizer.core.common_stats import CommonStats
from cognify.optimizer.trace.progress_info import pbar
from cognify.optimizer.evaluator import EvalTask, EvaluationResult

logger = logging.getLogger(__name__)

class TrialLog:
    """Log of a proposed configuration
    """
    def __init__(
        self,
        id: str,
        layer_name: str,
        params: dict[str, any],
        result: EvaluationResult = None,
        eval_task_dict: dict = None,
    ):
        self.layer_name = layer_name
        self.id: str = id or uuid.uuid4().hex
        self.params = params
        self.eval_task_dict = eval_task_dict
        if result is None:
            result = EvaluationResult(
                ids=[],
                scores=[],
                prices=[],
                exec_times=[],
                total_eval_cost=0,
                complete=False,
                reduced_price=0,
                reduced_score=0,
                reduced_exec_time=0,
            )
        self.result = result

    def to_dict(self):
        return {
            "layer_name": self.layer_name,
            "id": self.id,
            "params": self.params,
            "result": self.result.to_dict(),
            "eval_task": self.eval_task_dict,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            layer_name=data["layer_name"],
            id=data["id"],
            params=data["params"],
            result=EvaluationResult.from_dict(data["result"]),
            eval_task_dict=data.get("eval_task"),
        )
        
    def show_transformation(self) -> str:
        eval_task = EvalTask.from_dict(self.eval_task_dict)
        return eval_task.show_opt_trace()

def get_pareto_front(candidates: list[TrialLog]) -> list[TrialLog]:
    """Find the pareto-efficient points

    NOTE: Filter the candidate list before calling this function
    """
    if not candidates:
        return []
    score_cost_list = []
    for trial_log in candidates:
        score_cost_list.append(
            CommonStats.objectives.select_from(
                -trial_log.result.reduced_score, 
                trial_log.result.reduced_price, 
                trial_log.result.reduced_exec_time
            )
        )

    vectors = np.array(list(map(list, zip(*score_cost_list))))
    is_efficient = np.ones(vectors.shape[0], dtype=bool)
    for i, v in enumerate(vectors):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(
                vectors[is_efficient] < v, axis=1
            )  # Keep any point with a lower cost
            is_efficient[i] = True  # And keep self

    # return filtered [T_ParetoProgram]
    pareto_frontier = [
        log for log, eff in zip(candidates, is_efficient) if eff
    ]
    return pareto_frontier
    
class LayerStat:
    def __init__(
        self, 
        layer_name: str, 
        instance_id: str, 
        opt_log_path: str, 
        is_leaf: bool
    ):
        self.layer_name = layer_name
        # instance_id identifies the BO instance at each layer
        # by different upper-layer configs
        self.instance_id = instance_id
        self.opt_logs: dict[str, TrialLog] = {}
        self.best_score = None
        self.lowest_cost = None
        self.fastest_exec_time = None
        self.opt_cost = 0.0
        self.opt_log_path = opt_log_path
        self.is_leaf = is_leaf
        # if the logs already loaded from a file
        self._loaded = False
    
    def add_trial(self, params) -> str:
        # NOTE: hold study lock when calling this function
        id = self._inc_log_id()
        log = TrialLog(layer_name=self.layer_name, params=params, id=id)
        self.opt_logs[log.id] = log
        return id
    
    def _inc_log_id(self):
        max_id = None
        for log in self.opt_logs.values():
            id = int(log.id.split("_")[-1])
            max_id = id if max_id is None else max(max_id, id)
        new_trial_number = max_id + 1 if max_id is not None else 0
        return f"{self.instance_id}_{new_trial_number}"
    
    def report_result(self, trial_id: str, result: EvaluationResult):
        """Attach result to the config
        """
        # register result to log
        self.opt_logs[trial_id].result = result
        self.opt_cost += result.total_eval_cost
        
        # update best metrics - local to each layer
        score, price, exec_time = result.reduced_score, result.reduced_price, result.reduced_exec_time
        if self.best_score is None or score > self.best_score:
            self.best_score = score
        if self.lowest_cost is None or price < self.lowest_cost:
            self.lowest_cost = price
        if self.fastest_exec_time is None or exec_time < self.fastest_exec_time:
            self.fastest_exec_time = exec_time

        logger.debug(
            f"- {self.instance_id} - Trial id {trial_id} result: score= {score:.2f}, cost@1000= ${price*1000:.3f}, exec_time= {result.reduced_exec_time:.2f}s"
        )
    
    def load_existing_logs(self) -> dict[str, TrialLog]:
        """Load optimization trace for a BO instance
        """
        with open(self.opt_log_path, "r") as f:
            opt_trace = json.load(f)
        for trial_log_id, trial_meta in opt_trace.items():
            trial_log = TrialLog.from_dict(trial_meta)
            self.opt_logs[trial_log_id] = trial_log
            self.opt_cost += trial_log.result.total_eval_cost
            
        cancidates = self.get_all_candidates()
        if cancidates:
            self.best_score = max([log.result.reduced_score for log in cancidates])
            self.lowest_cost = min([log.result.reduced_price for log in cancidates])
            self.fastest_exec_time = min([log.result.reduced_exec_time for log in cancidates])
        self._loaded = True
        return self.opt_logs
    
    def save_opt_logs(self):
        """Save the optimization trace to a json file
        """
        opt_logs_json_obj = {}
        for k, v in self.opt_logs.items():
            if v.result and v.result.complete:
                opt_logs_json_obj[k] = v.to_dict()
        if not opt_logs_json_obj:
            logger.warning("No finished trials to save")
            return
        json.dump(opt_logs_json_obj, open(self.opt_log_path, "w+"), indent=4)
    
    def get_opt_summary(self):
        """Get the pareto frontier of the layer
        """
        candidates = self.get_all_candidates()
        pareto_frontier = get_pareto_front(candidates)
        return pareto_frontier

    @property
    def all_finished(self):
        return all([log.result and log.result.complete for log in self.opt_logs.values()])
    
    def get_completed_logs(self) -> list[TrialLog]:
        return [log for log in self.opt_logs.values() if log.result and log.result.complete]
    
    def get_all_candidates(self) -> list[TrialLog]:
        """Get logs that are qualified
        """
        cancidates = []
        for log_id, log in self.opt_logs.items():
            if not log.result or not log.result.complete:
                continue
            # if not meet the quality constraint, skip
            if (
                CommonStats.quality_constraint is not None
                and log.result.reduced_score < CommonStats.quality_constraint
            ):
                continue
            cancidates.append(log)
        return cancidates

    @property
    def best_metrics_so_far(self):
        return self.best_score, self.lowest_cost, self.fastest_exec_time
    
class LogManager:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(LogManager, cls).__new__(cls)
            cls._instance._init(*args, **kwargs)
        return cls._instance
    
    def _init(self, log_dir):
        # path to store the history of all real-evaluations
        self.opt_trace_log_path = os.path.join(log_dir, "opt_trace.json")
        # by-layer configuration logs
        self.layer_stats: dict[str, LayerStat] = {}
        # global best metrics
        self._glob_best_score = CommonStats.base_quality
        self._glob_lowest_cost = CommonStats.base_price
        self._glob_fastest_exec_time = CommonStats.base_exec_time
        self._glob_lock = threading.Lock()
        self._opt_trace = []
        # total cost of optimization
        self.total_opt_cost = 0.0
    
    def register_layer(
        self, 
        layer_name: str,
        layer_instance: str, 
        opt_log_path: str, 
        is_leaf: bool
    ):
        if layer_instance in self.layer_stats:
            # raise ValueError(f"Layer {layer_instance} already registered at LogManager")
            return
        self.layer_stats[layer_instance] = LayerStat(layer_name, layer_instance, opt_log_path, is_leaf)
    
    def remove_layer(
        self,
        layer_instance: str
    ):
        del self.layer_stats[layer_instance]
    
    def add_trial(self, layer_instance: str, params) -> str:
        """Add a proposed configuration to the layer log
        """
        return self.layer_stats[layer_instance].add_trial(params)

    def num_trials(self, layer_instance: str, finished: bool = True):
        return len(self.layer_stats[layer_instance].get_completed_logs()) if finished \
                else len(self.layer_stats[layer_instance].opt_logs)
    
    def report_trial_result(self, layer_instance: str, id: str, result: EvaluationResult):
        """Report the evaluation result of a configuration
        """
        if result is None or not result.complete:
            return
        self.layer_stats[layer_instance].report_result(id, result)
        if self.layer_stats[layer_instance].is_leaf:
            with self._glob_lock:
                # update global
                self._update_glob_best_score(result.reduced_score)
                self._update_glob_lowest_cost(result.reduced_price)
                self._update_glob_fastest_exec_time(result.reduced_exec_time)
                self.total_opt_cost += result.total_eval_cost
                
                # save the evaluation trace
                _result_cpy = copy.deepcopy(result)
                _result_cpy.meta = {
                    'best_score': self._glob_best_score, 
                    'lowest_cost': self._glob_lowest_cost, 
                    'fastest_exec_time': self._glob_fastest_exec_time
                }
                self._opt_trace.append(_result_cpy.to_dict())
                
                # update progress bar
                pbar.update_status(
                    self._glob_best_score, 
                    self._glob_lowest_cost, 
                    self._glob_fastest_exec_time, 
                    self.total_opt_cost
                )
                
    
    def load_existing_logs(self, layer_instance: str) -> dict[str, TrialLog]:
        """Load all history optimization logs for a BO instance
        """
        if self.layer_stats[layer_instance]._loaded:
            return self.layer_stats[layer_instance].opt_logs
        
        logs = self.layer_stats[layer_instance].load_existing_logs()
        # update global best score and lowest cost
        local_best_score = self.layer_stats[layer_instance].best_score
        local_lowest_cost = self.layer_stats[layer_instance].lowest_cost
        fast_exec_time = self.layer_stats[layer_instance].fastest_exec_time
        self._update_glob_best_score(local_best_score)
        self._update_glob_lowest_cost(local_lowest_cost)
        self._update_glob_fastest_exec_time(fast_exec_time)
        self.total_opt_cost += self.layer_stats[layer_instance].opt_cost
        return logs
    
    def get_global_summary(self, verbose: bool):
        candidates = []
        for layer_instance, layer_stat in self.layer_stats.items():
            if layer_stat.is_leaf:
                candidates.extend(layer_stat.get_all_candidates())
        pareto_frontier = get_pareto_front(candidates)
        if verbose:
            print(f"================ Optimization Results =================")
            print(f"Optimized for: {str(CommonStats.objectives)}")

            if len(pareto_frontier) == 0:
                print("Based on current optimization parameters, the best solution is the original workflow.")
                print("We recommend increasing the number of trials or relaxing constraints.")
            else:
                print(f"Number of Optimized Workflows Generated: {len(pareto_frontier)}")

            for i, trial_log in enumerate(pareto_frontier):
                print("--------------------------------------------------------")
                print("Optimization_{}".format(i + 1))
                # logger.info("  Params: {}".format(trial_log.params))
                score, price, exec_time = trial_log.result.reduced_score, trial_log.result.reduced_price, trial_log.result.reduced_exec_time
                if CommonStats.base_quality is not None:
                    print("  Quality improvement: {:.0f}%".format(
                        _report_quality_impv(score, CommonStats.base_quality)
                    ))
                if CommonStats.base_price is not None:
                    print("  Cost: {:.2f}x original".format(
                        _report_cost_reduction(price, CommonStats.base_price)
                    ))
                if CommonStats.base_exec_time is not None:
                    print("  Execution time: {:.2f}x original".format(
                        _report_latency_reduction(exec_time, CommonStats.base_exec_time)
                    ))
                print("  Quality: {:.2f}, Cost per 1K invocation: ${:.2f}, Execution time: {:.2f}s".format(
                    score, price * 1000, exec_time
                ))
                    # print("  Applied at: {}".format(trial_log.id))
                    # logger.info("  config saved at: {}".format(log_path))

                # logger.info("Opt Cost: {}".format(self.opt_cost))
                print("========================================================")
        
        finished_trials: dict[str, tuple[TrialLog, str]] = {}
        for layer_instance, layer_stat in self.layer_stats.items():
            if layer_stat.is_leaf:
                for log in layer_stat.get_completed_logs():
                    finished_trials[log.id] = (log, layer_stat.opt_log_path)
        
        return self.total_opt_cost, pareto_frontier, finished_trials
    
    def init_progress_bar(self, num_total_trials, num_existing_trials):
        initial_score = self._glob_best_score or 0
        initial_cost = self._glob_lowest_cost or 0
        initial_exec_time = self._glob_fastest_exec_time or 0
        pbar.init_pbar(
            total=num_total_trials,
            initial=num_existing_trials,
            initial_score=initial_score,
            initial_cost=initial_cost,
            initial_exec_time=initial_exec_time,
            opt_cost=self.total_opt_cost,
        )

    def get_log_by_id(self, log_id: str) -> TrialLog:
        """Retrieve a log by its id
        """
        layer_instance = log_id.rsplit("_", 1)[0]
        return self.layer_stats[layer_instance].opt_logs[log_id]
        
    def _update_glob_best_score(self, score):
        if score is None:
            return
        if self._glob_best_score is None or score > self._glob_best_score:
            self._glob_best_score = score
    
    def _update_glob_lowest_cost(self, cost):
        if cost is None:
            return
        if self._glob_lowest_cost is None or cost < self._glob_lowest_cost:
            self._glob_lowest_cost = cost
    
    def _update_glob_fastest_exec_time(self, exec_time):
        if exec_time is None:
            return
        if self._glob_fastest_exec_time is None or exec_time < self._glob_fastest_exec_time:
            self._glob_fastest_exec_time = exec_time
    
    def _save_opt_trace(self):
        json.dump(self._opt_trace, open(self.opt_trace_log_path, "w+"), indent=4)