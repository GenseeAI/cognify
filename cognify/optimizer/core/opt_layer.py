import os
import json
from typing import (
    Optional,
    Any,
    Tuple,
    Iterable,
    Callable
)
from abc import ABC, abstractmethod
from dataclasses import dataclass
import copy
import logging
import optuna
import numpy as np
from collections import defaultdict
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)
import threading
import traceback
import math
from concurrent.futures import Future

from cognify.optimizer.utils import _report_cost_reduction, _report_quality_impv, _report_latency_reduction
from cognify._signal import _should_exit
from cognify.graph.program import Module
from cognify.hub.cogs.common import (
    CogBase,
    DynamicCogBase,
    EvolveType,
    AddNewModuleImportInterface,
)
from cognify.hub.cogs.utils import dump_params, load_params
from cognify.optimizer.evaluator import (
    EvaluationResult,
    EvaluatorPlugin,
    EvalTask,
    GeneralEvaluatorInterface,
)
from optuna.samplers import TPESampler, _base
from optuna.trial import TrialState, FrozenTrial
from cognify.optimizer.bo.tpe import FrugalTPESampler
from cognify.optimizer.core.flow import (
    TrialLog,
    ModuleTransformTrace,
    TopDownInformation,
    OptConfig,
)
from cognify.optimizer.control_param import SelectedObjectives
from cognify.optimizer.progress_info import pbar
from cognify.optimizer.core.common_stats import CommonStats

logger = logging.getLogger(__name__)


qc_identifier = "_#cognify_quality_constraint"


def get_quality_constraint(trial: optuna.trial.FrozenTrial):
    return trial.user_attrs.get(qc_identifier, (1,))

optimize_directions = [
    "maximize", # quality
    "minimize", # cost
    "minimize", # exec time
]
        
class OptLayerInterface(GeneralEvaluatorInterface):
    name: str
    dedicate_params: list[CogBase]
    universal_params: list[CogBase]
    target_modules: list[str]
    
    hierarchy_level: int
    next_layer_factory: Callable[[], GeneralEvaluatorInterface]
    is_leaf: bool
    
    opt_logs: dict[str, TrialLog]

    @abstractmethod
    def _setup_opt_env(self):
        ...
    
    @abstractmethod
    def _propose(self) -> Tuple[
        optuna.trial.Trial, 
        list[Module],
        ModuleTransformTrace, 
        str
    ]:
        """Propse and apply the next set of params

        Returns:
            Tuple[
                new optuna trial,
                new program,
                transform trace,
                log_id
            ]
        """
        ...
    
    @abstractmethod
    def _prepare_eval_task(
        self,
        new_program: list[Module],
        new_trace: ModuleTransformTrace,
        trial_id: str,
    ) -> TopDownInformation:
        """create info for next level optimization or actual evaluation
        """
        ...
    
    @abstractmethod
    def _update(
        self,
        trial: optuna.trial.Trial,
        eval_result: EvaluationResult,
        log_id: str,
    ):
        """Update observation set with current evaluation
        """
        ...
    
    
    @abstractmethod
    def optimize(
        self,
        current_tdi: TopDownInformation, 
    ) -> list[tuple[TrialLog, str]]:
        """Run the optimization with the given upper-layer configs
        """
        ...
    
    @abstractmethod
    def _summarize_to_eval_result(
        self, 
        pareto_frontier: list[tuple[TrialLog, str]]
    ) -> EvaluationResult:
        """Summarize the optimization results to a single EvaluationResult for upper-layer
        """
    
    def evaluate(self, tdi, **kwargs):
        frontier = self.optimize(tdi)
        return self._summarize_to_eval_result(frontier)

def get_pareto_front(candidates: list[TrialLog]) -> list[TrialLog]:
    """Find the pareto-efficient points

    This function will not filter with constraints
    """
    if not candidates:
        return []
    score_cost_time_list = []
    for trial_log in candidates:
        score_cost_time_list.append((trial_log.score, trial_log.price, trial_log.exec_time))

    vectors = np.array([CommonStats.objectives.select_from(-score, price, exec_time) for score, price, exec_time in score_cost_time_list])
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

    
class OptLayer(OptLayerInterface):
    def __init__(
        self,
        name: str,
        opt_config: OptConfig,
        hierarchy_level: int,
        is_leaf: bool,
        next_layer_factory: Callable[[], GeneralEvaluatorInterface],
        dedicate_params: list[CogBase] = [],
        universal_params: list[CogBase] = [],
        target_modules: Iterable[str] = None,
    ):
        """
        The optimization will always try to minimize the price and exec time and maximize the score
        Please make sure the evaluator is consistent with this.

        Args:
            name: name of the optimization layer

            evaluator: the evaluator that will be used to evaluate the proposal

            dedicate_params: a list of params that is dedicated to pre-defined modules
                need to set `module_name` correctly

            universal_params: a list of params that will be broadcasted to all modules
                will ignore `module_name` field

            target_modules: if provided, only the modules in this list will be optimized
                this has higher priority than dedicated params
        """
        self.name = name
        self.opt_config = opt_config
        self.dedicate_params = dedicate_params
        self.universal_params = universal_params
        if len(self.dedicate_params) + len(self.universal_params) == 0:
            raise ValueError("No params provided for optimization")
        self.params: dict[str, list[CogBase]] = None
        self._search_space: dict[str, optuna.distributions.CategoricalDistribution] = None

        self.target_modules = (
            set(target_modules) if target_modules is not None else None
        )
        self.study: optuna.study.Study = None
        self._study_lock: threading.Lock = None
        self._local_best_score = None
        self._local_lowest_cost = None
        self._local_fastest_exec_time = None
        self._patience_budget = None if self.opt_config.patience is None else self.opt_config.patience.n_iterations
        self._converged = False
        self.opt_cost = 0
        self.opt_logs = {}
        
        self.hierarchy_level = hierarchy_level
        self.next_layer_factory = next_layer_factory
        self.is_leaf = is_leaf
        self.top_down_info: TopDownInformation = None
        logger.info(f"OptLayer {self.name} initialized")

    def optimize(self, current_tdi) -> list[tuple[TrialLog, str]]:
        current_tdi.initialize(self.opt_config)
        self.top_down_info = current_tdi
        if current_tdi.opt_config.patience is not None:
            self._patience_budget = current_tdi.opt_config.patience.n_iterations
        self._setup_opt_env()

        # load previous optimization logs if exists
        opt_log_path = self.top_down_info.opt_config.opt_log_path
        if os.path.exists(opt_log_path):
            self._load_opt_ckpt()

        # start optimization
        total_budget = self.top_down_info.opt_config.n_trials
        if total_budget > 0:
            logger.debug(f"Start optimization {self.name} with {total_budget} trials")
            
            self._optimize()
            
            logger.debug(f"Optimization {self.name} finished")
            self._save_ckpt()
        result = self._get_layer_opt_result()
        return result
    
    def _get_layer_opt_result(self):
        # Analysis optimization result
        candidates = self.get_all_candidates()
        pareto_frontier = get_pareto_front(candidates=candidates)
        if self.hierarchy_level == 0:
            _log_optimization_results(pareto_frontier)
        return pareto_frontier
        
    def _save_ckpt(self):
        # save opt logs
        opt_logs_json_obj = {}
        for k, v in self.opt_logs.items():
            if v.finished:
                opt_logs_json_obj[k] = v.to_dict()
        if not opt_logs_json_obj:
            logger.warning("No finished trials to save")
            return
        json.dump(opt_logs_json_obj, open(self.top_down_info.opt_config.opt_log_path, "w+"), indent=4)
        params = [param for params in self.params.values() for param in params]
        
        # save search space in case evolving params 
        params = [param for params in self.params.values() for param in params]
        dump_params(params, self.top_down_info.opt_config.param_save_path)

        
    def get_all_candidates(self):
        cancidates = []
        for log_id, log in self.opt_logs.items():
            if not log.finished:
                continue
            # if not meet the quality constraint, skip
            if (
                CommonStats.quality_constraint is not None
                and log.score < CommonStats.quality_constraint
            ):
                continue
            cancidates.append(log)
        return cancidates
    
    def add_constraint(self, score, trial: optuna.trial.Trial):
        # Soft constraint, if score is lower than the quality constraint, reject it
        if CommonStats.quality_constraint is not None:
            constraint_result = (CommonStats.quality_constraint - score,)
            trial.set_user_attr(qc_identifier, constraint_result)
            # NOTE: add system attr at loading time
            # trial.set_system_attr(_base._CONSTRAINTS_KEY, constraint_result)
        
    def _load_opt_ckpt(self):
        # Load logs from file
        with open(self.top_down_info.opt_config.opt_log_path, "r") as f:
            opt_trace = json.load(f)
        for trial_log_id, trial_meta in opt_trace.items():
            trial_log = TrialLog.from_dict(trial_meta)
            self.opt_logs[trial_log_id] = trial_log
            self.opt_cost += trial_log.eval_cost
            
        candidates = self.get_all_candidates()
        if candidates:
            self._local_best_score = max([log.score for log, _ in candidates])
            self._local_lowest_cost = min([log.price for log, _ in candidates])
            self._local_fastest_exec_time = min([log.exec_time for log, _ in candidates])
        
        for trial_log_id, trial_log in self.opt_logs.items():
            assert trial_log.finished, f"Trial {trial_log_id} is not finished"
            trial = optuna.trial.create_trial(
                params=trial_log.params,
                values=CommonStats.objectives.select_from(trial_log.score, trial_log.price, trial_log.exec_time),
                distributions=self._search_space,
            )
            if CommonStats.quality_constraint is not None:
                self.add_constraint(trial_log.score, trial)
                trial.set_system_attr(
                    _base._CONSTRAINTS_KEY, get_quality_constraint(trial)
                )
            self.study.add_trial(trial)
            
    def _summarize_to_eval_result(self, pareto_frontier: list[tuple[TrialLog, str]]):
        if not pareto_frontier:
            # If no trial is finished/qualified:
            #   return bad information instead of no information
            # consider this as a failed evaluation
            return EvaluationResult(
                ids=[],
                scores=[],
                prices=[],
                exec_times=[float(0xDEADBEEF)],
                total_eval_cost=self.opt_cost,
                complete=True,
                reduced_price=float(0xDEADBEEF),
                reduced_score=0,
                reduced_exec_time=float(0xDEADBEEF),
                meta={"converged": self._converged},
            )
            
        inner_log_ids, scores, prices, exec_times = [], [], [], []
        for trial_log, _ in pareto_frontier:
            inner_log_ids.append(trial_log.id)
            scores.append(trial_log.score)
            prices.append(trial_log.price)
            exec_times.append(trial_log.exec_time)

        reduced_score = max(scores)
        reduced_price = min(prices)
        reduced_exec_time = min(exec_times)
        result = EvaluationResult(
            ids=inner_log_ids,
            scores=scores,
            prices=prices,
            exec_times=exec_times,
            total_eval_cost=self.opt_cost,
            complete=True,
            reduced_score=reduced_score,
            reduced_price=reduced_price,
            reduced_exec_time=reduced_exec_time,
            meta={"converged": self._converged},
        )
        return result

    def _setup_opt_env(self):
        self.params = defaultdict(list)
        self.base_program = list(self.top_down_info.current_module_pool.values())
        # NOTE: if param file exists, will load params from file and ignore the given params
        param_save_path = self.top_down_info.opt_config.param_save_path
        if os.path.exists(param_save_path):
            logger.info(f"Loading {self.name} params from {param_save_path}")
            l_param = load_params(param_save_path)
            for param in l_param:
                self.params[param.module_name].append(param)
            allowed_lm_names = set(param.module_name for param in l_param)
        else:
            module_pool = self.top_down_info.current_module_pool
            # extract mappings from old_lm_name to all_new_lm_names
            old_2_new_lms: dict[str, list[str]] = defaultdict(list)
            allowed_lm_names = set()
            for new_module in module_pool.values():
                old_name, new_modules = (
                    self.top_down_info.module_ttrace.get_derivatives_of_same_type(
                        new_module
                    )
                )
                new_names = [x.name for x in new_modules]
                if self.target_modules and old_name not in self.target_modules:
                    continue
                old_2_new_lms[old_name].extend(new_names)
                allowed_lm_names.update(new_names)

            # broadcast universal params
            if self.universal_params:
                for lm_name in allowed_lm_names:
                    params_cpy = copy.deepcopy(self.universal_params)
                    for param in params_cpy:
                        param.module_name = lm_name
                    self.params[lm_name] = params_cpy

            # apply dedicated params
            if self.dedicate_params:
                for param in self.dedicate_params:
                    target_names = []
                    if param.module_name in old_2_new_lms:
                        target_names = old_2_new_lms[param.module_name]
                    elif param.module_name in allowed_lm_names:
                        target_names = [param.module_name]

                    for lm_name in target_names:
                        mapped_param = copy.deepcopy(param)
                        mapped_param.module_name = lm_name
                        self.params[lm_name].append(mapped_param)

        self._search_space = {
            param.hash: optuna.distributions.CategoricalDistribution(
                list(param.options.keys())
            )
            for _, params in self.params.items()
            for param in params
        }
        self.opt_target_lm_names = allowed_lm_names
        self.study = self.init_study()
        self._study_lock = threading.Lock()
    
    def init_study(
        self,
        old_study: optuna.Study = None,
        old_trials: list[optuna.trial.FrozenTrial] = None,
    ):
        """Create a new study and migrate old trials if provided

        For all provided trials, the params dist will be adjusted to the current
        self.params when this method is called.

        Recommand using name based options instead of index based options as the dynamic
        params update may change the mapping between option index and the option itself
        """
        qc_fn = get_quality_constraint if CommonStats.quality_constraint is not None else None
        if self.top_down_info.opt_config.frugal_eval_cost:
            sampler = FrugalTPESampler(
                cost_estimator=self.param_cost_estimator,
                multivariate=True,
                n_startup_trials=5,
                constraints_func=qc_fn,
            )
        else:
            sampler = TPESampler(
                multivariate=True, n_startup_trials=5, constraints_func=qc_fn
            )

        new_study = optuna.create_study(
            directions=optimize_directions,
            sampler=sampler,
        )

        f_trials: list[optuna.trial.FrozenTrial] = []
        if old_study:
            f_trials.extend(old_study.trials)
        if old_trials:
            f_trials.extend(old_trials)

        for trial in f_trials:
            # Modify previous trials.
            # user and system attr will be copied
            # no need to deal with them here
            dists = self._search_space
            trial.distributions = dists
            # Persist the changes to the storage (in a new study).
            new_study.add_trial(trial)
        return new_study
    
    def param_cost_estimator(self, trial_proposal: dict[str, Any]) -> float:
        """predict the cost of the trial proposal

        NOTE: trial proposal may not contain all params, e.g. if param only have single option or is sampled independently
        """
        total_cost = 0.0
        # convert to external params
        ext_trial_proposal = {}
        for param_name, dist in self._search_space.items():
            if dist.single():
                continue
            ext_trial_proposal[param_name] = dist.to_external_repr(
                trial_proposal[param_name]
            )
        for lm_name, params in self.params.items():
            agent_cost = 1.0
            # for param imposed on the same agent, multiply the cost
            for param in params:
                if param.hash not in ext_trial_proposal:
                    continue
                selected = ext_trial_proposal[param.hash]
                option = param.options.get(selected, None)
                if option:
                    agent_cost *= option.cost_indicator
            total_cost += agent_cost
        return total_cost
    
    def _apply_params(
        self,
        trial_params: dict[str, Any],
    ) -> tuple[list[Module], ModuleTransformTrace]:
        trace_for_next_level = copy.deepcopy(self.top_down_info.module_ttrace)
        program_copy = copy.deepcopy(self.base_program)
        
        opt_target_lms = Module.all_with_predicate(
            program_copy, lambda m: m.name in self.opt_target_lm_names
        )
        module_dict = {lm.name: lm for lm in opt_target_lms}
        new_modules = []
        changed_modules = set()
        for lm_name, params in self.params.items():
            for param in params:
                selected = trial_params[param.hash]
                try:
                    new_module, new_mapping = param.apply_option(
                        selected, module_dict[lm_name]
                    )

                    for old_name, new_name in new_mapping.items():
                        trace_for_next_level.add_mapping(old_name, new_name)
                    new_modules.append(new_module)
                    changed_modules.add(lm_name)
                    trace_for_next_level.register_proposal(
                        self.name, [(lm_name, param.name, selected)]
                    )
                except Exception as e:
                    logger.error(
                        f"Error in applying param {param.name} to {lm_name}: {e}"
                    )
                    logger.error(
                        f"Module dict {module_dict.keys()} and self params {self.params.keys()}"
                    )
                    raise

        for m_name, new_module in module_dict.items():
            if m_name not in changed_modules:
                new_modules.append(new_module)
        return new_modules, trace_for_next_level
    
    def generate_trial_id(self) -> tuple[str, int]:
        """Get the next trial id
        
        Optuna trial id start from num_trials, if previous run is interrupted
        so using optuna trial number will have conflict
        
        here we always increment the trial number
        """
        with self._study_lock:
            max_id = None
            for log in self.opt_logs.values():
                id = int(log.id.split("_")[-1])
                max_id = id if max_id is None else max(max_id, id)
        new_trial_number = max_id + 1 if max_id is not None else 0
        return ".".join(
            self.top_down_info.trace_back + [f"{self.name}_{new_trial_number}"]
        )

    def _propose(self):
        with self._study_lock:
            trial = self.study.ask(self._search_space)

        logger.debug(
            f"- {self.name} - apply param - Trial {trial.number} params: {trial.params}"
        )
        
        trial_id_str = self.generate_trial_id()
        trial_log = TrialLog(
            params=trial.params, bo_trial_id=trial.number, id=trial_id_str, layer_name=self.name
        )
        new_program, new_trace = self._apply_params(trial.params)
        self.opt_logs[trial_id_str] = trial_log
        return trial, new_program, new_trace, trial_id_str
    
    def _get_new_python_paths(self):
        new_python_paths = []
        for lm_name, params in self.params.items():
            for param in params:
                if isinstance(param, AddNewModuleImportInterface):
                    new_python_paths.extend(param.get_python_paths())
        return new_python_paths
    
    def _prepare_eval_task(
        self,
        new_program: list[Module],
        new_trace: ModuleTransformTrace,
        trial_log_id: str,
    ) -> TopDownInformation:

        # add new python paths incase new module imports are added
        python_paths = (
            self.top_down_info.other_python_paths + self._get_new_python_paths()
        )
        python_paths = list(set(python_paths))
        
        # set log path to next level opt
        current_layer_config = self.top_down_info.opt_config
        current_level_log_dir = current_layer_config.log_dir
        trial_number = trial_log_id.rsplit("_", 1)[-1]
        next_log_dir = os.path.join(
            current_level_log_dir,
            f"{self.name}_trial_{trial_number}",
        )
        next_opt_config = OptConfig._set_log_dir_for_next(log_dir=next_log_dir)
        next_opt_config.frac = current_layer_config.frac / current_layer_config.n_trials

        next_level_info = TopDownInformation(
            opt_config=next_opt_config,
            all_params=self.top_down_info.all_params.copy(),  # params from upper-levels will not be changed
            module_ttrace=new_trace,
            current_module_pool={m.name: m for m in new_program},
            script_path=self.top_down_info.script_path,
            script_args=self.top_down_info.script_args,
            other_python_paths=python_paths,
        )
        next_level_info.trace_back.append(trial_log_id)

        # add current level params for next level
        for lm_name, params in self.params.items():
            # NOTE: params might be updated when scheduling the current iteration
            # so we make a copy of the current params
            for param in params:
                next_level_info.all_params[param.hash] = copy.deepcopy(param)

        return next_level_info
    
    def _add_constraint(self, score, trial: optuna.trial.Trial):
        # Soft constraint, if score is lower than the quality constraint, reject it
        if CommonStats.quality_constraint is not None:
            constraint_result = (CommonStats.quality_constraint - score,)
            trial.set_user_attr(qc_identifier, constraint_result)
            # NOTE: add system attr at loading time
            # trial.set_system_attr(_base._CONSTRAINTS_KEY, constraint_result)
            
    def get_finished_bo_trials(self, need_copy: bool) -> list[FrozenTrial]:
        states_of_interest = (TrialState.COMPLETE,)
        return self.study.get_trials(deepcopy=need_copy, states=states_of_interest)
    

    def _update(
        self,
        trial: optuna.trial.Trial,
        eval_result: EvaluationResult,
        trial_log_id: str,
    ):
        # if eval is interrupted, the result will not be used
        if not eval_result.complete:
            return

        # update study if any dynamic params can evolve
        self.opt_logs[trial_log_id].update_result(eval_result)
        score, price, exec_time = eval_result.reduced_score, eval_result.reduced_price, eval_result.reduced_exec_time
        with self._study_lock:
            self._add_constraint(score, trial)
            frozen_trial = self.study.tell(trial, [score, price, exec_time])
            is_evolved = False
            for params in self.params.values():
                for param in params:
                    if isinstance(param, DynamicCogBase):
                        evolve_type = param.evolve(eval_result)
                        if evolve_type != EvolveType.ID:
                            is_evolved = True
            if is_evolved:
                # update param dist
                self._search_space = {
                    param.hash: optuna.distributions.CategoricalDistribution(
                        list(param.options.keys())
                    )
                    for _, params in self.params.items()
                    for param in params
                }
                # create new study and migrate all trials
                new_study = self.init_study(self.study)
                self.study = new_study

    def _optimize_iteration(self):
        next_trial, program, new_trace, log_id = self._propose()
        next_level_info = self._prepare_eval_task(program, new_trace, log_id)
        if self.is_leaf:
            next_level_info = EvalTask.from_top_down_info(next_level_info)
            self.opt_logs[log_id].eval_task_dict = next_level_info.to_dict()

        if _should_exit():
            return None, None

        try:
            next_layer_evaluator = self.next_layer_factory()
            _result = next_layer_evaluator.evaluate(
                next_level_info,
                frac=self.top_down_info.opt_config.frac,
            )
            self._update(next_trial, _result, log_id)
            return log_id, _result
        except Exception as e:
            logger.error(f"Error in opt iteration: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def if_early_stop(self, result: EvaluationResult):
        if self.top_down_info.opt_config.patience is None:
            return False
        if not result.complete:
            self._patience_budget -= 1
            self._converged = self._patience_budget <= 0
            return self._converged
        impv = False
        score_threshold = self.top_down_info.opt_config.patience.quality_min_delta
        cost_threshold = self.top_down_info.opt_config.patience.cost_min_delta
        time_threshold = self.top_down_info.opt_config.patience.exec_time_min_delta
        # reset budget if any dimension is improved
        if self._local_best_score is None or result.reduced_score >= self._local_best_score * (1 + score_threshold):
            self._local_best_score = result.reduced_score
            impv = True
        if self._local_lowest_cost is None or result.reduced_price <= self._local_lowest_cost * (1 - cost_threshold):
            self._local_lowest_cost = result.reduced_price
            impv = True
        if self._local_fastest_exec_time is None or result.reduced_exec_time <= self._local_fastest_exec_time * (1 - time_threshold):
            self._local_fastest_exec_time = result.reduced_exec_time
            impv = True
        if impv:
            self._patience_budget = self.top_down_info.opt_config.patience.n_iterations
        else:
            self._patience_budget -= 1
        # print(f"Patience budget: {self._patience_budget}")
        self._converged = self._patience_budget <= 0
        return self._converged
            
        
    def _optimize(self):
        opt_config = self.top_down_info.opt_config
        num_current_trials = len(self.opt_logs)

        initial_score = self._local_best_score if self._local_best_score is not None else 0.0
        initial_cost = self._local_lowest_cost if self._local_lowest_cost is not None else 0.0
        initial_exec_time = self._local_fastest_exec_time if self._local_fastest_exec_time is not None else 0.0

        if self.hierarchy_level == 0:
            pbar.init_pbar(
                total=float(num_current_trials + opt_config.n_trials),
                initial=float(num_current_trials),
                initial_score=initial_score,
                initial_cost=initial_cost,
                initial_exec_time=initial_exec_time,
                opt_cost=self.opt_cost
            )
        
        with ThreadPoolExecutor(max_workers=opt_config.throughput) as executor:
            futures = [
                executor.submit(self._optimize_iteration)
                for _ in range(opt_config.n_trials)
            ]
            visited = set()
            for f in as_completed(futures):
                try:
                    log_id, result = f.result()
                    if result and result.complete:
                        pbar.update_status(self._local_best_score, self._local_lowest_cost, self._local_fastest_exec_time, self.opt_cost)
                        self._save_ckpt()
                        visited.add(f)
                        if self.if_early_stop(result):
                            executor.shutdown(wait=False, cancel_futures=True)
                            pbar.finish()
                            break
                    if _should_exit():
                        executor.shutdown(wait=False, cancel_futures=True)
                        break
                except Exception as e:
                    logger.error(f"Error in evaluating task: {e}")
                    raise
            # visit other finished trials
            for f in futures:
                if f not in visited and f.done() and not f.cancelled():
                    log_id, result = f.result()
                    if result and result.complete:
                        self._save_ckpt()
    
    def easy_optimize(
        self,
        script_path: str,
        script_args: Optional[list[str]] = None,
        other_python_paths: Optional[list[str]] = None,
    ) -> list[tuple[TrialLog, str]]:
        tdi = TopDownInformation(
            opt_config=None,
            all_params=None,
            module_ttrace=None,
            current_module_pool=None,
            script_path=script_path,
            script_args=script_args,
            other_python_paths=other_python_paths,
        )
        
        return self.optimize(tdi)
    
def _log_optimization_results(pareto_frontier: list[TrialLog]):
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
        if CommonStats.base_quality is not None:
            print("  Quality improvement: {:.0f}%".format(_report_quality_impv(trial_log.score, CommonStats.base_quality)))
        if CommonStats.base_price is not None:
            print("  Cost: {:.2f}x original".format(_report_cost_reduction(trial_log.price, CommonStats.base_price)))
        if CommonStats.base_exec_time is not None:
            print("  Execution time: {:.2f}x original".format(_report_latency_reduction(trial_log.exec_time, CommonStats.base_exec_time)))
        print("  Quality: {:.2f}, Cost per 1K invocation: ${:.2f}, Execution time: {:.2f}s".format(trial_log.score, trial_log.price * 1000, trial_log.exec_time))
            # print("  Applied at: {}".format(trial_log.id))
            # logger.info("  config saved at: {}".format(log_path))

        # logger.info("Opt Cost: {}".format(self.opt_cost))
        print("========================================================")