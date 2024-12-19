import os
import json
from typing import (
    Tuple,
)
import copy
import logging
from concurrent.futures import ThreadPoolExecutor, Future

from cognify._signal import _should_exit
from cognify.optimizer.evaluator import (
    EvaluationResult,
    GeneralEvaluatorInterface,
)
from cognify.optimizer.core.flow import TrialLog, TopDownInformation, OptConfig
from cognify.optimizer.core.unified_layer_opt import (
    OptimizationLayer,
    BottomLevelTrialLog,
)

logger = logging.getLogger(__name__)


class LayerEvaluator(GeneralEvaluatorInterface):
    def __init__(
        self,
        target_layer: OptimizationLayer,
    ):
        self.target_layer = target_layer

    def evaluate(
        self,
        layer_task: TopDownInformation,
        **kwargs,
    ) -> EvaluationResult:
        # NOTE: optimization will change layer meta, make a copy
        target_layer_cpy = copy.deepcopy(self.target_layer)
        # NOTE: returned opt_logs is all finished
        eval_cost, pareto_frontier, opt_logs = target_layer_cpy.optimize(layer_task)
        if not opt_logs:
            # If no trial is finished:
            #   return bad information instead of no information
            # consider this as a failed evaluation
            return EvaluationResult(
                ids=[],
                scores=[],
                prices=[],
                exec_times=[],
                total_eval_cost=eval_cost,
                complete=False,
                reduced_price=float(0xDEADBEEF),
                reduced_score=0,
            )

        inner_log_ids, scores, prices, exec_times = [], [], [], []
        for trial_log in opt_logs.values():
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
            total_eval_cost=eval_cost,
            complete=True,
            reduced_score=reduced_score,
            reduced_price=reduced_price,
            reduced_exec_time=reduced_exec_time,
            demos={"config_log_path": layer_task.opt_config.opt_log_path},
        )
        return result


class SuccessiveHalving:
    """Policy for successive halving resource allocation

    If a layer (A) adopts this policy, it will allocate resources for the next layer (B)

    layer A will propose `layer_a.opt_config.throughput` trials in each iteration

    **SH routine**:
    1. each remaining trial at layer B will have a fixed step_budget = `layer_b.opt_config.n_trials`
    2. after all trials are evaluated, lower half of the trials will be pruned
    3. repeat from step 1 until no trials left
    """

    def __init__(
        self,
        selected_runs: list[Tuple[LayerEvaluator, TopDownInformation]],
    ):
        self.selected_runs = selected_runs
        self.ready_to_run = [i for i in range(len(selected_runs))]
        self.num_inner_trials = [0] * len(selected_runs)

    def run_and_prune(self):
        logger.info(f"SH next with {self.ready_to_run}")
        for i in self.ready_to_run:
            tdi = self.selected_runs[i][1]
            self.num_inner_trials[i] += tdi.opt_config.n_trials

        futures: list[Future] = []
        with ThreadPoolExecutor(max_workers=len(self.ready_to_run)) as executor:
            for i in self.ready_to_run:
                futures.append(
                    executor.submit(
                        self.selected_runs[i][0].evaluate, self.selected_runs[i][1]
                    )
                )

            try:
                outer_indicators = []
                for f in futures:
                    if _should_exit():
                        executor.shutdown(cancel_futures=True, wait=True)
                        break
                    eval_result: EvaluationResult = f.result()
                    outer_indicators.append(
                        (eval_result.reduced_score, eval_result.reduced_price)
                    )
            except Exception as e:
                logger.error(f"Error in SH: {e}")
                raise

        # outer_indicators = []
        # for i in self.ready_to_run:
        #     eval_result = self.selected_outer_runs[i](self.inner_step_budget)
        #     outer_indicators.append((eval_result.reduced_score, eval_result.reduced_price))

        # sort by score and price, higher score then lower price
        sorted_indicator_indices = sorted(
            range(len(outer_indicators)),
            key=lambda i: (-outer_indicators[i][0], outer_indicators[i][1]),
        )
        runs_left_to_run = sorted_indicator_indices[: len(self.ready_to_run) // 2]
        self.ready_to_run = [self.ready_to_run[i] for i in runs_left_to_run]

    def execute(self):
        while len(self.ready_to_run) > 0:
            if _should_exit():
                break
            self.run_and_prune()
        # Collect inner loop performance
        outer_run_evals = []
        for layer_eval, tdi in self.selected_runs:
            tdi.opt_config.n_trials = 0
            outer_run_evals.append(layer_eval.evaluate(tdi))
        return outer_run_evals, self.num_inner_trials


class UpperLevelTrialLog(TrialLog):
    def __init__(
        self,
        params,
        bo_trial_id,
        id=None,
        score=0,
        price=0,
        exec_time=0,
        eval_cost=0,
        finished=False,
        next_level_log_dir=None,
        num_next_level_trials=0,
    ):
        super().__init__(params, bo_trial_id, id, score, price, exec_time, eval_cost, finished)
        self.next_level_log_dir = next_level_log_dir
        self.num_next_level_trials = num_next_level_trials

    def to_dict(self):
        return {
            **super().to_dict(),
            "next_level_log_dir": self.next_level_log_dir,
            "num_next_level_trials": self.num_next_level_trials,
        }


class UpperLevelOptimization(OptimizationLayer):
    opt_logs: dict[str, UpperLevelTrialLog]
    evaluator: LayerEvaluator
    trial_log_cls = UpperLevelTrialLog

    def __init__(
        self,
        next_level_opt_config: OptConfig = None,
        use_SH_allocation: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.next_level_opt_config = next_level_opt_config
        self.use_SH_allocation = use_SH_allocation

    def prepare_next_level_tdi(self, new_program, new_trace, trial_id):
        next_level_info = super().prepare_next_level_tdi(
            new_program, new_trace, trial_id
        )
        # reset opt_config for next level
        if self.next_level_opt_config:
            next_level_info.opt_config.update(self.next_level_opt_config)
        # incase log_dir is not set
        if self.next_level_opt_config.log_dir is None:
            current_level_log_dir = self.top_down_info.opt_config.log_dir
            _trial_number = trial_id.split("_")[-1]
            next_level_info.opt_config.log_dir = os.path.join(
                current_level_log_dir,
                f"{self.name}_trial_{_trial_number}",
            )
        # set these path to None to let the next level to populate
        next_level_info.opt_config.opt_log_path = None
        next_level_info.opt_config.param_save_path = None
        return next_level_info

    def get_eval_feedback(self, eval_result: EvaluationResult):
        if eval_result.reduced_price == float(0xDEADBEEF):
            return None, None, None
        return eval_result.reduced_score, eval_result.reduced_price, eval_result.reduced_exec_time

    def _optimize_iteration(self, base_program):
        next_trial, program, new_trace, log_id = self.propose(base_program, 1)[0]
        next_level_info = self.prepare_next_level_tdi(
            program, new_trace, log_id,
        )
        self.opt_logs[log_id].next_level_log_dir = next_level_info.opt_config.log_dir

        if _should_exit():
            return None

        # run evaluation
        try:
            eval_result = self.evaluator.evaluate(next_level_info)
            self.update(next_trial, eval_result, log_id)
            return eval_result
        except Exception as e:
            logger.error(f"Error in evaluation: {e}")
            raise

    def _optimize_SH(self, base_program):
        opt_config = self.top_down_info.opt_config

        # propose `throughput` trials
        proposals_at_this_level = self.propose(base_program, opt_config.throughput)
        selected_runs = []
        for new_trial, new_program, new_trace, new_log_id in proposals_at_this_level:
            next_level_info = self.prepare_next_level_tdi(
                new_program, new_trace, new_log_id
            )
            selected_runs.append((self.evaluator, next_level_info))
            self.opt_logs[
                new_log_id
            ].next_level_log_dir = next_level_info.opt_config.log_dir
        sh = SuccessiveHalving(selected_runs)
        if _should_exit():
            return
        eval_results, num_inner_trials = sh.execute()
        for i, (trial, program, new_trace, log_id) in enumerate(
            proposals_at_this_level
        ):
            self.update(trial, eval_results[i], log_id)
            self.opt_logs[log_id].num_next_level_trials = num_inner_trials[i]

    def _optimize(self, base_program):
        if not self.use_SH_allocation:
            return super()._optimize(base_program)

        # use SH allocation
        opt_config = self.top_down_info.opt_config
        n_iters = opt_config.n_trials // opt_config.throughput
        for i in range(n_iters):
            if _should_exit():
                break
            self._optimize_SH(base_program)
            if _should_exit():
                break
            if self.save_ckpt_interval > 0 and i % self.save_ckpt_interval == 0:
                self.save_ckpt(opt_config.opt_log_path, opt_config.param_save_path)

    def get_all_candidates(self) -> list[TrialLog, str]:
        candidates = []
        for _, upper_trial_log in self.opt_logs.items():
            if not upper_trial_log.finished:
                continue
            bot_opt_log_path = os.path.join(
                upper_trial_log.next_level_log_dir, "opt_logs.json"
            )
            with open(bot_opt_log_path, "r") as f:
                opt_trace = json.load(f)

            for trial_log_id, trial_meta in opt_trace.items():
                trial_log = BottomLevelTrialLog.from_dict(trial_meta)
                # if not meet the quality constraint, skip
                if (
                    self.quality_constraint is not None
                    and trial_log.score < self.quality_constraint
                ):
                    continue
                candidates.append((trial_log, bot_opt_log_path))
        return candidates
