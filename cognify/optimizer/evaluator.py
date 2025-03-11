import os
import sys
import json
from typing import Union, Optional, Tuple, Callable, Iterable, Literal, Sequence
import time
import copy
import logging
from dataclasses import dataclass, field
import numpy as np
from abc import ABC, abstractmethod
import multiprocessing as mp
import textwrap
from cognify.optimizer.utils import _cognify_tqdm as tqdm
from cognify.optimizer.registry import get_registered_opt_score_fn

from cognify.graph.utils import get_function_kwargs
from cognify._signal import _should_exit, _be_quiet, _stay_alert
from cognify.graph.program import Workflow, Module
from cognify.llm import Model, Demonstration
from cognify.hub.cogs.common import CogBase
from cognify.hub.cogs.utils import build_param
from cognify.optimizer.plugin import OptimizerSchema, capture_module_from_fs
from cognify.optimizer.core.flow import TopDownInformation, ModuleTransformTrace

from termcolor import colored

logger = logging.getLogger(__name__)

def default_reducer(xs):
    return sum(xs) / len(xs)


# {module_name: demo}
TDemoInTrial = dict[str, Demonstration]

@dataclass
class EvalTaskResult:
    task_index: int
    score: float
    price: float
    exec_time: float
    lm_to_demo: dict
    finished: bool

def get_unfinished_eval_task_result(task_index: int) -> EvalTaskResult:
    return EvalTaskResult(
        task_index,
        score=0.0,
        price=0.0,
        exec_time=0.0,
        lm_to_demo={},
        finished=False
    )

class EvaluationResult:
    def __init__(
        self,
        ids: Sequence[str],
        scores: Sequence[float],
        prices: Sequence[float],
        exec_times: Sequence[float],
        total_eval_cost: float,
        complete: bool,
        reduced_score: Optional[float] = None,
        reduced_price: Optional[float] = None,
        reduced_exec_time: Optional[float] = None,
        demos: Optional[Sequence[TDemoInTrial]] = None,
        converged: bool = False,
        meta: Optional[dict] = None,
    ) -> None:
        self.ids = ids
        self.scores = scores
        self.prices = prices
        self.exec_times = exec_times
        self.total_eval_cost = total_eval_cost
        self.complete = complete
        self.reduced_score = reduced_score
        self.reduced_price = reduced_price
        self.reduced_exec_time = reduced_exec_time
        self.demos = demos
        self.converged = converged
        self.meta = meta

    def __str__(self) -> str:
        return (
            f"EvalResult: score: {self.reduced_score}, "
            f"price: {self.reduced_price}, "
            f"{len(self.scores)} samples, "
            f"latency: {self.reduced_exec_time}, "
            f"eval cost: {self.total_eval_cost}, "
            f"avg exec time: {sum(self.exec_times) / len(self.exec_times)} s"
        )

    def to_dict(self):
        """return result stats

        meta and demos are not included
        """
        stats = {}
        stats["summary"] = {
            "reduced_score": self.reduced_score,
            "reduced_price": self.reduced_price,
            "reduced_exec_time": self.reduced_exec_time,
            "total_eval_cost": self.total_eval_cost,
            "converged": self.converged,
            "complete": self.complete,
        }
        stats["detailed"] = []
        for id, score, price, exec_time in zip(
            self.ids, self.scores, self.prices, self.exec_times
        ):
            stats["detailed"].append(
                {
                    "id": id,
                    "score": score,
                    "price": price,
                    "exec_time": exec_time,
                }
            )
        return stats

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            ids=[d["id"] for d in data["detailed"]],
            scores=[d["score"] for d in data["detailed"]],
            prices=[d["price"] for d in data["detailed"]],
            exec_times=[d["exec_time"] for d in data["detailed"]],
            total_eval_cost=data["summary"]["total_eval_cost"],
            complete=data["summary"]["complete"],
            reduced_score=data["summary"]["reduced_score"],
            reduced_price=data["summary"]["reduced_price"],
            reduced_exec_time=data["summary"]["reduced_exec_time"],
            converged=data["summary"]["converged"],
        )


class EvalFn:
    def __init__(
        self,
        score_fn: Optional[Callable] = None,
        score_file_path: Optional[str] = None,
    ):
        self.score_fn = score_fn
        self.score_file_path = score_file_path
        if self.score_fn is not None:
            self.input_fields, self.defaults = get_function_kwargs(self.score_fn)
        else:
            self.input_fields, self.defaults = [], {}

    def _set_score_fn(self):
        assert self.score_file_path is not None, "score file path should be given"
        # load the score file
        dir = os.path.dirname(self.score_file_path)
        if dir not in sys.path:
            sys.path.append(dir)

        module = capture_module_from_fs(self.score_file_path, mode="score")
        score_fn = get_registered_opt_score_fn()
        if score_fn is None:
            raise ValueError("No score function found in the config file")
        self.score_fn = score_fn

        assert self.score_fn is not None, "score function not set properly"
        self.input_fields, self.defaults = get_function_kwargs(self.score_fn)

    def score(self, state: dict):
        # lazy load the score function
        # This to avoid pickling the score function
        if self.score_fn is None:
            self._set_score_fn()
        for field in self.input_fields:
            if field not in self.defaults and field not in state:
                raise ValueError(
                    f"Missing field `{field}` in state when calling the evaluator\nAvailable fields: {state.keys()}"
                )
        kargs = {k: state.get(k) for k in state if k in self.input_fields}
        return self.score_fn(**kargs)


@dataclass
class EvalTask:
    """Define a task to evaluate the score of a workflow"""

    # execution env
    script_path: str
    args: list[str]  # cmd args to the script
    other_python_paths: list[str]

    # transformation meta
    all_params: dict[str, CogBase]  # {param_hash: param}
    module_name_paths: dict[str, str]
    aggregated_proposals: dict[
        str, dict[str, list[tuple[str, str]]]
    ]  # {layer_name: {module_name: [(param, option)]}}
    trace_back: list[str] = field(default_factory=list)

    def __getstate__(self) -> object:
        state = copy.deepcopy(self.__dict__)
        state.pop("all_params")
        state["all_params_ser"] = {}
        param_hashes = self.all_params.keys()
        for hash in param_hashes:
            state["all_params_ser"][hash] = self.all_params[hash].to_dict()
        return state

    def __setstate__(self, state: dict) -> None:
        param_pool = state.pop("all_params_ser")
        self.__dict__.update(state)
        self.all_params = {}
        # restore params
        for hash, param_dict in param_pool.items():
            self.all_params[hash] = build_param(param_dict)

    def to_dict(self) -> dict:
        return self.__getstate__()

    @classmethod
    def from_dict(cls, state: dict) -> "EvalTask":
        obj = cls.__new__(cls)
        obj.__setstate__(state)
        return obj

    def add_PYTHON_PATH(self, evaluator_path: str):
        dirs = [os.path.dirname(self.script_path), os.path.dirname(evaluator_path)]
        for dir in dirs:
            if dir not in sys.path:
                sys.path.append(dir)
        if self.other_python_paths is not None:
            for path in self.other_python_paths:
                if path not in sys.path:
                    sys.path.append(path)

    def replay_module_transformations(self, ms: list[Module]) -> dict[str, Module]:
        """Replay the module transformations"""

        module_pool = {m.name: m for m in ms}
        module_ttrace = ModuleTransformTrace({m.name: type(m) for m in ms})
        # collect all module names that will be transformed
        all_opt_target_name = set()
        for proposal in self.aggregated_proposals.values():
            all_opt_target_name.update(proposal.keys())

        for layer_name, proposal in self.aggregated_proposals.items():
            for module_name, l_param_option in proposal.items():
                module = module_pool[module_name]
                for param_name, option_name in l_param_option:
                    param_hash = CogBase.chash(module_name, param_name)
                    param = self.all_params[param_hash]

                    new_module, new_mapping = param.apply_option(option_name, module)

                    for old_name, new_name in new_mapping.items():
                        module_ttrace.add_mapping(old_name, new_name)
                    module_pool[new_module.name] = new_module
                    for next_opt_target in Module.all_with_predicate(
                        [new_module], lambda m: m.name in all_opt_target_name
                    ):
                        module_pool[next_opt_target.name] = next_opt_target
                    module = new_module

        # check if modules are transformed correctly
        # check equivalence of current name path and registered name path
        assert module_ttrace.eq_transform_path(
            self.module_name_paths
        ), "Module transformation not consistent"

        module_ttrace.mflatten()
        new_modules_dict = {
            ori_name: module_pool[new_name]
            for ori_name, new_name in module_ttrace.flattened_name_paths.items()
        }
        return new_modules_dict

    def get_program_schema(self) -> OptimizerSchema:
        sys.argv = [self.script_path] + self.args
        schema = OptimizerSchema.capture(self.script_path)

        logger.debug(f"opt_target_modules = {schema.opt_target_modules}")
        assert schema.opt_target_modules, "No module to optimize"
        return schema

    def load_and_transform(self) -> Tuple[OptimizerSchema, dict[str, Module]]:
        """Load workflow into memory and transform it with the aggregated proposals
        """
        schema = self.get_program_schema()
        module_pool = {m.name: m for m in schema.opt_target_modules}

        # replace module invoke with new module
        # this does not replace the model but only the invoke function
        if self.aggregated_proposals:
            module_pool = self.replay_module_transformations(schema.opt_target_modules)
            for m in schema.opt_target_modules:
                new_module = module_pool[m.name]
                if isinstance(new_module, Workflow):
                    new_module.compile()
                logger.debug(f"replace {m} with {new_module}")
                m.invoke = new_module.invoke

        # clear execution state
        for m in module_pool.values():
            m.reset()
        return schema, module_pool

    def evaluate_program(
        self,
        evaluator: EvalFn,
        input,
        label,
        task_index,
        sema,
        q: mp.Queue,
    ):
        # disable interruption info for running each data
        _be_quiet()
        # directly raise interrupt signal
        _stay_alert()

        sema.acquire()
        try:
            if not isinstance(input, dict):
                raise ValueError(f"Input from data loader should be a dict, got {input}")
            if not isinstance(label, dict):
                raise ValueError(f"Label from data loader should be a dict, got {label}")

            schema, module_pool = self.load_and_transform()
            workflow_args, workflow_defaults = get_function_kwargs(schema.program)
            # check if all required fields are provided
            for field in workflow_args:
                if field not in workflow_defaults and field not in input:
                    raise ValueError(
                        f"Missing field `{field}` in input when calling the workflow\nAvailable fields: {input.keys()}"
                    )

            start_time = time.time()
            end_time = time.time()
            score = 0.0
            price = 0.0
            exec_time = 0.0
            lm_to_demo = {}

            try:
                result = schema.program(**input)
                end_time = time.time()
                # merge input/result/label to a single dict
                state = {**input, **result, **label}
                score = evaluator.score(state)
            except Exception as e:
                # catch any errors thrown during the workflow and treat as an invalid result by scoring 0
                # Note: scoring 0 may be problematic if the evaluator's range includes negative numbers
                logger.error(f"Workflow execution threw error for task {task_index}: {e}. Automatic score of 0")
                end_time = time.time()  # this isn't accurate if the process is interrupted
            finally:
                # get price and demo of running the program
                for lm in Module.all_of_type(module_pool.values(), Model):
                    price += lm.get_total_cost()
                    demo = lm.get_last_step_as_demo()
                    if demo is not None:
                        lm_to_demo[lm.name] = demo

                exec_time = end_time - start_time
                q.put(
                    EvalTaskResult(
                        task_index,
                        score,
                        price,
                        exec_time,
                        lm_to_demo,
                        finished=True
                    )
                )
        except KeyboardInterrupt:
            q.put(get_unfinished_eval_task_result(task_index))
        except Exception as e:
            q.put(get_unfinished_eval_task_result(task_index))
            raise
        finally:
            sema.release()

    def show_opt_trace(self) -> str:
        trace_lines = []
        trace_lines.append("********** Detailed Optimization Trace **********\n")

        for layer, proposals in self.aggregated_proposals.items():
            trace_lines.append(f"========== Layer: {layer} ==========")

            for module_name, param_options in proposals.items():
                trace_lines.append(f"\n  >>> Module: {module_name} <<<")

                for param_name, option_name in param_options:
                    param_hash = CogBase.chash(module_name, param_name)
                    param = self.all_params[param_hash]
                    class_path = (
                        f"{param.__class__.__module__}.{param.__class__.__name__}"
                    )
                    trace_lines.append(f"\n    - Parameter: <{class_path}>")
                    trace_lines.append(f"      Applied Option: {option_name}")
                    # Get the description with indentation for each line
                    option_description = param.options[option_name].describe()
                    option_description = textwrap.dedent(option_description)

                    indented_description = "\n".join(
                        [f"        {line}" for line in option_description.splitlines()]
                    )
                    trace_lines.append(
                        f"      Transformation Details:\n{indented_description}"
                    )

            trace_lines.append("\n" + "=" * 50 + "\n")

        # Combine all trace lines into a single string
        trace_dump = "\n".join(trace_lines)
        return trace_dump

    @classmethod
    def from_top_down_info(cls, tdi: TopDownInformation):
        return cls(
            script_path=tdi.script_path,
            args=tdi.script_args,
            other_python_paths=tdi.other_python_paths,
            all_params=tdi.all_params,
            module_name_paths=tdi.module_transformation_trace.module_name_paths,
            aggregated_proposals=tdi.module_transformation_trace.aggregated_proposals,
            trace_back=tdi.trace_back,
        )


class GeneralEvaluatorInterface(ABC):
    @abstractmethod
    def evaluate(
        self,
        task: Union[EvalTask, TopDownInformation],
        **kwargs,
    ) -> EvaluationResult: ...


class EvaluatorPlugin(GeneralEvaluatorInterface):
    def __init__(
        self,
        trainset: Optional[Iterable[Tuple[any, any]]],  # list of input data and labels
        evalset: Optional[Iterable[Tuple[any, any]]],  # list of input data and labels
        testset: Optional[Iterable[Tuple[any, any]]],  # list of input data and labels
        evaluator_fn: Optional[EvalFn] = None,
        evaluator_path: Optional[str] = None,
        n_parallel: int = 10,
        score_reducer: Callable = None,
        price_reducer: Callable = None,
        exec_time_reducer: Callable = None,
    ):
        """Specify the evaluation method

        If both score_fn and score_file_path are provided, score file will be used

        If you have non-copyable or non-picklable evaluation function, consider specifying the path to the score file
        """
        self.dataset = {
            "train": [trainset, None if not trainset else list(range(len(trainset)))],
            "eval": [evalset, None if not evalset else list(range(len(evalset)))],
            "test": [testset, None if not testset else list(range(len(testset)))],
        }

        self.n_parallel = n_parallel
        self.score_reducer = (
            score_reducer if score_reducer is not None else default_reducer
        )
        self.price_reducer = (
            price_reducer if price_reducer is not None else default_reducer
        )
        self.exec_time_reducer = (
            exec_time_reducer if exec_time_reducer is not None else default_reducer
        )

        self._evaluator = EvalFn(score_fn=evaluator_fn, score_file_path=evaluator_path)

    def evaluate(
        self,
        task: EvalTask,
        frac: float = None,
        **kwargs,
    ):
        return self.get_score(
            mode="train",
            task=task,
            frac=frac,
        )

    def get_score(
        self,
        mode: Literal["train", "eval", "test"],
        task: EvalTask,
        frac: float = None,
        show_progress_bar: bool = False,
        is_dry_run: bool = False,
    ):
        logger.debug(f"sys_path = {sys.path}")

        data, indices = self.dataset[mode]
        n_parallel = min(self.n_parallel, len(indices))

        # Task queue to limit the number of parallel tasks
        # avoid worker pool to avoid reusing the same process
        all_workers = []
        sema = mp.Semaphore(n_parallel)
        result_q = mp.Queue()


        results = []
        for task_index, pair_idx in enumerate(indices):
            if _should_exit():
                break

            input, label = data[pair_idx]
            worker = mp.Process(
                target=task.evaluate_program,
                args=(self._evaluator, input, label, task_index, sema, result_q),
            )
            worker.start()
            all_workers.append(worker)

        from cognify.optimizer.trace.progress_info import pbar
        for i in tqdm(range(len(all_workers)),
                    colour='green',
                    leave=False,
                    disable=not show_progress_bar):
            eval_task_result: EvalTaskResult = result_q.get()
            if not eval_task_result.finished:
                continue
            results.append(eval_task_result)
            if frac is not None:
                pbar.update_progress(frac/len(indices))
        
       
        for worker in all_workers:
            worker.join()

        if not results:
            return EvaluationResult(
                ids=[f"{mode}_{i}" for i in indices],
                scores=[0.0],
                prices=[0.0],
                exec_times=[0.0],
                total_eval_cost=0.0,
                complete=False,
            )

        # re-order the results according to the task index
        results = sorted(results, key=lambda x: x.task_index)

        data_ids = []
        prices = []
        scores = []
        demos = []
        exec_times = []

        for eval_task_result in results:
            assert eval_task_result.finished, "Only finished tasks should be collected"
            data_ids.append(indices[eval_task_result.task_index])
            prices.append(eval_task_result.price)
            scores.append(eval_task_result.score)
            exec_times.append(eval_task_result.exec_time)
            demos.append(eval_task_result.lm_to_demo)

        reduced_score = self.score_reducer(scores)
        reduced_price = self.price_reducer(prices)
        reduced_exec_time = self.exec_time_reducer(exec_times)

        if is_dry_run:
            print(f"Original result before optimization | quality: {reduced_score:.2f}, cost@1000: ${reduced_price*1000:.2f}, avg latency: {reduced_exec_time:.2f}s")

        return EvaluationResult(
            ids=[f"{mode}_{i}" for i in data_ids],
            scores=scores,
            prices=prices,
            exec_times=exec_times,
            total_eval_cost=sum(prices),
            complete=len(results) == len(indices),
            reduced_score=reduced_score,
            reduced_price=reduced_price,
            reduced_exec_time=reduced_exec_time,
            demos=demos,
        )

    def down_sample(
        self,
        mode: Literal["train", "eval", "test"],
        sample_size: int,
        task: EvalTask,
        sample_mode: Literal["random", "difficulty"],
        prob_convertor: Callable[[EvaluationResult], Sequence[int]] = None,
        log_dir: str = "eval_down_sample_logs",
    ):
        """Generate a subset of the dataset according to answer score

        The objective is to reduce the evaluation cost with the following two principles:

        1. subset should have good coverage of the input space, spanning from easy to hard
        2. harder questions are more important

        In case the task itself does not provide meaningful comparative scores (e.g. classification task), use `random` sample mode to randomly sample from the eval_set or use `difficulty` at your own risk.

        NOTE: since we only care about comparative score, maybe use the most efficient config with least bias (e.g. cheap models) to evaluate the subset

        The default prob_convertor works fine for score within range[0,1], but you can provide a custom one if needed

        also please be informed that we always assume score is higher the better
        """
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"{mode}_sub_ids.json")

        # validate
        full_indices = list(range(len(self.dataset[mode][0])))
        if sample_size > len(full_indices):
            raise ValueError(
                f"Sample size {sample_size} is larger than the full dataset size {len(full_indices)}"
            )

        if os.path.exists(log_path):
            logger.info(f"Loading downsampled indices from {log_path}")
            indices = json.load(open(log_path, "r"))
            if len(indices) != sample_size:
                raise ValueError(
                    f"Loaded data size {len(indices)} does not match sample size {sample_size}"
                )
            self.dataset[mode][1] = indices
            return

        if sample_mode == "random":
            indices = np.random.choice(
                full_indices, size=sample_size, replace=False
            ).tolist()
        else:
            logger.info("Down sampling with difficulty, start profiling...")
            dry_run_path = os.path.join(log_dir, f"dry_run_{mode}.json")
            if os.path.exists(dry_run_path):
                logger.info(f"Loading dry run results from {dry_run_path}")
                eval_result = EvaluationResult.from_dict(
                    json.load(open(dry_run_path, "r"))
                )
            else:
                print("Dry run with downsampling on the dataset...") # TODO: https://github.com/GenseeAI/cognify/issues/35
                eval_result = self.get_score(mode, task, show_progress_bar=True)
                with open(dry_run_path, "w+") as f:
                    json.dump(eval_result.to_dict(), f, indent=4)
            # if user provide a custom prob convertor
            if prob_convertor is not None:
                probs = prob_convertor(eval_result)
                indices = np.random.choice(
                    full_indices, size=sample_size, replace=False, p=probs
                ).tolist()
            else:
                # sampling prob is reverse to the score
                # also smooth it to reduce extremely easy or hard questions
                def transform(x):
                    return np.exp(-x)

                scaled_reverse_score = transform(np.array(eval_result.scores))
                # normalize to prob
                probs = scaled_reverse_score / scaled_reverse_score.sum()
                # sample according to the prob
                indices = np.random.choice(
                    full_indices, size=sample_size, replace=False, p=probs
                ).tolist()

        json.dump(sorted(indices), open(log_path, "w"), indent=4)
        self.dataset[mode][1] = indices
