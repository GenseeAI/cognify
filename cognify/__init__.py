import multiprocessing as mp
if mp.get_start_method(allow_none=True) != "spawn":
    print("Loading workflow...")

from .llm import (
    Model,
    StructuredModel,
    LMConfig,
    Input,
    OutputLabel,
    OutputFormat,
    Demonstration,
    FilledInput,
)
from .frontends.dspy.connector import PredictModel, as_predict
from .frontends.langchain.connector import RunnableModel, as_runnable

from cognify import llm, optimizer

from cognify.run.evaluate import evaluate
from cognify.run.run import run, load_workflow
from cognify.run.optimize import optimize
from cognify.run.inspect import inspect
from cognify.optimizer import (
    register_workflow,
    register_evaluator,
    register_data_loader,
    clear_registry,
)

from cognify.optimizer.evaluator import EvaluationResult
from cognify import _logging

from cognify.hub.search.default import create_search as create_default_search


__all__ = [
    "llm",
    "optimizer",
    "cogs",
    "evaluate",
    "load_workflow",
    "optimize",
    "inspect",
    "run",
    "EvaluationResult",
    "create_default_search",
    "register_workflow",
    "register_evaluator",
    "register_data_loader",
    "clear_registry",
    "Model",
]