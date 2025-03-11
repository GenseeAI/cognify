from typing import Literal

from cognify.llm import LMConfig
from cognify.optimizer.core import driver, flow
from cognify.hub.cogs import reasoning, ensemble, model_selection
from cognify.hub.cogs.common import NoChange
from cognify.hub.cogs.fewshot import LMFewShot
from cognify.hub.cogs.reasoning import *
from cognify.optimizer.control_param import ControlParameter, SelectedObjectives
from dataclasses import dataclass

from cognify._tracing import trace_default_search

@dataclass
class SearchParams:
    n_trials: int
    objectives: SelectedObjectives
    quality_constraint: float = 1.0
    evaluator_batch_size: int = 10
    opt_log_dir: str = "opt_results"
    model_selection_cog: model_selection.LMSelection = None


def create_testOpt_search(search_params: SearchParams) -> ControlParameter:
    # Reasoning Parameter
    reasoning_param = reasoning.LMReasoning([DebateReflection()])
    # Few Shot Parameter
    few_shot_params = LMFewShot(2)

    # Layer Config
    inner_opt_config = flow.OptConfig(
        n_trials=search_params.n_trials,
    )
    params = [reasoning_param] # for unit test, light layer config - few shot = testOpt reasoning
    if search_params.model_selection_cog is not None:
        params.append(search_params.model_selection_cog)
    inner_loop_config = driver.LayerConfig(
        layer_name="light_opt_layer",
        universal_params=params,
        opt_config=inner_opt_config,
    )

    # ================= Overall Control Parameter =================
    optimize_control_param = ControlParameter(
        opt_layer_configs=[inner_loop_config],
        objectives=search_params.objectives,
        opt_history_log_dir=search_params.opt_log_dir,
        evaluator_batch_size=search_params.evaluator_batch_size,
        quality_constraint=search_params.quality_constraint,
    )
    return optimize_control_param


def create_light_search(search_params: SearchParams) -> ControlParameter:
    # Reasoning Parameter
    reasoning_param = reasoning.LMReasoning([NoChange(), ZeroShotCoT()])

    # Few Shot Parameter
    few_shot_params = LMFewShot(2)

    # Layer Config
    inner_opt_config = flow.OptConfig(
        n_trials=search_params.n_trials,
    )
    params = [few_shot_params, reasoning_param]
    if search_params.model_selection_cog is not None:
        params.append(search_params.model_selection_cog)
    inner_loop_config = driver.LayerConfig(
        layer_name="light_opt_layer",
        universal_params=params,
        opt_config=inner_opt_config,
    )

    # ================= Overall Control Parameter =================
    optimize_control_param = ControlParameter(
        opt_layer_configs=[inner_loop_config],
        objectives=search_params.objectives,
        opt_history_log_dir=search_params.opt_log_dir,
        evaluator_batch_size=search_params.evaluator_batch_size,
        quality_constraint=search_params.quality_constraint,
    )
    return optimize_control_param


def create_medium_search(search_params: SearchParams) -> ControlParameter:
    # Assign resource to each layer
    inner_trials = 15
    outer_trials = int((search_params.n_trials / inner_trials + 1) // 2)

    if outer_trials == 0:
        outer_trials += 1

    # ================= Inner Loop Config =================
    # Reasoning Parameter
    reasoning_param = reasoning.LMReasoning([NoChange(), ZeroShotCoT()])

    # Few Shot Parameter
    few_shot_params = LMFewShot(2)

    # Layer Config
    inner_opt_config = flow.OptConfig(
        n_trials=inner_trials,
    )
    params = [few_shot_params, reasoning_param]
    if search_params.model_selection_cog:
        params.append(search_params.model_selection_cog)
    inner_loop_config = driver.LayerConfig(
        layer_name="medium_inner",
        universal_params=params,
        opt_config=inner_opt_config,
    )

    # ================= Outer Loop Config =================
    # Ensemble Parameter
    general_usc_ensemble = ensemble.UniversalSelfConsistency(3)
    general_ensemble_params = ensemble.ModuleEnsemble(
        [NoChange(), general_usc_ensemble]
    )
    # Layer Config
    outer_throughput = 2 if outer_trials > 2 else outer_trials
    outer_opt_config = flow.OptConfig(
        n_trials=outer_trials,
        num_parallel_proposal=outer_throughput
    )
    outer_loop_config = driver.LayerConfig(
        layer_name="medium_outer",
        universal_params=[general_ensemble_params],
        opt_config=outer_opt_config,
    )

    # ================= Overall Control Parameter =================
    optimize_control_param = ControlParameter(
        opt_layer_configs=[outer_loop_config, inner_loop_config],
        objectives=search_params.objectives,
        opt_history_log_dir=search_params.opt_log_dir,
        evaluator_batch_size=search_params.evaluator_batch_size,
        quality_constraint=search_params.quality_constraint,
    )
    return optimize_control_param


def create_heavy_search(search_params: SearchParams) -> ControlParameter:
    # Assign resource to each layer
    # Use SH resource allocation
    # Total trials = inner * (2 * outer - 1)
    inner_trials = 20
    outer_trials = int((search_params.n_trials / inner_trials + 1) // 2)

    if outer_trials == 0:
        outer_trials += 1

    # ================= Inner Loop Config =================
    # Reasoning Parameter
    reasoning_param = reasoning.LMReasoning([NoChange(), ZeroShotCoT(), PlanBefore()])

    # Few Shot Parameter
    few_shot_params = LMFewShot(4)

    # Layer Config
    inner_opt_config = flow.OptConfig(
        n_trials=inner_trials,
    )

    params = [few_shot_params, reasoning_param]
    if search_params.model_selection_cog:
        params.append(search_params.model_selection_cog)
    inner_loop_config = driver.LayerConfig(
        layer_name="heavy_inner",
        universal_params=params,
        opt_config=inner_opt_config,
    )

    # ================= Outer Loop Config =================
    # Ensemble Parameter
    general_usc_ensemble = ensemble.UniversalSelfConsistency(3)
    general_ensemble_params = ensemble.ModuleEnsemble(
        [NoChange(), general_usc_ensemble]
    )
    # Layer Config
    outer_throughput = 2 if outer_trials > 2 else outer_trials
    outer_opt_config = flow.OptConfig(
        n_trials=outer_trials,
        num_parallel_proposal=outer_throughput,
    )
    outer_loop_config = driver.LayerConfig(
        layer_name="heavy_outer",
        universal_params=[general_ensemble_params],
        opt_config=outer_opt_config,
    )

    # ================= Overall Control Parameter =================
    optimize_control_param = ControlParameter(
        opt_layer_configs=[outer_loop_config, inner_loop_config],
        objectives=search_params.objectives,
        opt_history_log_dir=search_params.opt_log_dir,
        evaluator_batch_size=search_params.evaluator_batch_size,
        quality_constraint=search_params.quality_constraint,
    )
    return optimize_control_param


def parse_objectives(objectives: list[Literal["quality", "cost", "latency"]]) -> SelectedObjectives:
    selected_quality = False
    selected_cost = False
    selected_latency = False

    for item in objectives:
        if selected_quality and selected_cost and selected_latency:
            break

        if item == "quality":
            selected_quality = True
        elif item == "cost":
            selected_cost = True
        elif item == "latency":
            selected_latency = True
        else:
            raise ValueError(f"Invalid objective '{item}', options are 'quality', 'cost', 'latency'")

    return SelectedObjectives(selected_quality, selected_cost, selected_latency)

def create_search(
    *,
    search_type: Literal["light", "medium", "heavy"] = "light",
    model_selection_cog: model_selection.LMSelection | list[LMConfig] | None = None,
    n_trials: int = None,
    quality_constraint: float = 1.0,
    objectives: list[Literal["quality", "cost", "latency"]] = ["quality", "cost", "latency"],
    evaluator_batch_size: int = 10,
    opt_log_dir: str = "opt_results",
):
    if model_selection_cog is not None:
        if isinstance(model_selection_cog, list):
            model_selection_options = model_selection.model_option_factory(
                model_selection_cog
            )
            model_selection_cog = model_selection.LMSelection(
                "model_selection",
                model_selection_options,
            )
        assert isinstance(model_selection_cog, model_selection.LMSelection)
    
    if n_trials is None:
        if search_type == "light":
            # n_trials = 10
            n_trials = 1 # for manual unit test
        elif search_type == "medium":
            n_trials = 45
        elif search_type == "heavy":
            n_trials = 140
        else:
            raise ValueError(f"Invalid search type: {search_type}")

    selected_objectives = parse_objectives(objectives)

    search_params = SearchParams(
        n_trials,
        selected_objectives,
        quality_constraint,
        evaluator_batch_size,
        opt_log_dir,
        model_selection_cog,
    )

    trace_default_search(search_type, quality_constraint, list(set(objectives)))

    if search_type == "light":
        #return create_light_search(search_params) 
        return create_testOpt_search(search_params) # for manual unit test
    elif search_type == "medium":
        return create_medium_search(search_params)
    elif search_type == "heavy":
        return create_heavy_search(search_params)
    else:
        raise ValueError(f"Invalid search type: {search_type}")
