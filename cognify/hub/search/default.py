from typing import Literal

from cognify.llm import LMConfig
from cognify.optimizer.core import driver, flow
from cognify.hub.cogs import reasoning, ensemble, model_selection
from cognify.hub.cogs.common import NoChange
from cognify.hub.cogs.fewshot import LMFewShot
from cognify.hub.cogs.reasoning import ZeroShotCoT, PlanBefore
from cognify.optimizer.control_param import ControlParameter
from dataclasses import dataclass

from cognify._tracing import trace_default_search

@dataclass
class SearchParams:
    n_trials: int
    quality_constraint: float = 1.0
    evaluator_batch_size: int = 10
    opt_log_dir: str = "opt_results"
    model_selection_cog: model_selection.LMSelection = None


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
        throughput=outer_throughput
    )
    outer_loop_config = driver.LayerConfig(
        layer_name="medium_outer",
        universal_params=[general_ensemble_params],
        opt_config=outer_opt_config,
    )

    # ================= Overall Control Parameter =================
    optimize_control_param = ControlParameter(
        opt_layer_configs=[outer_loop_config, inner_loop_config],
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
        throughput=outer_throughput,
        use_SH_allocation=True,
    )
    outer_loop_config = driver.LayerConfig(
        layer_name="heavy_outer",
        universal_params=[general_ensemble_params],
        opt_config=outer_opt_config,
    )

    # ================= Overall Control Parameter =================
    optimize_control_param = ControlParameter(
        opt_layer_configs=[outer_loop_config, inner_loop_config],
        opt_history_log_dir=search_params.opt_log_dir,
        evaluator_batch_size=search_params.evaluator_batch_size,
        quality_constraint=search_params.quality_constraint,
    )
    return optimize_control_param


def create_search(
    *,
    search_type: Literal["light", "medium", "heavy"] = "light",
    model_selection_cog: model_selection.LMSelection | list[LMConfig] | None = None,
    n_trials: int = None,
    quality_constraint: float = 1.0,
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
            n_trials = 10
        elif search_type == "medium":
            n_trials = 45
        elif search_type == "heavy":
            n_trials = 140
        else:
            raise ValueError(f"Invalid search type: {search_type}")

    search_params = SearchParams(
        n_trials,
        quality_constraint,
        evaluator_batch_size,
        opt_log_dir,
        model_selection_cog,
    )

    trace_default_search(search_type, quality_constraint)

    if search_type == "light":
        return create_light_search(search_params)
    elif search_type == "medium":
        return create_medium_search(search_params)
    elif search_type == "heavy":
        return create_heavy_search(search_params)
    else:
        raise ValueError(f"Invalid search type: {search_type}")
