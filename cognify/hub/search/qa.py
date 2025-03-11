from typing import Literal
from cognify.optimizer.core import driver, flow
from cognify.optimizer.control_param import ControlParameter
from cognify.hub.cogs import reasoning, ensemble, model_selection
from cognify.hub.cogs.common import NoChange
from cognify.hub.cogs.fewshot import LMFewShot
from cognify.hub.cogs.reasoning import ZeroShotCoT
from cognify.hub.search.default import SearchParams, parse_objectives
from cognify.llm import LMConfig
from cognify._tracing import trace_custom_search

def create_qa_search(search_params: SearchParams) -> ControlParameter:
    # ================= Inner Loop Config =================
    # Reasoning Parameter
    reasoning_param = reasoning.LMReasoning([NoChange(), ZeroShotCoT()])
    # Few Shot Parameter
    few_shot_params = LMFewShot(2)

    # Layer Config
    inner_opt_config = flow.OptConfig(
        n_trials=5,
    )
    inner_loop_config = driver.LayerConfig(
        layer_name="inner_loop",
        universal_params=[few_shot_params, reasoning_param],
        opt_config=inner_opt_config,
    )

    # ================= Outer Loop Config =================
    # Ensemble Parameter
    general_usc_ensemble = ensemble.UniversalSelfConsistency(3)
    general_ensemble_params = ensemble.ModuleEnsemble(
        [NoChange(), general_usc_ensemble]
    )
    # Layer Config
    outer_trials = int((search_params.n_trials / 5 + 1) // 2)
    if outer_trials == 0:
        outer_trials += 1

    outer_throughput = 2 if outer_trials > 2 else outer_trials
    outer_opt_config = flow.OptConfig(
        n_trials=outer_trials, 
        num_parallel_proposal=outer_throughput,
    )
    outer_loop_config = driver.LayerConfig(
        layer_name="outer_loop",
        universal_params=[general_ensemble_params],
        opt_config=outer_opt_config,
    )
    optimize_control_param = ControlParameter(
        opt_layer_configs=[outer_loop_config, inner_loop_config],
        opt_history_log_dir=search_params.opt_log_dir,
        evaluator_batch_size=search_params.evaluator_batch_size,
        quality_constraint=search_params.quality_constraint,
        objectives=search_params.objectives
    )
    return optimize_control_param


def create_search(
    *,
    n_trials: int = 15,
    quality_constraint: float = 1.0,
    evaluator_batch_size: int = 10,
    opt_log_dir: str = "opt_results",
    objectives: list[Literal["quality", "cost", "latency"]] = ["quality", "cost", "latency"],
    model_selection_cog: model_selection.LMSelection | list[LMConfig] | None = None,
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

    selected_objectives = parse_objectives(objectives)

    search_params = SearchParams(
        n_trials,
        selected_objectives,
        quality_constraint,
        evaluator_batch_size,
        opt_log_dir,
        model_selection_cog,
    )
    trace_custom_search("qa", n_trials, quality_constraint)
    return create_qa_search(search_params)
