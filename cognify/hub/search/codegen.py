from cognify.optimizer.core import driver, flow
from cognify.hub.cogs import reasoning, ensemble, model_selection
from cognify.hub.cogs.common import NoChange
from cognify.hub.cogs.fewshot import LMFewShot
from cognify.hub.cogs.reasoning import ZeroShotCoT
from cognify.hub.search.default import SearchParams
from cognify.optimizer.control_param import ControlParameter
from cognify.llm import LMConfig
from cognify._tracing import trace_custom_search


def create_codegen_search(search_params: SearchParams) -> ControlParameter:
    # ================= Inner Loop Config =================
    # Reasoning Parameter
    reasoning_param = reasoning.LMReasoning([NoChange(), ZeroShotCoT()])
    # Few Shot Parameter
    few_shot_params = LMFewShot(4)

    # Layer Config
    inner_opt_config = flow.OptConfig(
        n_trials=6,
    )

    params = [few_shot_params, reasoning_param]
    if search_params.model_selection_cog:
        params.append(search_params.model_selection_cog)
    inner_loop_config = driver.LayerConfig(
        layer_name="codegen_inner_loop",
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
    outer_trials = int((search_params.n_trials / 6 + 1) // 2)
    if outer_trials == 0:
        outer_trials += 1

    outer_throughput = 4 if outer_trials > 4 else outer_trials

    outer_opt_config = flow.OptConfig(
        n_trials=outer_trials,
        num_parallel_proposal=outer_throughput,
    )
    outer_loop_config = driver.LayerConfig(
        layer_name="codegen_outer_loop",
        universal_params=[general_ensemble_params],
        opt_config=outer_opt_config,
    )

    optimize_control_param = ControlParameter(
        opt_layer_configs=[outer_loop_config, inner_loop_config],
        opt_history_log_dir=search_params.opt_log_dir,
        evaluator_batch_size=search_params.evaluator_batch_size,
        quality_constraint=search_params.quality_constraint,
    )
    return optimize_control_param


def create_search(
    *,
    n_trials: int = 18,
    quality_constraint: float = 1.0,
    evaluator_batch_size: int = 10,
    opt_log_dir: str = "opt_results",
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

    search_params = SearchParams(
        n_trials,
        quality_constraint,
        evaluator_batch_size,
        opt_log_dir,
        model_selection_cog,
    )
    trace_custom_search("codegen", n_trials, quality_constraint)
    return create_codegen_search(search_params)
