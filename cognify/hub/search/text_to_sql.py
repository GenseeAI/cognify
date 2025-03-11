from cognify.optimizer.core import driver, flow
from cognify.hub.cogs import reasoning, ensemble, model_selection
from cognify.hub.cogs.common import NoChange
from cognify.hub.cogs.fewshot import LMFewShot
from cognify.hub.cogs.reasoning import ZeroShotCoT, PlanBefore
from cognify.hub.search.default import SearchParams
from cognify.optimizer.control_param import ControlParameter
from cognify.llm import LMConfig
from cognify._tracing import trace_custom_search


def create_text_to_sql_search(search_params: SearchParams) -> ControlParameter:
    # ================= Reasoning Options =================
    reasoning_param = reasoning.LMReasoning([NoChange(), ZeroShotCoT(), PlanBefore()] 
    )
    # ================= Few Shot Options =================
    few_shot_params = LMFewShot(4)

    # ================= Inner Loop Config =================
    inner_opt_config = flow.OptConfig(
        n_trials=5,
        num_parallel_proposal=2,
    )
    inner_loop_config = driver.LayerConfig(
        layer_name='inner_loop',
        universal_params=[few_shot_params, reasoning_param],
        opt_config=inner_opt_config,
    )

    # ================= Ensemble Options =================
    def add_ensemble_option(lm_name):
        usc_ensemble = ensemble.UniversalSelfConsistency(3, temperature=0.9)
        ensemble_param = ensemble.ModuleEnsemble(
            [usc_ensemble]
        )
        ensemble_param.module_name = lm_name
        return ensemble_param

    ensemble_params = [
        add_ensemble_option('table_selection'),
        add_ensemble_option('candidate_generation'),
        add_ensemble_option('revision'),
    ]

    # ================= Outer Loop Config =================
    outer_trials = search_params.n_trials // 5
    if outer_trials == 0:
        outer_trials += 1

    outer_throughput = 2 if outer_trials > 2 else outer_trials
    outer_opt_config = flow.OptConfig(
        n_trials=outer_trials,
        num_parallel_proposal=outer_throughput,
    )

    outer_loop_config = driver.LayerConfig(
        layer_name='outer_loop',
        dedicate_params=ensemble_params,
        opt_config=outer_opt_config,
    )

    opt_layer_configs = [outer_loop_config, inner_loop_config]

    optimize_control_param = ControlParameter(
        opt_layer_configs=opt_layer_configs,
        opt_history_log_dir=search_params.opt_log_dir,
        evaluator_batch_size=search_params.evaluator_batch_size,
        quality_constraint=search_params.quality_constraint,
        train_down_sample=50,
        val_down_sample=25
    )
    return optimize_control_param


def create_search(
    *,
    n_trials: int = 20,
    quality_constraint: float = 0.98,
    evaluator_batch_size: int = 20,
    opt_log_dir: str = "cognify_opt_results",
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
    trace_custom_search("text_to_sql", n_trials, quality_constraint)
    return create_text_to_sql_search(search_params)
