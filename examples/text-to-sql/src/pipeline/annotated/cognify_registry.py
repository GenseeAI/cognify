from . import candidate_generation
from . import column_filtering
from . import column_selection
from . import keyword_extraction
from . import revision
from . import table_selection
import cognify
from cognify.hub.cogs.reasoning import ZeroShotCoT
from llm.models import get_llm_params

add_cot = False

if add_cot:
    ZeroShotCoT.direct_apply(column_filtering.exec)
    ZeroShotCoT.direct_apply(table_selection.exec)
    ZeroShotCoT.direct_apply(column_selection.exec)
    ZeroShotCoT.direct_apply(candidate_generation.exec)
    ZeroShotCoT.direct_apply(revision.exec)

_cognify_lm_registry = {
    'keyword_extraction': keyword_extraction.exec,
    'column_filtering': column_filtering.exec,
    'table_selection': table_selection.exec,
    'column_selection': column_selection.exec,
    'candidate_generation': candidate_generation.exec,
    'revision': revision.exec  
}

def set_lm_config_statically(pipeline_cfg):
    """
    args should be the same across all inputs
    """
    # Set up the language model configurations
    for node_name, cfg in pipeline_cfg.items():
        if node_name in _cognify_lm_registry:
            lm: cognify.Model = _cognify_lm_registry[node_name]
            engine_name = cfg["engine"]
            temperature = cfg.get("temperature", 0)
            base_uri = cfg.get("base_uri", None)
            
            lm_params_cpy = get_llm_params(engine=engine_name, temperature=temperature, base_uri=base_uri).copy()
            model = lm_params_cpy.pop("model")
            lm.lm_config = cognify.LMConfig(
                custom_llm_provider='openai',
                model=model,
                kwargs=lm_params_cpy,
            )

_pipeline_cfg = {
    "keyword_extraction": {
        "engine": "o1-preview",
        # "temperature": 0.2,
        "base_uri": ""
    },
    "column_filtering": {
        "engine": "o1-preview",
        # "temperature": 0.0,
        "base_uri": ""
    },
    "table_selection": {
        "mode": "ask_model",
        "engine": "o1-preview",
        # "temperature": 0.0,
        "base_uri": "",
        "sampling_count": 1
    },
    "column_selection": {
        "mode": "ask_model",
        "engine": "o1-preview",
        # "temperature": 0.0,
        "base_uri": "",
        "sampling_count": 1
    },
    "candidate_generation": {
        "engine": "o1-preview",
        # "temperature": 0.0,
        "base_uri": "",
        "sampling_count": 1
    },
    "revision": {
        "engine": "o1-preview",
        # "temperature": 0.0,
        "base_uri": "",
        "sampling_count": 1
    }
}
set_lm_config_statically(_pipeline_cfg)