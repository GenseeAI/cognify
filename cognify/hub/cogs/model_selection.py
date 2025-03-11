from cognify.hub.cogs.common import CogBase, CogLayerLevel, OptionBase
from cognify.llm.model import Model, LMConfig
import copy


class LMSelection(CogBase):
    level = CogLayerLevel.NODE

    @classmethod
    def from_dict(cls, data: dict):
        name, module_name, default_option, options = (
            data["name"],
            data["module_name"],
            data["default_option"],
            data["options"],
        )
        options = [
            ModelOption(LMConfig.from_dict(dat["model_config"]), tag)
            for tag, dat in options.items()
        ]
        return cls(
            name=name,
            options=options,
            default_option=default_option,
            module_name=module_name,
        )


class ModelOption(OptionBase):
    def __init__(self, model_config: LMConfig, tag: str = None):
        # NOTE: this assumes provider + model is unique
        # use this as tag to increase config readability
        tag = tag or f"{model_config.custom_llm_provider}_{model_config.model}"
        super().__init__(tag)
        # NOTE: deepcopy is necessary in case module config is shared in memory
        self.model_config = copy.deepcopy(model_config)

    def _get_cost_indicator(self):
        return self.model_config.cost_indicator

    def apply(self, lm_module: Model):
        if lm_module.lm_config:
            lm_module.lm_config.update(self.model_config)
        else:
            lm_module.lm_config = self.model_config
        return lm_module

    def to_dict(self):
        base = super().to_dict()
        base["model_config"] = self.model_config.to_dict()
        return base


def model_option_factory(model_configs: list[LMConfig]):
    return [ModelOption(cfg) for cfg in model_configs]
