from cognify._imports import _LazyModule

NoChange = _LazyModule("cognify.hub.cogs.common.NoChange")
CogBase = _LazyModule("cognify.hub.cogs.common.CogBase")
OptionBase = _LazyModule("cognify.hub.cogs.common.OptionBase")
DynamicCogBase = _LazyModule("cognify.hub.cogs.common.DynamicCogBase")

DecomposeCandidate = _LazyModule("cognify.hub.cogs.decompose.DecomposeCandidate")
LMTaskDecompose = _LazyModule("cognify.hub.cogs.decompose.LMTaskDecompose")

ModuleEnsemble = _LazyModule("cognify.hub.cogs.ensemble.ModuleEnsemble")
UniversalSelfConsistency = _LazyModule("cognify.hub.cogs.ensemble.UniversalSelfConsistency")

LMFewShot = _LazyModule("cognify.hub.cogs.fewshot.LMFewShot")

model_option_factory = _LazyModule("cognify.hub.cogs.model_selection.model_option_factory")
LMSelection = _LazyModule("cognify.hub.cogs.model_selection.LMSelection")
ModelOption = _LazyModule("cognify.hub.cogs.model_selection.ModelOption")

LMReasoning = _LazyModule("cognify.hub.cogs.reasoning.LMReasoning")
ZeroShotCoT = _LazyModule("cognify.hub.cogs.reasoning.ZeroShotCoT")
PlanBefore = _LazyModule("cognify.hub.cogs.reasoning.PlanBefore")

TreeOfThought = _LazyModule("cognify.hub.cogs.tree_of_thoughts.tot.TreeOfThought")

__all__ = [
    "NoChange",
    "CogBase",
    "OptionBase",
    "DynamicCogBase",
    "DecomposeCandidate",
    "LMTaskDecompose",
    "ModuleEnsemble",
    "UniversalSelfConsistency",
    "LMFewShot",
    "model_option_factory",
    "LMSelection",
    "ModelOption",
    "LMReasoning",
    "ZeroShotCoT",
    "PlanBefore",
    "TreeOfThought",
]