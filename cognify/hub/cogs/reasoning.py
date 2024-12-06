from abc import ABCMeta
from typing import List, Union
from cognify.hub.cogs.common import CogBase, CogLayerLevel, OptionBase, NoChange
from cognify.llm import Model, StructuredModel, litellm_completion
from cognify.llm.model import APICompatibleMessage
from litellm import ModelResponse
import copy

import logging

logger = logging.getLogger(__name__)


class LMReasoning(CogBase):
    level = CogLayerLevel.NODE

    def __init__(
        self,
        options: list[OptionBase],
        name: str = "reasoning",
        default_option: Union[int, str] = 0,
        module_name: str = None,
        inherit: bool = True,
    ):
        return super().__init__(name, options, default_option, module_name, inherit)

    @classmethod
    def from_dict(cls, data: dict):
        name, module_name, default_option, options = (
            data["name"],
            data["module_name"],
            data["default_option"],
            data["options"],
        )
        options = [
            ReasonThenFormat.registry[dat["type"]].from_dict(dat)
            for name, dat in options.items()
        ]
        return cls(
            name=name,
            options=options,
            default_option=default_option,
            module_name=module_name,
        )


class ReasoningOptionMeta(ABCMeta):
    registry: dict[str, type] = {"NoChange": NoChange}

    def __new__(cls, name, bases, attrs):
        new_cls = super().__new__(cls, name, bases, attrs)
        cls.registry[name] = new_cls
        return new_cls


class ReasonThenFormat(OptionBase, metaclass=ReasoningOptionMeta):
    @classmethod
    def direct_apply(cls, lm_module: Model):
        reasoning = cls()
        reasoning.apply(lm_module)
        return reasoning

    def reasoning_step(
        self, model: str, chat_messages: List[APICompatibleMessage], model_kwargs: dict
    ) -> List[ModelResponse]:
        """Produce reasoning steps for the given chat prompt messages"""
        raise NotImplementedError

    def aggregate_reasoning_steps(self, responses: List[ModelResponse]) -> str:
        agg_messages = []
        for response in responses:
            agg_messages.append(f"\n: {response.choices[0].message.content}")
        return "\n".join(agg_messages)

    def forward(
        self, lm_module: Model, messages: List[APICompatibleMessage], model_kwargs: dict
    ) -> List[ModelResponse]:
        """
        If the orignal output has certain format, applying additional reasoning steps will break down
        it into two phases, first one allows free generation along with reasoning steps, and the second
        one will the formatting step
        """

        model: str = model_kwargs.pop("model")
        responses = []

        messages.append(
            {
                "role": "user",
                "content": "Don't give your final response to the instruction directly. We can start with some reasoning first.\n",
            }
        )
        reasoning_step_responses: List[ModelResponse] = self.reasoning_step(
            model, copy.deepcopy(messages), model_kwargs
        )

        responses.extend(reasoning_step_responses)
        rationale = self.aggregate_reasoning_steps(reasoning_step_responses)
        lm_module.rationale = rationale

        messages.append({"role": "assistant", "content": rationale})
        if lm_module.contains_custom_format_instructions():
            messages.append(
                {
                    "role": "user",
                    "content": f"Based on the reasoning, now please only give {lm_module.get_output_label_name()} as your final response, according to the following instructions:\n{lm_module.get_custom_format_instructions_if_any()}",
                }
            )
        else:
            messages.append(
                {
                    "role": "user",
                    "content": f"Based on the reasoning, now please form {lm_module.get_output_label_name()} as your final response.",
                }
            )

        full_messages = [lm_module.system_message.to_api()] + messages
        if isinstance(lm_module, StructuredModel):
            response = litellm_completion(
                model,
                full_messages,
                model_kwargs,
                response_format=lm_module.output_format.schema,
            )
            responses.append(response)
        else:
            response = litellm_completion(model, full_messages, model_kwargs)
            responses.append(response)
        return responses

    def apply(self, lm_module: Model):
        lm_module.reasoning = self
        return lm_module

    @classmethod
    def from_dict(cls, data: dict):
        return cls()


class ZeroShotCoT(ReasonThenFormat):
    def __init__(self):
        super().__init__("ZeroShotCoT")

    def _get_cost_indicator(self):
        return 2.0

    def describe(self):
        desc = """
        - ZeroShotCoT -
        Return step-by-step reasoning for the given chat prompt messages.
        
        Reasoning Prompt: 
            Let's solve this problem step by step before giving the final response.
        """
        return desc

    def reasoning_step(
        self, model: str, chat_messages: List[APICompatibleMessage], model_kwargs: dict
    ) -> List[ModelResponse]:
        chat_messages.append(
            {
                "role": "user",
                "content": "Let's solve this problem step by step before giving the final response\n",
            }
        )
        response = litellm_completion(model, chat_messages, model_kwargs)
        return [response]


class PlanBefore(ReasonThenFormat):
    def __init__(self):
        super().__init__("PlanBefore")

    def _get_cost_indicator(self):
        return 3.0

    def describe(self):
        desc = """
        - PlanBefore -
        Similar to the planner in the LLMCompiler paper. Plan sub-tasks and synthesize a response for each sub-task as the rationale. Focus more on the runtime query complexity.
        
        Reasoning Prompt: 
            Let's first break down the task into several simpler sub-tasks that each covers different aspect of the original task. Clearly state each sub-question and provide your response to each one of them.
        """
        return desc

    def reasoning_step(
        self, model: str, chat_messages: List[APICompatibleMessage], model_kwargs: dict
    ) -> List[ModelResponse]:
        # TODO: make this a workflow and parallelize the reasoning steps
        chat_messages.append(
            {
                "role": "user",
                "content": "Let's first break down the task into several simpler sub-tasks that each covers different aspect of the original task. Clearly state each sub-question and provide your response to each one of them.",
            }
        )
        response = litellm_completion(model, chat_messages, model_kwargs)
        return [response]
