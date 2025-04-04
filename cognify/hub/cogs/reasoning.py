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


class DebateReflection(ReasonThenFormat):
    def __init__(self):
        super().__init__("DebateReflection")
    
    def _get_cost_indicator(self):
        return 3.5
    
    def describe(self):
        desc = """
        - DebateReflection -
        Creates an internal debate between multiple perspectives before forming a final conclusion.
        Arguments from different viewpoints are considered and weighed against each other.
        
        Reasoning Prompt:
            Let's analyze this from multiple perspectives, as if different experts were debating the question.
            Present 2-3 different viewpoints, including any criticisms each might have of the others,
            then synthesize these perspectives into a balanced conclusion.
        """
        return desc
    
    def reasoning_step(
        self, model: str, chat_messages: List[APICompatibleMessage], model_kwargs: dict
    ) -> List[ModelResponse]:
        chat_messages.append(
            {
                "role": "user",
                "content": "Let's analyze this from multiple perspectives, as if different experts were debating the question. "
                "Present 2-3 different viewpoints, including any criticisms each might have of the others, "
                "then synthesize these perspectives into a balanced conclusion.",
            }
        )
        response = litellm_completion(model, chat_messages, model_kwargs)
        return [response]


class FirstPrinciplesReasoning(ReasonThenFormat):
    def __init__(self):
        super().__init__("FirstPrinciplesReasoning")
    
    def _get_cost_indicator(self):
        return 2.5
    
    def describe(self):
        desc = """
        - FirstPrinciplesReasoning -
        Breaks down complex problems to their fundamental truths and builds up from there,
        avoiding assumptions and establishing core principles before proceeding.
        
        Reasoning Prompt:
            Let's break this down to first principles. Identify the fundamental truths or principles that apply
            to this situation, explain why they're relevant, and then build a solution from these foundations.
        """
        return desc
    
    def reasoning_step(
        self, model: str, chat_messages: List[APICompatibleMessage], model_kwargs: dict
    ) -> List[ModelResponse]:
        chat_messages.append(
            {
                "role": "user",
                "content": "Let's break this down to first principles. Identify the fundamental truths or principles that apply "
                "to this situation, explain why they're relevant, and then build a solution from these foundations.",
            }
        )
        response = litellm_completion(model, chat_messages, model_kwargs)
        return [response]


class SequentialStepAnalysis(ReasonThenFormat):
    def __init__(self, steps=4):
        super().__init__("SequentialStepAnalysis")
        self.steps = steps
    
    def _get_cost_indicator(self):
        return 2.0 + 0.5 * self.steps
    
    def describe(self):
        desc = f"""
        - SequentialStepAnalysis -
        Performs a multi-stage analysis with {self.steps} distinct, sequential steps, each building on previous findings.
        Each step is processed separately and the results are compiled into a comprehensive analysis.
        
        Reasoning Prompt:
            Let's analyze this in {self.steps} sequential steps. For each step, provide your analysis before moving to the next step.
        """
        return desc
    
    def reasoning_step(
        self, model: str, chat_messages: List[APICompatibleMessage], model_kwargs: dict
    ) -> List[ModelResponse]:
        responses = []
        
        # Initial step prompt
        chat_messages.append(
            {
                "role": "user",
                "content": f"Let's analyze this in {self.steps} sequential steps. This is step 1:"
            }
        )
        
        working_messages = copy.deepcopy(chat_messages)
        response = litellm_completion(model, working_messages, copy.deepcopy(model_kwargs))
        responses.append(response)
        
        # Sequential steps
        for step in range(2, self.steps + 1):
            working_messages.append({"role": "assistant", "content": response.choices[0].message.content})
            working_messages.append(
                {"role": "user", "content": f"Now, for step {step}:"}
            )
            response = litellm_completion(model, working_messages, copy.deepcopy(model_kwargs))
            responses.append(response)
        
        return responses
        
    def aggregate_reasoning_steps(self, responses: List[ModelResponse]) -> str:
        # Override to format the sequential steps
        agg_messages = []
        for i, response in enumerate(responses):
            agg_messages.append(f"Step {i+1}: {response.choices[0].message.content}")
        return "\n\n".join(agg_messages)


class CounterfactualAnalysis(ReasonThenFormat):
    def __init__(self):
        super().__init__("CounterfactualAnalysis")
    
    def _get_cost_indicator(self):
        return 3.0
    
    def describe(self):
        desc = """
        - CounterfactualAnalysis -
        Explores alternative scenarios and examines how changes to key variables might affect outcomes.
        This approach identifies critical dependencies and improves robustness of conclusions.
        
        Reasoning Prompt:
            Let's consider some counterfactuals. What if key assumptions or conditions were different?
            Identify 2-3 alternative scenarios, explore how outcomes would change, and then use these
            insights to strengthen your analysis of the actual situation.
        """
        return desc
    
    def reasoning_step(
        self, model: str, chat_messages: List[APICompatibleMessage], model_kwargs: dict
    ) -> List[ModelResponse]:
        chat_messages.append(
            {
                "role": "user",
                "content": "Let's consider some counterfactuals. What if key assumptions or conditions were different? "
                "Identify 2-3 alternative scenarios, explore how outcomes would change, and then use these "
                "insights to strengthen your analysis of the actual situation."
            }
        )
        response = litellm_completion(model, chat_messages, model_kwargs)
        return [response]


class TreeOfThoughts(ReasonThenFormat):
    def __init__(self, branches=3, depth=2):
        super().__init__("TreeOfThoughts")
        self.branches = branches
        self.depth = depth
    
    def _get_cost_indicator(self):
        # Cost scales with number of branches and depth
        return 2.0 + self.branches * self.depth
    
    def describe(self):
        desc = f"""
        - TreeOfThoughts -
        Expands reasoning into a tree structure with {self.branches} branching paths at each of {self.depth} levels.
        Different solution approaches are explored in parallel, evaluated, and the best path is selected.
        
        Reasoning Prompt:
            Let's use a tree of thoughts approach with {self.branches} different reasoning paths at each step,
            exploring to a depth of {self.depth}. For each path, evaluate its promise before deciding which to explore further.
            Finally, select the most promising complete path to the solution.
        """
        return desc
    
    def reasoning_step(
        self, model: str, chat_messages: List[APICompatibleMessage], model_kwargs: dict
    ) -> List[ModelResponse]:
        chat_messages.append(
            {
                "role": "user",
                "content": f"Let's use a tree of thoughts approach with {self.branches} different reasoning paths at each step, "
                f"exploring to a depth of {self.depth}. For each path, evaluate its promise before deciding which to explore further. "
                "Finally, select the most promising complete path to the solution."
            }
        )
        response = litellm_completion(model, chat_messages, model_kwargs)
        return [response]