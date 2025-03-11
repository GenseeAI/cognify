from dataclasses import dataclass, field
from typing import List, Dict, Optional
from cognify._compat import override
from cognify.llm.litellm_wrapper import litellm_completion
from cognify.llm.prompt import (
    Input,
    FilledInput,
    CompletionMessage,
    Demonstration,
    Content,
    TextContent,
    ImageContent,
    get_image_content_from_upload,
)
from cognify.llm.output import OutputLabel, OutputFormat
import litellm
from litellm import completion, get_supported_openai_params
from pydantic import BaseModel
import json
import time
from openai.types import CompletionUsage
from cognify.llm.response import ResponseMetadata, aggregate_usages, StepInfo
from cognify.graph.base import Module, StatePool
import copy
import threading
import logging

litellm_logger = logging.getLogger("LiteLLM")
litellm_logger.disabled = True

logger = logging.getLogger(__name__)
APICompatibleMessage = Dict[str, str]  # {"role": "...", "content": "..."}
_thread_local_chain = threading.local()


def _local_forward(
    _local_lm: "Model",
    messages: List[APICompatibleMessage],
    inputs: Dict[str, str],
    model_kwargs: Optional[dict] = None,
) -> str:
    if _local_lm.reasoning:
        responses = _local_lm.reasoning.forward(
            _local_lm, messages, model_kwargs
        )
        _local_lm.response_metadata_history.extend(
            [
                ResponseMetadata(
                    model=response.model,
                    cost=response._hidden_params["response_cost"],
                    usage=response.usage,
                )
                for response in responses
            ]
        )
        response = responses[-1]
    else:
        response = _local_lm._forward(messages, model_kwargs)
        _local_lm.response_metadata_history.append(
            ResponseMetadata(
                model=response.model,
                cost=response._hidden_params["response_cost"],
                usage=response.usage,
            )
        )
    step_info = StepInfo(
        filled_inputs_dict=inputs,
        output=response.choices[0].message.content,
        rationale=_local_lm.rationale,
    )
    _local_lm.steps.append(step_info)
    _local_lm.rationale = None

    return response.choices[0].message.content


@dataclass
class LMConfig:
    model: str  # see https://docs.litellm.ai/docs/providers
    kwargs: dict = field(default_factory=dict)
    custom_llm_provider: Optional[str] = None
    cost_indicator: float = (
        1.0  # used to rank models during model selection optimization step
    )

    def to_dict(self):
        return self.__dict__

    @classmethod
    def from_dict(cls, data):
        obj = cls.__new__(cls)
        obj.__dict__.update(data)
        return obj

    def get_model_kwargs(self) -> dict:
        full_kwargs_dict = {}
        full_kwargs_dict.update(self.kwargs)
        full_kwargs_dict["model"] = self.model
        if self.custom_llm_provider:
            full_kwargs_dict["custom_llm_provider"] = self.custom_llm_provider
        return full_kwargs_dict

    def update(self, other: "LMConfig"):
        self.model = other.model
        self.custom_llm_provider = other.custom_llm_provider
        self.kwargs.update(other.kwargs)
        self.cost_indicator = other.cost_indicator


class Model(Module):
    def __init__(
        self,
        agent_name: str,
        system_prompt: str,
        input_variables: List[Input],
        output: OutputLabel,
        lm_config: Optional[LMConfig] = None,
        opt_register: bool = True,
    ):
        self._lock = threading.Lock()
        super().__init__(name=agent_name, kernel=None, opt_register=opt_register)

        self.system_message: CompletionMessage = CompletionMessage(
            role="system", content=[TextContent(text=system_prompt)]
        )
        self.input_variables: List[Input] = input_variables
        self.output_label: Optional[OutputLabel] = output
        self.demo_messages: List[CompletionMessage] = []
        self.response_metadata_history: List[ResponseMetadata] = []
        self.steps: List[StepInfo] = []
        self.reasoning = None
        self.rationale = None

        # TODO: improve lm configuration handling between agents. currently just unique config for each agent
        self.lm_config = copy.deepcopy(lm_config)
        self.input_fields = [input_var.name for input_var in self.input_variables]

        setattr(_thread_local_chain, agent_name, self)

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k != "_lock":
                setattr(result, k, copy.deepcopy(v, memo))
            else:
                setattr(result, k, threading.Lock())
        return result

    @override
    def reset(self):
        super().reset()
        self.response_metadata_history = []
        self.steps = []

    def get_thread_local_chain(self):
        try:
            if not hasattr(_thread_local_chain, self.name):
                # NOTE: no need to set to local storage bc that's mainly used to detect if current context is in a new thread
                _self = copy.deepcopy(self)
                _self.reset()
            else:
                _self = getattr(_thread_local_chain, self.name)
            return _self
        except Exception as e:
            logger.info(f"Error in get_thread_local_chain: {e}")
            raise

    def get_high_level_info(self) -> str:
        dict = {
            "agent_prompt": self.get_system_prompt(),
            "input_names": self._get_input_names(),
            "output_name": self.get_output_label_name(),
        }
        return json.dumps(dict, indent=4)

    def get_formatted_info(self) -> str:
        dict = {
            "agent_prompt": self.get_system_prompt(),
            "input_variables": self._get_input_names(),
            "output_schema": self.get_output_label_name(),
        }
        return json.dumps(dict, indent=4)

    def get_last_step_as_demo(self) -> Optional[Demonstration]:
        if not self.steps:
            return None
        else:
            last_step: StepInfo = self.steps[-1]
            filled_input_list: List[FilledInput] = []
            for input_variable in self.input_variables:
                input_value = last_step.filled_inputs_dict.get(
                    input_variable.name, None
                )
                filled_input_list.append(
                    FilledInput(input_variable=input_variable, value=input_value)
                )
            return Demonstration(
                filled_input_variables=filled_input_list,
                output=last_step.output,
                reasoning=last_step.rationale,
            )

    def add_demos(self, demos: List[Demonstration], demo_prompt_string: str = None):
        if not demos:
            raise Exception("No demonstrations provided")

        input_variable_names = [variable.name for variable in self.input_variables]
        demo_prompt_string = (
            demo_prompt_string or "Let me show you some examples following the format"
        )  # customizable
        demos_content: List[Content] = []

        demos_content.append(
            TextContent(
                text=f"""{demo_prompt_string}:\n\n{self._get_example_format()}--\n\n"""
            )
        )
        for demo in demos:
            # validate demonstration
            demo_variable_names = [
                filled.input_variable.name for filled in demo.filled_input_variables
            ]
            if set(demo_variable_names) != set(input_variable_names):
                raise ValueError(
                    f"Demonstration variables {demo_variable_names} do not match input variables {input_variable_names}"
                )
            else:
                demos_content.extend(demo.to_content())
        self.demo_messages.append(CompletionMessage(role="user", content=demos_content))

    def get_lm_response_metadata_history(self) -> List[ResponseMetadata]:
        return copy.deepcopy(self.response_metadata_history)

    def get_total_cost(self) -> float:
        return sum(
            [
                response_metadata.cost
                for response_metadata in self.response_metadata_history if response_metadata.cost is not None
            ]
        )

    def get_current_token_usages(self):
        return copy.deepcopy(
            [
                response_cost_usage.usage
                for response_cost_usage in self.response_metadata_history
            ]
        )

    def get_aggregated_token_usage(self) -> CompletionUsage:
        return aggregate_usages(self.get_current_token_usages())

    def _get_example_format(self):
        input_fields = []
        for variable in self.input_variables:
            input_fields.append(
                f"{variable.name}:\n${{{variable.name}}}"
            )  # e.g. "question: ${question}"

        return (
            "\n\n".join(input_fields)
            + "\n\nrationale:\nOptional(${reasoning})"
            + f"\n\n{self.get_output_label_name()}:\n${{{self.get_output_label_name()}}}"
        )

    def get_output_label_name(self) -> str:
        return self.output_label.name or "response"

    def _get_input_names(self) -> List[str]:
        return [variable.name for variable in self.input_variables]

    def get_system_prompt(self) -> str:
        return self.system_message.content[0].text

    def get_agent_role(self) -> str:
        return self.get_system_prompt()

    def contains_custom_format_instructions(self) -> bool:
        return self.output_label and self.output_label.custom_output_format_instructions

    def get_custom_format_instructions_if_any(self) -> Optional[str]:
        return (
            self.output_label.custom_output_format_instructions
            if self.contains_custom_format_instructions()
            else None
        )

    def _get_api_compatible_messages(
        self, messages: List[APICompatibleMessage]
    ) -> List[APICompatibleMessage]:
        if messages[0]["role"] == "system":
            messages = messages[1:]

        api_compatible_messages = [self.system_message.to_api()] + messages
        api_compatible_messages.extend(
            [demo_message.to_api() for demo_message in self.demo_messages]
        )
        if self.contains_custom_format_instructions():
            api_compatible_messages.append(
                {
                    "role": "user",
                    "content": self.get_custom_format_instructions_if_any(),
                }
            )
        return api_compatible_messages

    def aggregate_thread_local_meta(self, _local_self: "Model"):
        if self is _local_self:
            return
        with self._lock:
            self.steps.extend(_local_self.steps)
            self.response_metadata_history.extend(_local_self.response_metadata_history)

    @override
    def invoke(self, statep: StatePool):
        logger.debug(f"Invoking {self}")
        for field in self.input_fields:
            if field not in self.defaults and field not in statep.states:
                raise ValueError(
                    f"Missing field {field} in state when calling {self.name}, available fields: {statep.states.keys()}"
                )
        kargs = {
            field: statep.news(field)
            for field in statep.states
            if field in self.input_fields
        }
        messages = statep.news("messages", [])
        model_kwargs = statep.news("model_kwargs", None)
        # time the execution
        start = time.perf_counter()
        result = self.forward(
            messages=messages, inputs=kargs, model_kwargs=model_kwargs
        )
        dur = time.perf_counter() - start
        result_snapshot = copy.deepcopy(result)
        statep.publish(result_snapshot, self.version_id, self.is_static)
        self.outputs.append(result_snapshot)
        # update metadata
        self.exec_times.append(dur)
        self.version_id += 1

    def prepare_input(
        self,
        messages: List[APICompatibleMessage],
        inputs: Dict[Input | str, str],
        model_kwargs: Optional[dict],
    ):
        # input variables will always take precedence
        if not inputs:
            assert messages, f"Messages must be provided for '{self.name}'"
            final_message_list = messages
        else:
            if isinstance(list(inputs.keys())[0], Input):
                inputs = {input_var.name: value for input_var, value in inputs.items()}
            final_message_list = self._get_input_messages(inputs)

        # lm config will always take precedence
        if not self.lm_config:
            assert (
                model_kwargs
            ), f"Model kwargs must be provided if LM config is not set at initialization for '{self.name}'"
            full_kwargs = model_kwargs
        else:
            full_kwargs = self.lm_config.get_model_kwargs()

        return final_message_list, inputs, full_kwargs

    def forward(
        self,
        messages: List[APICompatibleMessage] = [],
        inputs: Dict[str, str] = None,
        model_kwargs: dict = None,
    ):
        _self = self.get_thread_local_chain()
        messages, inputs, model_kwargs = _self.prepare_input(
            messages, inputs, model_kwargs
        )
        result = _local_forward(_self, messages, inputs, model_kwargs)
        self.aggregate_thread_local_meta(_self)
        return {self.get_output_label_name(): result}

    def __call__(
        self,
        messages: List[APICompatibleMessage] = [],
        inputs: Dict[str, str] = None,
        model_kwargs: dict = None,
    ):
        """External interface to invoke the cognify.Model"""
        statep = StatePool()
        statep.init(
            {
                "messages": messages,
                "model_kwargs": model_kwargs,
                **inputs,
            }
        )
        self.invoke(statep)  # kwargs have already been set when initializing cog_lm
        result = statep.news(self.get_output_label_name())
        return result

    def _get_input_messages(self, inputs: Dict[str, str]) -> List[APICompatibleMessage]:
        initialized_input_names = [input.name for input in self.input_variables]
        assert set(inputs.keys()) == set(
            initialized_input_names
        ), f"Input variables do not match, '{self.name}' initialized with {initialized_input_names} but provided {inputs.keys()}"

        input_names = ", ".join(f"`{name}`" for name in initialized_input_names)
        messages = [
            CompletionMessage(
                role="user",
                content=[
                    TextContent(
                        text=f"Given {input_names}, please strictly provide `{self.get_output_label_name()}`"
                    )
                ],
            )
        ]

        input_fields = []
        for input_var in self.input_variables:
            if input_var.image_type:
                if input_var.image_type == "web":
                    image_content = ImageContent(image_url=inputs[input_var.name])
                else:
                    image_content = get_image_content_from_upload(
                        inputs[input_var.name], input_var.image_type
                    )
                messages.append(CompletionMessage(role="user", content=[image_content]))
            else:
                input_fields.append(f"{input_var.name}: {inputs[input_var.name]}")
        messages.append(
            CompletionMessage(
                role="user", content=[TextContent(text="\n".join(input_fields))]
            )
        )
        return [message.to_api() for message in messages]

    def _forward(
        self, messages: List[APICompatibleMessage], model_kwargs: dict
    ):
        model = model_kwargs.pop("model")
        response = litellm_completion(
            model, self._get_api_compatible_messages(messages), model_kwargs
        )
        return response


class StructuredModel(Model):
    def __init__(
        self,
        agent_name: str,
        system_prompt: str,
        input_variables: List[Input],
        output_format: OutputFormat,
        lm_config: Optional[LMConfig] = None,
        opt_register: bool = True,
    ):
        assert isinstance(
            output_format.schema, type(BaseModel)
        ), f"Output format must be a Pydantic `BaseModel` for {agent_name}"

        self.output_format: OutputFormat = output_format
        super().__init__(
            agent_name,
            system_prompt,
            input_variables,
            output=OutputLabel(name=output_format.schema.__name__),
            lm_config=lm_config,
            opt_register=opt_register,
        )

    # these parse methods are provided for convenience
    def parse_response_str(self, response: str) -> BaseModel:
        # expects response to be `response.choices[0].message.content`
        return self.output_format.schema.model_validate_json(response)

    def parse_response(self, response) -> BaseModel:
        return self.parse_response_str(response.choices[0].message.content)

    @override
    def get_output_label_name(self):
        return self.output_format.schema.__name__

    @override
    def get_formatted_info(self) -> str:
        dict = {
            "agent_prompt": self.get_system_prompt(),
            "input_variables": self._get_input_names(),
            "output_schema": self.output_format.schema.model_json_schema(),
        }
        return json.dumps(dict, indent=4)

    @override
    def contains_custom_format_instructions(self) -> bool:
        return self.output_format.custom_output_format_instructions is not None

    @override
    def get_custom_format_instructions_if_any(self) -> Optional[str]:
        return self.output_format.custom_output_format_instructions

    @override
    def _get_api_compatible_messages(
        self, messages: List[APICompatibleMessage]
    ) -> List[APICompatibleMessage]:
        api_compatible_messages = super()._get_api_compatible_messages(messages)
        api_compatible_messages.append(
            self.output_format.get_output_instruction_message().to_api()
        )
        return api_compatible_messages

    @override
    def _forward(
        self, messages: List[APICompatibleMessage], model_kwargs: dict
    ):
        litellm.enable_json_schema_validation = True

        params = get_supported_openai_params(
            model=model_kwargs["model"],
            custom_llm_provider=model_kwargs.get("custom_llm_provider", None),
        )
        if "response_format" not in params:
            raise ValueError(
                f"Model {model_kwargs['model']} from provider {model_kwargs.get('custom_llm_provider', None)} does not support structured output"
            )
        else:
            model = model_kwargs.pop("model")

            response = litellm_completion(
                model, 
                self._get_api_compatible_messages(messages), 
                model_kwargs, 
                response_format=self.output_format.schema
            )
            return response

    @override
    def forward(
        self,
        messages: List[APICompatibleMessage] = [],
        inputs: Dict[str, str] = None,
        model_kwargs: dict = None,
    ) -> str:
        result = super().forward(messages, inputs, model_kwargs)
        result_obj = self.parse_response_str(result[self.get_output_label_name()])
        return {self.get_output_label_name(): result_obj}
