from langchain_core.runnables import Runnable, RunnableSequence
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.language_models.chat_models import BaseChatModel
from cognify.llm import Model, Input, StructuredModel, OutputFormat, OutputLabel
from cognify.llm.model import LMConfig
from pydantic import BaseModel
from typing import Any, List, Dict
from dataclasses import dataclass
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

APICompatibleMessage = Dict[str, str]  # {"role": "...", "content": "..."}

langchain_message_role_to_api_message_role = {
    "system": "system",
    "human": "user",
    "ai": "assistant",
}


@dataclass
class LangchainOutput:
    content: str


UNRECOGNIZED_PARAMS = ["model_name", "_type"]


class RunnableModel(Runnable):
    def __init__(self, name: str, runnable: RunnableSequence = None):
        self.chat_prompt_template: ChatPromptTemplate = None
        self.runnable = runnable
        self.contains_output_parser: bool = False
        self.cog_lm: Model = self.cognify_runnable(name, runnable)

    """
  Connector currently supports the following units to construct a `cognify.Model`:
  - BaseChatPromptTemplate | BaseChatModel
  - BaseChatPromptTemplate | BaseChatModel | BaseOutputParser
  These indepedent units should be split out of more complex chains.
  """

    def cognify_runnable(self, name: str, runnable: RunnableSequence = None) -> Model:
        if not runnable:
            return None

        # parse runnable
        assert isinstance(
            runnable.first, ChatPromptTemplate
        ), f"First runnable in a sequence must be a `ChatPromptTemplate` instead got {type(runnable.first)}"
        self.chat_prompt_template: ChatPromptTemplate = runnable.first
        output_parser = None

        if runnable.middle is None or len(runnable.middle) == 0:
            assert isinstance(
                runnable.last, BaseChatModel
            ), f"Last runnable in a sequence with no middle must be a `BaseChatModel`, instead got {type(runnable.last)}"
            chat_model: BaseChatModel = runnable.last
        elif len(runnable.middle) == 1:
            assert isinstance(
                runnable.middle[0], BaseChatModel
            ), f"Middle runnable must be a `BaseChatModel`, instead got {type(runnable.middle[0])}"
            chat_model: BaseChatModel = runnable.middle[0]

            assert isinstance(
                runnable.last, BaseOutputParser
            ), f"Last runnable in a sequence with a middle `BaseChatModel` must be a `BaseOutputParser`, instead got {type(runnable.last)}"
            output_parser: BaseOutputParser = runnable.last
            self.contains_output_parser = True
        else:
            raise NotImplementedError(
                f"Only one middle runnable is supported at this time, instead got {runnable.middle}"
            )

        # system prompt
        if isinstance(
            self.chat_prompt_template.messages[0], SystemMessagePromptTemplate
        ):
            system_message_prompt_template: SystemMessagePromptTemplate = (
                self.chat_prompt_template.messages[0]
            )
            if system_message_prompt_template.prompt.input_variables:
                raise NotImplementedError(
                    "Input variables are not supported in the system prompt. Best practices suggest placing these in the following user messages."
                )
            system_prompt_content: str = system_message_prompt_template.prompt.template
        else:
            raise ValueError(
                "First message in the chat prompt template must be a system message."
            )

        # input variables (ignore partial variables)
        input_vars: List[Input] = [
            Input(name=input_var)
            for input_var in self.chat_prompt_template.input_variables
        ]

        # lm config
        full_kwargs = chat_model._get_invocation_params()

        # remove unrecognized params
        for param in UNRECOGNIZED_PARAMS:
            full_kwargs.pop(param, None)

        lm_config = LMConfig(model=full_kwargs.pop("model"), kwargs=full_kwargs)

        # StructuredModel only supports pydantic types
        # all other output formatting or parsing will be applied functionally to the result
        if output_parser is not None and isinstance(
            output_parser.OutputType, BaseModel
        ):
            # custom format instructions
            try:
                custom_format_instructions = output_parser.get_format_instructions()
            except NotImplementedError:
                custom_format_instructions = None

            output_format = OutputFormat(
                schema=output_parser.OutputType,
                should_hint_format_in_prompt=True,
                custom_output_format_instructions=custom_format_instructions,
            )
            return StructuredModel(
                agent_name=name,
                system_prompt=system_prompt_content,
                input_variables=input_vars,
                output_format=output_format,
                lm_config=lm_config,
            )
        else:
            return Model(
                agent_name=name,
                system_prompt=system_prompt_content,
                input_variables=input_vars,
                output=OutputLabel(name="response"),
                lm_config=lm_config,
            )

    def invoke(self, input: Dict) -> Any:
        assert self.cog_lm, "cognify.Model must be initialized before invoking"

        messages = None
        if self.chat_prompt_template:
            chat_prompt_value: ChatPromptValue = self.chat_prompt_template.invoke(input)
            messages: List[APICompatibleMessage] = []
            for message in chat_prompt_value.messages:
                if message.type in langchain_message_role_to_api_message_role:
                    messages.append(
                        {
                            "role": langchain_message_role_to_api_message_role[
                                message.type
                            ],
                            "content": message.content,
                        }
                    )
                else:
                    raise NotImplementedError(
                        f"Message type {type(message)} is not supported, must be one of `SystemMessage`, `HumanMessage`, or `AIMessage`"
                    )

        result = self.cog_lm(
            messages, input
        )  # kwargs have already been set when initializing cog_lm

        if isinstance(self.cog_lm, StructuredModel):
            return result
        elif self.contains_output_parser:
            type_cls = type(self.runnable.last)
            type_inst = type_cls()
            return type_inst.parse(result)
        else:
            return AIMessage(result)


def as_runnable(cog_lm: Model):
    runnable_cog_lm = RunnableModel(cog_lm.name)
    runnable_cog_lm.cog_lm = cog_lm
    return RunnableLambda(runnable_cog_lm.invoke)


