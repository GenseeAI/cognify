import dspy
from dspy.adapters.chat_adapter import ChatAdapter, prepare_instructions
from cognify.llm import Model, StructuredModel, Input, OutputFormat, OutputLabel
from cognify.llm.model import LMConfig
from pydantic import BaseModel, create_model
from typing import Any, Dict, Type
import warnings

APICompatibleMessage = Dict[str, str]  # {"role": "...", "content": "..."}


def generate_pydantic_model(
    model_name: str, fields: Dict[str, Type[Any]]
) -> Type[BaseModel]:
    # Generate a dynamic Pydantic model using create_model
    return create_model(
        model_name, **{name: (field_type, ...) for name, field_type in fields.items()}
    )


"""
Connector currently supports `Predict` with any signature and strips away all reasoning fields.
This is done because we handle reasoning via cogs for the optimizer instead of in a templated format. 
"""


class PredictModel(dspy.Module):
    def __init__(self, name: str, dspy_predictor: dspy.Module = None):
        super().__init__()
        self.chat_adapter: ChatAdapter = ChatAdapter()
        self.predictor: dspy.Module = dspy_predictor
        self.ignore_module = False
        self.cog_lm: StructuredModel = self.cognify_predictor(name, dspy_predictor)
        self.output_schema = None

    def cognify_predictor(
        self, name: str, dspy_predictor: dspy.Module = None
    ) -> StructuredModel:
        if not dspy_predictor:
            return None

        if not isinstance(dspy_predictor, dspy.Predict):
            warnings.warn(
                "Original module is NOT a `dspy.Predict`. This may result in lossy translation",
                UserWarning,
            )

        if isinstance(dspy_predictor, dspy.Retrieve):
            warnings.warn(
                "Original module is a `dspy.Retrieve`. This will be ignored", UserWarning
            )
            self.ignore_module = True
            return None

        # initialize cog lm
        input_names = list(dspy_predictor.signature.input_fields.keys())
        input_variables = [Input(name=input_name) for input_name in input_names]

        output_fields = dspy_predictor.signature.output_fields
        if "reasoning" in output_fields:
            # stripping the reasoning field may crash their workflow, so we warn users instead
            warnings.warn(
                f"DSPy module {name} contained reasoning. This may lead to undefined behavior.", 
                UserWarning,
            )
        system_prompt = prepare_instructions(dspy_predictor.signature)

        # lm config
        lm_client: dspy.LM = dspy.settings.get("lm", None)

        assert lm_client, "Expected lm to be configured in dspy"
        lm_config = LMConfig(model=lm_client.model, kwargs=lm_client.kwargs)

        # treat as cognify.Model, allow dspy to handle output parsing
        return Model(
            agent_name=name,
            system_prompt=system_prompt,
            input_variables=input_variables,
            output=OutputLabel("llm_output"),
            lm_config=lm_config
        )
    
    def construct_messages(self, inputs):
        messages = None
        if self.predictor:
            messages: APICompatibleMessage = self.chat_adapter.format(
                self.predictor.signature, self.predictor.demos, inputs
            )
        return messages
    
    def parse_output(self, result):
        values = []

        # from dspy chat adapter __call__
        value = self.chat_adapter.parse(self.predictor.signature, result, _parse_values=True)
        assert set(value.keys()) == set(self.predictor.signature.output_fields.keys()), f"Expected {self.predictor.signature.output_fields.keys()} but got {value.keys()}"
        values.append(value)

        return values

    def forward(self, **kwargs):
        assert (
            self.cog_lm or self.predictor
        ), "Either cognify.Model or predictor must be initialized before invoking"

        if self.ignore_module:
            return self.predictor(**kwargs)
        else:
            inputs: Dict[str, str] = {
                k.name: kwargs[k.name] for k in self.cog_lm.input_variables
            }
            messages = self.construct_messages(inputs)
            result = self.cog_lm(
                messages, inputs
            )  # kwargs have already been set when initializing cog_lm
            completions = self.parse_output(result)
            return dspy.Prediction.from_completions(completions, signature=self.predictor.signature)


def as_predict(cog_lm: Model) -> PredictModel:
    predictor = PredictModel(name=cog_lm.name)
    if isinstance(cog_lm, StructuredModel):
        predictor.cog_lm = cog_lm
        predictor.output_schema = cog_lm.output_format.schema
    else:
        output_schema = generate_pydantic_model(
            "OutputData", {cog_lm.get_output_label_name(): str}
        )
        predictor.cog_lm = StructuredModel(
            agent_name=cog_lm.name,
            system_prompt=cog_lm.get_system_prompt(),
            input_variables=cog_lm.input_variables,
            output_format=OutputFormat(
                output_schema,
                custom_output_format_instructions=cog_lm.get_custom_format_instructions_if_any(),
            ),
            lm_config=cog_lm.lm_config,
        )
    return predictor
