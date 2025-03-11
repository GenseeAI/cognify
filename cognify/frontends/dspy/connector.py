import dspy
from dspy.adapters.chat_adapter import ChatAdapter, prepare_instructions
from cognify.llm import Model, StructuredModel, Input, OutputFormat
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
                "Original module is not a `Predict`. This may result in lossy translation",
                UserWarning,
            )

        if isinstance(dspy_predictor, dspy.Retrieve):
            warnings.warn(
                "Original module is a `Retrieve`. This will be ignored", UserWarning
            )
            self.ignore_module = True
            return None

        # initialize cog lm
        system_prompt = prepare_instructions(dspy_predictor.signature)
        input_names = list(dspy_predictor.signature.input_fields.keys())
        input_variables = [Input(name=input_name) for input_name in input_names]

        output_fields = dspy_predictor.signature.output_fields
        if "reasoning" in output_fields:
            del output_fields["reasoning"]
            warnings.warn(
                "Original module contained reasoning. This will be stripped. Add reasoning as a cog instead",
                UserWarning,
            )
        output_fields_for_schema = {k: v.annotation for k, v in output_fields.items()}
        self.output_schema = generate_pydantic_model(
            "OutputData", output_fields_for_schema
        )

        # lm config
        lm_client: dspy.LM = dspy.settings.get("lm", None)

        assert lm_client, "Expected lm to be configured in dspy"
        lm_config = LMConfig(model=lm_client.model, kwargs=lm_client.kwargs)

        # always treat as structured to provide compatiblity with forward function
        return StructuredModel(
            agent_name=name,
            system_prompt=system_prompt,
            input_variables=input_variables,
            output_format=OutputFormat(schema=self.output_schema),
            lm_config=lm_config,
        )

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
            messages = None
            if self.predictor:
                messages: APICompatibleMessage = self.chat_adapter.format(
                    self.predictor.signature, self.predictor.demos, inputs
                )
            result = self.cog_lm(
                messages, inputs
            )  # kwargs have already been set when initializing cog_lm
            kwargs: dict = result.model_dump()
            for k,v in kwargs.items():
                if not v:
                    raise ValueError(f"{self.cog_lm.name} did not generate a value for field `{k}`, consider using a larger model for structured output")
            return dspy.Prediction(**kwargs)


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
