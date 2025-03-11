import json
from pydantic import BaseModel, Field
from datamodel_code_generator.parser.jsonschema import JsonSchemaParser
from cognify.llm.prompt import CompletionMessage, TextContent
from dataclasses import dataclass


@dataclass
class OutputLabel:
    name: str
    custom_output_format_instructions: str = None


@dataclass
class OutputFormat:
    schema: BaseModel
    should_hint_format_in_prompt: bool = True
    custom_output_format_instructions: str = None

    def get_output_instruction_message(self) -> CompletionMessage:
        content = ""
        if self.should_hint_format_in_prompt:
            content += "\n" + get_format_hint(self.schema)
        if self.custom_output_format_instructions:
            content += "\n" + self.custom_output_format_instructions
        return CompletionMessage(role="user", content=[TextContent(text=content)])


def pydantic_model_repr(model: type[BaseModel]) -> str:
    """Get str representation of a Pydantic model

    Will return the class definition of the Pydantic model as a string.
    """
    pydantic_str = JsonSchemaParser(json.dumps(model.model_json_schema())).parse(
        with_import=False
    )
    return pydantic_str


class InnerModel(BaseModel):
    """A nested model"""

    a: int = Field(description="An integer field")
    b: str = Field(description="A string field")


class ExampleModel(BaseModel):
    """An example output schema"""

    ms: list[InnerModel] = Field(description="A list of InnerModel")
    meta: dict[str, str] = Field(description="A dictionary of string to string")


example_output_json = """
```json
{
    "ms": [
        {"a": 1, "b": "b1"},
        {"a": 2, "b": "b2"}
    ],
    "meta": {"key1": "value1", "key2": "value2"}
}
```
"""


def get_format_hint(schema: type[BaseModel]) -> str:
    template = """\
Your answer should be formatted as a JSON instance that conforms to the output schema. The json instance will be used directly to instantiate the Pydantic model.

As an example, given the output schema:
{example_output_schema}

Your answer in this case should be formatted as follows:
{example_output_json}

Here's the real output schema for your reference:
{real_output_schema}

Please provide your answer in the correct json format accordingly. Especially make sure each field will respect the type and constraints defined in the schema.
Pay attention to the enum field in properties, do not generate answer that is not in the enum field if provided.
"""
    return template.format(
        example_output_schema=pydantic_model_repr(ExampleModel),
        example_output_json=example_output_json,
        real_output_schema=pydantic_model_repr(schema),
    )
