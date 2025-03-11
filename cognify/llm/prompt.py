from dataclasses import dataclass
from typing import List, Dict, Literal, Optional
import uuid
import json


@dataclass
class Input:
    name: str
    image_type: Optional[Literal["web", "jpeg", "png"]] = None

    def __hash__(self):
        return hash(self.name, self.image_type)


@dataclass
class FilledInput:
    input_variable: Input
    value: str


@dataclass
class TextContent:
    text: str
    type: Literal["text"] = "text"


@dataclass
class ImageContent:
    image_url: dict
    type: Literal["image_url"] = "image_url"


def get_image_content_from_upload(
    image_upload: str, file_type: Literal["jpeg", "png"]
) -> ImageContent:
    return ImageContent({"url": f"data:image/{file_type};base64,{image_upload}"})


Content = TextContent | ImageContent


@dataclass
class Demonstration:
    filled_input_variables: List[FilledInput]
    output: str
    id: str
    reasoning: str = None

    def __init__(
        self,
        filled_input_variables: List[FilledInput],
        output: str,
        id: str = None,
        reasoning: str = None,
    ):
        self.filled_input_variables = filled_input_variables
        self.output = output
        self.id = id or str(uuid.uuid4())
        self.reasoning = reasoning

    def to_content(self) -> List[Content]:
        demo_content: List[Content] = []
        demo_string = ""
        demo_string = "**Input:**\n"
        for filled in self.filled_input_variables:
            demo_string += f"{filled.input_variable.name}:\n"
            input_variable = filled.input_variable
            if input_variable.image_type:
                demo_content.append(TextContent(text=demo_string))
                demo_string = ""
                if input_variable.image_type == "web":
                    demo_content.append(ImageContent(image_url=filled.value))
                else:
                    demo_content.append(
                        get_image_content_from_upload(
                            image_upload=filled.value,
                            file_type=input_variable.image_type,
                        )
                    )
        if self.reasoning is not None:
            demo_string += f"**Reasoning:**\n{self.reasoning}\n"
        else:
            demo_string += "**Reasoning:**\nnot available"
        demo_string += f"**Answer:**\n{self.output}"
        demo_content.append(TextContent(text=demo_string))
        return demo_content

    def __repr__(self):
        """Naive string representation of the demonstration

        NOTE: This is not used for adding demo to the actual user message. check `add_demos_to_prompt` instead.

        Demonstration designer can have their own way to present the demonstration
        especially when the input contains other modalities
        """

        def truncate(text, max_length=200):
            """Truncate text if it exceeds max_length, appending '...' at the end."""
            return text if len(text) <= max_length else text[:max_length] + "..."

        inputs_truncated = {
            filled_input_var.input_variable.name: truncate(filled_input_var.value)
            for filled_input_var in self.filled_input_variables
        }
        input_str = "**Input**\n" + json.dumps(inputs_truncated, indent=4)
        if self.reasoning:
            input_str += f"\n\n**Reasoning**\n{truncate(self.reasoning)}"
        demo_str = f"{input_str}\n\n**Response**\n{truncate(self.output)}"
        return demo_str


@dataclass
class CompletionMessage:
    role: Literal["system", "user", "assistant"]
    content: List[Content]
    name: str = None

    def to_api(self) -> Dict[str, str]:
        msg = {
            "role": self.role,
            "content": [content.__dict__ for content in self.content],
        }
        if self.name:
            msg["name"] = self.name
        return msg
