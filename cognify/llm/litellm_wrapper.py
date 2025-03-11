from litellm import completion
from pydantic import BaseModel

def litellm_completion(model: str, messages: list, model_kwargs: dict, response_format: BaseModel = None):
    if response_format:
        model_kwargs["response_format"] = response_format

    # handle ollama
    if model.startswith("ollama"):
        for msg in messages:
            concatenated_text_content = ""
            if isinstance(msg["content"], list):
                for entry in msg["content"]:
                    # Ollama image API support: https://github.com/GenseeAI/cognify/issues/11
                    assert entry["type"] != "image_url", "Image support for ollama coming soon."
                    concatenated_text_content += entry["text"]
                msg["content"] = concatenated_text_content
    
        if response_format:
            del model_kwargs["response_format"]
            model_kwargs["format"] = response_format.model_json_schema()
    
    response = completion(
        model,
        messages,
        **model_kwargs
    )

    return response