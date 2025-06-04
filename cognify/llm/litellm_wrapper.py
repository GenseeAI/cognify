from litellm import completion
from litellm.types.utils import ModelResponse
from pydantic import BaseModel
import requests
import os

import zmq

class HTTPClient:
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(HTTPClient, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._initialized = True
            self.base_url = None

    def post_completion(self, data):
        if self.base_url is None:
            base_url = os.getenv("_cognify_rate_limit_base_url")
            if base_url is None:
                raise Exception("Rate limit base URL not found.")
            self.base_url = base_url
            
        url = f"{self.base_url}/completion_endpoint"
        response = requests.post(url, json=data)
        if response.status_code != 200:
            raise Exception(response.json().get("detail", "Unknown error"))
        result = response.json()["result"]
        hidden_params = result.pop('_hidden_params', None)
        response_headers = result.pop('_response_headers', None)
        response = ModelResponse(**result)
        if hidden_params:
            response._hidden_params = hidden_params
        if response_headers:
            response._response_headers = response_headers
        return response


_client = HTTPClient()


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
    
    # response = completion(
    #     model,
    #     messages,
    #     **model_kwargs
    # )
    response = _client.post_completion({
        "model": model,
        "messages": messages,
        "model_kwargs": model_kwargs
    })
    # print(response)
    return response