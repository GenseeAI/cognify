# Cognify Interface

## Detailed Usage Instructions

By default, `cognify.Model` will construct messages on your behalf based on the `system_prompt`, `inputs` and `output`. These messages are directly passed to model defined in the `lm_config`. This is the **recommended** interface, as Cognify will control the entire message construction process. However, for compatibility with existing codebases that rely on passing messages and keyword arguments directly, we allow the user to pass in optional `messages` and `model_kwargs` arguments when calling a `cognify.Model` like so:


```python
import cognify

system_prompt = "You are a helpful AI assistant that answers questions."
model_kwargs = {'model': 'gpt-4o-mini', 'temperature': 0.0, 'max_tokens': 100}

# define cognify agent
qa_question = cognify.Input(name="question")
cog_agent = cognify.Model(agent_name="qa_agent",
  system_prompt=system_prompt,
  input_variables=[qa_question],
  output=cognify.OutputLabel(name="answer")
)

@cognify.register_workflow
def call_qa_llm(question):
  messages = [
    {
      "role": "system",
      "content": system_prompt
    },
    {
      "role": "user",
      "content": f"Answer the following question: {question}"
    }
  ]
  result = cog_agent(
    inputs={qa_question: question}, 
    messages=messages, 
    model_kwargs=model_kwargs
  )
  ...
```

## Output Formatting

Cognify allows for additional output formatting. In the base `cognify.Model` class, you can specify custom formatting instructions when defining the output label like so:
```python
cog_agent = cognify.Model(
  ...
  output=cognify.OutputLabel(
    name="answer", 
    custom_output_format_instructions="Answer the question in less than 10 words."
  )
  ...
)
```

### Structured Output

When working with `cognify.StructuredModel`, you must provide a Pydantic `BaseModel` as the schema that will be used to format the response.
```python
import cognify
from pydantic import BaseModel

class ConfidentAnswer(BaseModel):
  answer: str
  confidence: float

struct_cog_agent = cognify.StructuredModel(
  ...
  output_format=cognify.OutputFormat(
    schema=ConfidentAnswer
  )
  ...
)

conf_answer: ConfidentAnswer = struct_cog_agent(...)
```

The `cognify.OutputFormat` class also supports custom formatting instructions, as well as an optional hint parameter: if `should_hint_format_in_prompt=True`, Cognify will construct more detailed hints for the model based on the provided schema.

## Image Inputs

The `cognify.Input` class supports an optional `image_type` parameter, which can take on the values of "web", "jpeg", or "png". If either "jpeg" or "png" is selected, the system expects the image to be Base64 image upload for consistency with the [OpenAI Vision API](https://platform.openai.com/docs/guides/vision). An image input variable can be specified like so:
```python
import cognify
...
# Typical function to encode the image into base64
import base64
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

...
image_input = cognify.Input(name="my_image", image_type="png")
base64_str = encode_image("my_image_path.png")
...

response = cog_agent(inputs={..., 
                            image_input: base64_str, 
                            ...})
```

## Local Models

Cognify calls the `litellm.completion` API under the hood. This means we pass all `kwargs` that you specify when initializing a `cognify.LMConfig` directly to the endpoint. Thus, specifying a local model is as simple as passing in an `api_base` when initializing your `cognify.LMConfig` like so:

```python
import cognify

# ollama: https://docs.litellm.ai/docs/providers/ollama#using-ollama-apichat
local_llama = cognify.LMConfig(
  model="ollama_chat/llama2",
  kwargs={"api_base": "http://localhost:11434"}
)

# vllm:  https://docs.litellm.ai/docs/providers/vllm
vllm_llama = cognify.LMConfig(
  model="hosted_vllm/facebook/opt-125m"
  kwargs={"api_base": "https://hosted-vllm-api.co"}
)