Agent Reasoning
===================

The **Reasoning Cog** (`LMReasoning`) in Cognify introduces reasoning steps to the LLM generation, allowing agents to produce responses conditioned on rationale tokens. This Cog provides varies methods to enhance the quality and interpretability of responses, especially in complex tasks that require clear logic and multi-step problem-solving.


ReasonThenFormat Methodology
----------------------------

In Cognify, all reasoning options follow the **ReasonThenFormat** approach, designed to leverage the full potential of reasoning tokens without compromising the quality of the model's output. Traditional implementations often generate reasoning tokens alongside the main output, typically in a structured JSON format where one key contains the reasoning tokens and another contains the original response. However, this approach has several limitations.

1. Generating reasoning tokens and final output in a single pass can diminish the model’s generation capability. Existing research suggests that constraining generation with strict formatting requirements can degrade output quality, particularly when models are required to follow a specific structure, such as JSON.
2. Embedding reasoning tokens within a structured format can complicate format instructions, especially if the original output itself has certain formatting requirements, such as Pydantic models or other structured responses.

.. rubric:: Cognify's approach

To address these limitations, Cognify adopts the **ReasonThenFormat** methodology. This approach separates reasoning generation from final output generation, allowing models to produce reasoning tokens freely before synthesizing a structured response.

1. **Free Generation of Reasoning Tokens**: In the first LLM call, the model generates reasoning tokens without any formatting constraints, preserving the model's generative capacity and encouraging more detailed and coherent reasoning.

2. **Concatenation and Final Output**: In the second LLM call, the reasoning tokens are appended to the original prompt, along with any specific output formatting instructions required for the final response. This lets the model synthesize a formal answer based on both the initial prompt and the freely generated reasoning tokens, ensuring that the final output is both well-reasoned and formatted as needed.

.. rubric:: Implementation

Cognify’s Intermediate Representation (IR) allows flexible control over output instructions. During the reasoning step, we remove any formatting constraints (e.g., “be concise,” “output in JSON format”) to avoid interference with reasoning quality. Once the reasoning tokens are generated, we append them to the original prompt and apply the output instructions only in the final call.

.. note::
   This method requires two consecutive LLM calls—one for reasoning tokens and one for the formatted output. However, prompt tokens from the reasoning call are often cacheable (a feature supported by many providers, including OpenAI and Anthropic), which mitigates the cost and overhead of the additional call.


ZeroShotCoT
-----------

The `ZeroShotCoT` option implements `Zero-Shot Chain-of-Thought <https://arxiv.org/pdf/2205.11916>`_, guiding the model to reason through a problem step-by-step before providing a final answer. This approach is useful for tasks that require multi-step reasoning or vertical problem-solving.

- **Cost Indicator**: By default 2.0. the extra reasoning step incurs moderate cost.
- **Reasoning Instruction**: "Let's solve this problem step by step before giving the final response."
  
PlanBefore
----------

The `PlanBefore` option encourages the model to break down a task into sub-tasks, providing responses for each sub-task as part of the reasoning process. This process largely resembles the agent architecture proposed in `LLMCompiler <https://arxiv.org/pdf/2205.11916>`_, which is originally designed to accelerate task execution. This approach is useful for complex questions that can be decomposed into smaller, parallel queries.

- **Cost Indicator**: By default 3.0. This is a modest estimation (assuming 2-subtask plan in average). You can adjust it based on the complexity of the task.
- **Reasoning Instruction**: "Let's first break down the task into several simpler sub-tasks that each covers different aspect of the original task. Clearly state each sub-question and provide your response to each one of them."

Other Reasoning Options
-----------------------

In addition to **ZeroShotCoT** and **PlanBefore**, Cognify offers other options. While we won’t go into detail for each here, these options allow for further customization of reasoning strategies within workflows, and more options are planned for future releases.

The other reasoning options currently available include:

- **Tree-of-Thought**: Structures reasoning in a tree-like format to explore multiple solution paths. See the paper: `Tree of Thoughts: Deliberate Problem Solving with Large Language Models <https://arxiv.org/abs/2305.10601>`_.
- **Meta-Prompting**: Guides the main agent to decompose complex tasks into subtasks handled by specialized "experts", whose outputs are then coordinated and integrated by the main worker. The persona and prompt for each expert is generated by the main agent. See the paper: `Meta-Prompting: Enhancing Language Models with Task-Agnostic Scaffolding <https://arxiv.org/abs/2305.10601>`_.

Example Usage
-------------

Here is an example of how to define and initialize an LMReasoning Cog with multiple options:

.. code-block:: python

   from cognify.hub.cogs import NoChange, LMReasoning, ZeroShotCoT, PlanBefore

   # NoChange option stands for NO transformation to the module
   reasoning_options = [NoChange(), ZeroShotCoT(), PlanBefore()]

   # Initialize the LMReasoning Cog
   reasoning_cog = LMReasoning(
      name="reasoning_example",
      options=reasoning_options,
   )

Cognify strives to provide a comprehensive set of reasoning options to cater to various reasoning requirements in generative AI workflows. Apart from registering the reasoning Cog in the search space, you can also apply it manually to your workflow to enhance the reasoning capability of your LLM agents. 

.. code-block:: python

   import cognify
   from cognify import Input, OutputFormat
   from pydantic import BaseModel

   # Define the response format schema
   class Response(BaseModel):
      supporting_facts: list[str]
      answer: str

   # Initialize a cognify.StructuredModel
   # Cognify will automatically inject format instructions to the prompt
   cognify_agent = cognify.StructuredModel(
      agent_name='qa_agent',
      system_prompt='You are an expert in responding to user questions based on provided context. Answer the question and also provide supporting facts from the context.',
      input_variables=[
         Input(name="question"),
         Input(name="context")
      ],
      output_format=OutputFormat(schema=Response),
   )

   output: Response = cognify_agent.forward(
      {
         "question": "What is the capital of France?",
         "context": "France is a country in Europe."
      }
   )

   # Applying ZeroShotCoT reasoning manually to the agent
   from cognify.hub.cogs import ZeroShotCoT

   cognify_agent = ZeroShotCoT().apply(cognify_agent)
   output: Response = cognify_agent.forward(
      {
         "question": "What is the capital of France?",
         "context": "France is a country in Europe."
      }
   )

   # Inspect the reasoning step result
   print(cognify_agent.rationale)