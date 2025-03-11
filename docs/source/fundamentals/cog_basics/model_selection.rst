Model Selection
===============

The **Model Selection Cog** (`LMSelection`) in Cognify enables the adjustment of language models for each agent within a workflow. This Cog allows the optimizer to choose between different model configurations to balance quality and cost based on the task's requirements. Each model configuration is encapsulated within a `ModelOption`.

ModelOption
-----------

Each `ModelOption` defines a unique language model configuration with the following key properties:

- **model_config** (`LMConfig`): Contains the configuration details for the model, such as the provider (`openai`, `fireworks`, etc.), model name, built-in cost indicator, and other standard parameters (e.g., `max_tokens`, `temperature`).
- **cost_indicator** (`float`): A property that reads the cost indicator from :attr:`LMConfig.cost_indicator`, helping the optimizer evaluate cost-effectiveness.
- **apply** (`Callable`): A method that changes the model configuration of a `cognify.Model` module, updating it with the selected model settings and reinitializing the predictor if necessary.

Example Usage
-------------

Below is an example of how to define and initialize a Model Selection Cog with multiple model options:

.. code-block:: python

   import cognify
   from cognify.hub.cogs import LMSelection, model_option_factory
   from cognify import LMConfig

   # Define model configurations, each encapsulated in a ModelOption
   model_configs = [
      # OpenAI model
      LMConfig(
         custom_llm_provider='openai',
         model='gpt-4o-mini',
         cost_indicator=1.0,
         kwargs={'temperature': 0.0}
      ),
      # Fireworks model
      LMConfig(
         custom_llm_provider='fireworks',
         model="accounts/fireworks/models/llama-v3p1-8b-instruct",
         cost_indicator=0.6,
         kwargs={'temperature': 0.0}
      ),
      # Self-hosted model with OpenAI-compatible API
      LMConfig(
         custom_llm_provider='local',
         model='llama-3.1-8b',
         cost_indicator=0.0,  # Indicates no cost for local models
         kwargs={
            'temperature': 0.0,
            'openai_api_base': 'http://192.168.1.16:30000/v1'
         }
      ),
   ]

   # Create Model Options from LM configurations
   options = model_option_factory(model_configs)

   # Initialize the Model Selection Cog; the optimizer will search from the above options
   model_selection_cog = LMSelection(
      name="model_selection_example",
      options=options,
   )