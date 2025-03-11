.. _cognify_interface:

The Cognify Programming Model
=============================

We introduce a simple workflow programming model, the **Cognify Programming Model**. It is an easy-to-use interface designed for implementing gen-AI workflows. You do not need to use the Cognify programming model to use Cognify. For example, you can run unmodified :ref:`LangChain <cognify_tutorial_interface_langchain>` and :ref:`DSPy <cognify_dspy_interface>` programs on Cognify and skip this section.  

The Cognify programming model centers around :code:`cognify.Model`, a class used for defining a model call (currently, Cognify only supports language models) for Cognify's optimization.
This class is designed to be a drop-in replacement for your calls to a model such as the OpenAI API endpoints. 
:code:`cognify.Model` abstracts away the complexity of model selection and prompt construction, allowing you to focus on your business logic. 
Model calls not specified with :code:`cognify.Model` will still run but will not be optimized. 

We do not restrict how different model calls interact with each other, allowing users to freely define their relationship and communication. 
For example, you can pass the generation of one model call as the input to another model call, to multiple downstream model calls, to a tool/function calling, etc.
You can also write your own control flow like loops and conditional branching.

**Workflow Code for Solving Maths Problems:**

.. include:: _cognify_front.rst

The math-solver example above has two model calls specified by :code:`cognify.Model`, the :code:`interpreter_agent` and the :code:`solver_agent`. 

.. note::

   The :code:`cognify.StructuredModel` class allows for more complex output formats specified in a ``pydantic`` schema.

The :code:`math_solver_workflow` function specifies the overall workflow process, with the generation of :code:`interpreter_agent` is passed as the input of :code:`solver_agent`.

As seen, :code:`cognify.Model` encapsulates four key components that you should specify:

#. **System prompt**: The :code:`system_prompt` field specifies the initial "system" part of a prompt sequence sent to a language model to define the model's role or provide context and instructions to the model. For example, "You are a math problem interpreter..." and "You are a math solver..." are the system prompts for the two model calls in our math example, as shown below. A language model call has one system prompt that is used regardless of the user inputs. We mandate this information in the Cognify programming model because Cogs like task decomposition rely on the system prompt.
#. **Request Inputs**: The :code:`input_variables` defines the parts of the prompt that change from request to request. For example, the :code:`solver_agent` has two input variables: the first being the end-user request math problem and the second being the generation of the :code:`interpreter_agent` step. The reason it is an input "variable" is that the actual argument changes from one workflow invocation to another. While you can concatenate all content into a single input variable, Cognify can achieve better optimization results if each variable represents a distinct piece of information.
#. **Output format**: The :code:`output_format` field specifies the format of the model output. It can simply be a label assigned to the output string or a complex schema that the response is expected to conform to. For the latter, you need to use the :code:`cognify.StructuredModel` class. 
#. **Language model configuration**: The :code:`lm_config` field specifies the initial set of language models and their configurations that Cognify uses as the starting point and as the baseline to compare for reporting its optimization improvement. You can add more models for Cognify to explore in the :ref:`optimization configuration file <config_search>`. 

.. hint::

   When including models from different providers in your configuration, make sure that all required API keys are provided in your environment for the optimizer to call the models.

For Cognify to properly capture your :code:`cognify.Model`, be sure to instantiate them as global variables. Cognify expects a stable set of optimization targets at initialization. Local instantiations create an unstable set of targets that can lead to inaccurate optimization results. However, once instantiated, they can be invoked anywhere in your program.

Invoking a :code:`cognify.Model` (or :code:`cognify.StructuredModel`) is straightforward. Simply pass in a dictionary of inputs that maps the variable name to its actual value. 
Cognify uses the system prompt, input variables, and output format to construct the messages to send to the model endpoints. 
We encourage users to let Cognify handle message construction and passing. However, for fine-grained control over the messages and arguments passed to the model and easy integration with your current codebase, you can optionally pass in a list of messages and your model keyword arguments. 
For more detailed usage instructions on output formatting, image inputs, and locally hosted models, check out our `GitHub repo <https://github.com/WukLab/Cognify/tree/main/cognify/llm>`_.

To integrate the workflow with Cognify, you need to register the function that invokes the workflow with our decorator ``@cognify.register_workflow`` like so:

.. code-block:: python

   import cognify
   ...

   @cognify.register_workflow
   def math_solver_workflow(workflow_input):
      math_model = interpreter_agent(inputs={"problem": workflow_input})
      answer = solver_agent(inputs={"problem": workflow_input, "math_model": math_model})
      return {"workflow_output": answer}


Language Model Configuration
----------------------------

The most common search customization is model selection, which asks Cognify to choose between different models. 
To provide the optimizer with a list of models to search over, you can define a list of :code:`cognify.LMConfig` objects like so:

.. code-block:: python

   import cognify

   gpt = cognify.LMConfig(model='gpt-4o-mini'),
   claude = cognify.LMConfig(model='claude-3.5-opus', kwargs={'max_tokens': 100),
   llama = cognify.LMConfig(
      custom_llm_provider='fireworks_ai',
      model="accounts/fireworks/models/llama-v3p1-8b-instruct",
      kwargs={'temperature': 0.7}
   )

   my_agent1 = cognify.Model(..., lm_config=gpt)
   my_agent2 = cognify.Model(..., lm_config=claude)
   my_agent3 = cognify.Model(..., lm_config=llama)

The only required parameter is :code:`model`. All other parameters are optional. In cases where multiple providers host the same model, you will need to provide :code:`custom_llm_provider` to specify the provider you are querying (e.g., :code:`'fireworks_ai'` or :code:`'together_ai'`). Under the hood, we support any model and provider combo that is supported by `LiteLLM <https://www.litellm.ai/>`_. You can specify the :code:`kwargs` parameter to pass in any additional keyword arguments to the model, such as :code:`temperature` or :code:`max_tokens`.
