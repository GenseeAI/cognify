.. _cognify_tutorial_interface_langchain:

LangChain
=========

Cognify supports unmodified LangChain programs. All you need to do is to **register the entry function** for Cognify to execute the workflow during optimization. The following is the LangChain code for the **Math Problem Solver** example:

.. include:: _langchain_front.rst

In LangChain, the :code:`Runnable` class is the primary abstraction for executing a task. When initializing the optimizer, Cognify will automatically translate all runnables that are instantiated as global variables into a :code:`cognify.Model` or :code:`cognify.StructuredModel`. 

.. tip::

  Your chain must follow the following format: :code:`ChatPromptTemplate | ChatModel | (optional) OutputParser`. This provides :code:`cognify.Model` with all the information it needs to optimize your workflow. The chat prompt template **must** contain a system prompt and at least one input variable. 

By default, Cognify will translate **all** runnables into valid optimization targets. For more fine-grained control over which :code:`Runnable` should be targeted, you can manually wrap your chain with our :code:`cognify.RunnableModel` class like so: 

.. code-block:: python

  import cognify
  ...

  solver_agent = solver_template | model | parser

  # -- manually wrap the chain with RunnableModel --
  solver_agent = cognify.RunnableModel("solver_agent", solver_agent)

  from cognify.optimizer.registry import register_workflow
  @cognify.register_workflow
  def math_solver_workflow(workflow_input):
    math_model = interpreter_agent.invoke({"problem": workflow_input}).content

    # -- invocation remains the same --
    answer = solver_agent.invoke({"problem": workflow_input, "math_model": math_model}).content
    return {"workflow_output": answer}

If you prefer to define your modules using our :code:`cognify.Model` interface but still want to utilize them with your existing LangChain infrastructure, you can wrap your :code:`cognify.Model` with an :code:`as_runnable()` call. This will convert your :code:`cognify.Model` into a :code:`cognify.RunnableModel` and follows the LangChain :code:`Runnable` protocol.

.. code-block:: python

  import cognify
  ...

  cognify_solver_agent = cognify.Model("solver_agent", ...)

  # -- manually wrap the cognify model with `as_runnable()` --
  solver_agent = cognify.as_runnable(cognify_solver_agent)

  @cognify.register_workflow
  def math_solver_workflow(workflow_input):
    math_model = interpreter_agent.invoke({"problem": workflow_input}).content

    # -- invocation remains the same --
    answer = solver_agent.invoke({"problem": workflow_input, "math_model": math_model}).content
    return {"workflow_output": answer}

Cognify is also compatible with **LangGraph**, a popular orchestration framework. It can be used to coordinate LangChain runnables, DSPy predictors, any other framework or even pure python. All you need to do to hook up your LangGraph code is use our decorator to **register** your invocation function.
