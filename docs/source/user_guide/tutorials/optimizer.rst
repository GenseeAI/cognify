.. _config_search:

*************************
Configuring Optimizations
*************************

Cognify uses a set of configurations for its optimizations, including the maximum number of optimization iterations, the set of models to use.

=====================
Specify the Model Set
=====================

During its optimization, Cognify explores combinations of different models for better quality and cost automatically. To specify the list of models you want Cognify to explore, you can define a list of :code:`cognify.LMConfig` objects as follows. Currently, Cognify only support language models.

.. code-block:: python

    # config.py
    import cognify

    model_configs = [
        # OpenAI models
        cognify.LMConfig(model='gpt-4o-mini', kwargs={'temperature': 0, 'max_tokens': 300}),
        cognify.LMConfig(model='gpt-4o', kwargs={'temperature': 0, 'max_tokens': 300}),
    ]

Each :code:`cognify.LMConfig` has to have a :code:`model` field. If the model is provided by several providers (e.g., Llama models), you also need to specify a :code:`custom_llm_provider` field.

.. hint::

    Ensure that you have the appropriate API keys for the providers you are using. 

There are two other optional configurations.

1. you can set a :code:`cost_indicator` for each :code:`LMConfig` to tell the optimizer to favor cheaper options when proposing a new configuration. By default, each :code:`LMConfig` has a :code:`cost_indicator = 1.0`, which tells the optimizer that all models are equally expensive (i.e. not to factor cost into its search).

.. note::

    The :code:`cost_indicator` does not need to reflect the true difference in prices between models. For example, if you are hosting models yourself, you can set the indicator to ``0`` if local GPU resources is treated as free.

    Additionally, this indicator only helps the optimizer to propose frugal configurations in the early stage. It still relies on observations of the actual cost to update the internal states.

2. you can specify the parameters used to call the model with :code:`kwargs`. These are the ones specified by your model API.

In this example, we ask Cognify to automatically choose between two OpenAI models, :code:`gpt-4o-mini` and :code:`gpt-4o`, for each agent in the Math workflow.

Configure Optimizer Settings
----------------------------

Apart from a model set, Cognify expects a set of configurations, encapsulated in the :code:`create_search` class.
After defining your model selection, the easiest way to configure your Cognify optimization process is to use our provided default configuration by importing :code:`default` as follows: 

.. code-block:: python

    from cognify.hub.search import default
    search_settings = default.create_search(
        model_selection_cog=model_configs # pass in the models we want to search over
    )

The default configuration internally uses the following set of values (you do not need to define :code:`create_search` if you are using the default; the values are listed below for your reference):

.. code-block:: python

    def create_search(
        *,
        search_type: Literal["light", "medium", "heavy"] = "light",
        model_selection_cog: model_selection.LMSelection | list[LMConfig] | None = None,
        n_trials: int = None,
        quality_constraint: float = 1.0,
        objectives: list[Literal["quality", "cost", "latency"]] = ["quality", "cost", "latency"],
        evaluator_batch_size: int = 10,
        opt_log_dir: str = "opt_results",
    ):
        ...

.. hint::

    If ``n_trials`` is not specified, Cognify will use the default number of trials based on the search type:

    - "light": 10
    - "medium": 45
    - "heavy": 140

Instead of using the default, you can customize your workflow optimization process to get the best out of Cognify.
This is done by simply setting up your :code:`create_search` function. Below we explain each configuration in :code:`create_search` in four categories.

Essential parameters:

* :code:`model_selection_cog (list[LMConfig])`: Specify the models that Cognify can explore with :code:`LMConfig` objects. For example, the :code:`search_settings` object specifies :code:`model_configs` as the model set in the above :code:`default` code block. If this parameter is not specified, Cognify will not explore multiple models and will simply use the models defined in your original workflow. Specifying this parameter will override the models in the original workflow.
* :code:`objectives (list[str])`: A list of objectives that Cognify will optimize for. The default is :code:`["quality", "cost", "latency"]`. You can remove any of these objectives if you do not want Cognify to optimize for them. For example, if you only want to optimize for quality and cost, you can set :code:`objectives=["quality", "cost"]`.
* :code:`opt_log_dir (str)`: The directory (under the workflow directory) where the optimization results will be stored. The default directory is named "opt_results". From :code:`opt_log_dir`, you can inspect the optimized workflow, use it in your code, or resume your optimization with more iterations (trials).

Parameters to determine the amount of exploration:

* :code:`search_type (str)`: Either **"light", "medium",** or **"heavy"**. This determines the amount of search Cognify performs within each iteration (trial), with "light" being the lightest and quickest, "heavy" being the most complex and the slowest, and "middle" being in between. While being the slowest, "heavy" usually yields the best optimization results.
* :code:`n_trials (int)`: A trial represents one iteration of Cognify's optimization. Each trial executes your training data once. More trials result in better optimization results but slower optimization and higher optimization cost (you need to pay to your model provider). This parameter allows you to roughly budget your optimization. 

.. hint::

    For complex workflows, we recommend a higher number of trials (e.g., 30) to allow the optimizer to effectively explore the search space.

Parameters for constraining Cognify's search:

* :code:`quality_constraint (float)`: In certain cases, you may want to only explore cost reductions if your workflow's generation quality is above a certain threshold. This configuration is designed for such cases. 
The quality constraint here represents the quality of the optimized workflow *relative* to the original workflow's generation quality. A value of 1.0 (the default) means that the optimized workflow must be at least the same quality as the original program. 
Setting a value below 1 allows for higher cost reduction. 
Note that the optimization results can (and will often) have quality higher than the quality constraint. Thus, a value below 1 does not necessarily mean lowered quality in Cognify's optimization results.

.. hint::

    A quality constraint of 1 or below will always yield optimization results, while a quality constraint above 1 may result in "no optimization found".

Parameters for controlling your optimization speed:

* :code:`evaluator_batch_size (int)`: This tells the optimizer how many training data points to evaluate at once. If you are using a cloud-based service, you can adjust this parameter to avoid rate limiting.

.. note::

     We also provide a few built-in domain-specific configurations that you can use directly for the `example workflows <https://github.com/WukLab/Cognify/tree/main/examples>`_ we provide, including QA :code:`qa`, code generation :code:`codegen`, and data visualization :code:`datavis`. You can use these settings like:

     .. code-block:: python

         from cognify.hub.search import codegen
         search_settings = codegen.create_search()
