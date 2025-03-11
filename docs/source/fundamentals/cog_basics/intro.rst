.. _cog_intro:

********************
Introduction to Cogs
********************

A core concept in Cognify is `Cogs`, a term we use to refer to all types of optimizations that could apply to gen AI workflows.
`Cogs` can be optimizations that work across components in a gen AI workflow by changing the workflow structure (e.g., decompose a component into two);
`Cogs` can also be optimizations that focus on single compoentns (e.g., improving the prompt of an LLM component). 
Cognify treats all `Cogs` as "hyperparameters" of gen AI workflows and apply an overall `Cog-tuning flow <optimizer_overview>`_.

.. tip::

   Think of each Cog in Cognify as similar to a learnable weight in a neural network. Just as neuron weight influence how a neural network predicts, Cogs control various aspects of a generative AI workflow by setting different value for each Cog, allowing it to adapt for optimal performance.

Key Concepts
============

- **Cog**: A parameter that needs to be optimized within a module.
- **Option**: A choice or setting available to a `Cog`, each offering different transformations to the module.
- **Dynamic Cog**: A `Cog` capable of evolving during optimization, allowing more advanced adaptability in the options it carries based on evaluation results.

.. _cog_basics:

Cog: The Unit of Optimization
-----------------------------

A `Cog` always contains the following attributes:

- **name** (`str`): The name of the `Cog`, developers can use this to identify the `Cog` easily. **NO** two `Cogs` should share the same name if they are in the same module.
- **options** (`list[OptionBase]`): A list of options available for this `Cog`, where each option represents a different way of transforming the module.
- **default_option** (`Union[int, str]`): The default option for the `Cog`, specified by index or name.
- **module_name** (`str`): The name of the module where the `Cog` is applied.
- **inherit** (`bool`): Specifies whether current `Cog` can be inherited by new modules derived from the current one during the optimization process. Set to `True` by default.

Option: Module Behavior Descriptor
----------------------------------

Each `Option` encapsulates a unique configuration or transformation for a module. Selecting an option within a `Cog` changes the module’s behavior according to the option’s implementation. The following core information is included in each `Option`:

- **name** (`str`): The name of the option. This helps developers select or reference specific configurations easily. **NO** two options should share the same name within a `Cog`.

.. _option_cost_indicator:

- **cost_indicator** (`float`): A pre-evaluation estimate of the relative execution cost of applying this option.  
   - **Purpose**: Helping the optimizer anticipate the expense of evaluating a configuration. This is especially useful when two options are expected to have a similar effect on quality, allowing the optimizer to favor a lower-cost option for a more efficient search.
   - **Scope**: The `cost_indicator` only provides a rough estimation to guide frugal search decisions, but it doesn’t replace actual execution costs. You may even set it to a large value to discourage using an option, while the optimizer still relies on real execution costs for final assessment.
   - **Usage**: You can override :func:`OptionBase._get_cost_indicator` to customize the cost penalty for each option. By default it returns `1.0`.

   .. rubric:: Example

   If a module originally costs `$0.5` to execute, and applying this option is expected to increase it to `$1.5`, a reasonable `cost_indicator` would be `3`.

In addition to these attributes, each `Option` provides an ``apply`` method that performs the actual transformation on the module. This method is responsible for changing the module based on the option’s configuration.

.. _dynamic_cog:

DynamicCog: Adaptive Parameter in Cognify
-----------------------------------------

A `DynamicCog` is a specialized type of `Cog` in Cognify that can evolve or adapt its options based on evaluation results during the optimization process. Unlike standard `Cogs`, which have a fixed set of options, `DynamicCogs` are designed to update or generate new options dynamically in response to performance feedback. 

Apart from standard attributes in normal `Cog`, each `DynamicCog` includes an `evolve` method, which defines how the `Cog` should adapt based on evaluation results. This method is customizable, allowing developers to tailor the evolution process to suit specific parameter types or optimization goals.

**Benefits of Using DynamicCog**

The adaptability of `DynamicCogs` allows for more granular control over parameters that benefit from dynamic refinement. By enabling parameters to evolve based on evaluation feedback, `DynamicCogs` make Cognify’s optimization process more efficient and effective, particularly for complex workflows requiring iterative improvements.

**Note**: The exact behavior of a `DynamicCog` depends on how the developer implements the `evolve` method. This customization provides flexibility, allowing `DynamicCogs` to be tailored to various types of parameters.

.. seealso:: :ref:`cognify_fewshot`
