Setting Up Search Space
===========================

In Cognify, the search space is defined by layers of cogs. This section explains how to structure cogs in each layer and organize them into a multi-level hierarchy.

1) Creating Cogs
---------------------

We start from creating one layer. Let's define all cogs involved in this layer first.

For example, we can add **reasoning** and **few-shot**:

.. code-block:: python

   from cognify.cog_hub import reasoning, fewshot, common

   # Reasoning Parameter
   reasoning_param = reasoning.LMReasoning(
      [common.NoChange(), reasoning.ZeroShotCoT()] 
   )
   # Few Shot Parameter
   few_shot_param = fewshot.LMFewShot(4)

.. seealso::
   
   :ref:`cog_examples` to include more cogs in your search space.

2) Defining a Layer
---------------------

Use ``LayerConfig`` to group the defined cogs and form the search space in this layer:

.. code-block:: python

   class LayerConfig:
      layer_name: str,
      dedicate_params: list[CogBase] = [],
      universal_params: list[CogBase] = [],
      target_modules: Iterable[str] = None,
      ...

To register the cogs in the layer, you can use either the ``dedicate_params`` or ``universal_params``:

**dedicate_params**
   This list includes cogs that apply to specific agents within the workflow. Cogs in this list require ``module_name`` to be set, and the optimizer will enforce this module-specific application.

   ::

      # post initialization
      reasoning_param.module_name = "Agent 0"

      # at initialization
      few_shot_param = fewshot.LMFewShot(4, module_name="Agent 1")
  
**universal_params**
   This list contains cogs that will be broadcast to all agents in the layer, disregarding any ``module_name`` setting that may be specified.

You can mix the usage of `dedicate_params` and `universal_params` to create a more flexible search space.

.. note::

   If ``target_modules`` is specified, only the listed modules will undergo any transformation, regardless of above two lists.

Example Configuration:

.. code-block:: python

   from cognify.optimizer.core import driver

   reasoning_param.module_name = "Agent 0"

   # example optimization Layer with Reasoning and Few-Shot Cogs
   layer_config = driver.LayerConfig(
      layer_name='example_opt_layer',
      dedicate_params=[reasoning_param],     # Specific to modules with set module_name
      universal_params=[few_shot_param],     # Applied across all agents
   )

This configuration creates an **layer** that:
   - Uses ``reasoning_param`` as a dedicated parameter, which is applied to "Agent 0" specifically.
   - Applies ``few_shot_param`` universally to all modules in the layer, including "Agent 0".


3) Organizing Layers into a Multi-Level Hierarchy 
--------------------------------------------------

You can follow the same steps to create multiple layers. Once you are comfortable with all layers, you can organize them into a multi-level hierarchy.

This step is done by passing a list of ``LayerConfig`` objects to the ``ControlParameter``:

.. code-block:: python

   from cognify.optimizer.core import driver
   from cognify.optimizer.control_param import ControlParameter

   layer_0 = driver.LayerConfig(...)  # Top layer (high-level structure)
   layer_1 = driver.LayerConfig(...)  # Mid layer (refinements on structure)
   layer_2 = driver.LayerConfig(...)  # Bottom layer (detailed fine-tuning)

   optimizer_control_param = ControlParameter(
      opt_layer_configs=[layer_0, layer_1, layer_2]
   )

.. admonition:: Layer Order

   Ensure that layers are ordered from top to bottom, with the last layer being the lowest layer, whose proposal generates a fully transformed workflow.
