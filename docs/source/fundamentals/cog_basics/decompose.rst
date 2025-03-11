Scaffolding
===========

The `LMScaffolding` Cog is designed to decompose complex tasks by creating multiple worker agents, each handling a more focused subtask. By breaking down a task into smaller, manageable parts, `LMScaffolding` enhances performance and interpretability, especially for ambiguous and intricate tasks.

DecomposeOption
---------------
Each option represents one possibility of the decomposed system.

Cognify provides three ways of specifying decomposition candidates:

- Write the new system in Cognify ``Workflow``.

.. seealso::
    `How to write in Cognify IR <cognify_ir>`_

- Provide a ``StructuredAgentSystem`` object that depicts high-level informations of the new system.
- Automatically bootstrap options using Cognify's built-in toolkit.

Define ``StructuredAgentSystem``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To do this, you need to provide the follwing information:

1. **Each sub-agent**: This includes their role (system prompt), input/output labels.
2. **Dependencies**: This decides the execution order of the sub-agents, also helps the runtime to maximize the parallelism.
3. **Aggregation Logic**: This is the final step that combines the outputs of all sub-agents to generate the final response. This is optional as one may directly use the output of one sub-agent as the final output.

Use Cognify's Decompose Toolkit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Cognify offers a streamlined toolkit to automatically bootstrap decomposition options for a complex workflow. This process leverages an LLM to analyze and generate the candidates.


.. rubric:: LLM-Assisted Analysis

Cognify begins by assessing the complexity of each agent within the current workflow. The complexity score ranges from 1 to 5. The rating criteria are as follows:

1. Straightforward, the task is simple and clear.
2. Given problem has multiple horizontal sub-tasks, but all are simple to tackle.
3. Given problem requires vertical planning, reasoning, decision-making, but all sub-tasks are very simple.
4. At least one sub-task is non-trivial to solve, can benefit from task decomposition for better interpretability and ease of reasoning.
5. Task is ambiguous, complex, and requires very fine-grained decomposition.

.. rubric:: Perform Decomposition

Once the complexity scores are obtained, Cognify selects the agents with scores above a predetermined threshold as decomposition targets. These agents are considered sufficiently complex to benefit from task decomposition.

For each target, Cognify prompts the LLM to generate a high-level `StructuredAgentSystem` object. This object encapsulates the suggested decomposition structure. Subsequently, Cognify’s compiler parses the information to materialize a new workflow. The compiler translates the LLM-generated decomposition structure into a runnable gents, wherein each agent performs a focused part of the original task.

.. hint::

   Cognify supports bootstrapping decomposition options for all agents in the workflow simultaneously. The complexity analysis of involved modules will be done in a single LLM call, ensuring a consistent basis for comparison and improves the analysis accuracy & efficiency.

Example Usage
-------------

The following example demonstrates how to create an `LMScaffolding` Cog using Cognify's bootstrapping toolkit:

.. code-block:: python

   import cognify
   from cognify.hub.cogs import LMScaffolding

   # Initial LLM agents in the workflow
   agent_0 = cognify.Model(...)
   agent_1 = cognify.Model(...)
   agent_2 = cognify.Model(...)

   # Bootstrap the LMScaffolding Cog for all agents
   scaffolding_cogs = LMScaffolding.bootstrap(
      lm_modules=[agent_0, agent_1, agent_2],
      # Decompose the agents with complexity score above 4
      decompose_threshold=4,
   )

   # Agents below the threshold will only have NoChange option in the Cog
   cog_0, cog_1, cog_2 = scaffolding_cogs