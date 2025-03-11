Module Ensemble
===============

The `ModuleEnsemble` Cog enables ensembling methods within generative AI workflows, allowing multiple independent sampler to generate diverse outputs that are subsequently aggregated to produce a final, refined response. This approach enhances the robustness and quality of generated content by leveraging the collective strengths of various solution path.

.. rubric:: Implementation

All options within the `ModuleEnsemble` Cog follow a common process: **sample then aggregate**. To facilitate this process, the option transforms the target module into a workflow, wherein the entry point is multiple independent samplers each initialized with a clone of the original module, followed by an aggregator that processes proposals of these samplers and to generate a final response.

.. note::
   The ensemble cog introduces significant structural change to the whole workflow, adding multiple modules that can be tuned independently. Take self-consistency as an example, choosing a different configuration for each sampling path can lead to a more diverse set of responses, increasing the chance of hitting the correct answer and reducing output bias.


Universal-Self-Consistency
--------------------------

The `UniversalSelfConsistency` option provides a simple implementation of the `paper <https://arxiv.org/pdf/2311.17311>`_. It operates by spawning multiple workers and adding a llm-based aggregator. The aggregator analyzes all proposals and synthesizes a final answer that reflects the majority consensus. This method is particularly effective for tasks requiring complex reasoning and ensures that the final output is both comprehensive and coherent.

Example Usage
-------------

To utilize the `ModuleEnsemble` Cog with the `UniversalSelfConsistency` option, you can define the search space as follows:

.. code-block:: python

   from cognify.hub.cogs import NoChange, ModuleEnsemble, UniversalSelfConsistency

   # Define the ensemble options
   ensemble_options = [
      NoChange(),
      # We can also ask the option to change the temperature during sampling
      UniversalSelfConsistency(num_path=5, temperature=0.7, change_temperature=True)
   ]

   # Initialize the ModuleEnsemble Cog
   ensemble_cog = ModuleEnsemble(
      name="ensemble_example",
      options=ensemble_options,
   )

Cognify also allows direct application of any ensemble options to your workflow:

.. code-block:: python

   import cognify
   from cognify.graph.program import Workflow
   from cognify.hub.cogs import UniversalSelfConsistency

   agent = cognify.Model(...)
   usc = UniversalSelfConsistency(num_path=3)
   ensembled_new_agent: Workflow = usc.apply(agent)

   # Easiest way to use the workflow with the same interface
   agent.invoke = ensembled_new_agent.invoke
   agent.forward(
      {
         "input_label_1": "value ...",
         "input_label_2": "value ..."
      }
   )