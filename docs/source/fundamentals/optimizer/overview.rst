.. _optimizer_overview:

Overview
========

The optimizer module is designed to optimize an agentic workflow by tuning the :ref:`hyper-parameters <cog_intro>`  of various workflow components. It operates hierarchically, leveraging multi-level parameter tuning to improve workflow efficiency. 

Problem Formalization
---------------------

The optimization problem in Cognify is a multi-objective optimization aimed at balancing the trade-offs between **generation quality** and **execution cost** by tuning the value of each Cog in the search space. The problem can be summarized as:

**Given**
   - A gen-AI workflow (or a set of modules to be optimized)
   - Evaluation criteria (any quality metrics)
   - A searh space (cogs and their options)

**Goal**:
   - Maximize workflow quality
   - Minimize execution cost
  
**Output**:
   A set of configurations along the quality-cost Pareto frontier, each represents a valid trade-off that's not dominoated by any other configuration.

Optimization Structure
----------------------

The optimizer is structured hierarchically, dividing the parameter space into multiple layers, each dedicated to optimizing a specific set of cogs.

Each layer proposes its configurations of all cogs at that hierarchy, which serves as the basis for the next layer to apply more transformations. This style resembles conventional multi-level optimization, e.g. neural architecture search (NAS).

Each layer has its own optimization kernel, which consumes feedbacks from the successor layer and gradually propose better configurations to try. The feedback includes the evaluation results of the selected cog combination. 

   * The **bottom layer**'s proposal is the fully transformed workflow, the evaluation is done by running the workflow on the training data.
   * The **higher layer**'s proposal is a partially transformed workflow, with rest of the cogs in lower-layers un-determined. Its evaluator is the optimization routine of its successor, which returns a paretro frontier reflecting quality-cost trade-offs. To indicate the potential of the proposal, we use the **best-score** and the **lowest-cost** achieved as the feedback.

.. _opt_process:

Optimization Process Explained
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To understand how the optimizer navigates this search space, consider each layer as a nested loop, performing the following steps in sequence:
   - **Propose**: Generates a configuration based on the search strategy.
   - **Evaluate**: Assesses the proposed configuration against target metrics (e.g., quality, cost).
   - **Learn**: Adjusts the search method based on existing observations to improve future proposals.

The following pseudo-code helps to illustrate the process:

.. code-block:: python

   for config_top in layer_top.propose():  # Top layer
      for config_next in layer_next.propose(config_top):  # Next layer
         ...
         for config_bottom in layer_bottom.propose(config_next):  # Bottom layer

            # Full workflow transformation
            new_workflow = config_bottom.apply(
                              ...
                              config_next.apply(
                                 config_top.apply(
                                    base_workflow
                                 )
                              )
                           )

            # Bottom layer generates evaluation result from actual execution
            result = new_workflow.run(test_data)
            layer_bottom.learn(result)
         
         # Upper layers evaluate based on feedback from the layer below
         result = (layer_bottom.best_score, layer_bottom.lowest_cost)
         layer_next.learn(result)
      
      result = (layer_next.best_score, layer_next.lowest_cost)
      layer_top.learn(result)


Each layer builds on the results of its successor, progressively refining configurations to balance quality and cost.

Benefits of layered structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Reduce Search Complexity**
   Decouple a factorial search space into loosely coupled subspaces, allowing each layer in the hierarchy to focus on a more manageable set of parameters. This leads to better scalability and modularity.

**Flexible Resource Allocation**
   Higher layers perform exploration with fewer trials, while lower layers handle detailed tuning with more iterations. This mirrors neural architecture search, where top layers set the basic structure, and lower layers refine it extensively. Users can place cogs by dependency and update frequency.

**Stabilized Convergence**
   Each upper layer provides a stable foundation for lower layers to fine-tune, reducing variability in configuration changes and noise in the feedback.
