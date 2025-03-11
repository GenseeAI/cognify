**********
Overview
**********

Cognify, a comprehensive, multi-objective gen-AI workflow optimizer. Cognify transforms gen-AI workflow programs into optimized workflows with multi-faceted optimization goals, including high workflow generation quality and low workflow execution cost. 
Cognify offers improvements of up to 2.8x in generation quality, up to 10x cost reduction, and up to 2.7x reduction in end-to-end latency.
It pushes the cost-quality Pareto frontier over SoTA solutions like DSPy, and it allows users to choose their preferred cost-quality combinations. Moreover, Cognify automates the entire optimization process with one click. 

The core idea of Cognify is to perform optimizations at the workflow level instead of at each individual workflow component. Since upstream components’ generation highly affects how downstream components perform, optimizing components in isolation does not work and could even negatively impact the final generation quality and workflow execution costs. 
Instead, Cognify optimizes the entire workflow by iteratively experimenting with various combinations of tuning methods (we call them "*cogs*") applied across workflow components and assessing the effectiveness of these combinations based on the quality of the final output. 
Cognify generalizes its optimization to different workflows by treating workflows as "*grey boxes*" and cogs as *hyper-parameters* to the workflows. The grey-box approach is in between white boxes and black boxes where we analyze and utilize workflows' internal structures but not what each workflow step does.

Cognify currently only optimizes language model calls. Cognify executes other model calls and other types of workflow components as is. The optimization of these non-language-model computation is left for future work, and we welcome your contribution in these areas.

To use Cognify, users provide a gen-AI workflow they write (we currently support LangChain, DSPy, and Cognify's own Python programming model). In addition, users supply the :ref:`training dataset <cognify_tutorials_data_loader>` and specify a workflow quality :ref:`evaluator <cognify_tutorials_evaluator>` that Cognify can use in its optimization iterations.
Instead of using default configurations, you can also :ref:`customize your optimization process <config_search>` by controlling the maximum number of iterations Cognify explores, the quality vs. cost goal constraints, and the set of models Cognify can explore.

The rest of this tutorial will explain each step involved in using Cognify with an example `Math Problem Solver workflow <https://github.com/GenseeAI/cognify/blob/main/examples/math/workflow.py>`_: 

* Porting workflows to Cognify
* Specifying evaluators for a workflow
* Specifying training and test dataset
* Configuring a Cognify optimization

.. seealso::

    For the full set of examples and templates, please refer to :ref:`examples <examples-toctree>`.

Let's get started!

