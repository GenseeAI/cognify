.. _cognify_introduction:

***************
What is Cognify
***************

Gen-AI Workflows and Key Difficulties
-------------------------------------

Today’s generative-AI (gen-AI) applications often involve creating pipelines of tasks known as gen-AI workflows.
These workflows can include various components, such as gen-AI model calls, tool calling, data retrieval, and other code executions. 
A key difficulty in developing gen-AI workflows is the manual testing and optimization phase, which is extremely time-consuming.
Moreover, the manually tuned workflows often have subpar generation quality or bloated execution cost.

Cognify: A Comprehensive, Multi-Faceted Gen-AI Workflow Optimizer
-----------------------------------------------------------------

Cognify is a fully automated gen-AI workflow optimizer that provides multi-faceted workflow optimizations.
Cognify performs workflow-level optimization by evaluating and optimizing an entire workflow instead of at each individual workflow component.
Built on a novel foundation of hierarchical, workflow-level optimization, Cognify improves gen-AI workflows' quality by up to 48% while reducing costs by as much as 9 times, setting a new standard in gen-AI workflow management.

To use Cognify, users supply their workflow source code, a sample input dataset, and an evaluator for determining the generation quality.
Cognify transforms the user-supplied workflow into a set of optimized versions with different quality-cost combinations that users can choose from and execute directly.


Key Features
------------

- **Multi-Objective Optimization**: Cognify targets both generation quality and workflow execution cost as its optimization goals. Cognify provides multiple optimized workflow versions with different quality-cost combinations for users to choose from.
- **Workflow-Level Architecture Tuning**: In addition to optimizations performed on individual steps of a workflow, Cognify explores modification of the workflow structure by adding, removing, reordering, or parallelizing its steps, while ensuring that users' workflow semantics remain the same. This approach allows for another level of quality improvement and cost reduction.
- **Hierarchical Optimization Framework**: Cognify separates tuning options, or *Cogs*, into architecture-maintaining (AM) and architecture-changing (AC) categories, with AM cogs in a higher, coarser-granularity layer and AC cogs in a lower, fine-tuned layer. This hierachical mechanism combined with Cognify's customized optimization search framework allows for Cognify to finish its optimization quickly.

Benchmark Results
-----------------

We compare Cognify to non-optimized workflows and `DSPy <https://github.com/stanfordnlp/dspy>`_ using the following workloads:

- `HotpotQA <https://hotpotqa.github.io/>`_
- `HoVer <https://github.com/hover-nlp/hover>`_
- `Code generation <https://github.com/openai/human-eval>`_
- `Data Visualization <https://github.com/thunlp/MatPlotAgent>`_
- `Text-to-SQL <https://github.com/ShayanTalaei/CHESS>`_
- `Math solver <https://github.com/hendrycks/math>`_
- `FinRobot <https://github.com/AI4Finance-Foundation/FinRobot>`_ with datasets from `FinGPT <https://github.com/AI4Finance-Foundation/FinGPT?tab=readme-ov-file>`_

The figures below show the generation quality and execution cost (closer a point to the uppper-left cornor, the better) of these results. Pareto frontiers that represent the best quality-cost trade-offs are connected by dashed lines.

Overall, Cognify improves these workflows' generation quality by up to **2.8x**, reduces execution monetary cost by up to **10x**.

.. image:: /_static/images/all_quality_cost.png
   :align: center
