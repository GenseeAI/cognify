.. _cognify_tutorials_cli:

*************
Executing Cognify
*************

Cognify provides a simple command-line interface (CLI) to optimize, evaluate, and inspect generative AI workflows.

Basic Syntax
====================

::

   cognify [mode_switch] <path_to_workflow> [optional common_options] [optional mode_options]

CLI Common Options
===================

Following options that are common across all modes:

.. list-table::
   :widths: 5 15 20 10 50
   :header-rows: 1

   * - Short
     - Long
     - Parameter
     - Default
     - Description

   * - -c
     - \-\-config
     - <config filepath>
     - See Description
     - Specify path to the configuration file for the workflow.
       If not provided, will look for ``config.py`` in the same directory as the workflow script.

   * - -l
     - \-\-log_level
     - DEBUG, INFO, WARNING, ERROR, CRITICAL
     - WARNING
     - **Sets the logging level**  

CLI Mode Switches
==================

.. list-table::
   :widths: 10 90
   :header-rows: 1

   * - Mode
     - Description

   * - optimize
     - Searches for cost-effective configurations of the given workflow. All intermediate results are saved for resuming.

   * - evaluate
     - Evaluates the a specific configuration on the test data.

   * - inspect
     - Lists all the Pareto-frontiers found during optimization so far.

.. _cognify_cli_opt_mode:

CLI Optimize Mode Options
--------------------------

::

   cognify optimize <path_to_workflow> [optional common_options] [optional optimize_options]

.. list-table::
   :widths: 5 15 20 10 50
   :header-rows: 1

   * - Short
     - Long
     - Parameter
     - Default
     - Description

   * - -r
     - \-\-resume
     - 
     - False
     - Resume optimization using saved intermediate results.

   * - -f
     - \-\-force
     - 
     - False
     - Force overwriting the existing result folder.

CLI Evaluate Mode Options
---------------------------

::

   cognify evaluate <path_to_workflow> [optional common_options] [optional evaluate_options]

.. list-table::
   :widths: 5 15 20 10 50
   :header-rows: 1

   * - Short
     - Long
     - Parameter
     - Default
     - Description

   * - -s
     - \-\-select
     - Optimization_1, Optimization_2, ..., or Original
     - Original
     - Select a specific configuration by ID for evaluation. Use ``Original`` to evaluate the original workflow.

   * - -j
     - \-\-n_parallel
     - 1, 2, ...
     - 10
     - Number of parallel data points to evaluate. Refer to the API rate limit of your model provider when setting this value.

   * - -o
     - \-\-output_path
     - <filename>
     - See Description
     - Path to save evaluation results. Defaults to ``eval_result.json`` in the workflow script's directory.


CLI Inspect Mode Options
------------------------

::

   cognify inspect <path_to_workflow> [optional common_options] [optional inspect_options]

.. list-table::
   :widths: 5 15 20 10 50
   :header-rows: 1

   * - Short
     - Long
     - Parameter
     - Default
     - Description

   * - -f
     - \-\-dump_optimization_results
     - 
     - False
     - Dump descriptive details of all optimizations found ``optimized_workflow_details`` directory in the optimization result folder.

Example CLI Usage
====================

.. rubric:: Check help info

::

   cognify -h

check help info for optimize mode
:: 
   
   cognify optimize -h

.. rubric:: Optimize a workflow

::

   cognify optimize workflow.py

If config file is named differently and want to overwite the result folder:
::
   
   cognify optimize workflow.py -c <path_to_config> -f


.. rubric:: Evaluate a configuration

::

   cognify evaluate workflow.py --select Optimization_1

Evaluate the original program with a batch size of 50:
:: 

   cognify evaluate workflow.py -j 50

.. rubric:: Inspect current optimization results

::

   cognify inspect workflow.py --dump_optimization_results
