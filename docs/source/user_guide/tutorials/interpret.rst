.. _cognify_tutorials_interpret:

******************************
Interpret Optimization Results
******************************

Get Optimization Results
========================

Before running Cognify, let's first evaluate the original workflow's generation quality and cost. 

This can be done by setting the ``--select Original`` flag with ``cognify evaluate`` to indicate that we are not optimizing the workflow.

.. code-block:: console

    $ cognify evaluate workflow.py --select Original

    ----- Testing Raw Program -----
    =========== Evaluation Results ===========
    Quality: 6.186, Cost per 1K invocation ($): 7.47 $
    ===========================================

Now, let's run Cognify's optimization:

.. code-block:: console

    $ cognify optimize workflow.py

Optimization Overview
=====================

The above optimize command will return a set of optimization results and their generation quality and cost. 

Below is a sample output we got when running it. Note that because of the non-deterministic nature of generative models, you may not get the exact same results.

.. code-block:: console

    ================ Optimization Results =================
    Number of Optimization Results: 4
    --------------------------------------------------------
    Optimization_1
      Quality improvement: 5%
      Cost is 1.06x original
      Quality: 6.47, Cost per 1K invocation: $7.90
    --------------------------------------------------------
    Optimization_2
      Quality improvement: 6%
      Cost is 1.52x original
      Quality: 6.53, Cost per 1K invocation: $11.39
    --------------------------------------------------------
    Optimization_3
      Quality improvement: 3%
      Cost is 0.11x original
      Quality: 6.37, Cost per 1K invocation: $0.80
    --------------------------------------------------------
    Optimization_4
      Quality improvement: 4%
      Cost is 1.05x original
      Quality: 6.43, Cost per 1K invocation: $7.82
    ========================================================

Here, Cognify finds four valid optimization results as different versions of the workflow. You can interprete each item as follows:

    Optimization_1 (config ID to select):
        Represents one of the Pareto-optimal solutions. It balances the trade-off between quality and cost effectively:

        - **Quality Improvement**: 5% higher compared to the original workflow. (higher the better).
        - **Relative Cost**: 1.06x of the original cost (lower the better).
        - **Quality**: 6.467 (average score on the training data).
        - **Cost**: $7.90 per 1K invocations (average invocation cost).

You can also get a summary of the optimization results afterwards with:

.. code-block:: console

   $ cognify inspect workflow.py

Detailed Transformation Trace
=============================

You can further inspect the optimizations Cognify applies by checking the :code:`.cog` files under the ``opt_results/optimized_workflow_details`` directory.

For example, the :code:`Optimization_3.cog` (corresponding to the third result) looks like:

.. note::

    You may not get the exact number of Pareto-frontiers. 
    
    Adjust the `ID` to view configurations in your case.

We show ``Optimization_3`` in the above run as an example:

.. code-block:: console

    $ cat opt_results/optimized_workflow_details/Optimization_3.cog 
    Trial - light_opt_layer_6
    Log at: opt_results/light_opt_layer/opt_logs.json
    Quality: 6.367, Cost per 1K invocation ($): 0.80 $
      Cost is 11.0% of the origin
    ********** Detailed Optimization Trace **********

    ========== Layer: light_opt_layer ==========

    >>> Module: solver_agent <<<

        - Parameter: <cognify.hub.cogs.fewshot.LMFewShot>
        Applied Option: solver_agent_demos_c4d0a1fc-c664-40ec-a7c2-879ede9a241a
        Transformation Details:
            - FewShot Examples -
            2 demos:
            Demonstration 1:
            **Input**
            {
                "math_model": "To solve this problem, we need to determine the number of sequences of length 10 consisting of 0s and 1s that do not contain two consecutive 1s. Let's define \\( a_n \\) as the number of such sequences ...",
                "problem": "A sequence of ten $0$s and/or $1$s is randomly generated. If the probability that the sequence does not contain two consecutive $1$s can be written in the form $\\dfrac{m}{n}$, where $m,n$ are relative..."
            }
            
            **Response**
            To solve the problem, we need to find the number of sequences of length 10 consisting of 0s and 1s that do not contain two consecutive 1s. We will use the recurrence relation given in the math model:
            ...
            ========================================
            Demonstration 2:
            **Input**
            {
                "math_model": "response: To solve this problem, we need to determine the number of distinguishable colorings of the octahedron using eight different colors, considering the symmetries of the octahedron.\n\n1. **Identi...",
                "problem": "Eight congruent equilateral triangles, each of a different color, are used to construct a regular octahedron. How many distinguishable ways are there to construct the octahedron? (Two colored octahedr..."
            }
            
            **Response**
            To solve the problem, we apply the steps outlined in the mathematical model using Burnside's Lemma.
            
            1. **Identify the Symmetries of the Octahedron:**
            The octahedron has 24 rotational symmetries.
            
            ...
            ========================================

        - Parameter: <cognify.hub.cogs.reasoning.LMReasoning>
        Applied Option: NoChange
        Transformation Details:
            NoChange

        - Parameter: <cognify.hub.cogs.model_selection.LMSelection>
        Applied Option: None_gpt-4o-mini
        Transformation Details:
            None_gpt-4o-mini

    >>> Module: interpreter_agent <<<

        - Parameter: <cognify.hub.cogs.fewshot.LMFewShot>
        Applied Option: interpreter_agent_demos_6acf03ae-763f-4357-bba2-0aea69b9f38d
        Transformation Details:
            - FewShot Examples -
            2 demos:
            Demonstration 1:
            **Input**
            {
                "problem": "A sequence of ten $0$s and/or $1$s is randomly generated. If the probability that the sequence does not contain two consecutive $1$s can be written in the form $\\dfrac{m}{n}$, where $m,n$ are relative..."
            }
            
            **Response**
            To solve this problem, we need to determine the number of sequences of length 10 consisting of 0s and 1s that do not contain two consecutive 1s. Let's define \( a_n \) as the number of such sequences ...
            ========================================
            Demonstration 2:
            **Input**
            {
                "problem": "Eight congruent equilateral triangles, each of a different color, are used to construct a regular octahedron. How many distinguishable ways are there to construct the octahedron? (Two colored octahedr..."
            }
            
            **Response**
            response: To solve this problem, we need to determine the number of distinguishable colorings of the octahedron using eight different colors, considering the symmetries of the octahedron.
            
            1. **Identi...
            ========================================

        - Parameter: <cognify.hub.cogs.reasoning.LMReasoning>
        Applied Option: ZeroShotCoT
        Transformation Details:
            
            - ZeroShotCoT -
            Return step-by-step reasoning for the given chat prompt messages.
            
            Reasoning Prompt: 
                Let's solve this problem step by step before giving the final response.

        - Parameter: <cognify.hub.cogs.model_selection.LMSelection>
        Applied Option: None_gpt-4o-mini
        Transformation Details:
            None_gpt-4o-mini

    ==================================================

With this configuration, all agents adopt ``gpt-4o-mini`` as the model, leading to significant cost savings. It also adds ``few-shot examples`` to both agents. The solver agent further benefits from ``Chain-of-Thought`` reasoning.

Overall, ``Optimization_3`` achieves a decent quality of ``6.367`` with a much low cost of ``$0.80`` per 1K invocations.

Evaluate and Use Optimized Workflow
===================================

You can evaluate the optimized workflow on the test data with:

.. code-block:: console

    $ cognify evaluate workflow.py --select Optimization_3

    ----- Testing select trial light_opt_layer_6 -----
      Params: {'solver_agent_few_shot': 'solver_agent_demos_c4d0a1fc-c664-40ec-a7c2-879ede9a241a', 'solver_agent_reasoning': 'NoChange', 'solver_agent_model_selection': 'None_gpt-4o-mini', 'interpreter_agent_few_shot': 'interpreter_agent_demos_6acf03ae-763f-4357-bba2-0aea69b9f38d', 'interpreter_agent_reasoning': 'ZeroShotCoT', 'interpreter_agent_model_selection': 'None_gpt-4o-mini'}

    =========== Evaluation Results ===========
      Quality improvement: 2%
      Cost is 0.11x original
      Quality: 6.31, Cost per 1K invocation: $0.80
    ===========================================

**To Use it in Your Code:**

.. code-block:: python
    
    import cognify

    problem = "A bored student walks down a hall that contains a row of closed lockers, numbered $1$ to $1024$. He opens the locker numbered 1, and then alternates between skipping and opening each locker thereafter. When he reaches the end of the hall, the student turns around and starts back. He opens the first closed locker he encounters, and then alternates between skipping and opening each closed locker thereafter. The student continues wandering back and forth in this manner until every locker is open. What is the number of the last locker he opens?\n"

    new_workflow = cognify.load_workflow(config_id='Optimization_3', opt_result_path='opt_results')
    answer = new_workflow(problem)

**Command-line Interface**

You can run a single request using the command line interface (CLI) as well:

.. code-block:: console

    $ cognify run workflow.py --select Optimization_3 --input "A bored student walks down a hall that contains a row of closed lockers..."
    Loading workflow with Optimization_3
    Running the following input: A bored student walks down a hall that contains a row of closed lockers...
    Output: The last locker he opens is 342.
