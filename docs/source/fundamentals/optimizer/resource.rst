Resource Management
===================

The Cognify optimizer offers various settings and strategies to control and optimize the resource allocation for each optimization layer. This section explains how to:

1. Manage the optimization budget
2. Apply resource-efficient strategies like Successive Halving (SH)
3. Leverage Cognify’s frugal optimization mode.

Simple Budget Limit Settings
----------------------------

Each layer in the optimization hierarchy can independently control the number of trials (evaluations) and the level of parallelism by configuring the ``OptConfig``:

- **n_trials**: Sets the maximum number of optimization iterations. This value determines the total budget for each **invocation** of the optimizer routine in that layer.
- **num_parallel_proposal**: Controls the level of parallelism by specifying the number of proposals to evaluate concurrently. Adjusting this level can speed up optimization but also impacts resource consumption.

Example:

.. code-block:: python

   from cognify.optimizer.core import driver, flow

   layer_opt_config = flow.OptConfig(
      n_trials=10,    # each invocation of this layer will run 10 trials
      num_parallel_proposal=2,   # 2 proposals running in parallel
   )

   # Apply this configuration in a layer
   layer_config = driver.LayerConfig(
      layer_name="example_layer",
      opt_config=layer_opt_config,
      ...
   )

.. rubric:: Estimating Total Optimization Cost

To estimate the overall resources required for a full multi-layer optimization, consider the product of `n_trials` values across layers. Also check :ref:`opt_process` for better understanding the optimization process.


Efficient Resource Allocation with Successive Halving
-----------------------------------------------------

The Successive Halving (SH) strategy in Cognify’s optimizer refines resource use by **gradually focusing on** top-performing configurations.

SH Breakdown
^^^^^^^^^^^^

When ``use_SH_allocation=True`` is set, the optimizer applies the following process:

1. **Initial Pool**: SH starts by proposing a set number of configurations (`num_parallel_proposal`).
2. **Additional Budget Per Round**: In each round, a fixed budget (`n_trials`) is added to each remaining configuration, allowing more resources for further evaluation.
3. **Prune the Pool**: After each round, the least promising configurations (by default the bottom half) are removed from the pool.
4. **Iterate Until Convergence**: This cycle continues, with each remaining configuration receiving the same budget increment, until only the entire pool is exhausted.

.. hint::
   This process helps to identify top-performing configurations without over-allocating resources to low-potential candidates, providing an efficient way to balance **exploration** and **exploitation**.

Example Configuration
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   opt_config_bottom = flow.OptConfig(
      n_trials=10,  # Total trials to try per invocation
   )

   opt_config_upper = flow.OptConfig(
      n_trials=6,             # Total trials to try per invocation
      num_parallel_proposal=2,           # Initial pool size
      use_SH_allocation=True  # Enable Successive Halving
   )

Estimate SH resource consumption
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this example, the upper layer will run ``(6 / 2) = 3`` SH passes. Within each pass, a pool size of 2 proposals will be initialized. 

By halfing the pool after each round, SH will converge in ``log2(2) + 1 = 2`` rounds.

   - **Round 1**: 2 proposals (each involving the bottom layer) are evaluated with a budget of 10 trials = **20 evaluations**
   - **Round 2**: 1 remaining proposal (from the pruned pool) is evaluated with an additional 10 trials += **10 evaluations**

   **Total Trials**: ``20 + 10 = 30`` trials are required for this two-layer optimization.

Frugal Optimization with Cost-Aware Search Strategy
---------------------------------------------------------

Cognify further optimizes resource use by introducing **cost-awareness** to the optimizer. Cognify normalizes the acquisition function by an **estimated evaluation cost**, allowing the optimizer to prioritize configurations that provide a high return on investment.

What is an Acquisition Function?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An acquisition function is a key component in `Bayesian Optimization` that guides the search for optimal configurations.

It evaluates candidate configurations to determine which ones are most promising to sample next. In simple terms, the acquisition function scores configurations based on their **expected improvement** relative to the current best-known performance, allowing the optimizer to focus on the most promising regions of the search space.

How Cognify’s Frugal Acquisition Works
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The frugal approach reduces evaluation cost by dynamically adjusting search priorities to focus on configurations that maximize both performance and resource efficiency, supporting a scalable, adaptable optimization process.

1. **Cost Estimation for the proposal**: For each candidate proposed, a cost estimator predicts the evaluation cost. This estimator considers factors like the number of active parameters and their :ref:`cost indicators <option_cost_indicator>`.
2. **Normalized Acquisition Function**: The acquisition function, which ranks configurations based on the expected improvement, is adjusted by dividing by the predicted cost. This normalization helps prioritize cost-effective configurations when having similar performance.
3. **Cost Decay**: To prevent the cost factor from overly dominating in the later stage, the cost effect is moderated with a decay function. Over multiple rounds, this decay reduces the cost influence, allowing a smoother transition to exploitation of high-potential configurations.


Example Configuration
^^^^^^^^^^^^^^^^^^^^^

Frugal optimization is enabled by default. To explicitly set this flag:

.. code-block:: python

   from cognify.optimizer.core import flow

   frugal_config = flow.OptConfig(
      n_trials=30,               
      frugal_eval_cost=True       # Activates cost-aware frugal optimization
   )
