.. _cognify_fewshot:

Few-Shot Learning
=====================

Overview
--------

The **Few-Shot Learning Cog** (`LMFewShot`) enables embedding of few-shot examples to guide the behavior of language models. This Cog allows the optimizer to select the best demonstrations from an evaluation dataset, helping the model achieve better task performance by using the most relevant examples. 

This Cog is a type of :ref:`DynamicCog <dynamic_cog>`. As a **DynamicCog**, it evolves over time by continuously updating its options based on new evaluation results. This adaptability allows the Few-Shot Learning Cog to maintain an optimal set of examples that maximize the alignment to the given task.

.. tip::

   Like a Python closure capturing variables, the Few-Shot Learning Cog can retain high-quality demonstrations from more expensive configurations during optimization. This allows the final solution to benefit from the reasoning or generation quality of those higher-cost trials, without needing to adopt the full expense of those configurations in production.


.. rubric:: Key Attributes

- **max_num** (`int`): The maximum number of demonstrations (few-shot examples) to include in each option. 

.. caution::

   In current implementation, Cognify limits the search space by always trying to select demonstration sets of size ``max_num`` as the options available to the optimizer.

- **allow_duplicate** (`bool`): Determines whether multiple demonstrations of the same data point can appear in the option set. This can be useful for controlling the expoitation and exploration trade-off in few-shot examples.

.. admonition:: Follow-up

   By enabling ``allow_duplicate``, a few-shot option can contain different solution paths for the same task. This can be benefitial if tasks are alike but can be solved in different ways.

   Disabling ``allow_duplicate`` means that demonstrations selected in each option cover different inputs. This ensures diversity in the few-shot examples, which can be useful for generalization.


- **Options** (`DemoOption`): Each option represents a unique combination of few-shot demonstrations, dynamically updated based on the evaluation scores.

A Glance into Demonstrations
-----------------------------

In Cognify’s few-shot learning approach, a **Demonstration** contains a complete example that includes inputs, an expected output, and optional reasoning. These demonstrations are used within prompts to guide and steer the agent's behavior, enabling the model to learn from high-quality examples and apply similar reasoning in new situations.

.. rubric:: What Information Does a Demonstration Carry?

- **Input Variables** (`filled_input_variables`): A list of ``FilledInput`` each carries the input label (a descriptive name) and the value in string.
- **Output** (`output`): The expected response for the input, which the model should strive to capture the pattern or logic behind.
- **Reasoning** (`reasoning`, optional): An optional explanation of the reasoning process behind the output. This field provides insight into the steps or thought process used to arrive at the answer, which can enhance the model’s understanding and improve its problem-solving ability.
- **ID** (`id`): A unique identifier for the demonstration, which helps manage and reference specific examples.

Few-Shot Example Evolving Process
----------------------------------

The Few-Shot Learning Cog tracks evaluation scores for each data point. It updates the options whenever the top-K combination changes the composition, adding a new option that includes the new top-K demonstrations.

1. **Initialization**: The Cog can start with a user-defined set or an empty set of few-shot examples.
2. **Tracking by Data Points**: The `Few-Shot Learning Cog` monitors the highest score for each data point in the evaluation dataset, with top-scoring data points serving as potential examples.
3. **Updating Option Set**: In each optimization iteration, if any data point achieves a new high score and meets specified criteria (e.g., uniqueness if ``allow_duplicate`` is ``False``), a new option is created with the current top-K highest-scoring data points as demonstrations, guiding the model’s behavior.

DemoOption
----------

Each `DemoOption` encapsulates a set of few-shot demonstrations used to guide the model generalization. Key elements include:

- **demos** (`list[Demonstration]`): A list of selected demonstrations, each 
- **cost_indicator** (`float`): A simple heuristic using the number of demonstrations to estimate the cost of applying this option. Assumption is that each example's input/output length is similar to the orignal task.
- **apply** (`Callable`): A emthod that embeds corresponding few-shot examples to LLM calls.

Example Usage
-------------

Below we show how to create a simple Few-Shot Learning Cog with empty initial demonstration set:

.. code-block:: python

   from cognify.hub.cogs import LMFewShot

   few_shot_cog = LMFewShot(
      max_num=5, # each option will have at most 5 demonstrations
      name="simple_few_shot_cog",
      allow_duplicate=False # no demonstrations from the same input in each option
   )


Cognify also provides a programmable interface to manage and apply your few-shot examples:

.. code-block:: python

   import cognify
   from cognify import Demonstration, FilledInput

   # A list of demos for task: extract keywords from a question given some hints for retrieval
   demos = [
      Demonstration(
         filled_input_variables=[
            FilledInput(
               InputVar("QUESTION"), 
               value="What is the annual revenue of Acme Corp in the United States for 2022?"
            ),
            FilledInput(
               InputVar("HINT"), 
               value="Focus on financial reports and U.S. market performance for the fiscal year 2022."
            )
         ],
         output='["annual revenue", "Acme Corp", "United States", "2022", "financial reports", "U.S. market performance", "fiscal year"]'
      ),

      # Can also optionally provide your reasoning text
      Demonstration(
         filled_input_variables=[
            FilledInput(
               InputVar("QUESTION"), 
               value="In the Winter and Summer Olympics of 1988, which game has the most number of competitors? Find the difference of the number of competitors between the two games."
            ),
            FilledInput(
               InputVar("HINT"), 
               value="the most number of competitors refer to MAX(COUNT(person_id)); SUBTRACT(COUNT(person_id where games_name = '1988 Summer'), COUNT(person_id where games_name = '1988 Winter'));"
            )
         ],
         output='["Winter Olympics", "Summer Olympics", "1988", "1988 Summer", "Summer", "1988 Winter", "Winter", "number of competitors", "difference", "MAX(COUNT(person_id))", "games_name", "person_id"]',
         reasoning='To extract keyword in the question, we first understand the hints ...' 
      ),
   ]

   # To add demos to your agent directly
   agent = cognify.Model()
   agent.add_demos(demos=demos)

   # To create a few-shot cog with these predefined demos
   ur_expert = LMFewShot(
      max_num=5,
      name="few_shot_cog_with_help",
      user_demos=demos,
      # Uncomment following of you want the optimizer 
      # to only select from the provided examples
      # disable_evolve=True 
   )
