.. _cognify_tutorials_evaluator:

******************
Workflow Evaluator
******************

Cognify evaluates your workflow throughout its optimization iterations. To tell Cognify how you want it to be evaluated, you should define an evaluator for your workflow that returns a score (a positive numerical value, higher being better generation quality) for a workflow's output. This can usually be done by comparing the output to the ground truth provided in the training dataset.

Cognify provides a few `sample evaluators <https://github.com/GenseeAI/cognify/tree/main/cognify/optimizer/evaluation>`_ to start with: F1 score, LLM-as-a-judge, exact match, and code execution.

The evaluator function signature and its implementation are both customizable. A common type of signature includes workflow input, workflow output generation, and ground truth as the function parameters as follows. But you can also define an evaluation function with other or fewer parameters, e.g., an evaluator that only needs the generation output and ground truth to measure the score. To register a function as your evaluator, simply add :code:`@cognify.register_evaluator` before it.

.. code-block:: python

   @cognify.register_evaluator
   def evaluate(workflow_input, workflow_output, ground_truth):
      # your evaluation logic here
      return score

For the math-solver example, we will use LLM-as-a-judge to be the evaluator. We have provided the evaluator implementation with both sending messages directly using the OpenAI API as well as using LangChain.

.. tab-set::

   .. tab-item:: OpenAI

      .. include:: evaluator_code/_openai_eval.rst

   .. tab-item:: LangChain

      .. include:: evaluator_code/_langchain_eval.rst

The evaluator agent uses `gpt-4o-mini` as the backbone model. It also returns a structured output, ``Assessment``, to enforce the output format since we require the evaluator to return a numerical value.

Recommendations
---------------

Depending on your task, it may be difficult to find or write a suitable evaluator. Here are some tips to help you get started:

* `LLM-as-a-judge`: among the `sample evaluators <https://github.com/GenseeAI/cognify/tree/main/cognify/optimizer/evaluation>`_, we provide a base implementation from which you can build upon. 
 
  * By default, our implementation uses GPT-4o as the judge model. To customize the judge, you can use our provided factory function like so:

   .. code-block:: python

      from cognify.hub.evaluators.llm_judge import llm_judge_factory
      gpt_4o_mini_judge = llm_judge_factory(model="gpt-4o-mini")

      @cognify.register_evaluator
      def evaluate(llm_output, ground_truth):
         # your evaluation logic here
         return gpt_4o_mini_judge(llm_output, ground_truth)
  
  * We **highly recommend** tailoring the criteria to your task. For example, if you are looking for conciseness, the system prompt should instruct the judge to rate the answer based on its length. 

  * We also recommend you provide some **few-shot examples** to the model with human evaluation at different quality levels.

  * For more information on how best to use LLM-as-a-judge, you can refer to this `survey paper <https://llm-as-a-judge.github.io/>`_.
* `Majority vote`: if you are unsure of the quality of an evaluator's output, you can use a majority vote from multiple evaluators. This can be done by averaging the scores from multiple evaluators or using a custom weighting scheme.
* `Training your own model`: if you have sufficient labeled examples in the format of ``(generated output, human evaluation)`` pairs, you can train a model of your choice as the evaluator. 