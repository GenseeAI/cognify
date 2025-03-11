.. _cognify_tutorials_all_in_one:

************************
Putting It All Together
************************

Now that we have gone through all the steps of using Cognify, let's put everything together for the math-solver example.

.. note::

    Code for this tutorial is also available at `<https://github.com/GenseeAI/cognify/tree/main/examples/math>`_.

Workflow Definition
===================

Defined in ``workflow.py``:

.. tab-set::

    .. tab-item:: LangChain

        .. include:: interface/_langchain_front.rst

    .. tab-item:: DSPy

        .. include:: interface/_dspy_front.rst

    .. tab-item:: Cognify

        .. include:: interface/_cognify_front.rst

Optimization Configuration
===========================

Evaluator, data loader, and optimization configurations as defined in ``config.py``:

.. code-block:: python

    #================================================================
    # Evaluator
    #================================================================

    import cognify

    from pydantic import BaseModel
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate

    # Initialize the model
    import dotenv
    dotenv.load_dotenv()
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    from langchain.output_parsers import PydanticOutputParser
    class Assessment(BaseModel):
        score: int
        
    parser = PydanticOutputParser(pydantic_object=Assessment)

    @cognify.register_evaluator
    def llm_judge(workflow_input, workflow_output, ground_truth):
        evaluator_prompt = """
    You are a math problem evaluator. Your task is to grade the the answer to a math proble by assessing its correctness and completeness.

    You should not solve the problem by yourself, a standard solution will be provided. 

    Please rate the answer with a score between 0 and 10.
        """
        evaluator_template = ChatPromptTemplate.from_messages(
            [
                ("system", evaluator_prompt),
                ("human", "problem:\n{problem}\n\nstandard solution:\n{solution}\n\nanswer:\n{answer}\n\nYou response format:\n{format_instructions}\n"),
            ]
        )
        evaluator_agent = evaluator_template | model | parser
        assess = evaluator_agent.invoke(
            {
                "problem": workflow_input, 
                "answer": workflow_output, 
                "solution": ground_truth, 
                "format_instructions": parser.get_format_instructions()
            }
        )
        return assess.score


    #================================================================
    # Data Loader
    #================================================================

    import json
    import random

    @cognify.register_data_loader
    def load_data():
        with open("data._json", "r") as f:
            data = json.load(f)
            
        random.seed(42)
        random.shuffle(data) 
        # format to (input, output) pairs
        new_data = []
        for d in data:
            input_sample = {
                'workflow_input': d["problem"],
            }
            ground_truth = {
                'ground_truth': d["solution"],
            }
            new_data.append((input_sample, ground_truth))
        return new_data[:30], None, new_data[30:]

    #================================================================
    # Optimizer Set Up
    #================================================================

    from cognify.hub.search import default

    model_configs = [
        # OpenAI models
        cognify.LMConfig(model='gpt-4o-mini', kwargs={'temperature': 0, 'max_tokens': 300}),
        cognify.LMConfig(model='gpt-4o', kwargs={'temperature': 0, 'max_tokens': 300}),
    ]

    search_settings = default.create_search(
        model_selection_cog=model_configs,
        opt_log_dir='with_ms_opt_log',
    )

Run Cognify 
=======================

.. rubric:: To evaluate the original workflow:

.. code-block:: console

    $ cognify evaluate workflow.py --select Original

    ----- Testing Raw Program -----
    =========== Evaluation Results ===========
    Quality: 6.186, Cost per 1K invocation ($): 7.47 $
    ===========================================

.. rubric:: To run Cognify's optimization:

.. code-block:: console

    $ cognify optimize workflow.py

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

.. rubric:: To check the detailed transformations:

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

Evaluate and Use Optimized Workflow
===================================

.. rubric:: To evaluate the optimized workflow on the test set:

.. code-block:: console

    $ cognify evaluate workflow.py --select Optimization_3

    ----- Testing select trial light_opt_layer_6 -----
      Params: {'solver_agent_few_shot': 'solver_agent_demos_c4d0a1fc-c664-40ec-a7c2-879ede9a241a', 'solver_agent_reasoning': 'NoChange', 'solver_agent_model_selection': 'None_gpt-4o-mini', 'interpreter_agent_few_shot': 'interpreter_agent_demos_6acf03ae-763f-4357-bba2-0aea69b9f38d', 'interpreter_agent_reasoning': 'ZeroShotCoT', 'interpreter_agent_model_selection': 'None_gpt-4o-mini'}

    =========== Evaluation Results ===========
      Quality improvement: 2%
      Cost is 0.11x original
      Quality: 6.31, Cost per 1K invocation: $0.80
    ===========================================

.. rubric:: To integrate the optimized workflow into your application:

.. code-block:: python
    
    import cognify

    problem = "A bored student walks down a hall that contains a row of closed lockers, numbered $1$ to $1024$. He opens the locker numbered 1, and then alternates between skipping and opening each locker thereafter. When he reaches the end of the hall, the student turns around and starts back. He opens the first closed locker he encounters, and then alternates between skipping and opening each closed locker thereafter. The student continues wandering back and forth in this manner until every locker is open. What is the number of the last locker he opens?\n"

    new_workflow = cognify.load_workflow(config_id='Optimization_3', opt_result_path='opt_results')
    answer = new_workflow(problem)
