.. _cognify_quickstart:

******************
Cognify Quickstart
******************

This section demonstrates the basic way to use Cognify using a simple example.

Run Cognify in Four Simple Steps
================================

1. Connect to Your Workflow
---------------------------

The first step of using Cognify is to connect it to your existing gen-AI workflow. 
We currently support unmodified programs written in LangChain and DSPy. 
You can also develop gen-AI workflows on our Python-based interface or modify your existing Python programs to this interface.
Read more about our supported programming interfaces :ref:`here <cognify_tutorials_interface>`.


2. Build the Evaluation Pipeline
--------------------------------

The next step is to create an evaluation pipeline. This involves providing a training dataset and an evaluator of your workflow.

- **Training Data**: Cognify relies on user supplied training data for its optimization process. Thus, you need to provide a data loader function that returns a sequence of input/output pairs served as the ground truth. Read more about training datasets and the data loader :ref:`here <cognify_tutorials_data_loader>`.

- **Workflow Evaluator**: We expect users (developers of workflows) to understand how to evaluate their workflows. Thus, you need to provide an evaluator function for determining the generation quality. We provide several common evaluators such as F1 and LLM-as-a-judge that you could use to start with. Read more about the evaluator :ref:`here <cognify_tutorials_evaluator>`.

3. Configure the Optimizer Behavior
-----------------------------------

The third step is to configure the optimization process. This step is optional. If not provided, Cognify will use default values to configure your optimization.
However, we highly encourage you to configure your optimization to achieve better results. You can configure your optimization in the following ways:

- **Select Model Set**: Define the set of models you want Cognify to try on your workflows. You are responsible for setting up your model API keys whenever they are needed.

- **Config Optimization Settings**: Establish the overall optimization strategy by defining the maximum number of search iterations, quality constraint, or cost constraint. These settings allow you to choose whether to prioritize quality improvement, cost reduction, or minimize Cognify's optimization time.

Read more about optimizer configuration :ref:`here <config_search>`.

4. Run the Cognify Command-Line Interface (CLI)
----------------------

The final step is to run Cognify, as simple as ``cognify optimize your-workflow-file``.
Read more about Cognify's full CLI :ref:`here <cognify_tutorials_cli>`.

A Minimal Example
=================

Now let's walk through the use of Cognify with a "hello-world" example: making a single call to an LLM to answer a user question.

Step 0: Provide the workflow to optimize
--------------------------------------------

To get started, let's first take a look at the original workflow that we will optimize. 
The code of this example is available at `examples/quickstart <https://github.com/WukLab/Cognify/tree/main/examples/quickstart>`_. This tutorial will explain each step in detail.

.. code-block:: python

   # examples/quickstart/workflow.py

   # ----------------------------------------------------------------------------
   # Define a single LLM agent to answer user question with provided documents
   # ----------------------------------------------------------------------------

   import dotenv
   from langchain_openai import ChatOpenAI
   # Load the environment variables
   dotenv.load_dotenv()
   # Initialize the model
   model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

   # Define system prompt
   system_prompt = """
   You are an expert at answering questions based on provided documents. Your task is to provide the answer along with all supporting facts in given documents.
   """

   # Define agent routine 
   from langchain_core.prompts import ChatPromptTemplate
   agent_prompt = ChatPromptTemplate.from_messages(
      [
         ("system", system_prompt),
         ("human", "User question: {question} \n\nDocuments: {documents}"),
      ]
   )

   qa_agent = agent_prompt | model

   # Define workflow
   def doc_str(docs):
      context = []
      for i, c in enumerate(docs):
         context.append(f"[{i+1}]: {c}")
      return "\n".join(docs)

   def qa_workflow(question, documents):
      format_doc = doc_str(documents)
      answer = qa_agent.invoke({"question": question, "documents": format_doc}).content
      return {'answer': answer}

This ``qa_workflow`` constructs a prompt with a ``system_prompt`` part, the user question, and the user supplied set of documents. It makes a single call to the GPT-4o-mini model and generates the answer to the user question along with the supporting facts.

As this example uses an OpenAI mode, you need to add your OpenAI API, e.g., by creating a ``.env`` file in the same directory with the following content:

::

   OPENAI_API_KEY=your_openai_api_key

You can try running this workflow by:

.. code-block:: python

   question = "What was the 2010 population of the birthplace of Gerard Piel?"
   documents = [
      'Gerard Piel | Gerard Piel (1 March 1915 in Woodmere, N.Y. – 5 September 2004) was the publisher of the new Scientific American magazine starting in 1948. He wrote for magazines, including "The Nation", and published books on science for the general public. In 1990, Piel was presented with the "In Praise of Reason" award by the Committee for Skeptical Inquiry (CSICOP).',
   ]

   result = qa_workflow(question=question, documents=documents)
   print(result)

A sample output looks like:

::
   
   {'answer': 'The birthplace of Gerard Piel is Woodmere, New York. However, the provided document does not include the 2010 population of Woodmere. To find that information, one would typically refer to census data or demographic reports from that year.'}

.. hint::

   This workflow is a minimal one, leaving little room for Cognify to optimize. In general, workflows are more complex and benefit more from Cognify's optimization.

Step 1: Register the workflow
-------------------------------

For LangChain programs like the above, you do not need to modify your code. But to tell Cognify how to invoke the workflow, you need to add a register annotation ``@cognify.register_workflow``.
In this example, the entry point of the workflow is the ``qa_workflow`` function. So we will add ``@cognify.register_workflow`` above the function definition as follow:

.. code-block:: python

   import cognify

   @cognify.register_workflow
   def qa_workflow(question, documents):

Step 2: Build the Evaluation Pipeline
-------------------------------------

Next, we will define the evaluator and training data used for the Cognify optimization. Both are defined in a single file, for this example, the ``config.py`` file under the same directory as ``workflow.py``.

2.1 Register evaluator
^^^^^^^^^^^^^^^^^^^^^^

Cognify evaluates your workflow throughout its optimization. To tell Cognify how you want it to be evaluated, you need to define the evaluator. Cognify provides several common evaluator implementation. If you choose one of them, you can simply import them from ``cognify.hub.evaluators``. In this example, we use the ``F1`` score to quantify the similarity between the predicted answer and the ground truth. Cognify already provides an implementation of this metric. So the evaluator looks like this:

.. code-block:: python

   import cognify
   from cognify.hub.evaluators import f1_score_str

   @cognify.register_evaluator
   def evaluate_answer(answer, label):
      return f1_score_str(answer, label)

Read more about the evaluator :ref:`here <cognify_tutorials_evaluator>`.

2.2 Register data loader
^^^^^^^^^^^^^^^^^^^^^^^^

The Cognify optimization process utilizes user-provided training datasets which include pairs of input and ground-truth output. Cognify expects users to define a data loader that provide the input-output pairs, with both the input and the output being a dictionary.
In this example, we use a sample dataset from from the `HotPotQA <https://hotpotqa.github.io>`_ dataset in :file:`data._json`. The data loader reads the file and returns the pairs in the form of training, validation, and test datasets like so:

.. code-block:: python

   import cognify
   import json

   @cognify.register_data_loader
   def load_data():
      with open("data._json", "r") as f:
         data = json.load(f)
            
      # format to (input, output) pairs
      new_data = []
      for d in data:
         input = {
               'question': d["question"], 
               'documents': d["docs"]
         }
         output = {
               'label': d["answer"],
         }
         new_data.append((input, output))
      
      # split to train, val, test
      return new_data[:5], None, new_data[5:]

.. hint::

   The dataset is small for a quick demonstration. In practice, you should provide a larger dataset for better generalization.

Read more about training datasets and the data loader :ref:`here <cognify_tutorials_data_loader>`.

Step 3: Configure the Optimizer Behavior
----------------------------------------

Cognify's optimization behavior can be configured by users, such as whether to perform light-weight, medium-weight, or heavy-weight optimizations, the maximum iterations of trials to perform, what models Cognify can choose from, etc. 
These configurations are defined in the :code:`create_search` construct in a configuration file.
By default, Cognify assumes that this file is ``config.py`` in the same workflow directory. You can also use another file name by specifying :code:`-c /path/to/custom_config.py` in the command line.

The simpliest way to set the configurations is the use Cognify's default as follows:

.. code-block:: python

   from cognify.hub.search import default

   search_settings = default.create_search()

.. hint::

   To achieve better optimization results that meet your requirements, you should customize your optimization configuration instead of using the default.

Cognify provides a set of `pre-defined configurations <https://github.com/WukLab/Cognify/blob/main/cognify/hub/search/default.py>`_ for you to start with.
Read more about optimizer configuration :ref:`here <config_search>`.

Wrap Up
-------

Now we have all the components in place. The final directory structure should look like this:

::

   .
   ├── config.py # evaluator + data loader + search settings
   ├── data._json
   ├── workflow.py
   └── .env


Run Cognify Optimization
------------------------

To run Cognify, simply use ``cognify optimize your-workflow-file``.

.. code-block:: bash
   
   cd examples/quickstart
   cognify optimize workflow.py

An example output looks like this:

.. code-block:: bash

   (my_env) user@hostname:/path/to/quickstart$ cognify optimize workflow.py 
   > light_opt_layer | (best score: 0.16, lowest cost@1000: 0.09 $): 100%|███████████████| 10/10 [01:53<00:00, 11.30s/it]
   ================ Optimization Results =================
   Number of Optimization Results: 2
   --------------------------------------------------------
   Optimization_1
   Quality improvement: 5%
   Cost is 0.95x original
   Quality: 0.160, Cost per 1K invocation: $0.28
   --------------------------------------------------------
   Optimization_2
   Quality improvement: 7%
   Cost is 0.98x original
   Quality: 0.154, Cost per 1K invocation: $0.09
   ========================================================

The optimizer found two optimized workflow versions on the Pareto frontier, i.e., they are the most cost-effective solutions within all searched optimizations.

.. note::

   It's not guaranteed that the optimizer will find any better solutions than the original one. You might get ``Number of Optimization Results: 0`` in the output.

Check detailed optimizations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can find all the output workflows' information under the ``opt_results/optimized_workflow_details`` directory (the default directory used by Cognify, which you can change in ``config.py``). 

Beflow is the transformations that ``Optimization_1`` applies to the original workflow: the Chain-of-Thought prompting is applied to the model call, while no few-shot demonstration is added.

::

   Trial - light_opt_layer_4
   Log at: opt_results/light_opt_layer/opt_logs.json
   Quality: 0.160, Cost per 1K invocation ($): 0.28 $
   ********** Detailed Optimization Trace **********

   ========== Layer: light_opt_layer ==========

   >>> Module: qa_agent <<<

      - Parameter: <cognify.hub.cogs.fewshot.LMFewShot>
         Applied Option: NoChange
         Transformation Details:
         NoChange

      - Parameter: <cognify.hub.cogs.reasoning.LMReasoning>
         Applied Option: ZeroShotCoT
         Transformation Details:
         
         - ZeroShotCoT -
         Return step-by-step reasoning for the given chat prompt messages.
         
         Reasoning Prompt: 
               Let's solve this problem step by step before giving the final response.

   ==================================================


Evaluate a Specific Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To see how well an optimized workflow peforms, you can load it into your code and run it on a sample input like so:

.. code-block:: python

   question = "What was the 2010 population of the birthplace of Gerard Piel?"
   documents = [
      'Gerard Piel | Gerard Piel (1 March 1915 in Woodmere, N.Y. – 5 September 2004) was the publisher of the new Scientific American magazine starting in 1948. He wrote for magazines, including "The Nation", and published books on science for the general public. In 1990, Piel was presented with the "In Praise of Reason" award by the Committee for Skeptical Inquiry (CSICOP).',
   ]

   # load optimized workflow
   optimized_workflow = cognify.load_workflow(
      config_id='Optimization_1',
      opt_result_path='opt_results'
   )
   result = optimized_workflow(question=question, documents=documents)
   print(result)

You can also evaluate an optimized workflow on your entire test dataset. 
When you define your dataloader, you should split the data into train, validation, and test sets. The following command will run the optimized workflow on your test data.

.. code-block:: bash

   cognify evaluate workflow.py --select Optimization_1

The sample output looks like:

.. code-block:: bash

   (my_env) user@hostname:/path/to/quickstart$ cognify evaluate workflow.py --select Optimization_1
   ----- Testing select trial light_opt_layer_4 -----
   Params: {'qa_agent_few_shot': 'NoChange', 'qa_agent_reasoning': 'ZeroShotCoT'}
   Training Quality: 0.160, Cost per 1K invocation: $0.28

   > Evaluation in light_opt_layer_4 | (avg score: 0.20, avg cost@1000: 0.28 $): 100%|███████10/10 [00:07<00:00,  1.42it/s]
   =========== Evaluation Results ===========
   Quality improvement: 25%
   Cost is 0.95x original
   Quality: 0.199, Cost per 1K invocation: $0.28
   ===========================================

You can also use Cognify to evaluate the original workflow with:

.. code-block:: bash

   cognify evaluate workflow.py --select Original
