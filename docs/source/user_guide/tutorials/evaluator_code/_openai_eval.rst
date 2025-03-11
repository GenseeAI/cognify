
.. code-block:: python

    import cognify
    from openai import OpenAI
    from pydantic import BaseModel

    # Initialize the model
    import dotenv
    dotenv.load_dotenv()

    class Assessment(BaseModel):
    score: int

    @cognify.register_evaluator
    def llm_judge(workflow_input, workflow_output, ground_truth):
        evaluator_prompt = """
        You are a math problem evaluator. Your task is to grade the the answer to a math proble by assessing its correctness and completeness.

        You should not solve the problem by yourself, a standard solution will be provided. 

        Please rate the answer with a score between 0 and 10.
        """

        # based on https://platform.openai.com/docs/guides/structured-outputs
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": evaluator_prompt},
                {"role": "human", "content": f"problem:\n{workflow_input}\n\nstandard solution:\n{ground_truth}\n\nanswer:\n{workflow_output}\n"},
            ],
            response_format=Assessment
        )
        assess = completion.choices[0].message.parsed
        return assess.score