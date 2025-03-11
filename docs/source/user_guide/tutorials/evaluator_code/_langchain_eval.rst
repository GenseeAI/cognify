
.. code-block:: python

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
            ("human", "problem:\n{problem}\n\nstandard solution:\n{solution}\n\nanswer:\n{answer}\n\nYour response format:\n{format_instructions}\n"),
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