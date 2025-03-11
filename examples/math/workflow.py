from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Initialize the model
import dotenv
dotenv.load_dotenv()
model = ChatOpenAI(model="gpt-4o", temperature=0, max_tokens=300)

interpreter_prompt = """
You are a math problem interpreter. Your task is to analyze the problem, identify key variables, and formulate the appropriate mathematical model or equation needed to solve it. Be concise and clear in your response.
"""

interpreter_template = ChatPromptTemplate.from_messages(
    [
        ("system", interpreter_prompt),
        ("human", "problem:\n{problem}\n"),
    ]
)

interpreter_agent = interpreter_template | model

solver_prompt = """
You are a math solver. Given a math problem, and a mathematical model for solving it, your task is to compute the solution and return the final answer. Be concise and clear in your response.
"""

solver_template = ChatPromptTemplate.from_messages(
    [
        ("system", solver_prompt),
        ("human", "problem:\n{problem}\n\nmath model:\n{math_model}\n"),
    ]
)

solver_agent = solver_template | model

import cognify

# Define Workflow
@cognify.register_workflow
def math_solver_workflow(workflow_input):
    math_model = interpreter_agent.invoke({"problem": workflow_input}).content
    answer = solver_agent.invoke({"problem": workflow_input, "math_model": math_model}).content
    return {"workflow_output": answer}


if __name__ == "__main__":
    problem = "A bored student walks down a hall that contains a row of closed lockers, numbered $1$ to $1024$. He opens the locker numbered 1, and then alternates between skipping and opening each locker thereafter. When he reaches the end of the hall, the student turns around and starts back. He opens the first closed locker he encounters, and then alternates between skipping and opening each closed locker thereafter. The student continues wandering back and forth in this manner until every locker is open. What is the number of the last locker he opens?\n"
    answer = math_solver_workflow(problem)
    print(answer)
    