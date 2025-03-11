from workflow import interpreter_agent, solver_agent

from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    problem: str
    math_model: str
    answer: str

def interpreter_node(state: State) -> State:
    math_model = interpreter_agent.invoke({"problem": state["problem"]}).content
    return {"math_model": math_model}

def solver_node(state: State) -> State:
    answer = solver_agent.invoke({"problem": state["problem"], 
                                           "math_model": state["math_model"]}).content
    return {"answer": answer}

# initialize graph
builder = StateGraph(State)
builder.add_node("interpreter_node", interpreter_node)
builder.add_node("solver_node", solver_node)

builder.add_edge(START, "interpreter_node")
builder.add_edge("interpreter_node", "solver_node")
builder.add_edge("solver_node", END)

graph = builder.compile()

# invoke graph
import cognify

@cognify.register_workflow
def math_solver_workflow(workflow_input):
    return {"workflow_output": graph.invoke({"problem": workflow_input})}

if __name__ == "__main__":
    problem = "A bored student walks down a hall that contains a row of closed lockers, numbered $1$ to $1024$. He opens the locker numbered 1, and then alternates between skipping and opening each locker thereafter. When he reaches the end of the hall, the student turns around and starts back. He opens the first closed locker he encounters, and then alternates between skipping and opening each closed locker thereafter. The student continues wandering back and forth in this manner until every locker is open. What is the number of the last locker he opens?\n"
    answer = math_solver_workflow(problem)
    print(answer)