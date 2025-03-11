from typing import List
from pydantic import BaseModel, Field
from cognify.llm import Input, StructuredModel, OutputFormat
from .prompts import complexity_system


class ComplexityEstimation(BaseModel):
    """complexity of each agent"""

    score: int = Field(description="complexity score of the agent")
    rationale: str = Field(description="rationale for the complexity score")


class ComplexityList(BaseModel):
    """complexity of all agents"""

    es: List[ComplexityEstimation] = Field(
        description="list of complexity descriptions"
    )


def estimate_complexity_kernel(agents: list[str]) -> List[ComplexityEstimation]:
    agent_input = Input(name="agents")
    complexity_agent = StructuredModel(
        agent_name="complexity_agent",
        system_prompt=complexity_system,
        input_variables=[agent_input],
        output_format=OutputFormat(
            schema=ComplexityList, should_hint_format_in_prompt=True
        ),
    )
    agent_prompts_list = [f"Prompt {i+1}:\n {agent}" for i, agent in enumerate(agents)]
    agent_prompts = "\n".join(agent_prompts_list)
    messages = [
        {
            "role": "user",
            "content": f"Here are the agent prompts {agent_prompts}\n\nNow please give your Complexity Analysis.\nPlease follow the format instructions:\n",
        }
    ]
    complexity_list: ComplexityList = complexity_agent.forward(
        messages,
        model_kwargs={"model": "gpt-4o", "temperature": 0.0},
        inputs={agent_input: agent_prompts},
    )
    return complexity_list.es
