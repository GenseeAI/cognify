from pydantic import BaseModel, Field
from typing import Dict, List
from cognify.llm import Model, StructuredModel, OutputFormat
from .prompts import decompose_refine_system, mid_level_system_format_instructions

# ================== Refine New Agent Workflow ==================


class AgentMeta(BaseModel):
    """Information about each agent"""

    inputs: List[str] = Field(description="list of inputs for the agent")
    output: str = Field(description="agent output variable name")
    prompt: str = Field(description="refined prompt for the agent")
    next_action: List[str] = Field(description="all possible next agents to invoke")
    dynamic_action_decision: str = Field(
        "python code for dynamically deciding the next action, put 'None' if not needed"
    )


class NewAgentSystem(BaseModel):
    """New agent system"""

    agents: Dict[str, AgentMeta] = Field(
        description="dictionary of agent name to information about that agent"
    )


def decompose_refine_kernel(
    new_agent_meta: dict[str, dict], high_level_info: str
) -> NewAgentSystem:
    refine_new_agents = Model("refine", decompose_refine_system, input_variables=[])
    messages = [
        {
            "role": "user",
            "content": f"""
      Now, this is the real user question for you:

      ## Information of the existing agent system
      {high_level_info}

      ## Metadata of suggested new agents
      {new_agent_meta}

      Your answer:""",
        }
    ]
    model_kwargs = {"model": "gpt-4o", "temperature": 0.0}
    new_interaction: str = refine_new_agents(messages, model_kwargs, inputs={})

    # format
    messages.extend(
        [
            {"role": "assistant", "content": f"{new_interaction}"},
            {
                "role": "user",
                "content": "Now please reformat your answer in the desired format.\n",
            },
        ]
    )
    model_kwargs["model"] = "gpt-4o-mini"
    reformatter = StructuredModel(
        "refine_struct",
        decompose_refine_system,
        input_variables=[],
        output_format=OutputFormat(
            schema=NewAgentSystem,
            custom_output_format_instructions=mid_level_system_format_instructions,
        ),
    )
    new_system: NewAgentSystem = reformatter(messages, model_kwargs, inputs={})
    return new_system
