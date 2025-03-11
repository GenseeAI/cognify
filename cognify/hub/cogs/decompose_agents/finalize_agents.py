from pydantic import BaseModel, Field
from typing import List, Dict, Union
from cognify.llm import Model, StructuredModel, OutputFormat
from cognify.hub.cogs.decompose_agents import NewAgentSystem
from .prompts import finalize_new_agents_system


# ================== Finalize New Agents ==================
class AgentSemantic(BaseModel):
    """Information about each agent"""

    agent_prompt: str = Field(description="prompt for the agent")
    inputs_variables: List[str] = Field(
        description="list of input variables for the agent"
    )
    output_json_schema: Union[Dict, str] = Field(
        description="json output schema for the agent, or the output variable name for simple text output"
    )
    next_action: List[str] = Field(description="all possible next agents to invoke")
    dynamic_action_decision: str = Field(
        "python code for dynamically deciding the next action, put 'None' if not needed"
    )


class StructuredAgentSystem(BaseModel):
    """Refined agent system with structured output schema"""

    agents: Dict[str, AgentSemantic] = Field(
        description="dictionary of agent name to information about that agent"
    )

    final_output_aggregator_code: str = Field(
        description="python code to combine the outputs of the new agents to generate the final output, put 'None' if not needed"
    )


def finalize_new_agents_kernel(old_info: str, mid_level_desc: NewAgentSystem):
    finalize_new_agents = Model(
        "finalize", finalize_new_agents_system, input_variables=[]
    )
    messages = [
        {
            "role": "user",
            "content": f"""
Now, this is the real task for you.

## Information of the old single-agent system
{old_info}

## Information of the suggested multi-agent system
{mid_level_desc.model_dump(indent=4)}

## Your answer:
""",
        }
    ]
    model_kwargs = {"model": "gpt-4o", "temperature": 0.0}
    new_interaction: str = finalize_new_agents(messages, model_kwargs, inputs={})

    reformatter = StructuredModel(
        "finalize_struct",
        finalize_new_agents_system,
        input_variables=[],
        output_format=OutputFormat(
            schema=StructuredAgentSystem, should_hint_format_in_prompt=True
        ),
    )
    messages.extend(
        [
            {"role": "assistant", "content": f"{new_interaction}"},
            {
                "role": "user",
                "content": "Now please reformat the new agent system to the desired JSON format\n",
            },
        ]
    )
    output = reformatter(messages, model_kwargs, inputs={})
    return output
