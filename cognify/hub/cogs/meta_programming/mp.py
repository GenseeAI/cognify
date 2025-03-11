question_suffix_or_path: str = "\n\nLet's first come up with a list of experts you may want to consult for this problem and then immediately start solving it."  # "" # "\n\nLet's think step by step."
intermediate_feedback = "Based on the information given, what are the most logical next steps or conclusions? Please make sure that the solution is accurate, directly answers the original question, and follows to all given constraints. Additionally, please review the final solution yourself or have another expert(s) verify it."

expert_python_message: str = 'You are an expert in Python and can generate Python code. To execute the code and display its output in the terminal using print statements, please make sure to include "Please run this code!" after the code block (i.e., after the closing code blocks)'

import os
import json

script_dir = os.path.dirname(__file__)

meta_config_path = os.path.join(
    script_dir, "prompts", "meta-v0-2023-08-14-baseline.json"
)
with open(meta_config_path, "r") as f:
    meta_prompt_config_dict = json.load(f)
meta_model_message_list = meta_prompt_config_dict["meta-model"]["message-list"]

question_prefix_path = os.path.join(
    script_dir, "prompts", "meta-prompting-instruction.txt"
)
with open(question_prefix_path, "r") as f:
    question_prefix = f.read()
question_prefix_or_path = question_prefix

meta_model_settings = meta_prompt_config_dict["meta-model"]
generator_settings = meta_prompt_config_dict["generator"]
verifier_settings = meta_prompt_config_dict["verifier"]
summarizer_settings = meta_prompt_config_dict["summarizer"]

# Get the error message and final answer indicator
error_message = meta_prompt_config_dict["meta-model"]["error-message"]
final_answer_indicator = meta_prompt_config_dict["meta-model"]["final-answer-indicator"]


from cognify.hub.cogs.reasoning import ReasonThenFormat
from .helper import MetaPromptingScaffolding
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    HumanMessage,
    BaseMessage,
)

meta_model = MetaPromptingScaffolding(
    generator_settings=generator_settings,
    verifier_settings=verifier_settings,
    summarizer_settings=summarizer_settings,
    error_message=error_message,
    final_answer_indicator=final_answer_indicator,
    expert_python_message=expert_python_message,
    intermediate_feedback=intermediate_feedback,
    fresh_eyes=True,
    include_expert_name_in_instruction=True,
    extract_output=False,
    use_zero_shot_cot_in_expert_messages=False,
)


class MetaPrompting(ReasonThenFormat):
    """
    Implementation adopted from https://github.com/suzgunmirac/meta-prompting
    """

    def __init__(self):
        super().__init__("MetaPrompting")

    def describe(self):
        desc = """
        - Meta Prompting -
        Dynamically spwan expert persona to tackle sub-problems. The original agent orchestrates the information flow and execution of the expert personas. The history of orchestrations is kept as rationale.
        """
        return desc

    def _get_cost_indicator(self):
        return 8.0

    def reasoning_step(
        self,
        chat_messages: list[BaseMessage],
        lm: ChatOpenAI,
    ) -> list[BaseMessage]:
        chat_messages.insert(
            max(len(chat_messages) - 1, 0), HumanMessage(question_prefix_or_path)
        )

        h = HumanMessage(question_suffix_or_path)
        chat_messages.append(h)

        reasoning_steps = [h]
        full_reasoning_history = meta_model.meta_model_generate(
            lm, reasoning_steps, chat_messages
        )
        return reasoning_steps
