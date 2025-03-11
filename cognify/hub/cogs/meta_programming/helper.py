# Import the necessary libraries
import random
import re
import time

# Import the typing library for type hints
from typing import Any, Dict, List

# Import the necessary classes from the utils folder
from .execute_code import execute_code_with_timeout
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    BaseMessage,
)

import logging

logger = logging.getLogger(__name__)

# TODO: refactor meta-programming to new interface
# Adapted from: https://github.com/suzgunmirac/meta-prompting


# Define the MetaPromptingScaffolding class
class MetaPromptingScaffolding:
    def __init__(
        self,
        generator_settings: Dict[str, Any],
        verifier_settings: Dict[str, Any],
        summarizer_settings: Dict[str, Any],
        error_message: str,
        final_answer_indicator: str,
        expert_python_message: str,
        intermediate_feedback: str,
        fresh_eyes: bool = False,
        include_expert_name_in_instruction: bool = False,
        extract_output: bool = False,
        use_zero_shot_cot_in_expert_messages: bool = False,
    ) -> None:
        # Set the generator and verifier parameters + summarizer parameters (optional)
        self.generator_settings = generator_settings
        self.verifier_settings = verifier_settings
        self.summarizer_settings = summarizer_settings

        # Set the error message and final answer indicator
        self.error_message = error_message
        self.final_answer_indicator = final_answer_indicator

        # Set the fresh_eyes flag
        self.fresh_eyes = fresh_eyes

        # Other helper variables and constants for the model
        self.triple_quotes = '"""'

        # Set the include_expert_name_in_instruction flag
        self.include_expert_name_in_instruction = include_expert_name_in_instruction
        self.extract_output = extract_output
        self.expert_python_message = expert_python_message
        self.intermediate_feedback = intermediate_feedback
        self.use_zero_shot_cot_in_expert_messages = use_zero_shot_cot_in_expert_messages

    def meta_model_generate(
        self,
        lm: ChatOpenAI,
        reasoning_steps: List[BaseMessage],
        chat_prompt: list[BaseMessage],
        counter=0,
        last_answer: str = None,
        original_question: str = None,
        trial_num: int = 0,
        **kwargs: Any,
    ):
        routine = lm
        try:
            # This step is defined to ensure that the meta model returns a response in less than 16 rounds.
            # Note: Please feel free to change the number of rounds as you see fit.
            if counter == 8:
                return chat_prompt

            # TODO(msuzgun)[improvement]: In the future, if the total content is to long, we can use the summarizer to summarize the whole content.

            while True:
                round_info = f"Round {counter+1}: "
                if counter == 6:
                    round_info += (
                        "This is the last round; so, please present your final answer."
                    )

                chat_prompt.append(HumanMessage(round_info))

                # Step 1: Generate an output from the meta model
                meta_model_output = routine.invoke(chat_prompt).content
                chat_prompt.append(AIMessage(meta_model_output))
                reasoning_steps.append(AIMessage(meta_model_output))
                logger.debug(f"Meta model output: {meta_model_output}")

                # Check if the meta_model_output contains a text of the form "Expert XYZ:\n" (where XYZ is an alphabanumeric string).

                # Step 2 (a): If we are not in the 0-shot CoT setting, check if the meta model output contains any text between triple quotes.
                # If it does, then generate an output from the corresponding model.
                pattern = r"Expert ((?:\w+ ?){1,5}):\n"
                if (self.fresh_eyes) and (
                    # f":\n{self.triple_quotes}" in meta_model_output
                    re.search(pattern, meta_model_output)
                ):
                    # There might be multiple instructions between the triple quotes; so, split the output by the triple quotes.
                    triple_quote_splits = meta_model_output.split(self.triple_quotes)
                    # Odd indices are the instructions, even indices contain the lines preceding the instructions (indicating which model to use).
                    len_triple_quote_splits = len(triple_quote_splits)

                    intermediate_output = ""
                    model_num_return_sequences = 1  # Feel free to ignore the model_num_return_sequences > 1 case for now.
                    # Iterate over the instructions.
                    for i in range(1, len_triple_quote_splits, 2):
                        # Get the instructions for the corresponding model, as well as the line preceding the instructions (indicating which Expert to use).
                        line_preceding_instruction = triple_quote_splits[i - 1].strip()
                        model_name = line_preceding_instruction.split("\n")[-1].strip()
                        if "Expert " in model_name:
                            if model_name[-1] == ":":
                                model_name = model_name[:-1]

                            model_instruction = triple_quote_splits[i].strip()

                            # Add the expert name to the instruction.
                            if self.include_expert_name_in_instruction:
                                model_instruction = (
                                    f"You are {model_name}.\n\n{model_instruction}"
                                )

                            # Add "Let's think step by step." to the instruction.
                            if self.use_zero_shot_cot_in_expert_messages:
                                model_instruction += "\n\nLet's think step by step."

                            if model_name == "Expert Python":
                                model_instruction = f"{self.expert_python_message}.\n\n{model_instruction}"

                            current_chat_prompt = [
                                SystemMessage(model_instruction),
                                SystemMessage(
                                    'Once you have determined the final answer, please present it using the format below:\n\n>> FINAL ANSWER:\n"""\n[final answer]\n"""'
                                ),
                            ]
                            expert_routine = lm
                            model_output = expert_routine.invoke(
                                current_chat_prompt
                            ).content

                            ## Special case for Expert Python
                            if model_name == "Expert Python":
                                # If the output contains the special substring, then we need to execute the code.
                                if "Please run this code!" in model_output:
                                    # Get the code #ADDED: 14-08-2023
                                    code_text = model_output.split(
                                        "Please run this code!"
                                    )[0].strip()
                                    # Get the output
                                    code_text = code_text.replace("```python", "```")
                                    try:
                                        code_text = code_text.split("```")[-2].strip()
                                    except:
                                        code_text = code_text.split("```")[1].strip()

                                    # print(f"We are going to execute the following code:\n{code_text}\n\n")
                                    code_text = rf"{code_text}"
                                    # Execute the code
                                    python_output = execute_code_with_timeout(code_text)
                                    # Add the output to the model output
                                    model_output += f"Here is the Python code used to solve the problem:\n\n{code_text}\n\nHere is the output of the code when executed:\n\n{python_output}"

                            else:
                                specicial_token = "* * *"
                                if self.extract_output:
                                    # FIXME: Temporary fix
                                    if specicial_token in model_output:
                                        model_output = model_output.split(
                                            specicial_token
                                        )[1].strip()

                                    if len(model_output.split(" ")) > 128:
                                        model_output = (
                                            "Solution too long. Please try again."
                                        )
                                else:
                                    model_output.replace(specicial_token, "")
                                    model_output.replace(
                                        "FINAL ANSWER:",
                                        f"{model_name}'s final answer:\n",
                                    )

                            intermediate_output += f"{model_name}'s output:\n{self.triple_quotes}\n{model_output}\n{self.triple_quotes}"

                            # Remove the last two newlines.
                            intermediate_output = intermediate_output.strip()

                    # Add the intermediate output to the full prompt or messages.
                    intermediate_output = (
                        f"{intermediate_output}\n\n{self.intermediate_feedback}"
                    )
                    logger.debug(f"Intermediate output: {intermediate_output}")

                    # Add the intermediate output to the full prompt or messages.
                    chat_prompt.append(HumanMessage(intermediate_output))
                    reasoning_steps.append(HumanMessage(intermediate_output))

                    # Prepare the prompt for the meta model
                    return self.meta_model_generate(
                        lm=lm,
                        reasoning_steps=reasoning_steps,
                        chat_prompt=chat_prompt,
                        counter=counter + 1,
                        last_answer=last_answer,
                        original_question=original_question,
                        **kwargs,
                    )
                # Step 2(b): Check if the meta_model_output contains the final answer indicator.
                elif self.final_answer_indicator in meta_model_output:
                    # The following code is commented out because we are not using the final answer indicator anymore.
                    # However, it is useful for debugging purposes.
                    # final_answer = meta_model_output.split(self.final_answer_indicator)[
                    #     -1
                    # ].strip()
                    # print(f"Final answer: {final_answer}")
                    return chat_prompt
                # Step 2(c): We need to continue the (meta-)conversation.
                else:
                    chat_prompt.append(HumanMessage(self.error_message))
                    return self.meta_model_generate(
                        lm=lm,
                        chat_prompt=chat_prompt,
                        counter=counter + 1,
                        last_answer=last_answer,
                        original_question=original_question,
                        **kwargs,
                    )

        except Exception as e:
            print(f"Houston, we have a problem in meta_model_generate: {e}")

            # If we have tried 7 times, then let's return the current prompt or messages.
            if trial_num == 7:
                return chat_prompt

            print("Meta-prompting retrying, wait for 1-5 seconds...")
            # Let's wait for 1-5 seconds before trying again.
            time.sleep(random.randint(1, 5))
            return self.meta_model_generate(
                lm=lm,
                chat_prompt=chat_prompt,
                counter=counter,
                last_answer=last_answer,
                trial_num=trial_num + 1,
                **kwargs,
            )
