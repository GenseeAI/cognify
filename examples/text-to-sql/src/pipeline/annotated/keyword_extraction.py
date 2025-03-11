import os
import sys

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', '..', '..', '..'))
import cognify
from llm.parsers import PythonListOutputParser

system_prompt = \
"""
Objective: Analyze the given question and hint to identify and extract keywords, keyphrases, and named entities. These elements are crucial for understanding the core components of the inquiry and the guidance provided. This process involves recognizing and isolating significant terms and phrases that could be instrumental in formulating searches or queries related to the posed question.

Instructions:

Read the Question Carefully: Understand the primary focus and specific details of the question. Look for any named entities (such as organizations, locations, etc.), technical terms, and other phrases that encapsulate important aspects of the inquiry.

Analyze the Hint: The hint is designed to direct attention toward certain elements relevant to answering the question. Extract any keywords, phrases, or named entities that could provide further clarity or direction in formulating an answer.

List Keyphrases and Entities: Combine your findings from both the question and the hint into a single Python list. This list should contain:

Keywords: Single words that capture essential aspects of the question or hint.

Keyphrases: Short phrases or named entities that represent specific concepts, locations, organizations, or other significant details.

Ensure to maintain the original phrasing or terminology used in the question and hint.
"""

inputs = ["QUESTION", "HINT"]

output_format = "list_keywords_keyphrases"

output_format_instructions = \
"""
Please provide your findings as a Python list, capturing the essence of both the question and hint through the identified terms and phrases.
Ensure that you output only the list itself, without any variable assignments, explanations, or additional text.
"""

demos = [
   cognify.Demonstration(
        filled_input_variables=[
           cognify.FilledInput(
                cognify.Input("QUESTION"), 
                value="What is the annual revenue of Acme Corp in the United States for 2022?"
            ),
           cognify.FilledInput(
                cognify.Input("HINT"), 
                value="Focus on financial reports and U.S. market performance for the fiscal year 2022."
            )
        ],
        output='["annual revenue", "Acme Corp", "United States", "2022", "financial reports", "U.S. market performance", "fiscal year"]'
    ),

   cognify.Demonstration(
        filled_input_variables=[
           cognify.FilledInput(
                cognify.Input("QUESTION"), 
                value="In the Winter and Summer Olympics of 1988, which game has the most number of competitors? Find the difference of the number of competitors between the two games."
            ),
           cognify.FilledInput(
                cognify.Input("HINT"), 
                value="the most number of competitors refer to MAX(COUNT(person_id)); SUBTRACT(COUNT(person_id where games_name = '1988 Summer'), COUNT(person_id where games_name = '1988 Winter'));"
            )
        ],
        output='["Winter Olympics", "Summer Olympics", "1988", "1988 Summer", "Summer", "1988 Winter", "Winter", "number of competitors", "difference", "MAX(COUNT(person_id))", "games_name", "person_id"]',
    ),

   cognify.Demonstration(
        filled_input_variables=[
           cognify.FilledInput(
                cognify.Input("QUESTION"), 
                value="How many Men's 200 Metres Freestyle events did Ian James Thorpe compete in?"
            ),
           cognify.FilledInput(
                cognify.Input("HINT"), 
                value="Men's 200 Metres Freestyle events refer to event_name = 'Swimming Men's 200 metres Freestyle'; events compete in refers to event_id;"
            )
        ],
        output='["Swimming Men\'s 200 metres Freestyle", "Ian James Thorpe", "Ian", "James", "Thorpe", "compete in", "event_name", "event_id"]'
    )

]

lm_config = cognify.LMConfig(model="gpt-4o-mini", kwargs={"temperature": 0.2})
exec = cognify.Model(agent_name="keyword_extraction",
             system_prompt=system_prompt, 
             input_variables=[cognify.Input(name=input) for input in inputs], 
             output=cognify.OutputLabel(name=output_format, custom_output_format_instructions=output_format_instructions),
             lm_config=lm_config)
runnable_exec = cognify.as_runnable(exec) | PythonListOutputParser()