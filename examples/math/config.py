#================================================================
# Evaluator
#================================================================

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
            ("human", "problem:\n{problem}\n\nstandard solution:\n{solution}\n\nanswer:\n{answer}\n\nYou response format:\n{format_instructions}\n"),
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


#================================================================
# Data Loader
#================================================================

import json
import random

@cognify.register_data_loader
def load_data():
    with open("data._json", "r") as f:
        data = json.load(f)
        
    random.seed(42)
    random.shuffle(data) 
    # format to (input, output) pairs
    new_data = []
    for d in data:
        input_sample = {
            'workflow_input': d["problem"],
        }
        ground_truth = {
            'ground_truth': d["solution"],
        }
        new_data.append((input_sample, ground_truth))
    # train, val, test split
    return new_data[:30], None, new_data[30:]

#================================================================
# Optimizer Set Up
#================================================================

from cognify.hub.search import default

model_configs = [
    # OpenAI models
    cognify.LMConfig(model='gpt-4o-mini', kwargs={'temperature': 0, 'max_tokens': 300}),
    # cognify.LMConfig(model='gpt-4o', kwargs={'temperature': 0, 'max_tokens': 300}),
]

search_settings = default.create_search(
    model_selection_cog=model_configs,
    opt_log_dir='with_ms_opt_log',
)