import os
import base64
import re


from .prompt import SYSTEM_PROMPT, USER_PROMPT, ERROR_PROMPT
from agents.openai_chatComplete import  completion_for_4v
from agents.utils import fill_in_placeholders, common_lm_config
import cognify
from cognify.hub.cogs.reasoning import ZeroShotCoT


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def get_code(response):
    all_python_code_blocks_pattern = re.compile(r'```python\s*([\s\S]+?)\s*```', re.MULTILINE)
    all_code_blocks = all_python_code_blocks_pattern.findall(response)
    all_code_blocks_combined = '\n'.join(all_code_blocks)
    return all_code_blocks_combined

VISUAL_FEEDBACK_SYSTEM_PROMPT = """
You are an expert in data visualization. Given a user query, a piece of code and an image of the current plot, please determine whether the plot has faithfully followed the user query. Your task is to provide instruction to refine the plot so that it can strictly completed the requirements of the query. Please output a detailed step by step instruction on how to enhance the plot.

Carefully read and analyze the user query to understand the specific requirements. Examine the provided Python code to understand how the current plot is generated. Check if the code aligns with the user query in terms of data selection, plot type, and any specific customization. Look at the provided image of the plot. Assess the plot type, the data it represents, labels, titles, colors, and any other visual elements. Compare these elements with the requirements specified in the user query. Note any differences between the user query requirements and the current plot. Based on the identified discrepancies, provide step-by-step instructions on how to modify the Python code to meet the user query requirements. Suggest improvements for better visualization practices, such as clarity, readability, and aesthetics, while ensuring the primary focus is on meeting the user's specified requirements.

You don't need to provide the complete code, just be very explicit in what changes are needed and how to make them.
"""
visual_refine_lm_config = cognify.LMConfig(
    custom_llm_provider='openai',
    model='gpt-4o-mini',
    kwargs= {
        'temperature': 0.0,
    }
)
visual_refinement_agent = cognify.Model(agent_name='visual_refinement', system_prompt=VISUAL_FEEDBACK_SYSTEM_PROMPT,
                                input_variables=[cognify.Input(name='query'), cognify.Input(name='code'), 
                                                 cognify.Input(name='plot_image', 
                                                          image_type='png')],
                                output=cognify.OutputLabel(name='refinement'),
                                lm_config=visual_refine_lm_config)

# ZeroShotCoT.direct_apply(visual_refinement_agent)

class VisualRefineAgent:
    def __init__(self, plot_file, config, code, query):
        self.chat_history = []
        self.plot_file = plot_file
        self.code = code
        self.query = query
        self.workspace = config['workspace']

    def run(self, model_type, query_type, file_name):
        plot = os.path.join(self.workspace, self.plot_file)
        base64_image1 = encode_image(f"{plot}")

        information = {
            'query': self.query,
            'code': self.code,
            'plot_image': base64_image1,
        }

        visual_feedback = visual_refinement_agent(inputs=information)
        return visual_feedback