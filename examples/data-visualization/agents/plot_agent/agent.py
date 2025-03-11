import os
import re


from .prompt import ERROR_PROMPT, INITIAL_SYSTEM_PROMPT, INITIAL_USER_PROMPT, VIS_SYSTEM_PROMPT, VIS_USER_PROMPT, ZERO_SHOT_COT_PROMPT
from agents.openai_chatComplete import completion_with_backoff
from agents.utils import fill_in_placeholders, get_error_message, is_run_code_success, run_code
from agents.utils import print_filesys_struture
from agents.utils import change_directory, common_lm_config
import cognify
import logging


class PlotAgent():


    def __init__(self, config, query, data_information=None):
        self.chat_history = []
        self.workspace = config['workspace']
        self.query = query
        self.data_information = data_information

    def generate(self, user_prompt, model_type, query_type, file_name):

        workspace_structure = print_filesys_struture(self.workspace)
        
        information = {
            'workspace_structure': workspace_structure,
            'file_name': file_name,
            'query': user_prompt
        }

        if query_type == 'initial':
            messages = []
            messages.append({"role": "system", "content": fill_in_placeholders(INITIAL_SYSTEM_PROMPT, information)})
            messages.append({"role": "user", "content": fill_in_placeholders(INITIAL_USER_PROMPT, information)})
        else:
            messages = []
            messages.append({"role": "system", "content": fill_in_placeholders(VIS_SYSTEM_PROMPT, information)})
            messages.append({"role": "user", "content": fill_in_placeholders(VIS_USER_PROMPT, information)})

        self.chat_history = self.chat_history + messages
        return completion_with_backoff(messages, model_type)

    def get_code(self, response):

        all_python_code_blocks_pattern = re.compile(r'```python\s*([\s\S]+?)\s*```', re.MULTILINE)


        all_code_blocks = all_python_code_blocks_pattern.findall(response)
        all_code_blocks_combined = '\n'.join(all_code_blocks)
        return all_code_blocks_combined
    
    def get_code2(self, response,file_name):

        all_python_code_blocks_pattern = re.compile(r'```\s*([\s\S]+?)\s*```', re.MULTILINE)


        all_code_blocks = all_python_code_blocks_pattern.findall(response)
        all_code_blocks_combined = '\n'.join(all_code_blocks)
        if all_code_blocks_combined == '':

            response_lines = response.split('\n')
            code_lines = []
            code_start = False
            for line in response_lines:
                if line.find('import') == 0 or code_start:
                    code_lines.append(line)
                    code_start = True
                if code_start and line.find(file_name)!=-1 and line.find('(') !=-1 and line.find(')')!=-1 and line.find('(') < line.find(file_name)< line.find(')'): #要有文件名，同时要有函数调用

                    return '\n'.join(code_lines)
        return all_code_blocks_combined


    def run(self, query, model_type, query_type, file_name):
        try_count = 0
        image_file = file_name
        result = self.generate(query, model_type=model_type, query_type=query_type, file_name=file_name)
        while try_count < 4:
            
            if not isinstance(result, str):  # 如果返回的不是字符串，那么就是出错了
                return 'TOO LONG FOR MODEL', code
            if model_type != 'gpt-4':
                code = self.get_code(result)
                if code.strip() == '':
                    code = self.get_code2(result,image_file) #第二次尝试获得代码
                    if code.strip() == '':
                        code = result  #只能用原始回答
                        if code.strip() == '' and try_count == 0: #有可能是因为没有extend query写好了代码，所以他不写代码
                            code = self.get_code(query)
            else:
                code = self.get_code(result)
            self.chat_history.append({"role": "assistant", "content": result if result.strip() != '' else ''})


            file_name = f'code_action_{model_type}_{query_type}_{try_count}.py'
            with open(os.path.join(self.workspace, file_name), 'w') as f:
                f.write(code)
            error = None
            log = run_code(self.workspace, file_name)

            if is_run_code_success(log):
                if print_filesys_struture(self.workspace).find('.png') == -1:
                    log = log + '\n' + 'No plot generated.'
                    
                    self.chat_history.append({"role": "user", "content": fill_in_placeholders(ERROR_PROMPT,
                                                                                          {'error_message': f'No plot generated. When you complete a plot, remember to save it to a png file. The file name should be """{image_file}""".',
                                                                                           'data_information': self.data_information})})
                    try_count += 1
                    result = completion_with_backoff(self.chat_history, model_type=model_type)


                else:
                    return log, code

            else:
                error = get_error_message(log) if error is None else error
                # TODO error prompt
                self.chat_history.append({"role": "user", "content": fill_in_placeholders(ERROR_PROMPT,
                                                                                          {'error_message': error,
                                                                                           'data_information': self.data_information})})
                try_count += 1
                result = completion_with_backoff(self.chat_history, model_type=model_type)

        return log, ''

    def run_initial(self, model_type, file_name):
        print('========Plot AGENT Expert RUN========')
        self.chat_history = []
        log, code = self.run(self.query, model_type, 'initial', file_name)
        return log, code

    def run_vis(self, model_type, file_name):
        print('========Plot AGENT Novice RUN========')
        self.chat_history = []
        log, code = self.run(self.query, model_type, 'vis_refined', file_name)
        return log, code

    def run_one_time(self, model_type, file_name,query_type='novice',no_sysprompt=False):
        
        print('========Plot AGENT Novice RUN========')
        message = []
        workspace_structure = print_filesys_struture(self.workspace)
        
        information = {
            'workspace_structure': workspace_structure,
            'file_name': file_name,
            'query': self.query
        }
        if no_sysprompt:
            message.append({"role": "system", "content": ''''''})
        message.append({"role": "user", "content": fill_in_placeholders(INITIAL_USER_PROMPT, information)})
        result = completion_with_backoff(message, model_type)
        if model_type != 'gpt-4':
            code = self.get_code(result)
            if code == '':
                code = self.get_code2(result,file_name)
                if code == '':
                    code = result
        else:
            code = self.get_code(result)


        file_name = f'code_action_{model_type}_{query_type}_0.py'
        with open(os.path.join(self.workspace, file_name), 'w') as f:
            f.write(code)
        log = run_code(self.workspace, file_name)
        return log, code
    def run_one_time_zero_shot_COT(self, model_type, file_name,query_type='novice',no_sysprompt=False):
        
        print('========Plot AGENT Novice RUN========')
        message = []
        workspace_structure = print_filesys_struture(self.workspace)
        
        information = {
            'workspace_structure': workspace_structure,
            'file_name': file_name,
            'query': self.query
        }
        message.append({"role": "system", "content": ''''''})
        message.append({"role": "user", "content": fill_in_placeholders(ZERO_SHOT_COT_PROMPT, information)})
        result = completion_with_backoff(message, model_type)
        if model_type != 'gpt-4':
            code = self.get_code(result)
            if code == '':
                code = self.get_code2(result,file_name)
                if code == '':
                    code = result
        else:
            code = self.get_code(result)

        file_name = f'code_action_{model_type}_{query_type}_0.py'
        with open(os.path.join(self.workspace, file_name), 'w') as f:
            f.write(code)
        log = run_code(self.workspace, file_name)
        return log, code

"""
Cognify Implementation
"""  

from pydantic import BaseModel, Field

lm_config = cognify.LMConfig(
    custom_llm_provider='openai',
    model='gpt-4o-mini',
    kwargs= {
        'temperature': 0.0,
    }
)

#==============================================================================
# Coder
#==============================================================================

INITIAL_SYSTEM_PROMPT_IR = '''
You are an expert Python coder specialized in data visualization. Your task is to generate Python code that fulfills the user’s data visualization request. You will receive the user query and a detailed plan outlining how to create the plot.

If the instruction requires data manipulation from a csv file, write code to process the data from the csv file and then draw the plot. Please place all your code in a single code block.

Your code should save the final plot to a png file with the give filename rather than displaying it.
'''
initial_coder_agent = cognify.Model(agent_name='initial code generation', system_prompt=INITIAL_SYSTEM_PROMPT_IR,
                            input_variables=[cognify.Input(name='query'), cognify.Input(name='expanded_query'), cognify.Input(name='plot_file_name')],
                            output=cognify.OutputLabel(name='code', custom_output_format_instructions="Please only give the code as your answer and format it in markdown code block, i.e. wrap it with ```python and ```."),
                            lm_config=lm_config)

#==============================================================================
# Debugger
#==============================================================================

DEBUG_SYSTEM_PROMPT_IR = """
You are an expert in debugging Python code, particularly for data visualization tasks. You will be given an user request, a piece of python code for completing the task and an error message associate with this code. 

Your task is to:
- Analyze the error messages and understand the issue in the context of the user’s request.
- Fix the code completely: Modify the code so that it runs without errors and correctly fulfills the user’s original requirements,

Always output the complete, updated Python code with all necessary corrections applied, ensuring it is ready to be executed successfully.
Your code should save the final plot to a png file with the give filename rather than displaying it.
"""
plot_debugger_agent = cognify.Model(agent_name='plot debugger', system_prompt=DEBUG_SYSTEM_PROMPT_IR,
                            input_variables=[cognify.Input(name='query'), cognify.Input(name='code'), cognify.Input(name='error_message')],
                            output=cognify.OutputLabel(name='code', custom_output_format_instructions="Please only give the code as your answer and format it in markdown code block, i.e. wrap it with ```python and ```."),
                            lm_config=lm_config)

#==============================================================================
# Refiner
#==============================================================================

VIS_SYSTEM_PROMPT_IR = """
You are an expert Python coder specialized in data visualization. Your task is to refine the existing Python code based on the feedback to ensure the plot fully meets the user's requirements.

You will receive:
- The user's query to understand the intended goal.
- The existing Python code used to generate the current plot.
- The feedback that contains specific instructions for improving the visualization.

Please analyze the feedback and apply the recommended changes to the Python code while ensuring the adjustments align with the user's original request. 
Your code should save the final plot to a png file with the give filename rather than displaying it.
"""

refine_plot_agent = cognify.Model(agent_name='visual refine coder', system_prompt=VIS_SYSTEM_PROMPT_IR,
                            input_variables=[cognify.Input(name='query'), cognify.Input(name='code'), cognify.Input(name='visual_refinement'), cognify.Input(name='plot_file_name')],
                            output=cognify.OutputLabel(name='code', custom_output_format_instructions="Please only return the python code. Wrap it with ```python and ``` to format it properly."),
                            lm_config=lm_config)

class PlotAgentModule:
    def __init__(self, data_information=None):
        self.data_information = data_information
    
    def get_code(self, response):
        all_python_code_blocks_pattern = re.compile(r'```python\s*([\s\S]+?)\s*```', re.MULTILINE)

        all_code_blocks = all_python_code_blocks_pattern.findall(response)
        all_code_blocks_combined = '\n'.join(all_code_blocks)
        if all_code_blocks_combined == '':
            return response
        return all_code_blocks_combined
        
    def forward(
        self,
        query_type: str,
        workspace: str,
        **kwargs,
    ):
        try_count = 0
        if query_type == 'initial':
            result = initial_coder_agent(inputs=kwargs)
        else:
            result = refine_plot_agent(inputs=kwargs)
        
        while try_count < 4:
            code = self.get_code(result)
            workspace_structure = print_filesys_struture(workspace)
            
            file_name = f'code_action_{query_type}_{try_count}.py'
            with open(os.path.join(workspace, file_name), 'w') as f:
                f.write(code)
            error = None
            log = run_code(workspace, file_name)
            
            if is_run_code_success(log):
                if print_filesys_struture(workspace).find(file_name) == -1:
                    log = log + '\n' + 'No plot generated.'
                    error_message = f'The expected file is not generated. When you complete a plot, remember to save it to a png file. The file name should be """{file_name}"""'
                    
                    # debug and retry
                    try_count += 1
                    result = plot_debugger_agent(inputs={'query': kwargs['query'], 'code': code, 'error_message': error_message})
                else:
                    return log, code
            else:
                error = get_error_message(log) if error is None else error
                try_count += 1
                result = plot_debugger_agent(inputs={'query': kwargs['query'], 'code': code, 'error_message': error})
        return log, ''
    
    def run(self, query_type, workspace, **kwargs):
        logging.debug(f'========Plot AGENT {query_type} RUN========')
        log, code = self.forward(
            query_type=query_type,
            workspace=workspace,
            **kwargs,
        )
        return log, code