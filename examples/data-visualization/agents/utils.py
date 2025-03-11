from collections import defaultdict
from copy import deepcopy
import fnmatch
import logging
import re
from typing import Dict

import os
from contextlib import contextmanager
import subprocess
import cognify

common_lm_config = cognify.LMConfig(
    custom_llm_provider='openai',
    model='gpt-4o-mini',
    kwargs= {
        'temperature': 0.0,
    }
)

@contextmanager
def change_directory(directory):
    current_directory = os.getcwd()
    
    try:

        print("Changing directory to:", directory)
        os.chdir(directory)
        yield 
    finally:

        print("Changing directory back to:", current_directory)
        os.chdir(current_directory)

def get_workspace_structure(work_directory):

    def generate_directory_structure(path):
        result = {'name':os.path.basename(path)}
        if os.path.isdir(path):
            result['type'] = 'directory'
            result['children'] = [generate_directory_structure(os.path.join(path,child)) for child in os.listdir(path)]
        else:
            result['type'] = 'file'
        return result
    return generate_directory_structure(work_directory)

def fill_in_placeholders(prompt_messages,placeholders: dict):

    filled_messages = deepcopy(prompt_messages)
    
    for key, value in placeholders.items():
        if value is not None:
            filled_messages = filled_messages.replace("{{" + str(key) + "}}", str(value))
    return filled_messages

def _check_ignorement(path:str,ignored_list)->bool:
    for pattern in ignored_list:
        if fnmatch.fnmatch(path,pattern):
            return True
    return False

def print_filesys_struture(work_directory,return_root=False,max_entry_nums_for_level=100,ignored_list=[])->str:
    full_repr = ''
    if return_root:
        full_repr += f'Global Root Work Directory: {work_directory}\n'

    folder_counts =  defaultdict(lambda: 0)
    for root, dirs, files in os.walk(work_directory):
        if _check_ignorement(root,ignored_list):
            continue
        level = root.replace(work_directory, '').count(os.sep)
        indent = ' ' * 4 * (level)
        
        folder_counts[root] += 1
        if folder_counts[root] > max_entry_nums_for_level:
            full_repr += f'{indent}`wrapped`\n'
        
        full_repr += f'{indent}- {os.path.basename(root)}/\n'
        
        idx = 0
        subindent = ' ' * 4 * (level + 1) + '- '
        for f in files:
            if _check_ignorement(f,ignored_list):
                continue
            
            idx += 1
            if idx > max_entry_nums_for_level:
                full_repr += f'{subindent}`wrapped`\n'
                break
            full_repr += f'{subindent}{f}\n'


    return full_repr


def get_code(response):

    all_python_code_blocks_pattern = re.compile(r'```python\s*([\s\S]+?)\s*```', re.MULTILINE)

    all_code_blocks = all_python_code_blocks_pattern.findall(response)
    all_code_blocks_combined = '\n'.join(all_code_blocks)
    return all_code_blocks_combined


def run_code(workspace, code_file, log_file=None)->str:
    if log_file is None:
        log_file = code_file + '.log'
        
    result = subprocess.run(['python', code_file],
                            cwd=workspace,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True)
    
    with open(os.path.join(workspace, log_file), 'w+') as f:
        f.write(result.stdout)

    return result.stdout


def is_run_code_success(log):
    if 'Traceback (most recent call last):' in log or 'Error:' in log:
        return False
    else:
        return True

def get_error_message(log):
    if 'Traceback (most recent call last):' in log:
        return log.split('Traceback (most recent call last):')[1]
    elif 'Error:' in log:
        return log.split('Error:')[1]
    else:
        return 'Unknown Error'

def print_chat_message(messages):
    for message in messages:
        logging.info(f"{message['role']}: {message['content']}")
