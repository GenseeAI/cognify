import argparse
import json
import re
import uuid
import dspy

from tqdm import tqdm
from agents.query_expansion_agent.agent import QueryExpansionAgent, query_expansion_agent
from agents.plot_agent.agent import PlotAgent, PlotAgentModule
from agents.visual_refine_agent import VisualRefineAgent
import logging
import os
import shutil
import glob
import sys
from agents.utils import is_run_code_success, run_code, get_code
from agents.dspy_common import OpenAIModel
from agents.config.openai import openai_kwargs
import cognify
import dotenv

# set to info level logging
logging.basicConfig(level=logging.INFO)

dotenv.load_dotenv()

@cognify.register_workflow
def mainworkflow(query, directory_path, example_id, input_path):
    # Prepare workspace
    workspace = f'{directory_path}/{example_id}_{uuid.uuid4().hex}'
    if not os.path.exists(workspace):
        # If it doesn't exist, create the directory
        os.makedirs(workspace, exist_ok=True)
        if os.path.exists(input_path):
            os.system(f'cp -r {input_path}/* {workspace}')
    else:
        logging.debug(f"Directory '{workspace}' already exists.")
        
    # Query expanding
    logging.debug('=========Query Expansion AGENT=========')
    config = {'workspace': workspace}
    expanded_simple_instruction = query_expansion_agent(inputs={'query': query})
    # logging.info('=========Expanded Simple Instruction=========')
    # logging.info(expanded_simple_instruction)
    logging.debug('=========Plotting=========')

    # GPT-4 Plot Agent
    # Initial plotting
    action_agent = PlotAgentModule()
    logging.debug('=========Novice 4 Plotting=========')
    novice_log, novice_code = action_agent.run(
        query_type='initial',
        workspace=workspace,
        query=query,
        expanded_query=expanded_simple_instruction,
        plot_file_name='novice.png',
    )
    # logging.info(novice_log)
    # logging.info('=========Original Code=========')
    # logging.info(novice_code)

    # Visual refinement
    if os.path.exists(f'{workspace}/novice.png'):
        logging.debug('========= Get visual feedback =======')
        visual_refine_agent = VisualRefineAgent('novice.png', config, novice_code, query)
        visual_feedback = visual_refine_agent.run('gpt-4o-mini', 'novice', 'novice_final.png')
        # logging.info('=========Visual Feedback=========')
        # logging.info(visual_feedback)
        final_instruction = '' + '\n\n' + visual_feedback
        
        novice_log, novice_code = action_agent.run(
            query_type='refinement',
            workspace=workspace,
            query=query,
            code=novice_code,
            visual_refinement=final_instruction,
            plot_file_name='novice_final.png',
        )
        # logging.info(novice_code)
    return {
        "image": f"{workspace}/novice_final.png",
        "rollback": f"{workspace}/novice.png",
    }

if __name__ == "__main__":
    pass

    # print("-- Running main workflow --")
    # data_path = 'benchmark_data'
    
    # # open the json file 
    # data = json.load(open(f'{data_path}/96.json'))
    
    # for item in tqdm(data):
    #     novice_instruction = item['simple_instruction']
    #     expert_instruction = item['expert_instruction']
    #     example_id = item['id']
    #     directory_path = "sample_runs_direct"

    #     # Check if the directory already exists
    #     if not os.path.exists(directory_path):
    #         os.makedirs(directory_path, exist_ok=True)
        
    #     mainworkflow(
    #         query=novice_instruction,
    #         directory_path=directory_path,
    #         example_id=example_id,
    #         input_path=f'{data_path}/data/{example_id}',
    #     )