import argparse
import json
from datetime import datetime
import os
import debugpy
import multiprocessing as mp
import sys



sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__))))

from runner.task import Task
from typing import Any, Dict, List, TypedDict, Callable

entity_retieval_mode='ask_model' # Options: 'corrects', 'ask_model'

context_retrieval_mode='vector_db' # Options: 'corrects', 'vector_db'
top_k=5

table_selection_mode='ask_model' # Options: 'corrects', 'ask_model'

column_selection_mode='ask_model' # Options: 'corrects', 'ask_model'

engine2='gpt-4o-mini'
engine3='gpt-4o-mini'

pipeline_setup=f'''{{
    "keyword_extraction": {{
        "engine": "{engine2}",
        "temperature": 0.2,
        "base_uri": ""
    }},
    "entity_retrieval": {{
        "mode": "{entity_retieval_mode}"
    }},
    "context_retrieval": {{
        "mode": "{context_retrieval_mode}",
        "top_k": {top_k}
    }},
    "column_filtering": {{
        "engine": "{engine2}",
        "temperature": 0.0,
        "base_uri": ""
    }},
    "table_selection": {{
        "mode": "{table_selection_mode}",
        "engine": "{engine3}",
        "temperature": 0.0,
        "base_uri": "",
        "sampling_count": 1
    }},
    "column_selection": {{
        "mode": "{column_selection_mode}",
        "engine": "{engine3}",
        "temperature": 0.0,
        "base_uri": "",
        "sampling_count": 1
    }},
    "candidate_generation": {{
        "engine": "{engine3}",
        "temperature": 0.0,
        "base_uri": "",
        "sampling_count": 1
    }},
    "revision": {{
        "engine": "{engine3}",
        "temperature": 0.0,
        "base_uri": "",
        "sampling_count": 1
    }}
}}'''

cmd_args = [
    '--data_mode', 'dev',
    '--pipeline_nodes', 'keyword_extraction+entity_retrieval+context_retrieval+column_filtering+table_selection+column_selection+candidate_generation+revision+evaluation',
    '--pipeline_setup', pipeline_setup
]

def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run the pipeline with the specified configuration."
    )
    parser.add_argument(
        "--data_mode", type=str, required=True, help="Mode of the data to be processed."
    )
    parser.add_argument(
        "--pipeline_nodes",
        type=str,
        required=True,
        help="Pipeline nodes configuration.",
    )
    parser.add_argument(
        "--pipeline_setup",
        type=str,
        required=True,
        help="Pipeline setup in JSON format.",
    )
    parser.add_argument(
        "--use_checkpoint", action="store_true", help="Flag to use checkpointing."
    )
    parser.add_argument(
        "--checkpoint_nodes",
        type=str,
        required=False,
        help="Checkpoint nodes configuration.",
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, required=False, help="Directory for checkpoints."
    )
    parser.add_argument(
        "--log_level", type=str, default="warning", help="Logging level."
    )
    args = parser.parse_args(cmd_args)


    if args.use_checkpoint:
        print("Using checkpoint")
        if not args.checkpoint_nodes:
            raise ValueError("Please provide the checkpoint nodes to use checkpoint")
        args.checkpoint_nodes = args.checkpoint_nodes.split("+")
        if not args.checkpoint_dir:
            raise ValueError("Please provide the checkpoint path to use checkpoint")

    return args


def read_from_file(data_path, args):
    
    with open(data_path, "r") as file:
        dataset = json.load(file)

    inputs = []
    for data in dataset:
        inputs.append(
            {
                'args': args,
                'dataset': [data],
            }
        )
    eval_data = [(input, {}) for input in inputs]
    return eval_data