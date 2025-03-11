import argparse
import json
from datetime import datetime
import os
import debugpy
import multiprocessing as mp
import sys
from src.utils import parse_arguments, read_from_file

import cognify
import numpy as np

@cognify.register_data_loader
def load_data():
    args = parse_arguments()
    all_train = read_from_file('data/dev/other_sub_sampled.json', args)
    test_set = read_from_file('data/dev/sub_sampled_bird_dev_set.json', args)
    
    # shuffle the data
    all_train = np.random.permutation(all_train).tolist()
    return all_train[:100], all_train[100:], test_set[:10]


@cognify.register_evaluator
def eval(counts):
    """
    Evaluate the statistics of the run.
    """
    correct = any(vs['correct'] == 1 for vs in counts.values())
    return 1.0 if correct else 0.0

from cognify.hub.search import text_to_sql
search_settings = text_to_sql.create_search()