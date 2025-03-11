import cognify
from cognify.hub.evaluators import f1_score_set

@cognify.register_evaluator
def doc_f1(pred_docs, gold_docs):
    pred_docs = set(pred_docs)
    gold_docs = set(gold_docs)
    return f1_score_set(pred_docs, gold_docs)

train_size = 100
val_size = 50
dev_size = 200
data_path = 'qas._json'
seed = 0

import json
import random

@cognify.register_data_loader
def load_data():
    data = []
    
    # only include difficult examples with 3 hops
    with open(data_path, 'r') as file:
        for line in file:
            obj = json.loads(line)
            if obj['num_hops'] == 3:
                data.append(obj)
                
    rng = random.Random(seed)
    rng.shuffle(data)
    
    def formatting(x):
        input = {'claim': x['question']}
        ground_truth = {'gold_docs': x['support_pids']}
        return (input, ground_truth)
    
    train_set = data[:train_size]
    val_set = data[train_size:train_size+val_size]
    dev_set = data[train_size+val_size:train_size+val_size+dev_size]
    
    train_set = [formatting(x) for x in train_set]
    val_set = [formatting(x) for x in val_set]
    dev_set = [formatting(x) for x in dev_set]
    return train_set, val_set, dev_set

from cognify.hub.search import default
search_settings = default.create_search(evaluator_batch_size=50)