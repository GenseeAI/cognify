import json
import os
from collections import defaultdict
import random
import numpy as np

with open("data/dev_full.json", "r") as file:
    full_dev = json.load(file)
    
with open("data/dev/sub_sampled_bird_dev_set.json", "r") as file:
    sds_dev = json.load(file)
    
sample_size = 150

by_difficulty = defaultdict(list)
for i, data in enumerate(full_dev):
    by_difficulty[data["difficulty"]].append((i, data))

sizes = {
    'simple': int(0.55 * sample_size),
    'moderate': int(0.37 * sample_size),
}
sizes['challenging'] = sample_size - sizes['simple'] - sizes['moderate']

sds_query_ids = set([f'{data['db_id']}_{data['question_id']}' for data in sds_dev])

np.random.seed(0)
sampled_data = []
# preserve the order
for d in ["simple", "moderate", "challenging"]:
    cand_i = np.random.choice(list(range(len(by_difficulty[d]))), int(sizes[d] * 1.3), replace=False).tolist()
    cand = [by_difficulty[d][i] for i in cand_i]
    # remove overlap with sds
    cand = [(i, c) for i, c in cand if f'{c['db_id']}_{c['question_id']}' not in sds_query_ids]
    cand = cand[:sizes[d]]
    print(f"Difficulty: {d}, candidates: {len(cand)}")
    sampled_data.extend(cand)
    
# sort by index and only keep data
sampled_data = [data for i, data in sorted(sampled_data)]

with open("data/dev/other_sub_sampled.json", "w+") as file:
    json.dump(sampled_data, file, indent=4)