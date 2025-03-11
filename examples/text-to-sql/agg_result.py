import os
import json
import sys
from collections import defaultdict

# Initialize the aggregated structure with default values
aggregated_data = {
    "counts": {
        "candidate_generation": {"correct": 0, "incorrect": 0, "error": 0, "total": 0},
        "revision": {"correct": 0, "incorrect": 0, "error": 0, "total": 0}
    },
    "ids": {
        "candidate_generation": {"correct": [], "incorrect": [], "error": []},
        "revision": {"correct": [], "incorrect": [], "error": []}
    }
}

def aggregate_counts(src, dest):
    for phase in ["candidate_generation", "revision"]:
        for key in ["correct", "incorrect", "error", "total"]:
            dest["counts"][phase][key] += src["counts"][phase].get(key, 0)

def aggregate_ids(src, dest):
    for phase in ["candidate_generation", "revision"]:
        for key in ["correct", "incorrect", "error"]:
            dest["ids"][phase][key].extend(src["ids"][phase].get(key, []))

def process_statistics_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
        if "candidate_generation" not in data["counts"]:
            print(f)
    aggregate_counts(data, aggregated_data)
    aggregate_ids(data, aggregated_data)

# Walk through the directory and process each statistics.json file
root_dir = sys.argv[1]
for dirpath, dirnames, filenames in os.walk(root_dir):
    for filename in filenames:
        if filename == "-statistics.json":
            file_path = os.path.join(dirpath, filename)
            process_statistics_file(file_path)

# Write the aggregated result to a file
output_file = os.path.join(root_dir, "aggregated_statistics.json")
with open(output_file, 'w') as f:
    json.dump(aggregated_data, f, indent=4)

print(f"Aggregated statistics saved to {output_file}")
