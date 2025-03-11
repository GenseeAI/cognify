import json
import sys

# Load the aggregated statistics files A and B
file_a = sys.argv[1]
file_b = sys.argv[2]

with open(file_a, 'r') as f:
    data_a = json.load(f)

with open(file_b, 'r') as f:
    data_b = json.load(f)

# Find instances that are incorrect in A's revision but correct in B's revision
incorrect_in_a_correct_in_b = []

# Get the 'incorrect' instances in A and 'correct' instances in B for revision
incorrect_a = set(tuple(instance) for instance in data_a["ids"]["revision"]["incorrect"])
correct_b = set(tuple(instance) for instance in data_b["ids"]["revision"]["correct"])

# Find instances present in 'incorrect' in A but 'correct' in B
incorrect_in_a_correct_in_b = incorrect_a.intersection(correct_b)

# Display the result as a list of instances
incorrect_in_a_correct_in_b = [list(instance) for instance in incorrect_in_a_correct_in_b]
input_pool = json.load(open('/mnt/ssd4/lm_compiler/examples/CHESS/data/dev/sub_sampled_bird_dev_set.json', 'r'))
input_ref_dict = {}
for e in input_pool:
    if e['db_id'] not in input_ref_dict:
        input_ref_dict[e['db_id']] = {}
    input_ref_dict[e['db_id']][e['question_id']] = e['difficulty']

for e in incorrect_in_a_correct_in_b:
    e.append(input_ref_dict[e[0]][e[1]])

# Print or save the result
output_file = "benefit.json"
with open(output_file, 'w') as f:
    json.dump(incorrect_in_a_correct_in_b, f, indent=4)

print(f"Instances that are incorrect in A's revision but correct in B's revision saved to {output_file}")

