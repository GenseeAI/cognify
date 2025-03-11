from cognify import register_data_loader
from cognify import register_evaluator

# sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__))))

from humaneval.humaneval import check_correctness_thread
from humaneval.humaneval import HumanEvalDataset
import random

@register_data_loader
def load_data():
    raw_dataset = HumanEvalDataset()
    size = len(raw_dataset.data)
    # shuffle the data
    random.seed(42)
    random.shuffle(raw_dataset.data)
    data = []
    for i in range(size):
        problem = raw_dataset.data[i]
        input = {'problem': problem}
        ground_truth = {}
        data.append((input, ground_truth))
    train, val, test = data[:40], data[40:60], data[60:]
    return train, val, test

@register_evaluator
def pass_test(problem, finalized_code):
    split_completion = finalized_code.split('\n')
    parsed_lines = []
    for line in split_completion:
        if "<result>" in line or "</result>" in line or "```" in line or "python" in line:
            continue
        parsed_lines.append(line)
    completion = '\n'.join(parsed_lines)

    result = check_correctness_thread(problem, completion, timeout=3.0)
    return 1.0 if result["passed"] else 0.0

## search
from cognify.hub.search import codegen
search_settings = codegen.create_search(evaluator_batch_size=40)
