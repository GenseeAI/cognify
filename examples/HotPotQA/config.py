import cognify
from cognify.hub.evaluators import f1_score_str
from cognify.hub.datasets.hotpot import HotPotQA

@cognify.register_evaluator
def answer_f1(answer: str, ground_truth: str):
    return f1_score_str(answer, ground_truth) # score function

def formatting(item):
    return (
        {'question': item.question},
        {'ground_truth': item.answer}
    )

@cognify.register_data_loader
def load_data():
    dataset = HotPotQA(train_seed=1, 
                       train_size=150, 
                       eval_seed=2023, 
                       dev_size=200, 
                       test_size=0)
    
    trainset = [formatting(x) for x in dataset.train[0:100]]
    valset = [formatting(x) for x in dataset.train[100:150]]
    devset = [formatting(x) for x in dataset.dev]
    return trainset, valset, devset # training and evaluation data

from cognify.hub.search import qa
search_settings = qa.create_search(
    evaluator_batch_size=12,
    opt_log_dir='hotpot_results'
)