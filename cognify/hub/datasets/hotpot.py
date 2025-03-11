import random

from datasets import load_dataset
from dspy.datasets.dataset import Dataset
import multiprocessing as mp


def set_gold_titles(example):
    example['gold_titles'] = set(example['supporting_facts']['title'])
    return example

class HotPotQA(Dataset):
    def __init__(self, *args, only_hard_examples=True, unofficial_dev=True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert only_hard_examples, "Care must be taken when adding support for easy examples." \
                                   "Dev must be all hard to match official dev, but training can be flexible."
        
        hf_official_train = load_dataset("hotpot_qa", 'fullwiki', split='train', trust_remote_code=True)
        hf_official_dev = load_dataset("hotpot_qa", 'fullwiki', split='validation', trust_remote_code=True)

        hf_official_train_hard = hf_official_train.filter(lambda x: x['level'] == 'hard', num_proc=mp.cpu_count())
        official_train = hf_official_train_hard.remove_columns(['id', 'type', 'level', 'context']).to_list()

        rng = random.Random(0)
        rng.shuffle(official_train)

        self._train = official_train[:len(official_train)*75//100]

        if unofficial_dev:
            self._dev = official_train[len(official_train)*75//100:]
        else:
            self._dev = None

        hf_official_dev_remapped = hf_official_dev.remove_columns(['level', 'context'])
        test = hf_official_dev_remapped.map(set_gold_titles, num_proc=mp.cpu_count(), remove_columns='supporting_facts').to_list()

        self._test = test