import pandas as pd
import cognify
import dotenv
from cognify.hub.evaluators import f1_score_str
import json
dotenv.load_dotenv()

import pandas as pd

# Data loader
def load_specific_data(task, mode):
    sentiment_df = pd.read_parquet(f"data/{task}.parquet")
    data = []
    for i, row in sentiment_df.iterrows():
        input = {
            'task': row['instruction'] + "\n" + row['input'],
            'mode': mode
        }
        output = {
            'label': row['output']
        }
        data.append((input, output))
        if i == 99:
            break
    return data

@cognify.register_data_loader
def load_all_data():
    sentiment_data = load_specific_data('sentiment', 'sentiment_analysis')
    headline_data = load_specific_data('headline', 'headline_classification')
    fiqa_data = load_specific_data('fiqa', 'fiqa')

    trainset = sentiment_data[:70] + headline_data[:70] + fiqa_data[:70]
    devset = sentiment_data[70:85] + headline_data[70:85] + fiqa_data[70:85]
    testset = sentiment_data[85:] + headline_data[85:] + fiqa_data[85:]
    return trainset, devset, testset


# Evaluators 
def evaluate_sentiment(answer, label):
    return f1_score_str(answer, label)


def evaluate_headline(answer, label):
    return f1_score_str(answer, label)

from pydantic import BaseModel
class Assessment(BaseModel):
    success: bool

import litellm
def evaluate_fiqa(answer, label, task):
    system_prompt="Given the question and the ground truth, evaluate if the response answers the question."
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": "You're given the following inputs:\n\nQuestion: " + task + "\n\nGround Truth: " + label + "\n\nResponse: " + answer}]
    response = litellm.completion('gpt-4o-mini', messages=messages, temperature=0, response_format=Assessment)
    assessment = json.loads(response.choices[0].message.content)
    return int(assessment['success'])


@cognify.register_evaluator
def evaluate(answer, label, mode, task):
    if mode == 'sentiment_analysis':
        return evaluate_sentiment(answer, label)
    elif mode == 'headline_classification':
        return evaluate_headline(answer, label)
    elif mode == 'fiqa':
        return evaluate_fiqa(answer, label, task)
    else:
        raise ValueError(f"Invalid mode: {mode}")
    
def evaluate_for_dspy_trace(true, prediction, trace=None):
    if true.mode == 'sentiment_analysis':
        return evaluate_sentiment(prediction.solution, true.label)
    elif true.mode == 'headline_classification':
        return evaluate_headline(prediction.solution, true.label)
    elif true.mode == 'fiqa':
        return evaluate_fiqa(prediction.solution, true.label, true.task)
    else:
        raise ValueError(f"Invalid mode: {true.mode}")


# Search settings
from cognify.hub.search import default

search_settings = default.create_search(
    search_type='light',
    n_trials=10,
    opt_log_dir='optimization_results',
    evaluator_batch_size=20,
)
