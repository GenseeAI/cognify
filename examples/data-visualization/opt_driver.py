from cognify.hub.cogs.fewshot import LMFewShot
from cognify.hub.cogs.scaffolding import LMScaffolding
from cognify.hub.cogs import reasoning, model_selection, common
from cognify.cognify.optimizer.evaluator import EvaluationResult, EvaluatorPlugin, EvalTask
import runpy
import uuid
import multiprocessing as mp
import json
import os
import random
import optuna
import numpy as np

import cognify
from cognify.hub.cogs.common import NoChange
from cognify.hub.cogs.reasoning import ZeroShotCoT, PlanBefore
from cognify.optimizer.plugin import OptimizerSchema
from cognify.optimizer.analysis.param_sensitivity import SensitivityAnalyzer
from cognify.optimizer.core import driver, flow

def load_data():
    def load_from_file(input_file):
        data_path = 'examples/IR_matplot_agent/benchmark_data'
        # open the json file 
        data = json.load(open(f'{data_path}/{input_file}'))
        
        all_data = []
        for item in data:
            novice_instruction = item['simple_instruction']
            expert_instruction = item['expert_instruction']
            example_id = item['id'] 
            # directory_path = f'examples/IR_matplot_agent/new_opt_runs_test_data_mini_reason'
            directory_path = f'examples/IR_matplot_agent/test_data_mini_query_refine_cot_new_prmopt'

            if not os.path.exists(directory_path):
                os.makedirs(directory_path, exist_ok=True)
            
            input = {
                'query': novice_instruction,
                "directory_path": directory_path,
                "example_id": example_id,
                "input_path": f'{data_path}/data/{example_id}',
            }
            label = {"ground_truth": f"/mnt/ssd4/lm_compiler/examples/IR_matplot_agent/benchmark_data/ground_truth/example_{example_id}.png"}
            all_data.append((input, label))
        return all_data
            
    all_train = load_from_file('train_data.json')
    test_data = load_from_file('test_data.json')
    train_indices = np.random.choice(range(len(all_train)), 40, replace=False).tolist()
    eval_indices = list(set(range(len(all_train))) - set(train_indices))
    
    train_data = [all_train[i] for i in train_indices]
    eval_data = [all_train[i] for i in eval_indices]
    return train_data, eval_data, test_data

def raw_test(data):
    evaluator = EvaluatorPlugin(
        trainset=None,
        evalset=None,
        testset=data,
        n_parallel=20,
    )
    eval_task = EvalTask(
        script_path='/mnt/ssd4/lm_compiler/examples/IR_matplot_agent/workflow.py',
        args=[],
        other_python_paths=[],
        all_params={},
        module_name_paths={},
        aggregated_proposals={},
    )
    result = evaluator.get_score('test', eval_task, show_process=True, show_tqdm_bar=False)
    print(result)
    meta = []
    with open('/mnt/ssd4/lm_compiler/examples/IR_matplot_agent/test_data_mini_query_refine_cot_new_prmopt/raw_test_result.log', 'w+') as f:
        for id, score, price, exec_time, input in zip(result.ids, result.scores, result.prices, result.exec_times, data):
            meta.append({
                'id': id,
                'score': score,
                'price': price,
                'exec_time': exec_time,
                'sample_id': input[0]['example_id'],
            })
        meta.append({
            'reduced score': result.reduced_score,
            'reduced price': result.reduced_price,
            'total eval cost': result.total_eval_cost,
        })
        json.dump(meta, f, indent=4)

def downsample(evaluator: EvaluatorPlugin):
    plain_task = EvalTask(
        script_path='/mnt/ssd4/lm_compiler/examples/IR_matplot_agent/workflow.py',
        args=[],
        other_python_paths=[],
        all_params={},
        module_name_paths={},
        aggregated_proposals={},
    )
    evaluator.down_sample(
        sample_size=40,
        mode='train',
        task=plain_task, 
        sample_mode='difficulty',
        log_dir='/mnt/ssd4/lm_compiler/examples/IR_matplot_agent/down_sample_logs',
    )
    evaluator.down_sample(
        sample_size=10,
        mode='eval',
        task=plain_task, 
        sample_mode='difficulty',
        log_dir='/mnt/ssd4/lm_compiler/examples/HotPotQA/down_sample_logs',
    )

def opt(train, val, test):
    evaluator = EvaluatorPlugin(
        trainset=train,
        evalset=val,
        testset=test,
        n_parallel=40,
    )
    # downsample(evaluator)
    
    lm_options = [
        cognify.LMConfig(
            provider='openai',
            model='gpt-4o-mini',
            cost_indicator=1.0,
            kwargs= {
                'temperature': 0.0,
            }
        ),
        cognify.LMConfig(
            provider='openai',
            model='gpt-4o',
            cost_indicator=1.0,
            kwargs= {
                'temperature': 0.0,
            }
        )
    ]
    model_param = model_selection.LMSelection(
        'lm_model', model_selection.model_option_factory(lm_options)
    )
    reasoning_param = reasoning.LMReasoning(
        "reasoning", [NoChange(), ZeroShotCoT()]
    )
    
    few_shot_params = LMFewShot("few_shot", 2)
    
    inner_opt_config = flow.OptConfig(
        n_trials=0,
        num_parallel_proposal=2,
        log_dir="/mnt/ssd4/lm_compiler/examples/IR_matplot_agent/opt_4omini_reason.log",
        evolve_interval=2,
        frugal_eval_cost=False,
    )
    inner_loop_config = driver.LayerConfig(
        layer_name='inner_loop',
        universal_params=[reasoning_param],
        opt_config=inner_opt_config,
        save_ckpt_interval=1,
        # target_modules=['query expansion', 'initial code generation', 'plot debugger', 'visual refine coder']
    )
    
    opt_driver = driver.MultiLayerOptimizationDriver(
        layer_configs=[inner_loop_config],
        # layer_configs=[outer_loop_config, inner_loop_config],
        quality_constraint=0.51,
    )
    cost, pareto_frontier, opt_logs = opt_driver.run(
        evaluator=evaluator,
        script_path='/mnt/ssd4/lm_compiler/examples/IR_matplot_agent/workflow.py',
    )
    return opt_driver

def get_test_score(opt_driver: driver.MultiLayerOptimizationDriver, test_data):
    result = opt_driver.evaluate(
        bot_trial_log_id='c5fbe93857014673a198a31847c5a5a9',
        opt_log_path='/mnt/ssd4/lm_compiler/examples/IR_matplot_agent/opt_4omini_reason.log/opt_logs.json',
    )
    print(result)
    meta = []
    with open('/mnt/ssd4/lm_compiler/examples/IR_matplot_agent/new_opt_runs_test_data_mini_reason/raw_test_result.log', 'w+') as f:
        for id, score, price, exec_time, input in zip(result.ids, result.scores, result.prices, result.exec_times, test_data):
            meta.append({
                'id': id,
                'score': score,
                'price': price,
                'exec_time': exec_time,
                'sample_id': input[0]['example_id'],
            })
        meta.append({
            'reduced score': result.reduced_score,
            'reduced price': result.reduced_price,
            'total eval cost': result.total_eval_cost,
        })
        json.dump(meta, f, indent=4)

if __name__ == '__main__':
    # mp.set_start_method('spawn')
    
    train, eval, test = load_data()
    print(f'Loaded {len(train)} train, {len(eval)} eval, {len(test)} test data')
    
    np.random.seed(0)
    # opt_driver = opt(train, eval, test)
    # get_test_score(opt_driver, test)
    raw_test(test)