import json
import os
import sys

import numpy as np
from fire import Fire
from tqdm import tqdm

from algorithms import *
from benchmarks import *
from models import *


LONGBENCH_TASKS = [
    "narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
            "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
            "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"
]

def main(
    model_name: str = 'together_ai/meta-llama/Llama-3-8b-chat-hf',
    log_dir: str = './benchmarks/results/',
    dataset: str = 'gsm8k',
    shots: int = 3,
    algo_name: str = 'cot'
):
    log_model_name = model_name.split('/')[-1].replace("[^a-zA-Z0-9\\.\\-]", "_")
    log_path = os.path.join(log_dir, dataset)
    os.makedirs(log_path, exist_ok=True)
    log_file = log_path + f'/{dataset}_{algo_name}_{log_model_name}.json'
    print("==> log will be saved to:", log_file)
    
    print('==> initializing model and algorithm...')

    if algo_name == "rpmad":
        model = [
            closesource_API_models(model=model_name, max_tokens=500) for _ in range(4)
        ]
    else:
        if os.path.exists(model_name):
            model = opensource_local_models(model=model_name, max_tokens=500)
        elif model_name.startswith("together_ai/"):
            model = opensource_API_models(model=model_name.split("together_ai/")[-1], max_tokens=500)
        else:
            model = closesource_API_models(model=model_name, max_tokens=500)
            
    if algo_name == 'react':
        algorithm = ReAct(method=algo_name)
    elif algo_name == 'rpmad':
        algorithm = RPMAD(method=algo_name)
    elif algo_name == 'reconcile':
        # TODO: check if the model names are the same
        model_names = model_name.split(',')
        models = []
        for model_name in model_names:
            if os.path.exists(model_name):
                model = opensource_local_models(model=model_name, max_tokens=500)
            elif model_name.startswith("together_ai/"):
                model = opensource_API_models(model=model_name.split("together_ai/")[-1], max_tokens=500)
            else:
                model = closesource_API_models(model=model_name, max_tokens=500)
            models.append(model)
        
        algorithm = ReConcile(method=algo_name, output_type='weighted_max')
        algorithm.set_models(models)
        algorithm.set_dataset(dataset)
    elif algo_name.lower() == 'coa':
        algorithm = CoA(method=algo_name)
        algorithm.set_dataset(dataset)
    else:
        algorithm = Prompt(method=algo_name)
    algorithm.set_model(model)
    print('==> Done.')

    print('==> initializing dataset...')
    dataset_class = get_data_class(dataset)
    dataloader = dataset_class.get_dataset()
    question_template = dataset_class.get_question_template()
    algorithm.set_question_template(question_template)
    examples = dataset_class.get_example(shots=shots, algorithm=algorithm.prompt_name)
    algorithm.set_example(examples)
    print('==> Done.')
    
    print('==> doing evaluation...')
    # TODO: continue evaluation after interruption
    # TODO: parallel reasoning to boost evaluation    
    # TODO: support wandb logging
    # raw_predictions, extracted_answers, true_answers = [], [], []
    logs = []
    for q_idx, data in tqdm(enumerate(dataloader), ncols=50, leave=False): 
        # parse data
        context = None
        if dataset=="BBH":
            question = data['input']
            true_ans = data['target']

        elif dataset in LONGBENCH_TASKS:
            question = data['input']
            context = data['context']
            true_ans = data['answers'][0]

        else:
            question = data['question']
            true_ans = data['answer'].split("####")[-1] # ! TODO: this is specialize for gsm8k
        
        # do reasoning, and get the raw prediction
        raw_pred = algorithm.do_reasoning(question, context=context)
        # extract the answer from the raw prediction
        ext_ans = answer_cleansing(dataset_name=dataset, pred=raw_pred)
        # compute evaluation metrics in dataset
        eval_res = dataset_class.evaluation(ext_ans, true_ans)
        
        # store the prediction to local log file
        logs.append({
            "index": q_idx,
            "question": question,
            "raw prediction": raw_pred,
            "extracted answer": ext_ans,
            "true answer": true_ans,
            "evaluation": eval_res
        })
        
        # update log
        with open(log_file, 'w+') as f:
            json.dump(logs, f)
        
        # TODO: remove this
        if len(logs) >= 100:
            print('\n==> [TMP] logs:', logs)
            break
    
    print('==> Done.')
    
    # TODO: compute the final metrisc with logs
    final_metrics = compute_logs_metrics(logs)
    print(f'==> The final evaluation results:\n {final_metrics}')
    
if __name__ == "__main__":
    Fire(main)
