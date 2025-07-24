import os
import re
import json
import numpy as np
import pandas as pd

from pathlib import Path
from collections import Counter
from datetime import datetime
from jinja2.sandbox import SandboxedEnvironment
from utils import OpenAICacheClient, get_unique_id, convert_sft_mcq_dataset, DOCCI, MultipleChoicesRandomizer, register_to_dataset_json


from tqdm import tqdm
tqdm.pandas()

force_use_cache = False
gpt_4o_mini_model_id = 'gpt-4o-mini-2024-07-18'
gpt_4o_model_id = 'gpt-4o-2024-08-06'


def check_MCQ_valid(example):
    parsed_choice_list = MultipleChoicesRandomizer.parse_choice_list(example["mcq_choices"])
    gt_answer = example["mcq_answer"]
    gt_answer_option = [c[0] for c in parsed_choice_list if c[1] == gt_answer]
    return len(gt_answer_option) == 1

            
def generate_mcq_from_captions(max_examples=1000, verbose=False):
    env = SandboxedEnvironment()
    mcq_gen_prompt_template = env.from_string(open("stage_1_mcq_gen/templates/caption_to_mcq.jinja2").read())
    parse_to_mcq_prompt_template = env.from_string(open("stage_1_mcq_gen/templates/parse_to_mcq.jinja2").read())

    openai_client = {}
    for model_name, model_id in zip(['gpt-4o-mini', 'gpt-4o'], [gpt_4o_mini_model_id, gpt_4o_model_id]):
        CACHE_DIR = Path('outputs/stage_1_mcq_gen/.cache') / model_id
        CACHE_DIR.mkdir(exist_ok=True, parents=True)
        openai_client[model_name] = OpenAICacheClient(model_id=model_id, cache_dir=CACHE_DIR, force_use_cache=force_use_cache, verbose=verbose)
        
        
    # Construct prompt
    def _construct_mcq_gen_prompt(example):
        prompt = mcq_gen_prompt_template.render(image_description=example["description"])
        return prompt
    
    def _construct_parse_to_mcq_prompt(example):
        prompt = parse_to_mcq_prompt_template.render(response=example["_mcq_gen_raw_response"])
        return prompt
    
    def parse_responses(example):
        res_list = re.findall(r'<question>\s*(.*?)\s*</question>.*?<choices>\s*(.*?)\s*</choices>.*?<answer>\s*(.*?)\s*</answer>.*?<type>(.*?)</type>', res, re.DOTALL)
        
        
    dataset = DOCCI('train', max_examples=max_examples)
    image_id_list = sorted(dataset.df["image_id"].unique())
    
    import random
    random.seed(42)
    random.shuffle(image_id_list)
    
    # Generate MCQs
    system_prompt = "You are a helpful assistant good at converting image descriptions to multi-choice visual questions."
    dataset.df["_mcq_gen_model_id"] = gpt_4o_model_id
    dataset.df["_mcq_gen_prompt"] = dataset.df.progress_apply(_construct_mcq_gen_prompt, axis=1)
    dataset.df["_mcq_gen_raw_response"] = dataset.df.progress_apply(lambda x: openai_client['gpt-4o'].query_openai(x["_mcq_gen_prompt"], system_prompt, temperature=0.7, n=1)[0], axis=1)
    
    # Parse to MCQs
    system_prompt = "You are a helpful assistant good at converting parsing text contain several multiple-choice questions into the specified format."
    dataset.df["_parse_to_mcq_model_id"] = gpt_4o_mini_model_id
    dataset.df["_parse_to_mcq_prompt"] = dataset.df.progress_apply(_construct_parse_to_mcq_prompt, axis=1)
    dataset.df["_parse_to_mcq_response"] = dataset.df.progress_apply(lambda x: openai_client['gpt-4o-mini'].query_openai(x["_parse_to_mcq_prompt"], system_prompt, temperature=0., n=1)[0], axis=1)
    dataset.df["_temp"] = dataset.df.progress_apply(lambda x: re.findall(r'<question>\s*(.*?)\s*</question>.*?<choices>\s*(.*?)\s*</choices>.*?<answer>\s*(.*?)\s*</answer>.*?<type>(.*?)</type>', x['_parse_to_mcq_response'], re.DOTALL), axis=1)
    print('#mcqs per image (#mcqs, count):', Counter(dataset.df["_temp"].apply(lambda x: len(x))).most_common())
    
    dataset.df = dataset.df.explode("_temp")
    
    
    from pandarallel import pandarallel
    pandarallel.initialize(progress_bar=True)
    dataset.df["mcq_question"] = dataset.df.parallel_apply(lambda x: x["_temp"][0].strip(), axis=1)
    dataset.df["mcq_choices"] = dataset.df.parallel_apply(lambda x: x["_temp"][1].strip(), axis=1)
    dataset.df["mcq_answer"] = dataset.df.parallel_apply(lambda x: x["_temp"][2].strip(), axis=1)
    dataset.df["mcq_question_type"] = dataset.df.parallel_apply(lambda x: x["_temp"][3].strip(), axis=1)
    dataset.df.drop(columns=["_temp"], inplace=True)
    
    print('Distribution of the question types:', Counter(dataset.df["mcq_question_type"]).most_common())
    
    dataset.df["mcq_unique_id"] = dataset.df.progress_apply(lambda x: get_unique_id(str((x["mcq_question"], x["mcq_choices"], x["mcq_answer"], x["mcq_question_type"])), image_path=x["image_path"]), axis=1)
    
    # Usually around 1 - 1.5 % of data is invalid
    dataset.df["mcq_is_valid"] = dataset.df.progress_apply(check_MCQ_valid, axis=1)
    dataset.df = dataset.df[dataset.df["mcq_is_valid"]]
    
    Path("outputs").mkdir(exist_ok=True)
    dataset.df.to_csv('outputs/docci_mcq.csv', index=False)
    
    # Create MCQs
    dataset_info = {}
    tag = f"{len(image_id_list)}_images"
    convert_sft_mcq_dataset(dataset.df, f"outputs/sft_docci_{tag}_mcqs.json")
    convert_sft_mcq_dataset(dataset.df, f"outputs/sft_direct_answer_docci_{tag}_mcqs.json", direct_answer=True)
    
    dataset_info[f"long_perceptual_thoughts_stage_1/{tag}"] = str(Path(f"outputs/sft_docci_{tag}_mcqs.json").absolute())
    dataset_info[f"long_perceptual_thoughts_stage_1_direct_answer/{tag}"] = str(Path(f"outputs/sft_direct_answer_docci_{tag}_mcqs.json").absolute())
    
    # Register to a json file so that llama-factory could see it
    register_to_dataset_json(dataset_info)
        