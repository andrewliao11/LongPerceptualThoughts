import os
import re
import gc
import json
import torch
import ast
import math
import random
import pandas as pd
from functools import partial
from collections import Counter
from pathlib import Path
from itertools import islice
from tqdm import tqdm

from utils import (
    infer_template, 
    OpenAICacheClient, 
    MultipleChoicesRandomizer, 
    convert_sft_extended_cot_dataset, 
    extract_options_from_user_prompt, 
    get_unique_id, 
    convert_dpo_extended_cot_dataset, 
    keep_one_per_simple_cot_length_weighted_subsample_df, 
    length_weighted_subsample_df, 
    string_to_seed, 
    register_to_dataset_json
)

SUBSAMPLE_RATIO = 1 / 10
tqdm.pandas()


BAD_WORDS = "description,descriptions,describe,describes,described,mention,mentions,mentioned,misread,text,state,states,stated,say,says,said,internal,mental,visualize,visualization,the image described,user's,detailed image"
SIMPLE_COT_MODEL_NAME = "qwen2.5_vl_instruct"
EXPAND_COT_MODEL_NAME = "r1_distilled_32b"
QWEN2_5_VL_INSTRUCT_PATH = os.environ["QWEN2_5_VL_INSTRUCT_PATH"]
R1_DISTILLED_QWEN_32_B = os.environ["R1_DISTILLED_QWEN_32_B"]

        
def initialize_dataset(config, simple_cot_dataset_name):
    from transformers import Seq2SeqTrainingArguments
    from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
    from llamafactory.hparams import get_infer_args
    from llamafactory.model import load_tokenizer

    # dataset_name = "long_perceptual_thoughts/sft_docci_thought_expansion"
    vllm_config = {
        "gpu_memory_utilization": 0.85,
        "enforce_eager": True,
        "max_model_len": config["max_model_len"],
    }
    vllm_config_str = json.dumps(vllm_config).replace("\"", "\'")
    model_args, data_args, _, _ = get_infer_args(dict(
        cutoff_len=config["cutoff_len"],
        # DO NOT change the following
        model_name_or_path=R1_DISTILLED_QWEN_32_B,
        dataset=simple_cot_dataset_name,
        vllm_config=f"\"{vllm_config_str}\"",
        dataset_dir=os.path.join(os.environ["LLAMAFACTORY_DIR"], "data"),
        template=infer_template(R1_DISTILLED_QWEN_32_B),
        preprocessing_num_workers=8,
        infer_dtype="half",
        trust_remote_code=True,
        # tokenized_path="outputs/tokenized_path/" + dataset_name
    ))
    training_args = Seq2SeqTrainingArguments(output_dir="dummy_dir")
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template_obj = get_template_and_fix_tokenizer(tokenizer, data_args)
    template_obj.mm_plugin.expand_mm_tokens = False  # for vllm generate
    dataset_module = get_dataset(template_obj, model_args, data_args, training_args, "ppo", **tokenizer_module)
    return model_args, data_args, template_obj, dataset_module, tokenizer

    
def initialize_vllm(config, template_obj, tokenizer):
    from llamafactory.extras.misc import get_device_count
    from vllm import LLM, SamplingParams
    
    sampling_kwargs = {
        "n": config["n"],
        "max_tokens": config["max_tokens"],
        "bad_words": config["bad_words"],
        # DO NOT change the following
        "repetition_penalty": 1.05, 
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 50, 
        "stop_token_ids": template_obj.get_stop_token_ids(tokenizer),
        "skip_special_tokens": False,
    }
    sampling_params = SamplingParams(**sampling_kwargs)
    
    # Initialize vllm engine
    engine_args = {
        "max_num_seqs": config["max_num_seqs"],
        "model": R1_DISTILLED_QWEN_32_B,
        "trust_remote_code": True,
        "dtype": "half",
        "tensor_parallel_size": get_device_count() or 1,
        "pipeline_parallel_size": 1, 
        "disable_log_stats": True,
    }
    if template_obj.mm_plugin.__class__.__name__ != "BasePlugin":
        engine_args["limit_mm_per_prompt"] = {"image": 1, "video": 0}
    
    vllm_config = {
        "gpu_memory_utilization": 0.85,
        "enforce_eager": True,
        "max_model_len": config["max_model_len"],
    }
    engine_args.update(vllm_config)
        
    print(f"Engine args: {engine_args}")
    llm = LLM(**engine_args)
    return llm, sampling_params
    

def yield_chunks(dataset, metadata_df, template_obj, tokenizer, chunk_size=50):
    from llamafactory.extras.constants import IGNORE_INDEX
    
    inputs, prompts, labels, metadata = [], [], [], []
    counter = 0
    for sample in tqdm(dataset, desc="Preparing data"):
        inputs.append({"prompt_token_ids": sample["input_ids"], "multi_modal_data": None})
        prompts.append(tokenizer.decode(sample["input_ids"], skip_special_tokens=False))
        labels.append(
            tokenizer.decode(list(filter(lambda x: x != IGNORE_INDEX, sample["labels"])), skip_special_tokens=False)
        )
        metadata_dict = metadata_df.iloc[counter].to_dict()
        metadata.append({
            "metadata": {
                "mcq_unique_id": metadata_dict["mcq_unique_id"],},
            # NOTE: A bit of inconsistent naming here. We'll fix it later
            "simple_cot_unique_id": metadata_dict["simple_cot_unique_id"],
        })
        counter += 1
        if chunk_size > 0 and len(inputs) >= chunk_size:
            yield inputs, prompts, labels, metadata
            inputs, prompts, labels, metadata = [], [], [], []

    if inputs:
        yield inputs, prompts, labels, metadata
        

def generate_extended_cot(cognitive_phrase, simple_cot_dataset_name):
    
    from llamafactory.extras.constants import register_custom_data_config
    register_custom_data_config(str(Path("outputs/long_perceptual_thoughts_dataset_info.json").absolute()))
    
    
    simple_cot_output_path = Path(f'outputs/stage_2_simple_cot/{SIMPLE_COT_MODEL_NAME}.csv')
    df = pd.read_csv(simple_cot_output_path, dtype=str, keep_default_na=False)
    
    # Initialize dataset
    # Only the following config may be changed
    config = {
        "cutoff_len": 2048,
        # vllm
        "max_model_len": 8192,
        # sampling
        "n": 5,
        "max_tokens": 256,
        "max_num_seqs": 16, 
        # chunking
        "chunk_size": 500, 
        # bad words
        "bad_words": BAD_WORDS.split(","),
    }
    
    # Register docci_mcq first
    model_args, data_args, template_obj, dataset_module, tokenizer = initialize_dataset(config, simple_cot_dataset_name)
    llm, sampling_params = initialize_vllm(config, template_obj, tokenizer)
    
    assert len(df) == len(dataset_module["train_dataset"]), f"Data provided in 'outputs/stage_2_simple_cot/{SIMPLE_COT_MODEL_NAME}.csv' \
        does not match the dataset size registered as `long_perceptual_thoughts/sft_docci_all_mcqs`."
    
    # Generate CoT
    intermediate_df_path = Path(f"outputs/stage_3_expand_cot/intermediate_df/{SIMPLE_COT_MODEL_NAME}/{EXPAND_COT_MODEL_NAME}/{cognitive_phrase.replace('|', '')}.jsonl")
    intermediate_df_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(config, open(intermediate_df_path.parent / "config.json", "w"))
    eos_word = tokenizer.special_tokens_map["eos_token"]
    
    n_processed_examples = 0
    if intermediate_df_path.exists():
        n_processed_examples += len(pd.read_json(intermediate_df_path, lines=True))
        
    dataset_module = islice(dataset_module["train_dataset"], n_processed_examples, None)
    for inputs, prompts, labels, metadata in yield_chunks(
        dataset_module, 
        df, 
        template_obj=template_obj, 
        tokenizer=tokenizer, 
        chunk_size=config["chunk_size"]
    ):
        
        # Add precondition words to prompt
        precondition_phrase_list = []
        simple_cot_list = []
        for i, (example, label) in enumerate(zip(inputs, labels)):
            simple_cot = label.split(eos_word)[0]  # EOS of R1-Distilled-Qwen series of models
            precondition_phrase = ""
            if not simple_cot.endswith("."):
                precondition_phrase += "."
                
            if random.random() < 0.5:
                precondition_phrase += "\n"
            else:
                precondition_phrase += " "
                
            precondition_phrase += cognitive_phrase.replace("|", "")
            inputs[i]["prompt_token_ids"] = example["prompt_token_ids"] + tokenizer.encode(simple_cot + precondition_phrase)
            
            precondition_phrase_list.append(precondition_phrase)
            simple_cot_list.append(simple_cot)
        
        # Force thinking
        results = llm.generate(inputs, sampling_params)
        
        # Add `precondition_phrase` back to response
        preds = [[precondition_phrase + o.text for o in result.outputs] for result, precondition_phrase in zip(results, precondition_phrase_list)]
        
        with open(intermediate_df_path, "a", encoding="utf-8") as f:
            decoded_inputs = [tokenizer.decode(i["prompt_token_ids"]) for i in inputs]
            for prompt, pred, simple_cot, meta in zip(prompts, preds, simple_cot_list, metadata):
                f.write(json.dumps({"prompt": prompt, "predict": pred, "simple_cot": simple_cot, "metadata": meta}, ensure_ascii=False) + "\n")
            
        torch.cuda.empty_cache()
        gc.collect()
    

think_answer_pattern = re.compile(r'<think>\s*(.*?)\s*</think>.*?<answer>\s*(.*?)\s*</answer>', re.DOTALL)
def extract_thought_and_answer(example):
    option_in_prompt = extract_options_from_user_prompt(example["_simple_cot_prompt"], example["mcq_question"])
    parsed_choice_list = MultipleChoicesRandomizer.parse_choice_list(option_in_prompt)
    parsed_choice_dict = {c[0]: c[1].lower() for c in parsed_choice_list}
    gt_answer = example["mcq_answer"]
    gt_answer_option = [c[0] for c in parsed_choice_list if c[1] == gt_answer]
    assert len(gt_answer_option) == 1 
        
    gt_answer_option = gt_answer_option[0]
    res = []
    
    for r in example["_expand_cot_raw_response"]:
        
        full_response = "<think> " + example["simple_cot_parsed_thought"] + r
        re_res = think_answer_pattern.findall(full_response)
        if len(re_res) == 0:
            res.append([None, None, None])
        else:
            thought = re_res[0][0].strip()
            answer = re_res[0][1].strip()
            
            hit = False
            if answer.lower() in parsed_choice_dict.values():
                # Answer is shown in text
                hit = True
            else:
                # Answer is show in options
                if answer == gt_answer_option:
                    # Possible answer: A
                    hit = True
                elif answer in [f.format(gt_answer_option) for f in MultipleChoicesRandomizer.answer_formats]:
                    # Possible answer: (A), A.
                    hit = True
                elif answer in [f.format(gt_answer_option, gt_answer) for f in MultipleChoicesRandomizer.choice_formats]:
                    # Possible answer: (A) text, A. text
                    hit = True
                elif answer.split(" ")[0] in [f.format(gt_answer_option) for f in MultipleChoicesRandomizer.answer_formats]:
                    # Possible answer: (A) XXXXX, A. XXXXX
                    hit = True
                    
                if hit:
                    # Fix the answer format based on the format specified in the prompt
                    answer = MultipleChoicesRandomizer.reformat_answer_option(answer, parsed_choice_list, input_prompt=example["_simple_cot_prompt"])
            
            # if not hit:
            #     print(f"GT: {gt_answer_option}, Parsed: {answer}")
            res.append([thought, answer, hit])
        
    return res


def collect_extended_cot(dataset_tag, preprocess_filter_inconsistency=False):
    
    from pandarallel import pandarallel
    pandarallel.initialize(progress_bar=True)

    simple_cot_output_path = Path(f'outputs/stage_2_simple_cot/{SIMPLE_COT_MODEL_NAME}.csv')
    df = pd.read_csv(simple_cot_output_path, keep_default_na=False) 
    df["mcq_messages"] = df["mcq_messages"].parallel_apply(ast.literal_eval)
    
    # Collect all intermediate files
    intermediate_df_paths = Path(f"outputs/stage_3_expand_cot/intermediate_df/{SIMPLE_COT_MODEL_NAME}/{EXPAND_COT_MODEL_NAME}").glob("*.jsonl")
    intermediate_df_list = []
    for p in tqdm(intermediate_df_paths, desc="Loading intermediate files"):
        _df = pd.read_json(p, lines=True)
        _df["cognitive_phrase"] = p.stem #.split("_")[0]
        # NOTE: Only include "Wait," and "Let's double-check the details." as cognitive phrases now
        # if any([w in p.stem.split("_")[0] for w in ["Wait,", "Let's double-check the details."]]):
        intermediate_df_list.append(_df)
        
    intermediate_df = pd.concat(intermediate_df_list, ignore_index=True)
    
    
    intermediate_df["mcq_unique_id"] = intermediate_df["metadata"].apply(lambda x: x["metadata"]["mcq_unique_id"])
    # NOTE: A bit of inconsistent naming here. We'll fix it later
    intermediate_df["simple_cot_unique_id"] = intermediate_df["metadata"].apply(lambda x: x["simple_cot_unique_id"])
    intermediate_df.drop(columns=["metadata", "simple_cot"], inplace=True)
    intermediate_df.rename(columns={
        "prompt": "_expand_cot_prompt",
        "predict": "_expand_cot_raw_response",
    }, inplace=True)
    
    merged_df = pd.merge(df, intermediate_df, on=['mcq_unique_id', 'simple_cot_unique_id'])
    
    merged_df["_thought,answer,hit"] = merged_df.progress_apply(extract_thought_and_answer, axis=1)
    merged_df["extended_cot_parsed_thought"] = merged_df["_thought,answer,hit"].apply(lambda x: [y[0] for y in x])
    merged_df["extended_cot_parsed_answer"] = merged_df["_thought,answer,hit"].apply(lambda x: [y[1] for y in x])
    merged_df["extended_cot_parsed_correct"] = merged_df["_thought,answer,hit"].apply(lambda x: [y[2] for y in x])
    merged_df.drop(columns=["_thought,answer,hit"], inplace=True)
    
    exploded_df = merged_df.explode(["_expand_cot_raw_response", "extended_cot_parsed_thought", "extended_cot_parsed_answer", "extended_cot_parsed_correct"])
    exploded_df = exploded_df[exploded_df.apply(lambda x: x["extended_cot_parsed_thought"] is not None, axis=1)]
    
    
    # Filter out `BAD_WORDS`
    bad_words_list = BAD_WORDS.split(",")
    prev_n = len(exploded_df)
    # NOTE: we are losing 50% of data here! Double check it
    exploded_df["extended_cot_only"] = exploded_df.apply(lambda x: x["extended_cot_parsed_thought"].replace(x["simple_cot_parsed_thought"], ""), axis=1)
    exploded_df = exploded_df[exploded_df["extended_cot_only"].progress_apply(lambda x: not any([w in x for w in bad_words_list]))]
    exploded_df.drop(columns=["extended_cot_only"], inplace=True)
    print(f"Filtered out {prev_n - len(exploded_df)} due to bad words.")
    
    prev_n = len(exploded_df)
    exploded_df.drop_duplicates(subset=["_expand_cot_raw_response"], keep='first', inplace=True)
    print(f"Filtered out {prev_n - len(exploded_df)} duplicates.")
    
    exploded_df["extended_cot_unique_id"] = exploded_df["_expand_cot_raw_response"].apply(get_unique_id)
    exploded_df["extended_cot_parsed_correct"] = exploded_df["extended_cot_parsed_correct"].astype(bool)
    exploded_df["simple_cot_parsed_correct"] = exploded_df["simple_cot_parsed_correct"].astype(bool)
    
    print(f"Collected {len(exploded_df)} examples.")
    print(f"#Unique images: {len(exploded_df['image_id'].unique())}")
    print(f"#Unique MCQs: {len(exploded_df['mcq_unique_id'].unique())}")
    print(f"#Unique Simple CoTs: {len(exploded_df['simple_cot_unique_id'].unique())}")
    print(f"Overall Accuracy: {exploded_df['extended_cot_parsed_correct'].mean():.3f}")

    exploded_df.to_csv(f"outputs/stage_3_expand_cot/{SIMPLE_COT_MODEL_NAME}-{EXPAND_COT_MODEL_NAME}.csv", index=False)
    
    # Create SFT and DPO dataset
    df = exploded_df
    
    if preprocess_filter_inconsistency:
        print("This will use additional API calls to filter out inconsistent thoughts and answers.")
        print("Please make sure you have enough quota.")
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(QWEN2_5_VL_INSTRUCT_PATH)
    
    preprocess_tag = ""
    if preprocess_filter_inconsistency:
        preprocess_tag = f"{preprocess_tag}_filter_inconsistency"


    # SFT dataset
    O_df = df[df["simple_cot_parsed_correct"]].drop_duplicates(subset=["simple_cot_unique_id"], keep='first')
    O_df["extended_cot_parsed_thought"] = O_df["simple_cot_parsed_thought"]
    O_df["extended_cot_parsed_answer"] = O_df["simple_cot_parsed_answer"]
    O_df["extended_cot_parsed_correct"] = O_df["simple_cot_parsed_correct"]
    O_df["extended_cot_unique_id"] = O_df["simple_cot_unique_id"]
    
    O_O_df = df[df["extended_cot_parsed_correct"] & df["simple_cot_parsed_correct"]]
    O_O_df["added_thought"] = O_O_df.apply(lambda x: x["extended_cot_parsed_thought"].replace(x["simple_cot_parsed_thought"], ""), axis=1)
    
    X_O_df = df[df["extended_cot_parsed_correct"] & ~df["simple_cot_parsed_correct"]]
    X_O_df["added_thought"] = X_O_df.apply(lambda x: x["extended_cot_parsed_thought"].replace(x["simple_cot_parsed_thought"], ""), axis=1)
    
    sft_df = _create_sft(None, O_df, O_O_df, X_O_df, "all", preprocess_filter_inconsistency)
    convert_sft_extended_cot_dataset(sft_df, None, f"outputs/sft_docci_{dataset_tag}_extended_cots.json") 
    
    register_to_dataset_json({
        f"long_perceptual_thoughts_stage_3/{dataset_tag}": str(Path(f"outputs/sft_docci_{dataset_tag}_extended_cots.json").absolute())
    }, contain_image=True)
    
    # DPO dataset
    if preprocess_filter_inconsistency:
        # Filter X -> O
        X_O_df = df[df["extended_cot_parsed_correct"] & ~df["simple_cot_parsed_correct"]]
        rest_df = df[df["simple_cot_parsed_correct"] | (~df["extended_cot_parsed_correct"] & ~df["simple_cot_parsed_correct"])]
        p_negative = Path("outputs/stage_3_expand_cot/inconsistent_thought_and_answer.csv")
        p_positive = Path("outputs/stage_3_expand_cot/consistent_thought_and_answer.csv")
        assert p_negative.exists() and p_positive.exists(), f"Please run `filter_inconsistent_thought_and_answer` first."
        inconsistent_thought_and_answer = pd.read_csv(str(p_negative))
        consistent_thought_and_answer = pd.read_csv(str(p_positive))
        consistent_thought_and_answer["valid"] = True
        X_O_df = X_O_df.merge(consistent_thought_and_answer, on="extended_cot_unique_id", how="left")
        X_O_df = X_O_df[X_O_df["valid"] == True]
        X_O_df.drop(columns=["valid"], inplace=True)
        
        df = pd.concat([X_O_df, rest_df], ignore_index=True)
    
    
    _, dpo_O_better_than_O_X_df, dpo_X_O_better_than_X_and_any_other_df, dpo_O_better_than_X_and_any_other_df = _create_dpo_dataset(df)

    # remove redundancy
    dpo_O_better_than_O_X_df.sample(frac=1., random_state=42).reset_index(drop=True)
    dpo_O_better_than_X_and_any_other_df.sample(frac=1., random_state=42).reset_index(drop=True)
    dpo_X_O_better_than_X_and_any_other_df.sample(frac=1., random_state=42).reset_index(drop=True)
    
    dpo_O = pd.concat([dpo_O_better_than_O_X_df, dpo_O_better_than_X_and_any_other_df], ignore_index=True)
    dpo_O.drop_duplicates(subset=["positive"], keep='first', inplace=True)
    
    dpo_X_O_better_than_X_and_any_other_df.drop_duplicates(subset=["positive"], keep='first', inplace=True)
    dpo_X_O_better_than_X_and_any_other_df.drop_duplicates(subset=["negative"], keep='first', inplace=True)
    dpo_X_O = dpo_X_O_better_than_X_and_any_other_df

    print(f"Collected {len(dpo_O)} DPO examples for O")
    print(f"Collected {len(dpo_X_O)} DPO examples for X O")
    convert_dpo_extended_cot_dataset(dpo_O, None, f"outputs/dpo_docci_{dataset_tag}_extended_cots_O.json")
    convert_dpo_extended_cot_dataset(dpo_X_O, None, f"outputs/dpo_docci_{dataset_tag}_extended_cots_X_O.json")
    
    
def _create_dpo_dataset(df):
    
    simple_cot_keys = ["simple_cot_parsed_thought", "simple_cot_parsed_answer", "simple_cot_parsed_correct"]
    extended_cot_keys = ["extended_cot_parsed_thought", "extended_cot_parsed_answer", "extended_cot_parsed_correct"]
    
    dpo_O_better_than_O_O = []                          # Compactness, ensure that the length of negative examples is 1.5 longer than positive examples
    dpo_O_better_than_O_X = []                          # Correctness
    dpo_X_O_better_than_X_and_any_other = []            # Correctness
    
    grouped = df.groupby(["simple_cot_unique_id"])
    unique_groups = grouped.groups.keys()
    
    for group in tqdm(unique_groups, desc="Processing groups for DPO data"):
        group_df = grouped.get_group(group).copy()
        
        mcq_unique_id = group_df["mcq_unique_id"].iloc[0]
        image_id = group_df["image_id"].iloc[0]
        image_path = group_df["image_path"].iloc[0]
        mcq_messages = group_df["mcq_messages"].iloc[0]
        
        simple_cot_data = group_df[simple_cot_keys].iloc[0]
        extended_cot_data = group_df[extended_cot_keys]
        correct_extended_cot_data = extended_cot_data[extended_cot_data["extended_cot_parsed_correct"]]
        
        wrong_extended_cot_data = extended_cot_data[~extended_cot_data["extended_cot_parsed_correct"]]
        
        if simple_cot_data["simple_cot_parsed_correct"]:
            # Simple CoT is correct
            if len(correct_extended_cot_data) > 0:
                dpo_O_better_than_O_O.extend([{
                    "image_path": image_path, 
                    "mcq_messages": mcq_messages, 
                    "mcq_unique_id": mcq_unique_id, 
                    "image_id": image_id,
                    "positive": (simple_cot_data["simple_cot_parsed_thought"], simple_cot_data["simple_cot_parsed_answer"]), 
                    "negative": (d["extended_cot_parsed_thought"], d["extended_cot_parsed_answer"])
                } for _, d in correct_extended_cot_data.iterrows()])
            if len(wrong_extended_cot_data) > 0:
                dpo_O_better_than_O_X.extend([{
                    "image_path": image_path, 
                    "mcq_messages": mcq_messages,  
                    "mcq_unique_id": mcq_unique_id, 
                    "image_id": image_id,
                    "positive": (simple_cot_data["simple_cot_parsed_thought"], simple_cot_data["simple_cot_parsed_answer"]), 
                    "negative": (d["extended_cot_parsed_thought"], d["extended_cot_parsed_answer"])
                } for _, d in wrong_extended_cot_data.iterrows()])
                
        elif len(correct_extended_cot_data) > 0:
            if len(wrong_extended_cot_data) > 0:
                dpo_X_O_better_than_X_and_any_other.extend([{
                    "image_path": image_path, 
                    "mcq_messages": mcq_messages,  
                    "mcq_unique_id": mcq_unique_id, 
                    "image_id": image_id,
                    "positive": (d1["extended_cot_parsed_thought"], d1["extended_cot_parsed_answer"]),
                    "negative": (d2["extended_cot_parsed_thought"], d2["extended_cot_parsed_answer"])
                } for _, d1 in correct_extended_cot_data.iterrows() for _, d2 in wrong_extended_cot_data.iterrows()])
            
            dpo_X_O_better_than_X_and_any_other.extend([{
                "image_path": image_path, 
                "mcq_messages": mcq_messages, 
                "mcq_unique_id": mcq_unique_id, 
                "image_id": image_id,
                "positive": (d1["extended_cot_parsed_thought"], d1["extended_cot_parsed_answer"]),
                "negative": (simple_cot_data["simple_cot_parsed_thought"], simple_cot_data["simple_cot_parsed_answer"])
            } for _, d1 in correct_extended_cot_data.iterrows()])
    
    
    dpo_O_better_than_X_and_any_other = []              # Correctness
    
    grouped = df.groupby(["mcq_unique_id"])
    unique_groups = grouped.groups.keys()
    for group in tqdm(unique_groups, desc="Processing groups for DPO data"):
        group_df = grouped.get_group(group).copy()
        
        mcq_unique_id = group_df["mcq_unique_id"].iloc[0]
        image_id = group_df["image_id"].iloc[0]
        image_path = group_df["image_path"].iloc[0]
        mcq_messages = group_df["mcq_messages"].iloc[0]
        
        
        simple_cot_data = group_df[simple_cot_keys]
        simple_cot_correctness = simple_cot_data["simple_cot_parsed_correct"]
        correct_simple_cot_data = simple_cot_data[simple_cot_data["simple_cot_parsed_correct"]]
        wrong_simple_cot_data = simple_cot_data[~simple_cot_data["simple_cot_parsed_correct"]]
        
        if simple_cot_correctness.sum() > 0 and (~simple_cot_correctness).sum() > 0:
            dpo_O_better_than_X_and_any_other.extend([{
                "image_path": image_path, 
                "mcq_messages": mcq_messages, 
                "mcq_unique_id": mcq_unique_id, 
                "image_id": image_id,
                "positive": (d1["simple_cot_parsed_thought"], d1["simple_cot_parsed_answer"]),
                "negative": (d2["simple_cot_parsed_thought"], d2["simple_cot_parsed_answer"])
            } for _, d1 in correct_simple_cot_data.iterrows() for _, d2 in wrong_simple_cot_data.iterrows()])
            
            wrong_simple_and_extended_cot_data = group_df[~group_df["extended_cot_parsed_correct"] & ~group_df["simple_cot_parsed_correct"]]
            dpo_O_better_than_X_and_any_other.extend([{
                "image_path": image_path, 
                "mcq_messages": mcq_messages, 
                "mcq_unique_id": mcq_unique_id, 
                "image_id": image_id,
                "positive": (d1["simple_cot_parsed_thought"], d1["simple_cot_parsed_answer"]),
                "negative": (d2["extended_cot_parsed_thought"], d2["extended_cot_parsed_answer"])
            } for _, d1 in correct_simple_cot_data.iterrows() for _, d2 in wrong_simple_and_extended_cot_data.iterrows()])
            

    dpo_O_better_than_O_O_df = pd.DataFrame.from_dict(dpo_O_better_than_O_O)
    dpo_O_better_than_O_X_df = pd.DataFrame.from_dict(dpo_O_better_than_O_X)
    dpo_X_O_better_than_X_and_any_other_df = pd.DataFrame.from_dict(dpo_X_O_better_than_X_and_any_other)
    dpo_O_better_than_X_and_any_other_df = pd.DataFrame.from_dict(dpo_O_better_than_X_and_any_other)
    
    print(f"Collected {len(dpo_O_better_than_O_O_df)} DPO examples for O > O O.")
    print(f"Collected {len(dpo_O_better_than_O_X_df)} DPO examples for O > O X.")
    print(f"Collected {len(dpo_X_O_better_than_X_and_any_other_df)} DPO examples for X O > X or X X.")
    print(f"Collected {len(dpo_O_better_than_X_and_any_other_df)} DPO examples for O > X or X X.")
    return dpo_O_better_than_O_O_df, dpo_O_better_than_O_X_df, dpo_X_O_better_than_X_and_any_other_df, dpo_O_better_than_X_and_any_other_df


def filter_inconsistent_thought_and_answer(df):
    
    openai_client = {}
    model_id = 'gpt-4o-mini-2024-07-18'
    CACHE_DIR = Path('outputs/stage_3_expand_cot/.cache') / model_id
    openai_client = OpenAICacheClient(model_id=model_id, cache_dir=CACHE_DIR, force_use_cache=False, verbose=True)
    
    system_prompt = """You are good at identifying inconsistencies between text."""
    
    def _build_filter_prompt(example):
        question = example["mcq_messages"][1]["content"].replace("<image>", "")
        
        answer_pattern = re.compile(r'<answer>\s*(.*?)\s*</answer>', re.DOTALL)
        answer = answer_pattern.findall(example["mcq_messages"][2]["content"])[0]
        reflection = example["extended_cot_parsed_thought"].replace(example["simple_cot_parsed_thought"], "")
        
        prompt = """You will be given a visual question, its answer, and the reflection on the initial thought. The answer is always correct, but the reflection may sometimes be inconsistent with the answer.

You will check if the reflection is consistent with the answer by following these steps: 
Step 1. Understand the question and the answer
Step 2. Derive the answer solely from the reflection text
Step 3. Check consistency between answer from the reflection and the provided answer
Output Requirement: At the end, answer \\boxed{Yes} if the Reflection is consistent with the Answer; otherwise, answer \\boxed{No}.

---"""

        prompt += f"""
# Question:
{question}

Answer: {answer}

# Reflection on the initial thought
Reflection: ... {' '.join(reflection.split(' ')[-30:])}

--- 

Please think step by step.""" 
        return prompt
        
    def parse_filter_res(example):
        responses = example["_filter_raw_response"]
        try:
            yes_or_no = [re.search(r"boxed\{(.*?)\}", r.lower(), re.DOTALL).group(1) for r in responses]
            yes_or_no = ["yes" in i for i in yes_or_no]
            yes_or_no = Counter(yes_or_no).most_common(1)[0][0]
            return yes_or_no
        except Exception:
            return True
        
        
    df["_filter_prompt"] = df.progress_apply(_build_filter_prompt, axis=1)
    prompts = df["_filter_prompt"].tolist()
    system_prompts = [system_prompt] * len(prompts)
    batch_responses = openai_client.query_openai_all(prompts, system_prompts, batch_size=500, max_concurrent=100, temperature=0.7, n=1)

    df["_filter_raw_response"] = batch_responses
    df["filter_result"] = df.progress_apply(parse_filter_res, axis=1)
    
    return df
    

def _create_sft(image_list, O_df, O_O_df, X_O_df, image_list_tag, preprocess_filter_inconsistency=False):
    if image_list is not None:
        sampled_O_df = O_df[O_df["image_id"].apply(lambda x: x in image_list)]
        sampled_O_O_df = O_O_df[O_O_df["image_id"].apply(lambda x: x in image_list)]
        sampled_X_O_df = X_O_df[X_O_df["image_id"].apply(lambda x: x in image_list)]
    else:
        sampled_O_df = O_df
        sampled_O_O_df = O_O_df
        sampled_X_O_df = X_O_df

            
    sampled_O_any_df = pd.concat([sampled_O_df, sampled_O_O_df], ignore_index=True)
    sampled_O_any_df.sample(frac=1., random_state=42).reset_index(drop=True)
    sampled_O_any_df = length_weighted_subsample_df(sampled_O_any_df, int(len(sampled_O_any_df) * SUBSAMPLE_RATIO), "extended_cot_parsed_thought", group_key="mcq_unique_id")
    sampled_X_O_df = length_weighted_subsample_df(sampled_X_O_df, int(len(sampled_O_any_df)), "added_thought", group_key="mcq_unique_id")
    if preprocess_filter_inconsistency:
        p_negative = Path("outputs/stage_3_expand_cot/inconsistent_thought_and_answer.csv")
        if p_negative.exists():
            negative_examples = pd.read_csv(str(p_negative))
            sampled_X_O_df["filter_result"] = sampled_X_O_df["extended_cot_unique_id"].apply(lambda x: x not in negative_examples["extended_cot_unique_id"].values)
            print(f"Remove {sum(~sampled_X_O_df['filter_result'])} examples due to inconsistent thought and snwer")
            sampled_X_O_df = sampled_X_O_df[sampled_X_O_df["filter_result"]]
            
        filter_inconsistent_thought_and_answer(sampled_X_O_df)
        inconsistent_thought_and_answer = sampled_X_O_df[~sampled_X_O_df["filter_result"]][["extended_cot_unique_id"]]
        consistent_thought_and_answer = sampled_X_O_df[sampled_X_O_df["filter_result"]][["extended_cot_unique_id"]]
        sampled_X_O_df = sampled_X_O_df[sampled_X_O_df["filter_result"]]
        
        print(f"Remove {len(inconsistent_thought_and_answer)} examples due to inconsistent thought and snwer")
        p_negative = Path("outputs/stage_3_expand_cot/inconsistent_thought_and_answer.csv")
        if p_negative.exists():
            negative_examples = pd.read_csv(str(p_negative))
            inconsistent_thought_and_answer = pd.concat([negative_examples, inconsistent_thought_and_answer], ignore_index=True)
            inconsistent_thought_and_answer.drop_duplicates(subset=["extended_cot_unique_id"], keep='first', inplace=True)
            
        p_positive = Path("outputs/stage_3_expand_cot/consistent_thought_and_answer.csv")
        if p_positive.exists():
            positive_examples = pd.read_csv(str(p_positive))
            consistent_thought_and_answer = pd.concat([positive_examples, consistent_thought_and_answer], ignore_index=True)
            consistent_thought_and_answer.drop_duplicates(subset=["extended_cot_unique_id"], keep='first', inplace=True)
        
        
        inconsistent_thought_and_answer.to_csv(str(p_negative), index=False)
        consistent_thought_and_answer.to_csv(str(p_positive), index=False)

    combined_df = pd.concat([sampled_X_O_df, sampled_O_any_df], ignore_index=True)
    
    print(f"Size of O any: {len(sampled_O_any_df)}")
    print(f"Size of X O: {len(sampled_X_O_df)}")
    return combined_df
    