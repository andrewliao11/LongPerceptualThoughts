import os
import re
import ast
import json
import fire
import pandas as pd
from pathlib import Path
from itertools import islice
from copy import deepcopy
from tqdm import tqdm
from collections import Counter
from data_gen.utils import MultipleChoicesRandomizer
from llamafactory.data.parser import get_dataset_list


def initialize_dataset(dataset_name, model_path, config, use_tokenized_dataset=True):
    from transformers import Seq2SeqTrainingArguments
    from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
    from llamafactory.hparams import get_infer_args
    from llamafactory.model import load_tokenizer

    vllm_config = {
        "gpu_memory_utilization": 0.85,
        "enforce_eager": True,
        "max_model_len": config["max_model_len"],
    }
    vllm_config_str = json.dumps(vllm_config).replace("\"", "\'")
    model_args, data_args, _, _ = get_infer_args(dict(
        image_max_pixels=config["image_max_pixels"],
        image_min_pixels=config["image_min_pixels"],
        cutoff_len=config["cutoff_len"],
        # DO NOT change the following
        model_name_or_path=model_path,
        dataset=dataset_name,
        vllm_config=f"\"{vllm_config_str}\"",
        dataset_dir=os.path.join(os.environ["LLAMAFACTORY_DIR"], "data"),
        template="qwen2_vl",
        preprocessing_num_workers=8,
        infer_dtype="half",
        trust_remote_code=True,
        tokenized_path="benchmark_data/outputs/tokenized_path/" + dataset_name if use_tokenized_dataset else None
    ))
    training_args = Seq2SeqTrainingArguments(output_dir="dummy_dir")
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template_obj = get_template_and_fix_tokenizer(tokenizer, data_args)
    template_obj.mm_plugin.expand_mm_tokens = False  # for vllm generate
    dataset_module = get_dataset(template_obj, model_args, data_args, training_args, "ppo", **tokenizer_module)
    return model_args, data_args, template_obj, dataset_module, tokenizer

    
def initialize_vllm(config, model_path, template_obj, tokenizer):
    from llamafactory.extras.misc import get_device_count
    from vllm import LLM, SamplingParams
    
    sampling_kwargs = {
        "n": config["n"],
        "max_tokens": config["max_tokens"],
        # DO NOT change the following
        "repetition_penalty": config["repetition_penalty"], 
        "temperature": config["temperature"],
        "top_p": config["top_p"], 
        "top_k": config["top_k"],
        "stop_token_ids": template_obj.get_stop_token_ids(tokenizer),
        "skip_special_tokens": False,
    }
    sampling_params = SamplingParams(**sampling_kwargs)
    
    # Initialize vllm engine
    engine_args = {
        "max_num_seqs": config["max_num_seqs"],
        "model": model_path,
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
    

def yield_chunks(dataset, registered_df, template_obj, tokenizer, image_max_pixels, image_min_pixels, chunk_size=50):
    from llamafactory.extras.constants import IGNORE_INDEX
    
    inputs, prompts, labels, metadata = [], [], [], []
    counter = 0
    for sample in tqdm(dataset, desc="Preparing data"):

        if sample["images"]:
            multi_modal_data = {
                "image": template_obj.mm_plugin._regularize_images(sample["images"], 
                                                                   image_max_pixels=image_max_pixels, 
                                                                   image_min_pixels=image_min_pixels)["images"]
            }
        else:
            multi_modal_data = None

        inputs.append({"prompt_token_ids": sample["input_ids"], "multi_modal_data": multi_modal_data})
        prompts.append(tokenizer.decode(sample["input_ids"], skip_special_tokens=False))
        labels.append(
            tokenizer.decode(list(filter(lambda x: x != IGNORE_INDEX, sample["labels"])), skip_special_tokens=False)
        )
        metadata.append({
            "index": registered_df.iloc[counter].to_dict().get("index", counter)
        })
        counter += 1
        if chunk_size > 0 and len(inputs) >= chunk_size:
            yield inputs, prompts, labels, metadata
            inputs, prompts, labels, metadata = [], [], [], []

    if inputs:
        yield inputs, prompts, labels, metadata
        

def predict_and_eval(
    model_path: str,
    eval_dataset: str,
    prediction_dir: str,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    n_samples: int,
    force_thinking: bool = False, 
    do_eval: bool = True,
    use_tokenized_dataset: bool = True,
):
    
    from llamafactory.extras.constants import register_custom_data_config
    register_custom_data_config(str(Path("benchmark_data/benchmark_dataset_info.json").absolute()))
    
    Path(prediction_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize dataset
    # Only the following config may be changed
    config = {
        "image_max_pixels": 512*512,
        "image_min_pixels": 16*16,
        "cutoff_len": 8192, 
        # vllm
        "max_model_len": 8192,
        "max_tokens": 2048,
        "max_num_seqs": 256,        # NOTE: tune this if you have OOM issues
        # sampling
        "n": n_samples,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
        # chunking
        "chunk_size": 500
    }
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    thinking_bos_token = tokenizer.encode("<think>")
    
    if isinstance(eval_dataset, str):
        if "," in eval_dataset:
            eval_dataset = eval_dataset.split(",")
        else:
            eval_dataset = (eval_dataset, )
    
    model_args, data_args, template_obj, dataset_module, tokenizer = initialize_dataset(eval_dataset[0], model_path, config, use_tokenized_dataset=use_tokenized_dataset)
    llm, sampling_params = initialize_vllm(config, model_path, template_obj, tokenizer)
    
    for single_eval_dataset in eval_dataset:
        prediction_path = os.path.join(prediction_dir, f"predictions_{single_eval_dataset}.jsonl")
        Path(prediction_path).parent.mkdir(parents=True, exist_ok=True)
        
        _, _, _, dataset_module, _ = initialize_dataset(single_eval_dataset, model_path, config, use_tokenized_dataset=use_tokenized_dataset)
        
        n_processed_examples = 0
        if os.path.exists(prediction_path):
            df = pd.read_json(prediction_path, lines=True)
            n_processed_examples += len(df)
            
        
        json_filename = str(get_dataset_list([single_eval_dataset], os.path.join(os.environ["LLAMAFACTORY_DIR"], "data"))[0])
        registered_df = pd.read_json(json_filename)
                    
        # Run predictions
        dataset_module["train_dataset"] = islice(dataset_module["train_dataset"], n_processed_examples, None)
        registered_df = registered_df.iloc[n_processed_examples:]
        predictions = []
        for inputs, prompts, labels, metadata in yield_chunks(
            dataset_module["train_dataset"], 
            registered_df, 
            template_obj=template_obj, 
            tokenizer=tokenizer, 
            image_max_pixels=config["image_max_pixels"], 
            image_min_pixels=config["image_min_pixels"],
            chunk_size=config["chunk_size"]
        ):
        
            if force_thinking:
                for i, example in enumerate(inputs):
                    inputs[i]["prompt_token_ids"] = example["prompt_token_ids"] + thinking_bos_token
            
            results = llm.generate(inputs, sampling_params)
            
            if force_thinking:
                preds = [["<think>" + o.text for o in result.outputs] for result in results]
            else:
                preds = [[o.text for o in result.outputs] for result in results]
            
            # Dump the predictions as well as the index
            with open(prediction_path, "a", encoding="utf-8") as f:
                for prompt, pred, label, meta in zip(prompts, preds, labels, metadata):
                    f.write(json.dumps({"prompt": prompt, "predict": pred, "label": label, "index": meta["index"]}, ensure_ascii=False) + "\n")


        if do_eval:
            try:
                benchmark_df_path = os.path.join("./benchmark_data/outputs/tsv_files", single_eval_dataset.replace("benchmark_sampled_", "").replace("benchmark_", "").replace("direct_answer_", "") + ".tsv")
                print(benchmark_df_path)
                benchmark_df = pd.read_csv(benchmark_df_path, sep="\t")
                benchmark_df["choices"] = benchmark_df["choices"].apply(ast.literal_eval)
                pred_df = pd.read_json(prediction_path, lines=True)
                pred_df.drop(columns=["prompt", "label"], inplace=True)
                pred_df = pd.merge(pred_df, benchmark_df, on="index")
                
                pred_df["parsed_predict"] = pred_df["predict"].apply(extract_predict)
                pred_df["agg_parsed_predict"] = pred_df["parsed_predict"].apply(lambda x: Counter([i[0] if len(i) > 0 else "" for i in x]).most_common(1)[0][0])
                pred_df["hit"] = pred_df.apply(compute_mcq_hit, axis=1)
                
                # group
                grouped = pred_df.groupby(["category"])
                unique_groups = list(grouped.groups.keys())

                print(f"Dataset: {single_eval_dataset}")
                group_acc = []
                for group in unique_groups:
                    group_df = grouped.get_group(group).copy()
                    acc = group_df["hit"].mean()
                    group_acc.append(acc)
                    print(f"\033[92m{group} - Hit rate: {acc:.4f}\033[0m")
                    
                print(f"\033[92mOverall Hit rate: {(sum(group_acc) / len(group_acc)):.4f}\033[0m")

                with open(prediction_dir / f"{single_eval_dataset}-evaluation_results.txt", "w") as f:
                    for _acc, _grp in zip(group_acc, unique_groups):
                        f.write(f"{single_eval_dataset} - {_grp} Hit rate: {_acc:.4f}\n")
                    f.write(f"{single_eval_dataset} - Overall Hit rate: {(sum(group_acc) / len(group_acc)):.4f}\n")
                    
                pred_df.to_csv(prediction_path.replace(".jsonl", "_parsed.csv"), index=False)
            except Exception as e:
                print(f"Evaluation error: {e}")
        
        
answer_pattern = re.compile(r'.*?<answer>\s*(.*?)\s*</answer>', re.DOTALL)
def extract_predict(examples):
    predictions = []
    for i in examples:
        match = answer_pattern.findall(i)
        if len(match) > 0:
            predictions.append(match)
        else:
            # Answer: (A) xxx
            if "Answer:" in i:
                i = i.split("Answer:")[1].strip()
                
            predictions.append([i])
    
    return predictions
    

def compute_mcq_hit(example):
    
    # Extract letter; example["answer"]: (A)
    parsed_choice_list = MultipleChoicesRandomizer.parse_choice_list("\n".join(example["choices"]))
    parsed_choice_dict = {c[0]: c[1].lower() for c in parsed_choice_list}
    gt_answer_option = example["answer"][1]
    try:
        gt_answer = parsed_choice_dict[gt_answer_option]
    except KeyError:
        # There is bugs in the dataset
        print(f"KeyError: {gt_answer_option} not in {parsed_choice_dict}")
        gt_answer = "XXX"
    
    agg_parsed_predict = example["agg_parsed_predict"].strip()
    
    hit = False
    if agg_parsed_predict.lower() in parsed_choice_dict.values():
        # Answer is shown in text
        hit = True
    else:
        # Answer is show in options
        if agg_parsed_predict == gt_answer_option:
            # Possible answer: A
            hit = True
        elif agg_parsed_predict in [f.format(gt_answer_option) for f in MultipleChoicesRandomizer.answer_formats]:
            # Possible answer: (A), A.
            hit = True
        elif agg_parsed_predict.lower() in [f.format(gt_answer_option, gt_answer) for f in MultipleChoicesRandomizer.choice_formats]:
            # Possible answer: (A) text, A. text
            hit = True
        elif agg_parsed_predict.split(" ")[0] in [f.format(gt_answer_option) for f in MultipleChoicesRandomizer.answer_formats]:
            # Possible answer: (A) XXXXX, A. XXXXX
            hit = True
        
    return hit
    

if __name__ == '__main__':
    fire.Fire({
        'predict_and_eval': predict_and_eval,
    })