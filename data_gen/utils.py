import openai
import hashlib
import pickle
import os
import re
import pandas as pd
import json
import random
import asyncio
import httpx
from tqdm import tqdm

from copy import deepcopy
from pathlib import Path
from jinja2.sandbox import SandboxedEnvironment


# Set your OpenAI API key
if "OPENAI_API_KEY" in os.environ:
    api_key = os.environ['OPENAI_API_KEY']
else:
    api_key = None


DATASET_ROOT = Path('caption_datasets')
DOCCI_DATASET_ROOT = DATASET_ROOT / 'docci'
    
class DOCCI:
    def __init__(self, split, max_examples=-1, verbose=False):
        assert os.path.exists(DOCCI_DATASET_ROOT / 'docci_descriptions.jsonlines'), "Please download the DOCCI dataset from https://github.com/docci-ai/docci-dataset"
            
        df = pd.read_json(DOCCI_DATASET_ROOT / 'docci_descriptions.jsonlines', lines=True)
        df = df[df.split == split]
        n_total_images = len(df)
        
        # Unifying the keys
        df['image_path'] = df.apply(lambda x: str(DOCCI_DATASET_ROOT / 'images' / f"{x['image_file']}"), axis=1)
        df = df[df.apply(lambda x: Path(x['image_path']).exists(), axis=1)]             # Filter out missing images
        if verbose and (n_total_images - len(df)) > 0:
            print(f"Filtered out {n_total_images - len(df)} missing images.")
            n_total_images = len(df)
            
        df["image_id"] = df["example_id"]
        df.drop(columns=['split', 'example_id', 'image_file'], inplace=True)
        if max_examples > 0:
            df = df.iloc[:max_examples]
        
        self.df = df
        
        self.n_total_images = len(df)
        self.image_dir = DOCCI_DATASET_ROOT / "images"


def register_to_dataset_json(dataset_info, contain_image=True):
    path = os.path.join("outputs", "long_perceptual_thoughts_dataset_info.json")
    prev_dataset_info = {}
    if os.path.exists(path):
        prev_dataset_info = json.load(open(path))
        
    for k, v in dataset_info.items():
        prev_dataset_info[k] = get_sharegpt_template(path=v, contain_image=contain_image)
        
    json.dump(prev_dataset_info, open(path, 'w'), indent=4)
    
    
def get_sharegpt_template(path, contain_image):
    
    data = {
        "file_name": path,
        "formatting": "sharegpt",
        "columns": {
            "messages": "messages", 
        },
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant", 
            "system_tag": "system"
        }
    }
    if contain_image:
        data["columns"]["images"] = "images"
        
    return data
    
    
def infer_template(model_name_or_path):
    if "DeepSeek-R1" in model_name_or_path:
        return "deepseek3"
    elif "Qwen2-VL" in model_name_or_path or "Qwen2.5-VL" in model_name_or_path:
        return "qwen2_vl"
    elif "Qwen" in model_name_or_path or "QwQ" in model_name_or_path:
        return "qwen"
    else:
        raise NotImplementedError(f"Template for model {model_name_or_path} is not implemented.")
    
    
class MultipleChoicesRandomizer:
    answer_formats = ["({})", "{}."]
    choice_formats = ["({}) {}", "{}. {}"]
    
    def __init__(self, seed):
        random.seed(seed)
       
    @staticmethod
    def parse_choice_list(choices):
        # Parse choices into a list of (letter, text) tuples
        choice_pattern = r'(?:\(([A-J])\)|([A-J])[\.\)])\s+(.*?)(?=\s+(?:\([A-J]\)|[A-J][\.\)])\s+|$)'
        parsed_choices = re.findall(choice_pattern, choices, re.DOTALL)
        
        # Clean up parsed choices
        choice_list = []
        for match in parsed_choices:
            letter = match[0] if match[0] else match[1]
            text = match[2].strip()
            choice_list.append((letter, text))
        
        return choice_list
    
    def __call__(self, choices: str, answer: str):
        
        choice_list = self.parse_choice_list(choices)
        
        # Determine the original answer format and content
        answer_text = None
        answer_letter = None
        
        # Check if answer is a letter only (A, B, C...)
        if re.match(r'^[A-J]$', answer):
            answer_letter = answer
            for letter, text in choice_list:
                if letter == answer_letter:
                    answer_text = text
                    break
        # Check if answer is formatted like (A), A), or A.
        elif re.match(r'^\([A-J]\)$|^[A-J]\)$|^[A-J]\.$', answer):
            answer_letter = re.search(r'[A-J]', answer).group(0)
            for letter, text in choice_list:
                if letter == answer_letter:
                    answer_text = text
                    break
        # Otherwise, assume answer is the text content
        else:
            answer_text = answer
            for letter, text in choice_list:
                if text == answer_text:
                    answer_letter = letter
                    break
        
        # Randomize the order of choices
        random.shuffle(choice_list)
        
        # Reassign letters to shuffled choices
        letters = [chr(65 + i) for i in range(len(choice_list))]
        new_choice_list = []
        new_answer_letter = None
        
        for i, (_, text) in enumerate(choice_list):
            new_letter = letters[i]
            new_choice_list.append((new_letter, text))
            if text == answer_text:
                new_answer_letter = new_letter
        
        # Randomize choice format (A. XXX, (A) XXX)
        choice_format = random.choice(self.choice_formats)
        
        # Randomize answer format (A, (A), or XXX)
        answer_format_idx = random.randint(0, 2)  # 0-1 for format, 2 for text
        
        # Format choices with the selected format
        formatted_choices = "\n".join([choice_format.format(letter, text) for letter, text in new_choice_list])
        
        # Format answer based on selected format
        if answer_format_idx == 2:
            formatted_answer = answer_text
        else:
            formatted_answer = self.answer_formats[answer_format_idx].format(new_answer_letter)
        
        # Create the prompt message for the model
        if answer_format_idx == 2:
            format_specific_message = "Please answer with the full text of the correct option."
        else:
            format_example = self.answer_formats[answer_format_idx].format("X")
            format_specific_message = f"Please provide your answer as {format_example}, where X is the letter of the correct option."
        
        return format_specific_message, formatted_choices, formatted_answer
    
    @staticmethod
    def reformat_answer_option(option, parsed_choice_list, input_prompt):
        pattern = r'(?:\(([A-J])\)|([A-J])[\.\)]|^([A-J])$)'
        match = re.search(pattern, option.strip())
        
        assert match is not None, f"Could not parse answer option from {option}"
        # Return the first non-None group (whichever format matched)
        for group in match.groups():
            if group:
                option = group
                break
    
        for answer_format in MultipleChoicesRandomizer.answer_formats:
            format_example = answer_format.format("X")
            format_specific_message = f"Please provide your answer as {format_example}, where X is the letter of the correct option."
            if format_specific_message in input_prompt:
                return answer_format.format(option)
        
        
        text = [x for x in parsed_choice_list if x[0] == option][0][1]
        return text


def extract_options_from_user_prompt(user_prompt, mcq_question):
    if "Select from the following choices.\n" in user_prompt:
        option_in_prompt = user_prompt.split("Select from the following choices.\n")[-1].strip()
        option_in_prompt = option_in_prompt.split("<|im_end|>")[0].strip()
    else: 
        option_in_prompt = user_prompt.split(mcq_question)[-1].strip()
        option_in_prompt = option_in_prompt.split("<|im_end|>")[0].strip()
    return option_in_prompt


def string_to_seed(s, max_value=2**32 - 1):
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % max_value


def keep_one_per_simple_cot_length_weighted_subsample_df(df):

    df = df.reset_index()
    grouped = df.groupby(["mcq_unique_id", "simple_cot_unique_id"])
    unique_groups = list(grouped.groups.keys())
    n_groups = len(unique_groups)
    
    sampled_dfs = []
    """
    For each simple CoT, we only keep one sample.
    """
    for group in tqdm(unique_groups, desc=f"Processing groups. Limits to one sample per group."):
        group_df = grouped.get_group(group).copy()
        
        group_df['_rank'] = group_df['#tokens of extended_cot_parsed_thought'].rank(method='first', ascending=False).astype(int) - 1  # rank starts from 1, subtract 1
        group_df['_weight'] = 0.5 ** group_df['_rank']
        sample = group_df.sample(n=1, weights=group_df['_weight'], random_state=string_to_seed(group[0] + group[1]))
        
        sampled_dfs.append(sample)
        
    combined = pd.concat(sampled_dfs, ignore_index=True)
    combined.drop(columns=["_rank", "_weight"], inplace=True)
    return combined


def length_weighted_subsample_df(df, n_total, thought_key, group_key="mcq_unique_id", secondary_group_key=None):
    """
    For each MCQ, we perform weighted subsample based on 2^{rank of length},
    """
    df = df.reset_index()
    grouped = df.groupby([group_key])
    unique_groups = grouped.groups.keys()
    n_groups = len(unique_groups)

    # Uniform base allocation
    n_per_group = n_total // n_groups

    sampled_dfs = []
    sampled_indices = set()

    for group in unique_groups:
        group_df = grouped.get_group(group).copy()
        n = min(n_per_group, len(group_df))  # in case group is too small
        
        # Calculated weight
        group_df["_length"] = group_df[thought_key].apply(lambda x: len(x))
        if secondary_group_key is not None:
            # If secondary_group_key is provided, we need to rank within each secondary group
            _df_list = []
            for k in group_df[secondary_group_key].unique():
                _df = group_df[group_df[secondary_group_key] == k].copy()
                _df["_rank"] = _df['_length'].rank(method='first', ascending=False).astype(int) - 1  # rank starts from 1, subtract 1
                _df_list.append(_df)

            group_df = pd.concat(_df_list)
        else:
            group_df['_rank'] = group_df['_length'].rank(method='first', ascending=False).astype(int) - 1  # rank starts from 1, subtract 1
            
        group_df['_weight'] = 0.5 ** group_df['_rank']
        
        sample = group_df.sample(n=n, weights=group_df['_weight'], random_state=string_to_seed(group))
        sample.drop(columns=['_length', '_rank', '_weight'], inplace=True)
        sampled_dfs.append(sample)
        sampled_indices.update(sample.index)

    combined = pd.concat(sampled_dfs)

    # Fill the gap to exactly N
    gap = n_total - len(combined)
    if gap > 0:
        remaining_pool = df.drop(index=sampled_indices)
        filler = remaining_pool.sample(n=gap, random_state=42)
        combined = pd.concat([combined, filler])

    # just to shuffle the dataframes
    return combined.sample(frac=1, random_state=42).reset_index(drop=True)


def convert_dpo_extended_cot_dataset(dpo_df, image_list, filename):
    
    if image_list is not None:
        dpo_df = dpo_df[dpo_df["image_id"].apply(lambda x: x in image_list)]

    data = []
    for i, row in tqdm(dpo_df.iterrows(), desc="Process DPO dataset for extended CoT dataset"):
        images = [str(Path(row["image_path"]).absolute())]
        
        messages = deepcopy(row["mcq_messages"])
        messages.pop(-1)
        
        data.append(dict(
            messages=messages, 
            images=images, 
            chosen={
                "role": "assistant", 
                "content": f"<think> {row['positive'][0]} </think> <answer> {row['positive'][1]} </answer>"
            }, 
            rejected={
                "role": "assistant", 
                "content": f"<think> {row['negative'][0]} </think> <answer> {row['negative'][1]} </answer>"
            }, 
            metadata=dict(
                image_id=row["image_id"],
                mcq_unique_id=row["mcq_unique_id"], 
            )
        ))

    if filename is not None:
        json.dump(data, open(filename, 'w'), indent=4)
    
    return data


def convert_sft_extended_cot_dataset(df, image_list, filename, include_simple_cot=False, weighted_sample=False, size=-1, seed=42):
    
    if image_list is not None:
        df = df[df["image_id"].apply(lambda x: x in image_list)]
    
    if include_simple_cot:
        simple_cot_df = df.drop_duplicates(subset=["simple_cot_unique_id"]).copy()
        simple_cot_df["extended_cot_parsed_thought"] = simple_cot_df["simple_cot_parsed_thought"]
        simple_cot_df["extended_cot_parsed_answer"] = simple_cot_df["simple_cot_parsed_answer"]
        simple_cot_df["extended_cot_parsed_correct"] = simple_cot_df["simple_cot_parsed_correct"]
        simple_cot_df["extended_cot_unique_id"] = simple_cot_df["simple_cot_unique_id"]
        df = pd.concat([df, simple_cot_df], ignore_index=True)
    
    if weighted_sample:
        assert size > 0, "Size must be greater than 0 for weighted sampling."
        df = length_weighted_subsample_df(df, size, "extended_cot_parsed_thought", secondary_group_key="simple_cot_unique_id")
            
    # long_perceptual_thoughts/sft_docci_500_images_extended_cots_O_any_df_all_weighted_sample
    
    data = []
    env = SandboxedEnvironment()
    for i, row in tqdm(df.iterrows(), desc="Process SFT dataset for extended CoT dataset"):
        images = [str(Path(row["image_path"]).absolute())]
        
        messages = deepcopy(row["mcq_messages"])
        # Step 1. Replace the system prompt with the think system prompt
        messages[2]["content"] = f"<think> {row['extended_cot_parsed_thought']} </think> <answer> {row['extended_cot_parsed_answer']} </answer>"
        
        if not row["simple_cot_parsed_correct"]:
            assistant_prefix = f"<think> {row['simple_cot_parsed_thought']}"
        else:
            assistant_prefix = ""
            
        data.append(dict(
            messages=messages, 
            images=images, 
            assistant_prefix=assistant_prefix,
            metadata=dict(
                image_id=row["image_id"],
                mcq_unique_id=row["mcq_unique_id"], 
                simple_cot_unique_id=row["simple_cot_unique_id"],
                extended_cot_unique_id=row["extended_cot_unique_id"],
            )
        ))

    if filename is not None:
        json.dump(data, open(filename, 'w'), indent=4)
    
    return data


def convert_sft_thought_expansion_dataset(df, filename, seed=42):
    data = []
    env = SandboxedEnvironment()
    vlm_system_prompt = env.from_string(open("templates/think_system_prompt.jinja2").read())
    reasoning_llmsystem_prompt = env.from_string(open("templates/reasoning_system_prompt.jinja2").read())
    
    for i, row in tqdm(df.iterrows(), desc="Process dataset for thought-expansion"):
        
        # Step 1. Replace the system prompt with the think system prompt
        messages = row["mcq_messages"]
        task_specific_message = messages[0]["content"].replace(vlm_system_prompt.render(), "")
        system_prompt = reasoning_llmsystem_prompt.render(task_specific_message=task_specific_message)
        
        # Step 2. Extract the options from the messages
        option_in_prompt = extract_options_from_user_prompt(row["_simple_cot_prompt"], row["mcq_question"])
        parsed_choice_list = MultipleChoicesRandomizer.parse_choice_list(option_in_prompt)
        
        options = option_in_prompt
        simple_cot = row["simple_cot_parsed_thought"]
        user_prompt = f"""Image description: {row['description']}\nQuestion: {row['mcq_question']}\nSelect from the following choices.\n{options}"""
        
        options_in_one_line = options.replace("\n", " ")
        # assistant_prompt = f"""<think>\nOkay, let's tackle this question. The user is asking "{row['mcq_question']}". The options are {options_in_one_line}. The user also asks to reason without referencing the fact that the text descriptions are revealed.\n\nFirst, I need to visualize the image described. {simple_cot}"""
        assistant_prompt = f"""<think>\nOkay, let's tackle this question. The user is asking "{row['mcq_question']}". The options are {options_in_one_line}. The user also asks to reason without referencing the fact that the text descriptions are revealed.\n\n{simple_cot.capitalize()}"""
        # R1-series model tips:
        # https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B#usage-recommendations
        messages = [
            # {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"},
            {"role": "assistant", "content": assistant_prompt},
        ]
        
        data.append(dict(
            messages=messages, 
            metadata=dict(
                image_id=row["image_id"],
                mcq_unique_id=row["mcq_unique_id"], 
                simple_cot_unique_id=row["simple_cot_unique_id"], 
                simple_cot=simple_cot
            )
        ))

    if filename is not None:
        json.dump(data, open(filename, 'w'), indent=4)
        

def convert_sft_simple_cot_dataset(df, image_list, filename, weighted_sample=False, sample_ratio=-1, seed=42):
    
    if image_list is not None:
        df = df[df["image_id"].apply(lambda x: x in image_list)]
    
    if weighted_sample:
        assert sample_ratio > 0, "Sample ratio must be greater than 0 for weighted sampling."
        assert sample_ratio <= 1, "Sample ratio must be less than or equal to 1."
        
        size = int(len(df) * sample_ratio)
        df = length_weighted_subsample_df(df, size, "simple_cot_parsed_thought")
            
    data = []
    env = SandboxedEnvironment()
    for i, row in tqdm(df.iterrows(), desc="Process dataset for simple CoT dataset"):
        images = [str(Path(row["image_path"]).absolute())]
        
        messages = deepcopy(row["mcq_messages"])
        messages[2]["content"] = f"<think> {row['simple_cot_parsed_thought']} </think> <answer> {row['simple_cot_parsed_answer']} </answer>"
        
        data.append(dict(
            messages=messages, 
            images=images, 
            metadata=dict(
                image_id=row["image_id"],
                mcq_unique_id=row["mcq_unique_id"], 
                description=row["description"],
            )
        ))

    if filename is not None:
        json.dump(data, open(filename, 'w'), indent=4)
    
    
def convert_sft_mcq_dataset(df, filename, direct_answer=False, seed=42):
    
    multiple_choices_randomizer = MultipleChoicesRandomizer(seed=seed)
    
    data = []
    env = SandboxedEnvironment()
    if direct_answer:
        system_prompt = env.from_string(open("templates/direct_answer_system_prompt.jinja2").read())
    else:
        system_prompt = env.from_string(open("templates/think_system_prompt.jinja2").read())
    
    
    for i, row in tqdm(df.iterrows(), desc="Process dataset for MCQ dataset"):
        images = [str(Path(row["image_path"]).absolute())]
        
        task_specific_message = f"The following question requires the capability of \"{row['mcq_question_type']}\""
        format_specific_message, mcq_choices, mcq_answer = multiple_choices_randomizer(row['mcq_choices'], row['mcq_answer'])
        task_specific_message += f" {format_specific_message}"
        
        user_prompt = f"<image>{row['mcq_question']}\nSelect from the following choices.\n{mcq_choices}"
        messages = (
            {"role": "system",
            "content": system_prompt.render(task_specific_message=task_specific_message)}, 
            {"role": "user",
            "content": user_prompt}, 
            {"role": "assistant",
            "content": f"<answer>{mcq_answer}</answer>"}, 
        )
        
        data.append(dict(
            messages=messages, 
            images=images, 
            metadata=dict(
                image_id=row["image_id"],
                mcq_unique_id=row["mcq_unique_id"], 
                description=row["description"],
            )
        ))

    if filename is not None:
        json.dump(data, open(filename, 'w'), indent=4)
    

def get_unique_id(prompt, image_path=None):
    if image_path is None:
        return hashlib.sha256(prompt.encode()).hexdigest()
    else:
        return hashlib.sha256(f"{prompt}{image_path}".encode()).hexdigest()
    

async def query_single(prompt, system_prompt, model_id, client: httpx.AsyncClient):
    OPENAI_URL = "https://api.openai.com/v1/chat/completions"
    MODEL_ID = model_id

    HEADERS = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    payload = {
        "model": MODEL_ID,
        "messages": messages,
        "temperature": 0.7,
    }
    try:
        response = await client.post(OPENAI_URL, headers=HEADERS, json=payload, timeout=30.0)
        response.raise_for_status()
        #return #response.json()['choices'][0]['message']['content']
        return [choice['message']['content'] for choice in response.json()['choices']]
    except Exception as e:
        return f"[ERROR] {e}"
    

async def query_all(prompts_with_sys, model_id, max_concurrent=10):
    results = []
    semaphore = asyncio.Semaphore(max_concurrent)  # Limit concurrency to stay within rate limits

    async with httpx.AsyncClient() as client:
        async def sem_task(prompt, system_prompt):
            async with semaphore:
                return await query_single(prompt, system_prompt, model_id, client)

        tasks = [sem_task(p["user"], p["system"]) for p in prompts_with_sys]
        results = await asyncio.gather(*tasks)
    return results
    
    
"""
- OpenAI API key is required to use this class.
- Main features:
    - Cache to disk
    - Batching of requests
"""
class OpenAICacheClient:
    def __init__(self, model_id, cache_dir='openai_cache', force_use_cache=False, verbose=False):
        '''
        Initialize the OpenAI client with a cache directory.

        :param api_key: OpenAI API key
        :param cache_dir: Directory to store cached responses
        '''
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.force_use_cache = force_use_cache
        self.verbose = verbose
        os.makedirs(self.cache_dir, exist_ok=True)  # Ensure cache directory exists
        self.client = openai.Client(api_key=api_key)

    def _get_cache_filename(self, system_prompt, prompt, temperature, n):
        '''Generate a unique cache filename based on prompt and parameters.'''
        unique_str = f'{prompt}|temp={temperature}|n={n}'
        prompt_hash = hashlib.sha256(unique_str.encode()).hexdigest()
        return os.path.join(self.cache_dir, f'{prompt_hash}.pkl')

    def _load_from_cache(self, system_prompt, prompt, temperature, n):
        '''Load cached response if available.'''
        cache_file = self._get_cache_filename(system_prompt, prompt, temperature, n)
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                return None
        return None

    def _save_to_cache(self, system_prompt, prompt, temperature, n, response):
        '''Save response to cache.'''
        cache_file = self._get_cache_filename(system_prompt, prompt, temperature, n)
        with open(cache_file, 'wb') as f:
            pickle.dump(response, f)

    def query_openai_all(self,
            prompts, 
            system_prompts, 
            batch_size=100, 
            max_concurrent=10, 
            temperature=0.7, n=1):
        
        results = []
        all_inputs = []
        for prompt, system_prompt in zip(prompts, system_prompts):
            
            cached_response = self._load_from_cache(system_prompt, prompt, temperature, n)
            if cached_response:
                if self.verbose:
                    print('Loaded from cache.')
                
                results.append(cached_response)
            else:
                all_inputs.append({
                    "system": system_prompt,
                    "user": prompt
                })
                results.append(None)
        
        # Batch them into a single request
        for batch_start_ind in tqdm(range(0, len(all_inputs), batch_size), desc="Running batched requests"):
            batch_prompts_with_sys = all_inputs[batch_start_ind:batch_start_ind+batch_size]

            # Query OpenAI API
            batch_responses = asyncio.run(query_all(batch_prompts_with_sys, self.model_id, max_concurrent=max_concurrent)) 
            
            # Save to cache
            for i, response in enumerate(batch_responses):
                if response is not None:
                    self._save_to_cache(batch_prompts_with_sys[i]["system"], batch_prompts_with_sys[i]["user"], temperature, n, response)
                    results[batch_start_ind + i] = response
                    
        return results
        
    def query_openai(self, 
            prompt, 
            system_prompt='You are an assistant that converts image descriptions to multi-choice visual questions.', 
            temperature=0.7, n=1):
        '''
        Query OpenAI API with temperature and multiple samples, using disk caching.

        :param prompt: Text prompt to query the OpenAI API
        :param temperature: Controls randomness (0.0 = deterministic, 1.0 = highly creative)
        :param n: Number of responses to generate
        :return: List of responses
        '''
        cached_response = self._load_from_cache(system_prompt, prompt, temperature, n)

        if self.force_use_cache:
            assert cached_response is not None, "Cache not found, but force_use_cache is set to True."
            
        if cached_response:
            if self.verbose:
                print('Loaded from cache.')
            return cached_response

        # Query OpenAI API if not cached
        assert n == 1, "Haven't tested for n > 1 yet"
        if self.verbose:
            print('Querying OpenAI API...')
        
        if len(system_prompt) > 0:
            messages = [{'role': 'system', 'content': system_prompt},]    
        else:
            messages = []
            
        messages.append({'role': 'user', 'content': prompt})
        
        response = self.client.chat.completions.create(
            messages=messages,
            temperature=temperature,
            n=n, 
            model=self.model_id
        )

        # Extract the message content(s)
        responses = [choice.message.content for choice in response.choices]

        # Save response to cache
        self._save_to_cache(system_prompt, prompt, temperature, n, responses)

        return responses