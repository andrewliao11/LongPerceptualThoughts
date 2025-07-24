import io
import re
import base64
import json
import random
import ast
import os
from PIL import Image
from io import BytesIO
import pandas as pd
from pathlib import Path
from datasets import load_dataset
from jinja2.sandbox import SandboxedEnvironment


def _generate_jsonl_file(df, system_prompt_template, dataset_name):
    data = []
    for i, row in df.iterrows():
        
        question = row["question"]
        choices = row["choices"]
        answer = row["answer"]
        
        # build prompt
        task_specific_message = f"Format the answer with the letter of the correct option in parentheses."
        system_prompt = system_prompt_template.render(task_specific_message=task_specific_message)
            
        if "image_path" in row:
            user_prompt = f"<image>{question}"
        else:
            user_prompt = question
        
        user_prompt += "\nSelect from the following choices.\n"
        user_prompt += "\n".join(choices)
        
        
        answer_prompt = f"<answer>{answer}</answer>"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": answer_prompt}
        ]
        
        if "image_path" in row:
            image_path = row["image_path"]
            images = [image_path]
            
            data.append({
                "messages": messages,
                "images": images,
                "index": row["index"]
            })
        else:
            data.append({
                "messages": messages,
                "index": row["index"]
            })
    return data
        
    
def create_sharegpt_dataset():
    
    def _generate_jsonl_file_given_system_prompt_path(prefix, dataset_name, df, system_prompt_template_path):
        system_prompt_template = env.from_string(open(system_prompt_template_path).read())
        data = _generate_jsonl_file(df, system_prompt_template, dataset_name)
        # save to json
        print(f"Dump {len(data)} samples to {dataset_name}.json")
        (Path("outputs") / f"benchmark_{prefix}{dataset_name}.json").write_text(json.dumps(data, indent=4))
        
    env = SandboxedEnvironment()
    
    for tsv_file in Path("outputs/tsv_files").glob("*.tsv"):
        dataset_name = tsv_file.stem
        
        df = pd.read_csv(tsv_file, sep="\t")
        df["choices"] = df["choices"].apply(ast.literal_eval)
        print(tsv_file)
        print(df.keys())
        
        prefix = "direct_answer_"
        system_prompt_template_path = "../data_gen/templates/direct_answer_system_prompt.jinja2"
        _generate_jsonl_file_given_system_prompt_path(prefix, dataset_name, df, system_prompt_template_path)
        
        prefix = ""
        system_prompt_template_path = "../data_gen/templates/think_system_prompt.jinja2"
        _generate_jsonl_file_given_system_prompt_path(prefix, dataset_name, df, system_prompt_template_path)
        
            
    # Prepare the dataset_info.json used in llama-factory
    dataset_info = {}
    for json_file in Path("outputs").glob("benchmark_*.json"):
        dataset_name = json_file.stem
        info = {
            "file_name": str(json_file.absolute()),
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
        if "images" in json.load(open(json_file))[0]:
            info["columns"]["images"] = "images"
            
        dataset_info.update({dataset_name: info})
    
    Path("benchmark_dataset_info.json").write_text(json.dumps(dataset_info, indent=4))
    
    
def _prepare_v_star_bench():
    # V* Bench
    def _parse_v_star_choices(x):
        hint = "Answer with the option's letter from the given choices directly."
        choices = x["text"].replace(x["question"], "").replace(hint, "")
        choices = choices.strip().split("\n")
        return choices
    
    assert os.path.exists("./outputs/vstar_bench"), \
        "Please download vstar bench images to `./outputs/vstar_bench`" \
            "git clone https://huggingface.co/datasets/craigwu/vstar_bench" \
            "and run `python main.py prepare_bench` again."
        
    v_star_image_dir = Path("./outputs/vstar_bench")
    v_star_image_dir = str(v_star_image_dir.absolute())
    dataset = load_dataset("craigwu/vstar_bench")
    
    df = dataset["test"].to_pandas()
    df["image_path"] = df.apply(lambda x: os.path.join(v_star_image_dir, x["image"]), axis=1)
    df["question"] = df["text"].apply(lambda x: x.split("\n(A)")[0])
    df["choices"] = df.apply(_parse_v_star_choices, axis=1)
    df["answer"] = df["label"].apply(lambda x: f"({x})")
    df.drop(columns=["text", "label", "image"], inplace=True)
    df["index"] = list(range(len(df)))
    df.to_csv(os.path.join(os.environ["BENCHMARK_DATASET_DIR"], "tsv_files", "v_star_bench.tsv"), sep="\t")
    
    
def _prepare_cv_bench():
    
    def _parse_cv_bench_choices(x):
        hint = "Select from the following choices."
        choices = x["prompt"].replace(x["question"], "").replace(hint, "")
        choices = choices.strip().split("\n")
        return choices
    
    assert os.path.exists("./outputs/CV-Bench"), \
        "Please download CV bench images to `./outputs/CV-Bench`" \
            "https://huggingface.co/datasets/nyu-visionx/CV-Bench" \
            "and run `python main.py prepare_bench` again."
        
    cv_bench_dir = Path("./outputs/CV-Bench")
    df_2d = pd.read_parquet(cv_bench_dir / "test_2d.parquet")
    df_2d["idx"] = list(range(len(df_2d)))
    df_2d["image_path"] = df_2d["idx"].apply(lambda x: str((cv_bench_dir / "img/2D" / f"{x:06}.png").absolute()))
    
    df_3d = pd.read_parquet(cv_bench_dir / "test_3d.parquet")
    df_3d["idx"] = list(range(len(df_3d)))
    df_3d["image_path"] = df_3d["idx"].apply(lambda x: str((cv_bench_dir / "img/3D" / f"{x:06}.png").absolute()))
    
    df = pd.concat([df_2d, df_3d], ignore_index=True)
    df["question"] = df["prompt"].apply(lambda x: x.split("Select from the following choices.")[0].split("\n")[0].strip())
    df["choices"] = df.apply(_parse_cv_bench_choices, axis=1)
    df["index"] = list(range(len(df)))
    df.rename(columns={"task": "category"}, inplace=True)
    df.drop(columns=["idx", "filename", "target_class", "prompt", "image"], inplace=True)
    df.to_csv(os.path.join(os.environ["BENCHMARK_DATASET_DIR"], "tsv_files", "cv_bench.tsv"), sep="\t")
    
    
def _prepare_mmvp_bench():
    
    pat = re.compile(
        r'\([a-zA-Z]\)\s*'          # literal “(a)” / “(b)” / … (case-insensitive)
        r'([^()]+?)'                # …followed by “content”, stop before next “(x)”
        r'(?=\s*\([a-zA-Z]\)|$)'    # look-ahead: another option marker OR end-of-string
    )
    def parse_options(text: str):
        # grab the raw option texts
        pieces = pat.findall(text)
        # prepend capital-letter tags: (A), (B), …
        return [f"({chr(65+i)}) {piece.strip()}" for i, piece in enumerate(pieces)]

    
    assert os.path.exists("./outputs/MMVP"), \
        "Please download CV bench images to `./outputs/MMVP`" \
            "https://huggingface.co/datasets/MMVP/MMVP" \
            "and run `python main.py prepare_bench` again."
        
    mmvp_dir = Path("./outputs/MMVP")
    df = pd.read_csv(mmvp_dir / "Questions.csv")
    
    df["image_path"] = df["Index"].apply(lambda x: str((mmvp_dir / "MMVP Images" / f"{x}.jpg").absolute()))
    df.rename(columns={"Question": "question"}, inplace=True)
    df["choices"] = df["Options"].apply(parse_options)
    df["answer"] = df["Correct Answer"].apply(lambda x: x.upper())
    df["index"] = list(range(len(df)))
    df.drop(columns=["Correct Answer", "Index", "Options"], inplace=True)
    df.to_csv(os.path.join(os.environ["BENCHMARK_DATASET_DIR"], "tsv_files", "mmvp.tsv"), sep="\t")
    

def _prepare_mmstar_bench():
    
    def _parse_mmstar_bench_choices(x):
        
        question = x["question"]
        question = re.sub(r'\b([A-D]):', r'(\1)', question)
        # Match options like (A) ..., (B) ..., supporting multiline content
        pattern = r'\(([A-D])\)\s*(.*?)(?=\s*\([A-D]\)|\Z)'  # \Z = end of string
        matches = re.findall(pattern, question, re.DOTALL)

        # Format final output
        return [f'({label}) {text.strip()}' for label, text in matches if text not in ["nan"]]
        
        """
        pattern = r'([A-D]):(.*?)(?=(?: [A-D]:|$))'

        matches = re.findall(pattern, x["question"], re.DOTALL)

        # Format the results as desired
        result = [f'({label}){text.strip()}' for label, text in matches]

        # hint = "\nOptions: "
        
        # choices = x["question"].split(hint)[-1]
        # pattern = r'([A-Z]): (.*?)(?= [A-Z]:|$)'

        # matches = re.findall(pattern, choices)
        # result = [(label, text.strip().rstrip(",")) for label, text in matches]
        # result = [f'({label}) {text}' for label, text in result if text not in ["nan"]]
        return result"""
    
    assert os.path.exists("./outputs/MMStar"), \
        "Please download MMStar bench images to `./outputs/MMStar`"
        
    mmstar_bench_dir = Path("./outputs/MMStar")
    df = pd.read_parquet(mmstar_bench_dir / "mmstar.parquet")
    df["choices"] = df.apply(_parse_mmstar_bench_choices, axis=1)
    df["question"] = df["question"].apply(lambda x: x.split("Options:")[0].split("Choices:")[0].split("Question:")[-1].strip())
    df["answer"] = df["answer"].apply(lambda x: f"({x})")
    
    SELECTED_CATEGORIES = ['coarse perception', 'fine-grained perception', 'instance reasoning']
    df = df[df["category"].apply(lambda x: x in SELECTED_CATEGORIES)]
    df["index"] = list(range(len(df)))
    
    image_paths = []
    (mmstar_bench_dir / "images").mkdir(parents=True, exist_ok=True)
    for i, row in df.iterrows():
        image_path = mmstar_bench_dir / "images" / f"{i}.jpeg"
        if not image_path.exists():
            pil_image = Image.open(BytesIO(row["image"]))
            pil_image.save(image_path, format="JPEG")
        
        image_paths.append(str(image_path.absolute()))
    
    df["image_path"] = image_paths
    df.drop(columns=["image"], inplace=True)
    
    df.to_csv(os.path.join(os.environ["BENCHMARK_DATASET_DIR"], "tsv_files", "mmstar_bench.tsv"), sep="\t")
    
       
def _prepare_mme_rw_bench():
    
    mme_rw_bench_dir = Path("./outputs/MME-RealWorld")
    df = pd.read_json(mme_rw_bench_dir / "MME_RealWorld.json")
    
    df["image_path"] = df["Image"].apply(lambda x: str((mme_rw_bench_dir / x).absolute()))
    df.rename(columns={"Text": "question", "Answer choices": "choices"}, inplace=True)
    df["answer"] = df["Ground truth"].apply(lambda x: f"({x})")
    df["category"] = df.apply(lambda x: f"{x['Task']}.{x['Subtask']}", axis=1)
    df = df[df["Subtask"].apply(lambda x: x in ['Autonomous_Driving', 'Monitoring', 'Remote Sensing'])]
    
    df["index"] = list(range(len(df)))
    df.drop(columns=['Question_id', 'Question Type', 'Image',
       'Ground truth', 'Task', 'Subtask', 'Category', 'Question type'], inplace=True)
    
    df.to_csv(os.path.join(os.environ["BENCHMARK_DATASET_DIR"], "tsv_files", "mme_rw_bench.tsv"), sep="\t")
    
    
# Common columns: image_path, question, choices, answer
def prepare_bench():
    
    _prepare_v_star_bench()
    _prepare_cv_bench()
    _prepare_mmvp_bench()
    _prepare_mmstar_bench()
    _prepare_mme_rw_bench()
    
    create_sharegpt_dataset()