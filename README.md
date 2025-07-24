# LongPerceptualThoughts

A framework for enriching visual reasoning with long chain-of-thoughts. We introduce a synthetic dataset that distills System-2-style reasoning into System-1 visual tasks, improving perceptual grounding and transfer to language tasks.

[**paper**](https://arxiv.org/abs/2504.15362) |
[**website**](https://andrewliao11.github.io/LongPerceptualThoughts/) |
[**dataset host on Huggingface**](https://huggingface.co/datasets/andrewliao11/LongPerceptualThought) |
[**X post**](https://x.com/andrewliao11/status/1917602672493973818)

![](./assets/overall_pipeline.gif)

## News
- ⭐ 2025/07/xx: Full release (including model weights)
- ⭐ 2025/07/07: LongPerceptualThoughts is accepted to COLM2025
- ⭐ 2025/05/26: updated LLaMA-Factory version for DPO training
- ⭐ 2025/05/23: released train and eval code 
- ⭐ 2025/05/09: released code for data generation
- ⭐ 2025/04/21: released paper and dataset

## Prerequisite
1. CUDA==12.4
2. torch==2.6.0
3. transformers>=4.51.3 (tested on 4.51.3 and 4.53.2)
4. vllm==0.8.5

## 🔧 Usage

This codebase provides you scripts to 
1. [Synthesize custom LongPerceptualThoughts]()
2. [Download and evaluate our checkpoints]()
3. [Train Qwen2.5-VL yourself!]()


### Environment
<details>
<summary>Conda env setup</summary>

Here is the line-by-line commands to install conda environment:
<pre><code>conda create -n LongPerceptualThoughts python==3.10 -y
conda install gcc=9 gxx=9 cmake -c conda-forge -y
conda install fire openai pandarallel -c conda-forge -y
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install --no-cache-dir vllm==0.8.5.post1

git clone 
cd LLaMA-Factory/
pip install -e . --no-build-isolation --no-deps -v

pip install accelerate datasets multiprocess xxhash peft trl omegaconf cachetools --no-deps -v
conda install pyarrow -c conda-forge -y
</code></pre>

Alternatively, you can install conda environment using the provided <code>.yml</code> file
<pre><code>conda env create -f environment.yml -n LongPerceptualThoughts

git clone -b LongPerceptualThoughts https://github.com/andrewliao11/LLaMA-Factory.git
cd LLaMA-Factory/
pip install -e . --no-build-isolation --no-deps -v
</code></pre>

</details>

Note: Both LLaMA-Factory and vllm are actively developed open-source projecets and the code might break when there are version mismatches. We recommend you to start a fresh conda environment.



### Data synthesis

We provide a three-stage data synthesis pipeline using image-caption datasets (e.g., [google/DOCCI](https://huggingface.co/datasets/google/docci)) to generate:

- Multiple-choice questions (MCQs)
- Short chain-of-thoughts (CoTs)
- Long CoTs
The output is a JSON format compatible with LLaMA-Factory.
For details, see the data generation README at [here](./data_gen/README.md)


### SFT/DPO Training using LLaMA-Factory

**IMPORTANT**
Before SFT/DPO training using LLaMA-Factory, you need to register the custom dataset by modifying `LLaMA-Facotry/data/dataset_info.json`. 

Here is an example:
```json
"long_perceptual_thoughts/sft_docci_all_extended_cots": {
      "file_name": /PATH/TO/DATASET/JSON,
      "formatting": "sharegpt",
      "columns": {
            "messages": "messages",
            "images": "images",
            "assistant_prefix": "assistant_prefix"
      },
      "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
            "system_tag": "system"
      }
}
```

#### Train and Evaluate

We are resource poor. We use **four A40 GPUs** (4 8GB per GPU) for training and defer the evaluation jobs by instantiate eval jobs using another instance to speedup training. More specifically, we use **4 RTX6000 GPUs** (24 GB per GPU) to perform inference and evaluation jobs. 

To disable this evaluation scheme, disable `` and `` in train config.

1. **Train jobs**
```bash
export DISABLE_VERSION_CHECK=1
llamafactory-cli train config/llama_factory_sft_train_config.yaml     # SFT training
llamafactory-cli train config/llama_factory_dpo_train_config.yaml     # DPO training
```



2. **Evaluation jobs**

By default, we evaluate on [V* bench](https://vstar-seal.github.io). Please download the images in V* Bench from [here](https://huggingface.co/datasets/craigwu/vstar_bench).

```bash
export DISABLE_VERSION_CHECK=1
export PROJECT_ROOT="/PATH/TO/GITHUB/ROOT/"
export LLAMAFACTORY_DIR="${PROJECT_ROOT}/third_party_packages/LLaMA-Factory"

cd benchmark_data/
python main.py prepare_bench
python main.py create_dataset_info

cd ../
python vllm_eval.py predict_and_eval --model_path /PATH/TO/CHECKPOINT --eval_dataset benchmark_v_star_bench --prediction_dir test/eval_sampled_greedy --temperature 0.0 --top_p 1.0 --top_k -1 --repetition_penalty 1.0 --n_samples 1 --force_thinking False --do_eval True --use_tokenized_dataset False
```


## 📚 Citation

If you find this repository helpful, please cite:

```bibtex
@misc{liao2025longperceptualthoughtsdistillingsystem2reasoning,
      title={LongPerceptualThoughts: Distilling System-2 Reasoning for System-1 Perception}, 
      author={Yuan-Hong Liao and Sven Elflein and Liu He and Laura Leal-Taixé and Yejin Choi and Sanja Fidler and David Acuna},
      year={2025},
      eprint={2504.15362},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2504.15362}, 
}
```
