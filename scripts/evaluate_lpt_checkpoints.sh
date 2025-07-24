#!/bin/bash
export DISABLE_VERSION_CHECK=1
export LLAMAFACTORY_DIR="LLaMA-Factory" 

mkdir -p outputs/download_weights/
cd outputs/downloaded_weights
# Download the weights from Hugging Face
git clone andrewliao11/LongPerceptualThought-SFT
git clone andrewliao11/LongPerceptualThought-SFT_then_DPO

# Feel free to swtich to the full benchmark if you want to evaluate on all benchmarks
FULL_BENCHMARK="benchmark_v_star_bench,benchmark_cv_bench,benchmark_mmvp,benchmark_mmstar_bench,benchmark_mme_rw_bench"
VSTAR_BENCHMARK="benchmark_v_star_bench"

echo "Evaluating LongPerceptualThoughts-SFT on v*star benchmark"
python vllm_eval.py predict_and_eval --model_path outputs/download_weights/LongPerceptualThoughts-SFT --eval_dataset ${VSTAR_BENCHMARK} --prediction_dir predictions/eval_sampled_greedy --temperature 0.0 --top_p 1.0 --top_k -1 --repetition_penalty 1.0 --n_samples 1 --force_thinking False --do_eval True --use_tokenized_dataset False

echo "Evaluating LongPerceptualThoughts-SFT-then-DPO on v*star benchmark"
python vllm_eval.py predict_and_eval --model_path outputs/download_weights/LongPerceptualThoughts-SFT_DPO --eval_dataset ${VSTAR_BENCHMARK} --prediction_dir predictions/eval_sampled_greedy --temperature 0.0 --top_p 1.0 --top_k -1 --repetition_penalty 1.0 --n_samples 1 --force_thinking False --do_eval True --use_tokenized_dataset False
