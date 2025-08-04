
conda install gcc=9 gxx=9 cmake -c conda-forge -y
conda install fire openai pandarallel -c conda-forge -y
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install --no-cache-dir vllm==0.8.5.post1

cd LLaMA-Factory
pip install -e . --no-build-isolation --no-deps -v
cd ../

pip install accelerate==1.9.0 datasets==4.0.0 peft==0.16.0 trl==0.9.6 deepspeed==0.16.4 --no-deps -v
pip install omegaconf cachetools multiprocess xxhash tyro liger-kernel wandb
conda install pyarrow -c conda-forge -y
