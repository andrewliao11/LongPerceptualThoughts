import os
import fire
from data_gen import (
    prepare_bench
)

os.environ["DISABLE_VERSION_CHECK"] = "1"
os.environ["BENCHMARK_DATASET_DIR"] = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(os.path.join(os.environ["BENCHMARK_DATASET_DIR"], "tsv_files"), exist_ok=True)

    
if __name__ == '__main__':
    fire.Fire({
        'prepare_bench': prepare_bench,
    })