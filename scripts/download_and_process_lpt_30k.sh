#!/bin/bash
export DISABLE_VERSION_CHECK=1
export LLAMAFACTORY_DIR="LLaMA-Factory" 


cd hf_lpt_dataset/
git clone https://huggingface.co/datasets/andrewliao11/LongPerceptualThoughts-30k
cd LongPerceptualThoughts-30k/


# Download DOCCI
mkdir docci
cd docci/
# Check and download docci_descriptions.jsonlines
if [ ! -f docci_descriptions.jsonlines ]; then
    echo "Downloading docci_descriptions.jsonlines..."
    wget https://storage.googleapis.com/docci/data/docci_descriptions.jsonlines
else
    echo "docci_descriptions.jsonlines already exists. Skipping download."
fi

# Check and extract docci_images.tar.gz if "images" directory doesn't exist
if [ ! -d images ]; then
    echo "Downloading docci_images.tar.gz..."
    wget https://storage.googleapis.com/docci/data/docci_images.tar.gz
    rm docci_images.tar.gz
else
    echo "'images' directory already exists. Skipping extraction."
fi

cd ../
python merge_with_docci_and_convert_to_sharegpt.py