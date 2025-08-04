#!/bin/bash

# Download dataset
cd data_gen/caption_datasets/docci
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

cd ../..


# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[1;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

export LLAMAFACTORY_DIR="../LLaMA-Factory"
export DISABLE_VERSION_CHECK=1 # Disable version check for llama_factory
# Explicily specify the number of GPUs to use if needed
# export CUDA_VISIBLE_DEVICES=0,1,2,3

# TODO: Set the number of test images you want to generate MCQs for
N_IMGS=5
DATASET_TAG=${N_IMGS}_images
FILTER_INCONSISTENT_COT=False


# Stage 1: Generate MCQs from captions
echo -e "${YELLOW}Starting MCQ generation for $N_IMGS test images...${NC}"
# NOTE: we implement cache when querying API, so you can run this stage multiple times without re-generating the same images
python main.py generate_mcq_from_captions --max_examples $N_IMGS
echo -e "${GREEN}✔ MCQ generation complete${NC}"


# # Stage 2: Generate simple CoT
echo -e "${YELLOW}Starting Simple CoT generation...${NC}"
python main.py generate_simple_cot long_perceptual_thoughts_stage_1/${N_IMGS}_images
python main.py collect_simple_cot long_perceptual_thoughts_stage_1/${N_IMGS}_images $DATASET_TAG
echo -e "${GREEN}✔ Simple CoT generation complete${NC}"


# # Stage 3: Generate long CoT
echo -e "${YELLOW}Starting Extended CoT generation...${NC}"
COGNITIVE_PHRASES=("Wait," "Hmm," "Alternatively,")
for PHRASE in "${COGNITIVE_PHRASES[@]}"; do
    echo "Running with cognitive phrase: $PHRASE"
    python main.py generate_extended_cot "|${PHRASE}|" long_perceptual_thoughts_stage_2_thought_expansion/${N_IMGS}_images
done

python main.py collect_extended_cot $DATASET_TAG --preprocess_filter_inconsistency=$FILTER_INCONSISTENT_COT
echo -e "${GREEN}✔ Extended CoT generation complete${NC}"

echo -e "${GREEN}✔ All stages complete${NC}"
echo -e "${GREEN}✔ Now, you could find the generated dataset at outputs/ ${NC}"
