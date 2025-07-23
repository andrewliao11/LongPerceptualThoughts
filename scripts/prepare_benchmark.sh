#!/bin/bash

echo "Preparing benchmark datasets..."
cd benchmark_data/outputs
git clone https://huggingface.co/datasets/craigwu/vstar_bench
git clone https://huggingface.co/datasets/nyu-visionx/CV-Bench
git clone https://huggingface.co/datasets/MMVP/MMVP
git clone https://huggingface.co/datasets/Lin-Chen/MMStar
git clone https://huggingface.co/datasets/yifanzhang114/MME-RealWorld


cd MME-RealWorld
echo "Extracting datasets... (this may take a while)"
tar -xvf AutonomousDriving.tar.gz
tar -xvf monitoring_images.tar.gz.part_aa
tar -xvf remote_sensing.tar.gz.part_aa
tar -xvf remote_sensing.tar.gz.part_ab
tar -xvf remote_sensing.tar.gz.part_ac
tar -xvf remote_sensing.tar.gz.part_ad
tar -xvf remote_sensing.tar.gz.part_ae
tar -xvf remote_sensing.tar.gz.part_af
tar -xvf remote_sensing.tar.gz.part_ag
tar -xvf remote_sensing.tar.gz.part_ah
tar -xvf remote_sensing.tar.gz.part_ai
tar -xvf remote_sensing.tar.gz.part_aj
tar -xvf remote_sensing.tar.gz.part_ak
tar -xvf remote_sensing.tar.gz.part_al
echo "Extraction complete."
cd ../


cd CV-Bench/
echo "Dumping images..."
python build_img.py
echo "Dumping images complete."
cd ../


cd ../
echo "Running main.py to convert benchmark datasets to ShareGPT format..."
python main.py prepare_bench
echo "Benchmark datasets prepared successfully."


