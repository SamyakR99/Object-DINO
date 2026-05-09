#!/bin/bash
#SBATCH --job-name=object_discovery
#SBATCH --output=object_discovery_%j.log
#SBATCH --error=object_discovery_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00

# Activate environment
conda activate dinov3_env

cd /scratch/bcyh/samyakr99/Object-DINO/unsupervised_object_discovery

# VOC12 trainval
python main_tokencut_copy.py \
    --dataset VOC12 \
    --set trainval \
    --which_features object_dino \
    --arch vit_base \
    --tau -0.35

# VOC07 trainval
python main_tokencut_copy.py \
    --dataset VOC07 \
    --set trainval \
    --which_features object_dino \
    --arch vit_base \
    --tau -0.35

# COCO20k train
python main_tokencut_copy.py \
    --dataset COCO20k \
    --set train \
    --which_features object_dino \
    --arch vit_base \
    --tau -0.35
