#!/bin/bash
#SBATCH --job-name=samyak_job1
#SBATCH --output=/scratch/bcyh/samyakr99/sbatch/output_%j.out
#SBATCH --error=/scratch/bcyh/samyakr99/sbatch/error_%j.err

#SBATCH --account=bcyh-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=1200
#SBATCH --threads-per-core=1

#SBATCH --time=10:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=samyakr2@illinois.edu



# Activate conda env
source $(dirname "${CONDA_PYTHON_EXE}")/activate dinov3_env
python=/scratch/bcyh/miniconda3/envs/dinov3_env/bin/python




cd /scratch/bcyh/samyakr99/middle_layers_indicating_hallucinations


# Pretraining

CUDA_VISIBLE_DEVICES="0" $python guidance_pope_qwen.py