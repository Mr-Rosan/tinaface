#!/bin/bash 
#SBATCH --partition=SCSEGPU_M2
#SBATCH --qos=q_dmsai
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=5G
#SBATCH --job-name=init
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

module load cuda/11.8
pip install -v -e .