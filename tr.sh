#!/bin/bash 
#SBATCH --partition=SCSEGPU_M1
#SBATCH --qos=q_amsai
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --mem=5G
#SBATCH --job-name=init
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

module load cuda/11.8
tools/dist_trainval.sh configs/trainval/tinaface/tinaface_r50_fpn_bn.py "0,1,2,3"