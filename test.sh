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
python configs/trainval/tinaface/test_widerface.py configs/trainval/tinaface/tinaface_r50_fpn_gn_dcn.py /home/msai/jwang098/tinaface/workdir/tinaface_r50_fpn_bn/tinaface_r50_fpn_bn.pth