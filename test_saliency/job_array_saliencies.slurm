#!/bin/bash

#SBATCH --job-name=saliency-array

#SBATCH --time=00:40:00

#SBATCH --nodes=1

#SBATCH --gres=gpu:1

#SBATCH --ntasks=14

#SBATCH -p lyceum

module load conda/py2-latest

source activate localenv

module load cuda

i=${SLURM_ARRAY_TASK_ID}

THEANO_FLAGS='floatX=float32,device=cuda,force_device=True' python saliency.py --seq $i --label S
