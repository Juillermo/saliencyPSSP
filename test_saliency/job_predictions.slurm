#!/bin/bash

#SBATCH --job-name=predictions

#SBATCH --time=00:20:00

#SBATCH --nodes=1

#SBATCH --gres=gpu:1

#SBATCH --ntasks=14

#SBATCH -p lyceum

module load conda/py2-latest

source activate localenv

module load cuda

THEANO_FLAGS='floatx=float32,device=cuda,force_device=True' python saliency.py
