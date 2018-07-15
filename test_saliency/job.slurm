#!/bin/bash

#SBATCH --job-name=saliency

#SBATCH --time=04:00:00

#SBATCH --nodes=1

#SBATCH --gres=gpu:1

#SBATCH --ntasks=14

#SBATCH -p lyceum

source activate localenv

python saliency_testing.py
