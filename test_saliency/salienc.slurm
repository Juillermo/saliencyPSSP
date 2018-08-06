#!/bin/bash

#SBATCH --job-name=salienjurtz

#SBATCH --time=1:00:00

#SBATCH --nodes=1

#SBATCH --gres=gpu:1

#SBATCH --ntasks=14

#SBATCH -p lyceum

module load conda/py2-latest

source activate localenv

module load cuda

THEANO_FLAGS='floatX=float32,device=cuda,force_device=True' python saliency.py
