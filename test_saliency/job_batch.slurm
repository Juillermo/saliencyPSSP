#!/bin/bash

#SBATCH --job-name=saliency-batch

#SBATCH --time=04:40:00

#SBATCH --nodes=1

#SBATCH --gres=gpu:1

#SBATCH --ntasks=14

#SBATCH -p lyceum

module load conda/py2-latest

source activate localenv

module load cuda

i=${SLURM_ARRAY_TASK_ID}

THEANO_FLAGS='floatX=float32,device=cuda,force_device=True' python saliency.py --batch $i --label H
THEANO_FLAGS='floatX=float32,device=cuda,force_device=True' python saliency.py --batch $i --label E
THEANO_FLAGS='floatX=float32,device=cuda,force_device=True' python saliency.py --batch $i --label L
THEANO_FLAGS='floatX=float32,device=cuda,force_device=True' python saliency.py --batch $i --label T
THEANO_FLAGS='floatX=float32,device=cuda,force_device=True' python saliency.py --batch $i --label S
THEANO_FLAGS='floatX=float32,device=cuda,force_device=True' python saliency.py --batch $i --label G
THEANO_FLAGS='floatX=float32,device=cuda,force_device=True' python saliency.py --batch $i --label B
THEANO_FLAGS='floatX=float32,device=cuda,force_device=True' python saliency.py --batch $i --label I
