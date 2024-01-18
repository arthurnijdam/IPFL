#!/bin/env bash
#SBATCH -A NAISS2023-22-980
#SBATCH -t 0-04:30:00
#SBATCH --gpus-per-node=A40:1   # If I need more RAM, I'll use: A40:1
#SBATCH -J test
#SBATCH -o test.out

python IFCA_MNIST.py #main.py