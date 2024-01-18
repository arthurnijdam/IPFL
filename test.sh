#!/bin/env bash
#SBATCH -A SNIC2022-22-830
#SBATCH -t 0-00:30:00
#SBATCH --gpus-per-node=A40:1   # If I need more RAM, I'll use: A40:1
#SBATCH -J test
#SBATCH -o test.out

python test.py