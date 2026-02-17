#!/bin/bash
#SBATCH --job-name=dummy_b300
#SBATCH --partition=gpu-b300-288g-ellis
#SBATCH --time=5-00:00:00
#SBATCH --gres=gpu:b300:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=512G
#SBATCH --output=logs/dummy_b300_%j.out
#SBATCH --error=logs/dummy_b300_%j.err

sleep 5d
