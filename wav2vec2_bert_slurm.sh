#!/bin/bash

#NOTE: If you want to reuse this Slurm script, adjust the setting below for your infrastructure.
#Usually, each university has a guide to its computing infrastructure.
#For the U of Saskatchewan guide, see https://wiki.usask.ca/display/ARC/Plato+HPC+Cluster

#SBATCH --account=<your_hpc_account>
#SBATCH --job-name=<your_job_name>
#SBATCH --constraint=<name_of_the_computing_node_you_wish_to_use>
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --mail-user=<your_email_for_notifications>
#SBATCH --mail-type=ALL

module purge

module load python/3.10
module load StdEnv/2023    #for cuda
module load intel/2023.2.1 #for cuda
module load cuda/11.8
module load StdEnv/2020    #for gcc
module load gcc/9.3.0
module load arrow/11.0.0

source $HOME/bert/bin/activate                             #My virtual environment with all the packages necessary for Whisper. Replace with the name of your virtual environment.

python3.10 Wav2Vec2-Bert_finetune_DS.py --device cuda      #The name of my script. Replace with the name of your script.
