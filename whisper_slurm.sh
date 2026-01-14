#!/bin/bash

#NOTE: If you want to reuse this Slurm script, adjust the setting below for your infrastructure.
#Usually, each university has a guide to its computing infrastructure.
#For the U of Saskatchewan guide, see https://wiki.usask.ca/display/ARC/Plato+HPC+Cluster


#SBATCH --account=<your_hpc_account>
#SBATCH --job-name=<your_job_name>
#SBATCH --constraint=<name_of_the_computing_node_you_wish_to_use>
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=70G
#SBATCH --cpus-per-task=8
#SBATCH --time=30:00:00
#SBATCH --ntasks=1
#SBATCH --mail-user=<your_email_for_notifications>
#SBATCH --mail-type=ALL

module purge

module load arch/avx2
module load StdEnv/2020
module load gcc/9.3.0
module load rust/1.75.0
module load cuda/11
module load python/3.10.2
module load scipy-stack/2023b
module load arrow/12.0.1
module load git-lfs/3.3.0


source $HOME/whisper/bin/activate                  #My virtual environment with all the packages necessary for Whisper. Replace with the name of your virtual environment.
echo "Modules loaded"

python Whisper-med_finetune_DS.py --device cuda    #The name of my script. Replace with the name of your script.
#OR
#python Whisper-lrg_finetune_DS.py --device cuda

