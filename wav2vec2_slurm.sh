#NOTE: If you want to reuse this Slurm script, adjust the settings below for your infrastructure.
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


module load arch/avx2
module load StdEnv/2020
module load gcc/9.3.0
module load cuda/11
module load openmpi/3.1.2
module load python-build-bundle/2023a
module load ipykernel/2023a
module load arrow/10.0.1
module load python/3.11.5
module load StdEnv/2023
module load arrow/21.0.0
module load rust/1.91.0

source $HOME/w2v2/bin/activate                              #My virtual environment with all the packages necessary for Wav2Vec2-XLS-R. Replace with the name of your virtual environment.
echo "Modules loaded"

python3.11 Wav2Vec2-XLS-R_finetune-DS.py --device cuda      #The name of my script. Replace with the name of your script.
