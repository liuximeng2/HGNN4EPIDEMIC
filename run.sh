#!/bin/bash --login
########## SBATCH Lines for Resource Request ##########

#SBATCH --time=4:00:00             # limit of wall clock time - how long the job will run (same as -t)
#SBATCH --nodes=1                 # number of different nodes - could be an exact number or a range of nodes (same as -N)
#SBATCH --ntasks=1                  # number of tasks - how many tasks (nodes) that you require (same as -n)
#SBATCH --cpus-per-task=4         # number of CPUs (or cores) per task (same as -c)
#SBATCH --mem-per-cpu=48G            # memory required per allocated CPU (or core) - amount of memory (in bytes)
#SBATCH --job-name hgnn4epi      # you can give your job a name for easier identification (same as -J)
#SBATCH --gres=gpu:k80:1

########## Command Lines for Job Running ##########
ssh dev-amd20-v100 
conda activate torch18

cd /mnt/home/jinwei2/simon/HGNN4EPIDEMIC  ### change to the directory where your code is located.

# # Set flag for --location_aware
# LOC_FLAG=""
# if [[ "$LOCATION_AWARE" -eq 1 ]]; then
#     LOC_FLAG="--location_aware"
# fi

# python train.py --dataset "$DATASET" --timestep_hidden $timestep_hidden --device "cuda" --model "DTHGNN" $LOC_FLAG

python train.py --dataset "UVA" --timestep_hidden 5 --device "cuda" --model "DTHGNN" --forecast

scontrol show job $SLURM_JOB_ID     ### write job information to SLURM output file.conda env export --no-builds | grep -v "prefix" > environment.yml