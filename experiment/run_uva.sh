#!/bin/bash --login
########## SBATCH Lines for Resource Request ##########

#SBATCH --time=3:00:00             # limit of wall clock time - how long the job will run (same as -t)
#SBATCH --nodes=1                 # number of different nodes - could be an exact number or a range of nodes (same as -N)
#SBATCH --ntasks=1                  # number of tasks - how many tasks (nodes) that you require (same as -n)
#SBATCH --cpus-per-task=4         # number of CPUs (or cores) per task (same as -c)
#SBATCH --mem-per-cpu=48G            # memory required per allocated CPU (or core) - amount of memory (in bytes)
#SBATCH --job-name hgnn4epi      # you can give your job a name for easier identification (same as -J)
#SBATCH --gres=gpu:v100:1

########## Command Lines for Job Running ##########
conda activate torch18
cd /mnt/scratch/lihang4/simon/HGNN4EPIDEMIC  ### change to the directory where your code is located.

# Set flag for --location_aware
LOC_FLAG=""
if [[ "$location_aware" -eq 1 ]]; then
    LOC_FLAG="--location_aware"
fi

# python train.py --dataset "UVA" --model "MSTGCN" --timestep_hidden 20 --hidden_channels 64 --lr 0.0005 --weight_decay 0.0001 --kernal_size 2 --epochs 100
# python train.py --dataset "UVA" --model "MSTGCN" --timestep_hidden 20 --lr 0.0005 --weight_decay 0.0001 --epochs 100


python train.py --dataset "UVA" --model "DTHGNN" --timestep_hidden $timestep_hidden \
    --hidden_channels $hidden_channels --lr $lr --weight_decay $weight_decay \
    --kernal_size $kernal_size --alpha $alpha $LOC_FLAG

scontrol show job $SLURM_JOB_ID     ### write job information to SLURM output file.conda env export --no-builds | grep -v "prefix" > environment.yml