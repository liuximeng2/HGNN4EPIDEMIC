#!/bin/bash

# Define parameter combinations
datasets=("UVA" "EpiSim")
timestep_hidden=(5 10 20)
location_aware_flags=(0 1)  # Binary flag (0=off, 1=on)


# Loop over parameter combinations and submit a job for each
for dataset in "${datasets[@]}"; do
    for timestep_hidden in "${timestep_hidden[@]}"; do
        for location_aware in "${location_aware_flags[@]}"; do
            sbatch --export=DATASET=$dataset,timestep_hidden=$timestep_hidden,LOCATION_AWARE=$location_aware run.sh
        done
    done
done