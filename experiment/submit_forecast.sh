#!/bin/bash

# Define parameter combinations
datasets=("UVA")

# Define parameter combinations
num_for_predict=(5)
location_aware_flags=(0 1)  # Binary flag (0=off, 1=on)
hidden_channels=(128)
lr=(0.005 0.001 0.0005)
#lr=(0.001)

weight_decay=(0.001 0.0001) 
#weight_decay=(0.0005)
kernal_size=(2 4 8)
#kernal_size=(3)

# Loop over parameter combinations and submit a job for each
for location_aware in "${location_aware_flags[@]}"; do
    if [ "$location_aware" -eq 1 ]; then
        alpha=(0.5 0.7 0.9)
    else
        alpha=(1.0)
    fi
    for timestep_hidden in "${timestep_hidden[@]}"; do
        for hc in "${hidden_channels[@]}"; do
            for l in "${lr[@]}"; do
                for a in "${alpha[@]}"; do
                    for wd in "${weight_decay[@]}"; do
                        for ks in "${kernal_size[@]}"; do
                            sbatch --export=datasets=$datasets,hidden_channels=$hc,lr=$l,alpha=$a,weight_decay=$wd,kernal_size=$ks,num_for_predict=$num_for_predict,location_aware=$location_aware experiment/run_forecast.sh
                        done
                    done
                done
            done
        done
    done
done