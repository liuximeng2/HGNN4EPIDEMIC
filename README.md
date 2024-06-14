# HGNN4EPIDEMIC

This is the official repo for our paper [Hypergraph Neural Network for Epidemic Modeling](https://openreview.net/forum?id=BTzbVsgoyx&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DKDD.org%2F2024%2FWorkshop%2FepiDAMIK%2FAuthors%23your-submissions)). The dataset can be accessed through the google drive [link](https://drive.google.com/file/d/1GSK52o9AofOzxS2dHcrEWcebClBNyZhH/view?usp=sharing). 

# Running the Code

This guide provides detailed instructions to set up and run the our. Follow the steps below to create a conda environment, simulate data, and run the tasks.

## Step 1: Conda Environment Creation

First, ensure you have Anaconda or Miniconda installed on your system. Then, follow these steps to create and activate a conda environment using the provided `env.yml` file.

1. **Navigate to the project directory:**
    ```bash
    cd HGNN4EPIDEMIC
    ```

2. **Create the conda environment:**
    ```bash
    conda env create -f env.yml
    ```

3. **Activate the conda environment:**
    ```bash
    conda activate hgnn4epi
    ```

## Step 2: Data Simulation

Next, simulate the data required for running the tasks. You have two options for data simulation: `simulation/random_simulate.py` or `simulate.py`. The random_simulate.py will generate the synthetic dataset. The simulate.py will generate based on real world dataset.

1. **Navigate to the simulation directory:**
    ```bash
    cd simulation
    ```

2. **Run the random simulation script:**
    ```bash
    python random_simulate.py
    python simulate.py
    ```

## Step 3: Run the Task

Once the data is simulated, you can run the task using `train.py`.

1. **Run the task with the specified model and dataset:**
    ```bash
    python train.py --model THGNN --dataset "h2abm"
    ```
