# Running the Code

This guide provides detailed instructions to set up and run the our code. Follow the steps below to create a conda environment, simulate data, and run the tasks.

## Step 1: Conda Environment Creation

First, ensure you have Anaconda or Miniconda installed on your system. Then, follow these steps to create and activate a conda environment using the provided `env.txt` file.

1. **Navigate to the project directory:**
    ```bash
    cd HGNN4EPIDEMIC
    ```

2. **Create the conda environment:**
    ```bash
    conda create --name hgnn4epi --file env.txt
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
    python train.py --model THGNN --dataset "UVA"
    ```
