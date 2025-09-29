# RL Stage of CodeV-R1

This document describes the environment setup and training procedures for the Reinforcement Learning (RL) stage of the CodeV-R1 model.

## Environment Installation

### Create Conda Environment
```bash
conda create -n verl-codev-r1 python=3.10 conda-forge::iverilog
conda activate verl-codev-r1
```

### Install Dependencies
```bash
pip install torch==2.6.0
pip install -r requirements.txt
pip install -e .
```

Among the packages, `flash-attn` might require manual download a .whl file from Github and install.

## Training Execution

We provide both slurm script and bash script for running on multiple nodes with multiple GPUs (it is also applicable for a single node).

Our RL configuration script is located at `recipe/dapo/run_dapo_codev_7b_3.1k.sh`. You should modify the `MODEL_PATH` and `DATA_PATH` in it to your own path. You may also modify `YOUR_WANDB_API_KEY` in `train-multigpu.sh`.
You should put the RL data file (download from https://huggingface.co/datasets/zhuyaoyu/CodeV-R1-dataset, `codev_r1_rl_train.parquet` and `codev_r1_rl_val.parquet`), rename them as `train.parquet` and `val.parquet` and put them in the same folder (`DATA_PATH` in `recipe/dapo/run_dapo_codev_7b_3.1k.sh`).

### Using Slurm (Recommended for cluster environments)
You may modify the slurm script `train-multigpu.slurm` to fit your cluster environment (for example, load environments). Then just run:

```bash
sbatch train-multigpu.slurm
```

### Direct Run
You may modify SLURM_NNODES, USER_GPUS_PER_NODE, MASTER_ADDR, MASTER_PORT and some other environment varibales in `run-direct.sh`.
Then on each node just run:

```bash
bash run-direct.sh
```