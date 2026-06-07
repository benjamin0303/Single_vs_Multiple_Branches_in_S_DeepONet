#!/bin/bash
#SBATCH -J AI
#SBATCH --output=./slurm_output/AdaptiveLearning%j.log
#SBATCH --account=<YOUR_SLURM_ACCOUNT>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16    # <- match to OMP_NUM_THREADS
#SBATCH --partition=<YOUR_GPU_PARTITION>     # set to a GPU partition on your cluster
#SBATCH --time=03:10:00      # hh:mm:ss for the jo
#SBATCH --mem=31g #199?

# # ### GPU options ###
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1
module load anaconda3_gpu/23.3.1
# (site-specific) the original runs prepended a CUDA lib path here.
# Set this for your own environment if needed, e.g.:
# export LD_LIBRARY_PATH=/path/to/cuda/lib:$LD_LIBRARY_PATH
which python

srun python3 -u GRU-1br_uncouple.py
