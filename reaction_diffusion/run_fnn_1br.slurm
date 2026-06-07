#!/bin/bash
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8   # <- match to OMP_NUM_THREADS
## (set --partition above to a GPU partition available on your cluster)
#SBATCH --partition=<YOUR_GPU_PARTITION>   # set to a GPU partition on your cluster
#SBATCH --account=<YOUR_SLURM_ACCOUNT>
#SBATCH --job-name=RD_DeepXDE_FNN
#SBATCH --output=RD_DeepXDE_GRU_no_sigm_1br_FNN_%j.out
#SBATCH --error=RD_DeepXDE_GRU_no_sigm_1br_FNN__%j.err
##SBATCH --time=08:00:00      # hh:mm:ss for the job
##SBATCH --reservation=no_scratch_requirements
#SBATCH --time=01:00:00
### GPU options ###
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=verbose,per_task:1
##SBATCH --gpu-bind=none 

#echo $MODULEPATH
module reset
#module load anaconda3_gpu
module load anaconda3_gpu/22.10.0
module list 
#pip3 list | grep tensorflow
#pip3 list | grep scikit-learn
#pip3 install scikit-image
#pip3 install deepxde
#pip3 install tensorflow-probability
which python

export DDE_BACKEND=tensorflow
srun python3 RD_GRU_DeepXDE_no_sigm_FNN.py 
