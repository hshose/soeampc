#!/bin/bash

### Job name
#SBATCH --job-name=soeampc_samplempc
#SBATCH --account=rwth1288

#SBATCH --array=1-80%80

### Time your job needs to execute, e. g. 15 min 30 sec
#SBATCH --time=16:00:00

### CPUS
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24

### File for the output
#SBATCH --output=/home/hh753317/projects/dsme/soeampc/examples/chain_mass/logs/Cluster.%J.log

### The last part consists of regular shell commands:
source /home/hh753317/.bashrc

cd /home/hh753317/projects/dsme/soeampc/examples/chain_mass

python3 10_samplempc.py parallel_sample_mpc \
    --instances=24 --samplesperinstance=5000 --prefix=Cluster_$(date +"%Y_%m_%d_%I_%M_%p")_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}