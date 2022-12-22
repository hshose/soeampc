#!/bin/bash

### Job name
#SBATCH --job-name=soeampc_samplempc

#SBATCH --array=1-50%50

### Time your job needs to execute, e. g. 15 min 30 sec
#SBATCH --time=08:00:00

### CPUS
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24

### File for the output
#SBATCH --output=/home/hh753317/projects/dsme/soeampc/examples/quadcopter/logs/Cluster.%J.log

### The last part consists of regular shell commands:
source /home/hh753317/.bashrc

cd /home/hh753317/projects/dsme/soeampc/examples/quadcopter

python3 01_runparallel.py \
    --instances=24 --samplesperinstance=5000 --prefix=Cluster_$(date +"%Y_%m_%d_%I_%M_%p")_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}