# Examples

## Stirtank RMPC
```
cd path/to/soeampc/examples/stirtank_rmpc
```
### Dataset creation

Compute terminal ingredients and stabilizing controller (optional):
```
env WRITEOUT=1 matlab -nodisplay
run offlinempcingredients.m
```

Test if acados finds solution with given parameters, `--generate=True` is important before running dataset creation in parallel:
```
python3 10_samplempc.py sample_mpc \
    --showplot=True \
    --randomseed=42 \
    --numberofsamples=500 \
    --nlpiter=1000 \
    --withstabilizingfeedback=True \
    --generate=True \
    --verbose=False
```

Now, run dataset creation in parallel (e.g., 16 cores and 5000 samples each):
```
python3 10_samplempc.py parallel_sample_mpc \
    --instances=16 \
    --samplesperinstance=5000
```

If you need even more speedup, you can use the RWTH Cluster (TODO: REMOVE HARDCODED PARAMETERS):
```
python3 10_samplempc.py sample_mpc \
    --numberofsamples=500 \
    --withstabilizingfeedback=True \
    --generate=True 

sbatch cluster_createdataset.sh
```

Now wait, until array job has finished, then you can merge all the datasets with:
```
python3 10_samplempc.py merge_single_parallel_job \
    --dataset_name=SLURM_ARRAY_JOB_ID
```

### Train AMPC
Some architectures and hyperparameters are provided in `20_approximatempc.py`. You can train the NN with:
```
python3 20_approximatempc.py find_approximate_mpc \
    --dataset=DATASET_NAME  \
    2>&1 | tee -a log.txt
```
If you didn't run the dataset generation, you can manually download some datasets from [here]()

### Safe online evaluation
You can compare a naive evaluation of the AMPC (always taking NN output) with the safe online evaluation (only taking AMPC output, when it is safe and improves cost over candidate sequence with appended terminal controller):
```
python3 30_safeonlineevaluation.py safe_online_vs_naive \
    --model=MODEL_NAME \
    --dataset=DATASET_NAME
```