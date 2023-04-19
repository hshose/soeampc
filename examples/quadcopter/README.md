# Quadcopter
A ten state quadcopter model.

## MPC ingredients
To compute the MPC ingredients, run the `offlinempcingredients.m` file in MATLAB.
The output of the MATLAB file are already available in the folder `mpc_parameters`.

```
matlab -nodisplay -r "run('offlinempcingredients.m')"
```

The mpc parameters should be safed in human-readible `.txt` form in the folder `mpc_parameters`.

## MPC dataset generation
To generate samples of the MPC, call
```
python samplempc.py sample_mpc \\
    --numberofsamples=10
```
The results of this would be saved in a folder called `datasets/quadcopter_N_{numberofsamples}_{date}-{time}`.


You can similarly create a larger dataset, by calling this function in parallel
```
python samplempc.py parallel_sample_mpc \\
    --instances=24 \\
    --samplesperinstance=10 \\
    --prefix=Cluster_test
```
The results of this would be saved in a folder called `datasets/quadcopter_N_{instances*samplesperinstance}_merged_{prefix}_{date}-{time}`

If you downloaded the precomputed dataset for this example, you should find it under `datasets/quadcopter_N_9600000_merged_20221223-161206`.

## Training a NN

If you want to train an approximator on the precomputed dataset for this example, call
```
python3 approximatempc.py find_approximate_mpc \\
    --dataset=quadcopter_N_9600000_merged_20221223-161206
```
The models will be saved in a `models` folder.

If you downloaded the pretrained NN, you should finde it under `models/10-200-400-600-600-400-200-30_mu=0.12_20230104-232806`

## Testing the NN
You can run closed loop test with the model calling
```
python3 safeonlineevaluation.py closed_loop_test_on_dataset \\
    --dataset=quadcopter_N_120000_merged_Cluster_2023_01_06_05_51_PM_31926010_2_20230106-175146 \\
    --model_name=10-200-400-600-600-400-200-30_mu=0.12_20230104-232806 \\
    --N_samples=3000 \\
    --N_sim=1000
```