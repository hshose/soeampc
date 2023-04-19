# Chain Mass System
The classic chain mass benchmark system for MPC.

## MPC ingredients
To compute the MPC ingredients, run the `offlinempcingredients.m` file in MATLAB.
The output of the MATLAB file are already available in the folder `mpc_parameters`.

```
matlab -nodisplay -r "run('offlinempcingredients.m')"
```

The mpc parameters should be saved in human readable .txt form in the folder `mpc_parameters`.

## MPC dataset generation
To generate samples of the MPC, call
```
python3 samplempc.py sample_mpc \\
    --n_mass=3 \\
    --numberofsamples=10
```
The results of this would be saved in a folder called `datasets/chain_mass_{n_mass}_N_{numberofsamples}_{date}-{time}`.


You can similarly create a larger dataset, by calling this function in parallel
```
python3 samplempc.py parallel_sample_mpc \\
    --instances=24 \\
    --samplesperinstance=10 \\
    --prefix=Cluster_test
```
The results of this would be saved in a folder called `datasets/chain_mass_{n_mass}_N_{instances*samplesperinstance}_merged_{prefix}_{date}-{time}`

If you downloaded the precomputed dataset for this example, you should find it under `datasets/chain_mass_3_N_19080000`.

## Training a NN

If you want to train an approximator on the precomputed dataset for this example, call
```
python3 approximatempc.py find_approximate_mpc
    --dataset=chain_mass_3_N_19080000
```
the models will be saved in a `models` folder.

If you downloaded the pretrained NN, you should find it under `models/9-200-400-600-600-400-200-30_mu=0.09_20230131-021341`

## Testing the NN
You can run closed loop test with the model calling
```
python3 safeonlineevaluation.py closed_loop_test_on_dataset \\
    --dataset=chain_mass_3_N_120000_test \\
    --model_name=9-200-400-600-600-400-200-30_mu=0.09_20230131-021341 \\
    --N_samples=10000 \\
    --N_sim=200
```
