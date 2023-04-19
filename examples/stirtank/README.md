# Stir Tank
A stir tank reactor.

## MPC ingredients
To compute the MPC ingredients, run the `offlinempcingredients.m` file in MATLAB.
The output of the MATLAB file are already available in the folder `mpc_parameters`.

```
matlab -nodisplay -r "run('offlinempcingredients.m')"
```

The MPC parameters should be saved in human readable `.txt` form in the folder `mpc_parameters`.

## MPC dataset generation
To generate samples of the MPC, call
```
python samplempc.py sample_mpc \\
    --numberofsamples=10
```
The results of this would be saved in a folder called `datasets/stirtank_N_{numberofsamples}_{date}-{time}`.


You can similarly create a larger dataset, by calling this function in parallel
```
python samplempc.py parallel_sample_mpc \\
    --instances=24 \\
    --samplesperinstance=10 \\
    --prefix=Cluster_test
```
The results of this would be saved in a folder called `datasets/stirtank_N_{instances*samplesperinstance}_merged_{prefix}_{date}-{time}`

If you downloaded the precomputed dataset for this example, you should find it under `datasets/stirtank_N_960000`.

## Training a NN

If you want to train an approximator on the precomputed dataset for this example, call
```
python3 approximatempc.py find_approximate_mpc \\
    --dataset=stirtank_N_960000
```
The models will be saved in a `models` folder.

If you downloaded the pretrained NN, you should find it under `models/10-200-400-600-600-400-200-30_mu=0.12_20230104-232806`

## Testing the NN
You can run closed loop test with the model calling
```
python3 safeonlineevaluation.py closed_loop_test_on_dataset \\
    --dataset=stirtank_N_240000_test \\
    --model_name=2-50-50-10_mu=1.00_20230106-170148 \\
    --N_samples=50000 \\
    --N_sim=1000
```