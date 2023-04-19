import numpy as np
# from sklearn.model_selection import train_test_split

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..','..'))
from pathlib import Path

fp = Path(os.path.dirname(__file__))
os.chdir(fp)

import fire

from soeampc.safeonline import closed_loop_test_on_dataset, closed_loop_test_on_sampler, closed_loop_test_reason
from soeampc.sampler import RandomSampler

# def evaluate_on_sampler(dataset="latest", model="latest", N_samples=int(1e3)):
#     mpc, X, V, _, _ = mpc_dataset_import(dataset)
#     sampler = ...

#     model = import_model(modelname=model)
    
#     naive_controller = AMPC(mpc, model)
#     safe_controller = SafeOnlineEvaluationAMPC(mpc, model)

#     controllers = [ naive_controller, safe_controller ]
#     closed_loop_test(sampler, controllers)

def closed_loop_test_on_sampler_stirtank_rmpc(model_name="latest", N_samples=int(1e3), random_seed=None):
    x_min = np.array([-0.2,-0.2])
    x_max = np.array([0.2,0.2])
    sampler = RandomSampler(N_samples, 2, random_seed, x_min, x_max)
    closed_loop_test_on_sampler(model_name, sampler, N_samples)

if __name__=="__main__":
    fire.Fire({
        "closed_loop_test_on_dataset": closed_loop_test_on_dataset,
        "closed_loop_test_on_sampler_stirtank_rmpc": closed_loop_test_on_sampler_stirtank_rmpc,
        "closed_loop_test_reason": closed_loop_test_reason,
    })