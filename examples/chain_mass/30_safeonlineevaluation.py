import numpy as np
import math

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..','..'))
from pathlib import Path

fp = Path(os.path.dirname(__file__))
os.chdir(fp)

import fire

from soeampc.safeonline import closed_loop_test_on_dataset, closed_loop_test_on_sampler, closed_loop_test_wtf, closed_loop_test_reason
from soeampc.sampler import RandomSampler

# from samplempc import export_acados_sim

def closed_loop_test_on_sampler_chain_mass(model_name="latest", N_samples=int(1e3), random_seed=None):
    """test closed loop on sampled initial conditions

    Args:
        model_name:
            name of the tensorflow model
        N_samples:
            number of samples to be evaluated
        random_seed:
            seed for random initial conditions
    """
    n_mass = 3
    M = n_mass - 2 # number of intermediate masses
    nx = (2*M + 1)*3  # differential states
    if n_mass == 3:
        x_min = -0.5*np.ones(nx)
        x_max = 0.5*np.ones(nx)
        x_min[:3*(M+1)] = -0.25*np.ones(3*(M+1))
        x_max[:3*(M+1)] = 0.25*np.ones(3*(M+1))
    for i in range(M+1):
        x_min[i*3+1] = -0.1

    sampler = RandomSampler(N_samples, nx, random_seed, x_min, x_max)
    closed_loop_test_on_sampler(model_name, sampler, N_samples, N_sim=200)


if __name__=="__main__":
    fire.Fire({
        "closed_loop_test_on_dataset": closed_loop_test_on_dataset,
        "closed_loop_test_on_sampler_chain_mass": closed_loop_test_on_sampler_chain_mass,
        "closed_loop_test_wtf": closed_loop_test_wtf,
        "closed_loop_test_reason": closed_loop_test_reason,
    })