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

from samplempc import export_acados_sim


# from plot import plot_chain_mass_cl

# def evaluate_on_sampler(dataset="latest", model="latest", N_samples=int(1e3)):
#     mpc, X, V, _, _ = mpc_dataset_import(dataset)
#     sampler = ...

#     model = import_model(modelname=model)
    
#     naive_controller = AMPC(mpc, model)
#     safe_controller = SafeOnlineEvaluationAMPC(mpc, model)

#     controllers = [ naive_controller, safe_controller ]
#     closed_loop_test(sampler, controllers)

def closed_loop_test_on_sampler_chain_mass(model_name="latest", N_samples=int(1e3), random_seed=None):
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

def closed_loop_test_on_dataset_plot(dataset="latest", model_name="latest", N_samples=1000):
    results, controllers, mpc = closed_loop_test_on_dataset(dataset, model_name, N_samples)

    x_min = np.array([None, None, None, None,    None, None,             1/mpc.Lx[1,6],  None, 1/mpc.Lx[1,6], None]) 
    x_max = np.array([ 1/mpc.Lx[0,0],  None,  None,  None,  None,  None, 1/mpc.Lx[3,6],  None, 1/mpc.Lx[3,6],  None]) 
    
    nxconstr = 5
    u_max = np.array([1/mpc.Lu[nxconstr+i, i] for i in range(3)])
    u_min = np.array([1/mpc.Lu[nxconstr+mpc.nu+i, i] for i in range (3)])
    
    limits = { "xmin": x_min, "xmax": x_max, "umin": u_min, "umax": u_max }

    plot_controllers = [0,2]

    for i in range(len(results)):
        res = results[i]
        Utraj       = [res[c]["U"][:-1] for c in plot_controllers]
        Xtraj       = [res[c]["X"][:-1] for c in plot_controllers]
        feasible    = [res[c]["feasible"][:-1] for c in plot_controllers]
        feasible[0] = np.ones(feasible[0].shape)
        labels      = [controllers[c] for c in plot_controllers]
        path = fp = Path(os.path.dirname(__file__)).joinpath("figures", model_name)
        plot_quadcopter_cl(mpc, Utraj, Xtraj, feasible, labels=labels, limits=limits, plt_show=False, path=path, filename=f"{i}")

if __name__=="__main__":
    fire.Fire({
        "closed_loop_test_on_dataset": closed_loop_test_on_dataset,
        "closed_loop_test_on_sampler_chain_mass": closed_loop_test_on_sampler_chain_mass,
        "closed_loop_test_on_dataset_plot": closed_loop_test_on_dataset_plot,
        "closed_loop_test_wtf": closed_loop_test_wtf,
        "closed_loop_test_reason": closed_loop_test_reason,
    })