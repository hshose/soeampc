#
# Copyright 2019 Gianluca Frison, Dimitris Kouzoupis, Robin Verschueren,
# Andrea Zanelli, Niels van Duijkeren, Jonathan Frey, Tommaso Sartor,
# Branimir Novoselnik, Rien Quirynen, Rezart Qelibari, Dang Doan,
# Jonas Koenemann, Yutao Chen, Tobias Schöls, Jonas Schlagenhauf, Moritz Diehl
#
# This file is part of acados.
#
# The 2-Clause BSD License
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.;
#

# authors: Florian Messerer, Jonathan Frey
#
# This implementation of the nonlinear chain example follows the problem formulation:
# https://github.com/dkouzoup/hanging-chain-acado
# and the publication
# Recent Advances in Quadratic Programming Algorithmsfor Nonlinear Model Predictive Control
# https://cdn.syscop.de/publications/Kouzoupis2018.pdf
#
# The problem therein is extended by applying disturbances at each sampling time.
# These disturbances are the parameters of the model described in
# export_disturbed_chain_mass_model.py

# Modifications by Henrik Hose
# * added terminal constraint and terminal controller
# * export and run function for use with soeampc


from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver, AcadosModel, AcadosSim
from casadi import SX, vertcat, sin, cos, tan, Function, sign, tanh, jacobian
import numpy as np
import scipy.linalg
from scipy.integrate import odeint
import math

np.set_printoptions(edgeitems=3)
np.core.arrayprint._line_width = 200

import subprocess
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..','..'))
from pathlib import Path

fp = Path(os.path.dirname(__file__))
os.chdir(fp)

from soeampc.sampler import RandomSampler
from soeampc.samplempc import sample_dataset_from_mpc, computetime_test_fwd_sim
from soeampc.mpcproblem import MPCQuadraticCostLxLu
from soeampc.datasetutils import import_dataset, merge_parallel_jobs, get_date_string, merge_single_parallel_job, print_dataset_statistics

import fire
import importlib
from plot import *

def export_chain_mass_model(n_mass):
    rho       = float(np.genfromtxt(fp.joinpath('mpc_parameters',f'rho_c_{n_mass}.txt'), delimiter=',')) # 10
    w_bar     = float(np.genfromtxt(fp.joinpath('mpc_parameters',f'wbar_{n_mass}.txt'), delimiter=',')) # 4.6e-1

    # import n_mass specific dynamics f
    # this can be rendered from `dynamics/f.template.py` and `mpc_parameters/xref_{{n_mass}}.txt` with:
    # ```$ python 01_renderdynamics --n_mass={{n_mass}}```
    spec = importlib.util.spec_from_file_location("f", fp.joinpath("dynamics",f"f_{n_mass}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    f = mod.f
    M = n_mass - 2 # number of intermediate masses

    model_name = f'chain_mass_{n_mass}'
    x0 = np.array([0, 0, 0]) # fix mass (at wall)

    nx = (2*M + 1)*3  # differential states
    nu = 3            # control inputs
    Kdelta = np.reshape(np.genfromtxt(fp.joinpath('mpc_parameters',f'Kdelta_{n_mass}.txt'), delimiter=','), (nx,nu)).T


    x = SX.sym('x', nx, 1) # position of fix mass eliminated
    u = SX.sym('u', nu, 1)
    xdot = SX.sym('xdot', nx, 1)
    s     =  SX.sym('s')
    sdot     =  SX.sym('sdot')

    # print(x.shape)
    v = SX.sym('u', nu, 1)
    u = Kdelta @ x + v

    # dynamics     
    fx = f(x,u)
    # f_expl = vertcat(vertcat(*fx), -rho*s+w_bar)
    f_impl = vertcat(vertcat(*fx)-xdot, -rho*s+w_bar-sdot)

    model = AcadosModel()

    model.f_impl_expr = f_impl
    # model.f_expl_expr = f_expl
    model.x = vertcat(x, s)
    model.xdot = vertcat(xdot, sdot)
    model.u = v
    model.p = []
    model.name = model_name
    
    return model

def sample_mpc(
    n_mass,
    showplot=True,
    experimentname="",
    numberofsamples=5000,
    randomseed=42,
    verbose=False,
    generate=True,
    nlpiter=1000):
    
    # import n_mass specific dynamics f
    # this can be rendered from `dynamics/f.template.py` and `mpc_parameters/xref_{{n_mass}}.txt` with:
    # ```$ python 01_renderdynamics --n_mass={{n_mass}}```
    spec = importlib.util.spec_from_file_location("f", fp.joinpath("dynamics",f"f_{n_mass}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    f = mod.f

    M = n_mass - 2 # number of intermediate masses

   # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    # set model
    model = export_chain_mass_model(n_mass)
    ocp.model = model

    N = 40
    Tf = float(np.genfromtxt(fp.joinpath('mpc_parameters',f'Tf_{n_mass}.txt'), delimiter=','))
    nx = model.x.size()[0]-1
    nx_ = model.x.size()[0]
    nu = model.u.size()[0]
    ny = nx_ + nu
    ny_e = nx_

    rho       = float(np.genfromtxt(fp.joinpath('mpc_parameters',f'rho_c_{n_mass}.txt'), delimiter=',')) # 10
    w_bar     = float(np.genfromtxt(fp.joinpath('mpc_parameters',f'wbar_{n_mass}.txt'), delimiter=',')) # 4.6e-1
    Kdelta    = np.reshape(np.genfromtxt(fp.joinpath('mpc_parameters',f'Kdelta_{n_mass}.txt'), delimiter=','), (nx,nu)).T
    print("Kdelta=\n",Kdelta,"\n")

    Sinit = odeint(lambda y,t: -rho*y+w_bar, 0, np.linspace(0,Tf,N+1))
    print("Sinit =\n",Sinit,"\n")


    Q = np.reshape(np.genfromtxt(fp.joinpath('mpc_parameters',f'Q_{n_mass}.txt'), delimiter=','), (nx,nx))
    P = np.reshape(np.genfromtxt(fp.joinpath('mpc_parameters',f'P_{n_mass}.txt'), delimiter=','), (nx,nx))
    R = np.reshape(np.genfromtxt(fp.joinpath('mpc_parameters',f'R_{n_mass}.txt'), delimiter=','), (nu,nu))
    Q_ = scipy.linalg.block_diag(Q, 1)
    P_ = scipy.linalg.block_diag(P, 1)
    K = np.reshape(np.genfromtxt(fp.joinpath('mpc_parameters',f'K_{n_mass}.txt'), delimiter=','), (nx,nu)).T
    # Kinit = np.reshape(np.genfromtxt(fp.joinpath('mpc_parameters',f'Kinit_{n_mass}.txt'), delimiter=','), (nx,nu)).T
    alpha_f = float(np.genfromtxt(fp.joinpath('mpc_parameters',f'alpha_{n_mass}.txt'), delimiter=','))

    ocp.dims.N = N

    print("P = \n", P, "\n")
    print("Q = \n", Q, "\n")
    print("R = \n", R, "\n")

    ocp.cost.W_e = P_
    ocp.cost.W = scipy.linalg.block_diag(Q_, R)

    # set dimensions
    ocp.dims.N = N

    # set cost module
    ocp.cost.cost_type = 'LINEAR_LS'
    ocp.cost.cost_type_e = 'LINEAR_LS'

    # mpc = MPCQuadraticCostBoxConstr(lambda x,u: f(x, u), nx, nu, N, Tf, Q, R, P, alpha_f, K, xmin, xmax, umin, umax, Vx, Vu)


    ocp.cost.Vx = np.zeros((ny, nx_))
    ocp.cost.Vx[:nx,:nx] = np.eye(nx)

    ocp.cost.Vu = np.zeros((ny, nu))
    ocp.cost.Vu[nx_:,:nu] = np.eye(nu)

    ocp.cost.Vx_e = np.zeros((nx_, nx_))
    ocp.cost.Vx_e[:nx,:nx] = np.eye(nx)

    ocp.cost.yref = np.zeros((ny,))
    ocp.cost.yref_e = np.zeros((ny_e,))

    nxconstr = M+1
    nuconstr = 2*nu
    nconstr = nxconstr + nuconstr

    Lx = np.reshape(np.genfromtxt(fp.joinpath('mpc_parameters',f'Lx_{n_mass}.txt'), delimiter=','), (nx,nconstr)).T
    Lu = np.reshape(np.genfromtxt(fp.joinpath('mpc_parameters',f'Lu_{n_mass}.txt'), delimiter=','), (nu,nconstr)).T
    Ls = np.reshape(np.genfromtxt(fp.joinpath('mpc_parameters',f'Ls_{n_mass}.txt'), delimiter=','), (1,nconstr)).T
    print("Lx = \n", Lx,"\n")
    print("Lu = \n", Lu,"\n")
    print("Ls = \n", Ls,"\n")
    
    print("C = \n", np.hstack((Lx,Ls)), "\n")
    print("D = \n", Lu, "\n")

    # set constraints
    # ocp.constraints.constr_type = 'BGH'
    ocp.constraints.lbu = -np.ones((nu,))
    ocp.constraints.ubu = np.ones((nu,))
    ocp.constraints.idxbu = np.array(range(nu))

    ocp.constraints.C   = np.hstack((Lx,Ls))
    ocp.constraints.D   = Lu
    ocp.constraints.lg  = -100000*np.ones(nconstr)
    ocp.constraints.ug  = np.ones(nconstr)

    alpha_s = float(np.genfromtxt(fp.joinpath('mpc_parameters',f'alpha_s_{n_mass}.txt'), delimiter=','))
    ocp.constraints.lh_e = np.array([-10000])

    ocp.model.con_h_expr_e = ocp.model.x[:nx].T @ P @ ocp.model.x[:nx]
    alpha = alpha_f - alpha_s*(1-math.exp(-rho*Tf))/rho*w_bar
    print("\nalpha_f=", alpha_f, "\n")
    print("\nalpha_s=", alpha_s, "\n")
    print("\nalpha=", alpha, "\n")
    if alpha < 0:
        raise Exception("Terminal set size alpha_f - alpha_s * s_T is negative: ", alpha )
    else: 
        ocp.constraints.uh_e = np.array([ alpha**2 ])

    ocp.constraints.x0 = np.zeros(nx_)

    # ocp.constraints.Jsg = np.eye(nconstr)
    # L2_pen = 1e3
    # L1_pen = 1
    # ocp.cost.Zl = L2_pen * np.ones((nconstr,))
    # ocp.cost.Zu = L2_pen * np.ones((nconstr,))
    # ocp.cost.zl = L1_pen * np.ones((nconstr,))
    # ocp.cost.zu = L1_pen * np.ones((nconstr,))


    mpc = MPCQuadraticCostLxLu( f, nx, nu, N, Tf, Q, R, P, alpha_f, K, Lx, Lu, Kdelta, alpha_reduced=alpha, S=Sinit, Ls=Ls)
    mpc.name = model.name

    # solver options
    ocp.solver_options.qp_solver = 'FULL_CONDENSING_DAQP' # FULL_CONDENSING_QPOASES
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'IRK'
    ocp.solver_options.nlp_solver_type = 'SQP' # SQP_RTI
    ocp.solver_options.nlp_solver_max_iter = nlpiter

    if n_mass == 3:
        ocp.solver_options.levenberg_marquardt = 20.0
        ocp.solver_options.line_search_use_sufficient_descent = 1
        ocp.solver_options.alpha_reduction = 0.1
        ocp.solver_options.alpha_min = 0.001
        ocp.solver_options.regularize_method = 'MIRROR'
    elif n_mass == 4:
        ocp.solver_options.levenberg_marquardt = 20.0
        ocp.solver_options.line_search_use_sufficient_descent = 1
        ocp.solver_options.alpha_reduction = 0.1
        ocp.solver_options.alpha_min = 0.0001
        ocp.solver_options.regularize_method = 'CONVEXIFY'

    ocp.solver_options.hpipm_mode='ROBUST'
    # ocp.solver_options.qp_solver_iter_max=100
    ocp.solver_options.globalization='MERIT_BACKTRACKING'
    ocp.solver_options.globalization_use_SOC=1

    ocp.solver_options.sim_method_num_stages = 2
    ocp.solver_options.sim_method_num_steps = 2

    nlp_tol = 1e-4
    ocp.solver_options.qp_solver_cond_N = N
    ocp.solver_options.qp_tol = nlp_tol
    ocp.solver_options.tol = nlp_tol
    # ocp.solver_options.nlp_solver_tol_eq = 1e-9

    # set prediction horizon
    ocp.solver_options.tf = Tf

    if generate:
        acados_ocp_solver = AcadosOcpSolver(ocp, json_file = 'acados_ocp_' + model.name + '.json')


    xmin = -0.5*np.ones(nx)
    xmax = 0.5*np.ones(nx)
    xmin[:3*(M+1)] = -0.25*np.ones(3*(M+1))
    xmax[:3*(M+1)] = 0.25*np.ones(3*(M+1))
    for i in range(M+1):
        xmin[i*3+1] = -0.1

    umax = np.array([1/Lu[nxconstr+i, i] for i in range(nu)])
    umin = np.array([1/Lu[nxconstr+nu+i, i] for i in range(nu)])
    print("\numin =\n ",umin)
    print("\numax =\n ",umax)

    # sampler = RandomSampler(int(100),mpc.nx, 42)
    sampler = RandomSampler(numberofsamples, mpc.nx, randomseed, xmin, xmax)


    def run(x0):
            # reset to avoid false warmstart
            acados_ocp_solver = AcadosOcpSolver(ocp, json_file = 'acados_ocp_' + model.name + '.json', build = False, generate=False)
            
            # solve ocp
            acados_ocp_solver.set(0, "lbx", np.append(x0,0))
            acados_ocp_solver.set(0, "ubx", np.append(x0,0))

            Xinit = np.linspace(x0,np.zeros(nx), N+1)
            Uinit = np.zeros((N,nu))
            
            for i in range(N):
                Uinit[i] = K @ Xinit[i]
                Uinit[i] = np.clip(Uinit[i], umin-Kdelta@Xinit[i], umax-Kdelta@Xinit[i])
                Xinit[i+1] = mpc.forward_simulate_single_step(Xinit[i], Uinit[i])

            for i in range(N):
                acados_ocp_solver.set(i, "x", np.append(Xinit[i], Sinit[i]))
                acados_ocp_solver.set(i, "u", Uinit[i])

            status = acados_ocp_solver.solve()

            # if status == 1 or status == 2 or status == 4:
            #     status = acados_ocp_solver.solve()


            # if status != 0 and status !=2:
            #     print('acados acados_ocp_solver returned status {}. Exiting.'.format(status))
            #     print(x0)
            # acados_ocp_solver.print_statistics()

            X = np.ndarray((N+1, nx))
            U = np.ndarray((N, nu))
            for i in range(N):
                X[i,:] = acados_ocp_solver.get(i, "x")[:-1]
                U[i,:] = acados_ocp_solver.get(i, "u")
            X[N,:] = acados_ocp_solver.get(N, "x")[:-1]
            # print(S)
            computetime = float(acados_ocp_solver.get_stats('time_tot'))
            print(status)
            return X,U, status, computetime

    # print(run([-5, -5, -10, 0,0,0, 0,0,0,0 ]))
    # print(run([1, 1, 0, 0,0,0, 0,0,0,0 ]))
    # print(run([1, 1, 0, 0,0,0, 0,0,0,0 ]))

    _,_,_,_, outfile = sample_dataset_from_mpc(mpc, run, sampler, experimentname, verbose=verbose)
    print("Outfile", outfile)

    if showplot:
        x0dataset, Udataset, Xdataset, computetimes = import_dataset(mpc, outfile)
        
        dimx = 0
        dimy = 1
        plot_feas(x0dataset[:,dimx],x0dataset[:,dimy])
    
        dimx = 0
        dimy = 2
        plot_feas(x0dataset[:,dimx],x0dataset[:,dimy])

    return outfile

def parallel_sample_mpc(n_mass=3, instances=16, samplesperinstance=int(1e5), prefix="Cluster"):
    now = get_date_string()

    fp = Path(os.path.abspath(os.path.dirname(__file__)))
    print("\n\n===============================================")
    print("Running", instances, "as process to produce", samplesperinstance,"datapoints each")
    print("===============================================\n")

    os.chdir(fp)
    datasetpath = str(fp.joinpath(os.path.abspath(fp),'datasets'))
    print("datasetpath = ", datasetpath)
    processes = []
    parallel_experiments_common_name = prefix+"_"+str(now)+"_"
    for i in range(instances):
        # command = ["python3", "01_samplempc.py", "--showplot=False", "--randomseed=None", "--experimentname=Docker_"+str(now)+"_"+str(i)+"_", "--numberofsamples="+str(samplesperinstance)]
        experimentname = parallel_experiments_common_name+"_"+str(i)+"_"
        command = [
            "python3",
            "10_samplempc.py",
            "sample_mpc",
            f"--n_mass={n_mass}",
            "--showplot=False",
            "--randomseed=None",        # all processes run with different random seed
            f"--experimentname={experimentname}",
            f"--numberofsamples={samplesperinstance}",
            "--generate=False"]         # don't export acados ocp json (which might cause file access issues in parallel)

        with open(fp.joinpath('logs',experimentname+".log"),"wb") as out:
            p = subprocess.Popen(command,
                stdout=out, stderr=out)
            processes.append(p)

    for p in processes:
        p.wait()

    merge_parallel_jobs([parallel_experiments_common_name], new_dataset_name=parallel_experiments_common_name[:-1])

def computetime_test_fwd_sim_chainmass(n_mass, dataset="latest"):
    name = f'chain_mass_{n_mass}'
    model = export_chain_mass_model(n_mass)
    Tf = float(np.genfromtxt(fp.joinpath('mpc_parameters','Tf.txt'), delimiter=','))
    N = 10
    sim = AcadosSim()
    sim.model = model

    sim.solver_options.T = Tf/N
    sim.solver_options.integrator_type = 'IRK'
    sim.solver_options.num_stages  = 4                     # number of stages in the integrator
    sim.solver_options.num_steps   = 1                     # number of steps in the integrator
    sim.solver_options.newton_iter = 3                     # number of Newton iterations in simulation method

    acados_integrator = AcadosSimSolver(sim, 'acados_ocp_' + name + '_sim.json')

    def run(x0, V):
        X = np.zeros((V.shape[0]+1,x0.shape[0]+1 ))
        X[0] = np.copy(np.append(x0,0))
        for i in range(len(V)):
            X[i+1] = acados_integrator.simulate(x=X[i], u=V[i])
        return X
    computetime_test_fwd_sim(run, dataset)

if __name__ == "__main__":
    fire.Fire({
        'sample_mpc': sample_mpc,
        'parallel_sample_mpc':parallel_sample_mpc,
        'merge_single_parallel_job':merge_single_parallel_job,
        'print_dataset_statistics':print_dataset_statistics,
        'computetime_test_fwd_sim_chainmass': computetime_test_fwd_sim_chainmass,
        })