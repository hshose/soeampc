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


from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver, AcadosModel
from casadi import SX, vertcat, sin, cos, tan, Function, sign, tanh, jacobian
import numpy as np
import scipy.linalg
from scipy.integrate import odeint
import math

np.set_printoptions(edgeitems=3)
np.core.arrayprint._line_width = 200

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..','..'))
from pathlib import Path

fp = Path(os.path.dirname(__file__))
os.chdir(fp)

from soeampc import RandomSampler, sampledataset, MPCQuadraticCostBoxConstr, import_dataset

import fire
import importlib

def samplempc(n_mass, showplot=True, experimentname="", numberofsamples=int(1e5), randomseed=42, verbose=False):

    # import n_mass specific dynamics f
    # this can be rendered from `dynamics/f.template.py` and `mpc_parameters/xref_{{n_mass}}.txt` with:
    # ```$ python 01_renderdynamics --n_mass={{n_mass}}```
    spec = importlib.util.spec_from_file_location("f", fp.joinpath("dynamics","f_"+str(n_mass)+".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    f = mod.f
    
    M = n_mass - 2 # number of intermediate masses

   # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    def export_chain_mass_model(n_mass):

        model_name = 'chain_mass_' + str(n_mass)
        x0 = np.array([0, 0, 0]) # fix mass (at wall)


        nx = (2*M + 1)*3  # differential states
        nu = 3            # control inputs

        x = SX.sym('x', nx, 1) # position of fix mass eliminated
        u = SX.sym('u', nu, 1)
        xdot = SX.sym('xdot', nx, 1)

        # dynamics     
        fx = f(x,u)
        f_expl = vertcat(*fx)
        f_impl = xdot - f_expl

        model = AcadosModel()

        model.f_impl_expr = f_impl
        # model.f_expl_expr = f_expl
        model.x = x
        model.xdot = xdot
        model.u = u
        model.p = []
        model.name = model_name
        
        return model


    # set model
    model = export_chain_mass_model(n_mass)
    ocp.model = model

    nx = model.x.size()[0]
    nu = model.u.size()[0]
    ny = nx + nu
    ny_e = nx
    N = 40
    Ts = 0.2
    Tf = N * Ts

    xmin = -0.5*np.ones(nx)
    xmax = 0.5*np.ones(nx)
    Vx = np.zeros(nx)

    for i in range(M+1):
        xmin[i*3+1] = -0.1
        Vx[i*3+1] = 1

    umin = -np.ones(nu)
    umax = np.ones(nu)
    Vu = np.ones(nu)


    # set dimensions
    ocp.dims.N = N

    # set cost module
    ocp.cost.cost_type = 'LINEAR_LS'
    ocp.cost.cost_type_e = 'LINEAR_LS'

    Q = np.reshape(np.genfromtxt(fp.joinpath('mpc_parameters','Q_'+str(n_mass)+'.txt'), delimiter=','), (nx,nx))
    P = np.reshape(np.genfromtxt(fp.joinpath('mpc_parameters','P_'+str(n_mass)+'.txt'), delimiter=','), (nx,nx))
    R = np.reshape(np.genfromtxt(fp.joinpath('mpc_parameters','R_'+str(n_mass)+'.txt'), delimiter=','), (nu,nu))
    K = np.reshape(np.genfromtxt(fp.joinpath('mpc_parameters','K_'+str(n_mass)+'.txt'), delimiter=','), (nx,nu))
    alpha_f = float(np.genfromtxt(fp.joinpath('mpc_parameters','alpha_'+str(n_mass)+'.txt'), delimiter=','))
    
    mpc = MPCQuadraticCostBoxConstr(lambda x,u: f(x, u), nx, nu, N, Tf, Q, R, P, alpha_f, K, xmin, xmax, umin, umax, Vx, Vu)

    ocp.cost.W = scipy.linalg.block_diag(Q, R)
    ocp.cost.W_e = Q

    ocp.cost.Vx = np.zeros((ny, nx))
    ocp.cost.Vx[:nx,:nx] = np.eye(nx)

    Vu = np.zeros((ny, nu))
    Vu[nx:nx+nu, :] = np.eye(nu)
    ocp.cost.Vu = Vu

    ocp.cost.Vx_e = np.eye(nx)

    ocp.cost.yref = np.zeros((ny,))
    ocp.cost.yref_e = np.zeros((ny_e))

    # set constraints
    ocp.constraints.constr_type = 'BGH'
    ocp.constraints.lbu = -np.ones((nu,))
    ocp.constraints.ubu = np.ones((nu,))
    ocp.constraints.x0 = np.zeros((nx,))
    ocp.constraints.idxbu = np.array(range(nu))

    # alpha_s = float(np.genfromtxt(fp.joinpath('mpc_parameters','alpha_s.txt'), delimiter=','))
    # ocp.model.con_h_expr_e = ocp.model.x[:nx].T @ P @ ocp.model.x[:nx] # + alpha_s*ocp.model.x[-1]
    ocp.model.con_h_expr_e = ocp.model.x[:nx].T @ P @ ocp.model.x[:nx]
    # ocp.model.con_h_expr_e = ocp.model.x[:nx].T @ P @ ocp.model.x[:nx]
    ocp.constraints.lh_e = np.array([-1])
    # ALPHA
    ocp.constraints.uh_e = np.array([alpha_f])

    # wall constraint
    with_wall = True
    if with_wall:
        nbx = M + 1
        Jbx = np.zeros((nbx,nx))
        for i in range(nbx):
            Jbx[i, 3*i+1] = 1.0

        yPosWall = -0.1

        ocp.constraints.Jbx = Jbx
        ocp.constraints.lbx = yPosWall * np.ones((nbx,))
        ocp.constraints.ubx = 1e9 * np.ones((nbx,))

        # slacks
        ocp.constraints.Jsbx = np.eye(nbx)
        L2_pen = 1e3
        L1_pen = 1
        ocp.cost.Zl = L2_pen * np.ones((nbx,))
        ocp.cost.Zu = L2_pen * np.ones((nbx,))
        ocp.cost.zl = L1_pen * np.ones((nbx,))
        ocp.cost.zu = L1_pen * np.ones((nbx,))


    # solver options
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'IRK'
    ocp.solver_options.nlp_solver_type = 'SQP' # SQP_RTI
    ocp.solver_options.nlp_solver_max_iter = 10000

    ocp.solver_options.hpipm_mode='ROBUST'
    ocp.solver_options.qp_solver_iter_max=1000
    ocp.solver_options.globalization='MERIT_BACKTRACKING'
    ocp.solver_options.globalization_use_SOC=1

    ocp.solver_options.sim_method_num_stages = 2
    ocp.solver_options.sim_method_num_steps = 2

    nlp_tol = 1e-5
    ocp.solver_options.qp_solver_cond_N = N
    ocp.solver_options.qp_tol = nlp_tol
    ocp.solver_options.tol = nlp_tol
    # ocp.solver_options.nlp_solver_tol_eq = 1e-9

    # set prediction horizon
    ocp.solver_options.tf = Tf

    acados_ocp_solver = AcadosOcpSolver(ocp, json_file = 'acados_ocp_' + model.name + '.json')


    def run(x0):
            # reset to avoid false warmstart
            acados_ocp_solver.reset()
            print("x0:",x0)

            # x0 = np.zeros(nx)
            # x0[4] = 0.1

            # solve ocp
            acados_ocp_solver.set(0, "lbx", x0)
            acados_ocp_solver.set(0, "ubx", x0)

            Xinit = np.linspace(x0,np.zeros(nx), N+1)
            Uinit = np.array([K.T@x for x in Xinit])

            for i in range(N):
                acados_ocp_solver.set(i, "x", Xinit[i])
                acados_ocp_solver.set(i, "u", Uinit[i])

            status = acados_ocp_solver.solve()

            if status == 1 or status == 2 or status == 4:
                status = acados_ocp_solver.solve()


            if status != 0 and status !=2:
                print('acados acados_ocp_solver returned status {}. Exiting.'.format(status))
                print(x0)
                acados_ocp_solver.print_statistics()

            X = np.ndarray((N+1, nx))
            U = np.ndarray((N, nu))
            for i in range(N):
                X[i,:] = acados_ocp_solver.get(i, "x")
                U[i,:] = acados_ocp_solver.get(i, "u")
            X[N,:] = acados_ocp_solver.get(N, "x")
            # print(S)
            computetime = float(acados_ocp_solver.get_stats('time_tot'))
            print(status)
            return X,U, status, computetime


    # sampler = RandomSampler(int(100),mpc.nx, 42)
    sampler = RandomSampler(numberofsamples,mpc.nx, randomseed)

    # print(run([-5, -5, -10, 0,0,0, 0,0,0,0 ]))
    # print(run([1, 1, 0, 0,0,0, 0,0,0,0 ]))
    # print(run([1, 1, 0, 0,0,0, 0,0,0,0 ]))

    _,_,_,_, outfile = sampledataset(mpc, run, sampler, experimentname, runtobreak=True, verbose=verbose)
    print("Outfile", outfile)

    if showplot:
        x0dataset, Udataset, Xdataset, computetimes = import_dataset(mpc, outfile)
        
        dimx = 3
        dimy = 4
        plot_feas(x0dataset[:,dimx],x0dataset[:,dimy],np.array([mpc.xmin[dimx], mpc.xmax[dimx]]), np.array([mpc.xmin[dimy], mpc.xmax[dimy]]))
    
    return outfile


if __name__ == "__main__":
    fire.Fire(samplempc)