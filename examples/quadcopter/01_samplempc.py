from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver, AcadosModel
from casadi import SX, vertcat, sin, cos, tan, Function, sign, tanh
import numpy as np
import scipy.linalg
import math

np.set_printoptions(edgeitems=3)
np.core.arrayprint._line_width = 200

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..','..'))
from pathlib import Path

fp = Path(os.path.dirname(__file__))
os.chdir(fp)

from soeampc import RandomSampler, sampledataset, MPCQuadraticCostBoxConstr
from dynamics.f import f

from plot import *

import fire

def samplempc(showplot=True, experimentname="", numberofsamples=int(1e5), randomseed=42):

    def export_quadcopter_ode_model():

        model_name = 'quadcopter'

        # set up states & controls
        nx = 10
        nu = 3
        x = SX.sym('x', nx, 1)
        u = SX.sym('u', nu, 1)
        xdot = SX.sym('xdot', nx, 1)
        s     =  SX.sym('s')
        sdot     =  SX.sym('sdot')

        # xdot
        fx = f(x,u)
        rho       = float(np.genfromtxt(fp.joinpath('mpc_parameters','rho_c.txt'), delimiter=',')) # 10
        # rho       = 10
        # we set eta = 0.022
        w_bar     = float(np.genfromtxt(fp.joinpath('mpc_parameters','wbar.txt'), delimiter=',')) # 4.6e-1
        # w_bar     = 4.6e-1
        f_impl = vertcat(vertcat(*fx)-xdot, -rho*s+w_bar-sdot)

        model = AcadosModel()

        model.f_impl_expr = f_impl
        # model.f_expl_expr = f_expl
        model.x = vertcat(x, s)
        model.xdot = vertcat(xdot, sdot)
        model.u = u
        # model.z = z
        model.p = []
        model.name = model_name

        return model

    print("\n\n===============================================")
    print("Setting up ACADOS OCP problem")
    print("===============================================\n")

    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    # set model
    model = export_quadcopter_ode_model()
    ocp.model = model

    # model parameters
    Tf = float(np.genfromtxt(fp.joinpath('mpc_parameters','Tf.txt'), delimiter=','))
    N = 10
    nx = model.x.size()[0]-1
    nx_ = model.x.size()[0]
    nu = model.u.size()[0]
    ny = nx_ + nu
    ny_e = nx_
    alpha_f = float(np.genfromtxt(fp.joinpath('mpc_parameters','alpha.txt'), delimiter=','))



    Q = np.reshape(np.genfromtxt(fp.joinpath('mpc_parameters','Q.txt'), delimiter=','), (nx,nx))
    P = np.reshape(np.genfromtxt(fp.joinpath('mpc_parameters','P.txt'), delimiter=','), (nx,nx))
    R = np.reshape(np.genfromtxt(fp.joinpath('mpc_parameters','R.txt'), delimiter=','), (nu,nu))
    Q_ = scipy.linalg.block_diag(Q, 1)
    P_ = scipy.linalg.block_diag(P, 1)
    K = np.reshape(np.genfromtxt(fp.joinpath('mpc_parameters','K.txt'), delimiter=','), (nx,nu))

    xmin = np.array([-3, -8, -8, -5, -5, -5, -math.pi/4, -4*math.pi, -math.pi/4, -4*math.pi]) 
    xmax = np.array([ 1,  8,  8,  5,  5,  5,  math.pi/4,  4*math.pi,  math.pi/4,  4*math.pi]) 

    umin = np.array([ -math.pi/9, -math.pi/9, -9.8/0.91       ])
    umax = np.array([  math.pi/9,  math.pi/9,  2*9.8-9.8/0.91 ])

    mpc = MPCQuadraticCostBoxConstr(f, nx, nu, N, Tf, Q, R, P, alpha_f, K, xmin, xmax, umin, umax)
    mpc.name = model.name

    ocp.dims.N = N

    print("P", P)
    print("Q", Q)
    print("R", R)

    ocp.cost.W_e = P_
    ocp.cost.W = scipy.linalg.block_diag(Q_, R)

    ocp.cost.cost_type = 'LINEAR_LS'
    ocp.cost.cost_type_e = 'LINEAR_LS'

    ocp.cost.Vx = np.zeros((ny, nx_))
    ocp.cost.Vx[:nx,:nx] = np.eye(nx)

    ocp.cost.Vu = np.zeros((ny, nu))
    ocp.cost.Vu[nx_:,:nu] = np.eye(nu)

    ocp.cost.Vx_e = np.zeros((nx_, nx_))
    ocp.cost.Vx_e[:nx,:nx] = np.eye(nx)

    ocp.cost.yref         = np.zeros((ny, ))
    ocp.cost.yref_e       = np.zeros((ny_e, ))

    # setting general constraints --> lg <= C*x +D*u <= ug
    # xmin <= x - Lx s 
    #         x + Lx s <= xmax
    # 1 <= x/xmin - s 
    #      x/xmax + s <= 1

    nconstr = 11
    Lx = np.reshape(np.genfromtxt(fp.joinpath('mpc_parameters','Lx.txt'), delimiter=','), (nx,nconstr)).T
    Lu = np.reshape(np.genfromtxt(fp.joinpath('mpc_parameters','Lu.txt'), delimiter=','), (nu,nconstr)).T
    Ls = np.reshape(np.genfromtxt(fp.joinpath('mpc_parameters','Ls.txt'), delimiter=','), (1,nconstr)).T

    lg = -100000*np.ones(nconstr)
    ug = 100*np.ones(nconstr)

    print("Lx", Lx)
    print("Lu", Lu)
    print("Ls", Ls)

    print("C", np.hstack((Lx,Ls)) )
    print("D", Lu )

    ocp.constraints.C   = np.hstack((Lx,Ls))
    ocp.constraints.D   = Lu
    ocp.constraints.lg  = -100*np.ones(nconstr)
    ocp.constraints.ug  = np.ones(nconstr)

    alpha_s = float(np.genfromtxt(fp.joinpath('mpc_parameters','alpha_s.txt'), delimiter=','))
    ocp.model.con_h_expr_e = ocp.model.x[:nx].T @ P @ ocp.model.x[:nx] + alpha_s*ocp.model.x[-1]
    # ocp.model.con_h_expr_e = ocp.model.x[:nx].T @ P @ ocp.model.x[:nx]
    ocp.constraints.lh_e = np.array([-1])
    # ALPHA
    ocp.constraints.uh_e = np.array([alpha_f])

    ocp.constraints.x0 = np.zeros(nx_)

    # set options
    ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES
    # ocp.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES' # FULL_CONDENSING_QPOASES
    # PARTIAL_CONDENSING_HPIPM, FULL_CONDENSING_QPOASES, FULL_CONDENSING_HPIPM,
    # PARTIAL_CONDENSING_QPDUNES, PARTIAL_CONDENSING_OSQP
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    # ocp.solver_options.hessian_approx = 'EXACT'
    ocp.solver_options.integrator_type = 'IRK'
    ocp.solver_options.nlp_solver_type = 'SQP' # SQP_RTI, SQP
    # ocp.solver_options.levenberg_marquardt = 0.1
    # ocp.solver_options.hpipm_mode = 'ROBUST'
    # ocp.solver_options.regularize_method = 'CONVEXIFY'
    ocp.solver_options.hpipm_mode='ROBUST'
    ocp.solver_options.qp_solver_iter_max=200

    # set prediction horizon
    ocp.solver_options.tf = Tf

    # ocp.solver_options.print_level = 2
    ocp.solver_options.nlp_solver_max_iter = 100
    # ocp.solver_options.sim_method_num_stages = 6
    ocp.solver_options.sim_method_newton_iter = 10
    # ocp.solver_options.sim_method_num_steps = 100

    acados_ocp_solver = AcadosOcpSolver(ocp, json_file = 'acados_ocp_' + model.name + '.json')

    # for i in range(N):
    #     print(acados_ocp_solver.get(i,'x'))
    #     print(acados_ocp_solver.get(i,'u'))

    def run(x0):
            # reset to avoid false warmstart
            # acados_ocp_solver.reset()

            # solve ocp
            acados_ocp_solver.set(0, "lbx", np.append(x0,0))
            acados_ocp_solver.set(0, "ubx", np.append(x0,0))

            status = acados_ocp_solver.solve()

            if status == 1 or status == 2 or status == 4:
                status = acados_ocp_solver.solve()


            # if status != 0 and status !=2:
                # print('acados acados_ocp_solver returned status {}. Exiting.'.format(status))
                # print(x0)
                # acados_ocp_solver.print_statistics()

            X = np.ndarray((N+1, nx))
            S = np.ndarray(N+1)
            U = np.ndarray((N, nu))
            for i in range(N):
                X[i,:] = acados_ocp_solver.get(i, "x")[:-1]
                S[i] = acados_ocp_solver.get(i, "x")[-1]
                U[i,:] = acados_ocp_solver.get(i, "u")
            X[N,:] = acados_ocp_solver.get(N, "x")[:-1]
            S[N] = acados_ocp_solver.get(N, "x")[-1]
            # print(S)
            computetime = float(acados_ocp_solver.get_stats('time_tot'))
            # print(status)
            return X,U, status, computetime


    # sampler = RandomSampler(int(100),mpc.nx, 42)
    sampler = RandomSampler(numberofsamples,mpc.nx, randomseed)

    # print(run([-5, -5, -10, 0,0,0, 0,0,0,0 ]))
    # print(run([1, 1, 0, 0,0,0, 0,0,0,0 ]))
    # print(run([1, 1, 0, 0,0,0, 0,0,0,0 ]))

    _,_,_,_, outfile = sampledataset(mpc, run, sampler, experimentname,runtobreak=True)
    print("Outfile", outfile)

    if showplot:
        x0dataset, Udataset, Xdataset, computetimes = import_dataset(mpc, outfile)
        
        dimx = 0
        dimy = 1
        plot_feas(x0dataset[:,dimx],x0dataset[:,dimy],np.array([mpc.xmin[xdim], mpc.xmax[xdim]]), np.array([mpc.xmin[ydim], mpc.xmax[ydim]]))
    
    return outfile


if __name__ == "__main__":
    fire.Fire(samplempc)