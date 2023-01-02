from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver, AcadosModel
from casadi import SX, vertcat, sin, cos, Function, sign, tanh
import numpy as np
import scipy.linalg
import math

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..','..'))
from pathlib import Path

fp = Path(os.path.dirname(__file__))
os.chdir(fp)

from soeampc.sampler import RandomSampler
from soeampc.samplempc import sampledataset
from soeampc.mpcproblem import MPCQuadraticCostLxLu

from dynamics.f import f
from plot import *

import fire

def samplempc(
        showplot=True,
        experimentname="",
        numberofsamples=int(5000),
        randomseed=42,
        verbose=False,
        withstabilizingfeedback=True,
        generate=True,
        nlpiter=400
        ):


    print("\n\n===============================================")
    print("Setting up ACADOS OCP problem")
    print("===============================================\n")

    rho       = float(np.genfromtxt(fp.joinpath('mpc_parameters','rho_c.txt'), delimiter=',')) # 10
    w_bar     = float(np.genfromtxt(fp.joinpath('mpc_parameters','wbar.txt'), delimiter=',')) # 4.6e-1
    nx = 2
    nu = 1
    Kdelta = None
    if withstabilizingfeedback:
        Kdelta = np.reshape(np.genfromtxt(fp.joinpath('mpc_parameters','Kdelta.txt'), delimiter=','), (nx,nu)).T
        print("Kdelta=\n",Kdelta,"\n")

    def export_stirtank_ode_model():

        model_name = 'stirtank'

        # set up states & controls
        x = SX.sym('x', nx, 1)
        u = SX.sym('u', nu, 1)       
        xdot = SX.sym('xdot', nx, 1)
        s     =  SX.sym('s')
        sdot     =  SX.sym('sdot')

        if withstabilizingfeedback:
            v = SX.sym('u', nu, 1)
            u = Kdelta @ x + v

        fx = f(x,u)
        # rho       = 10
        # w_bar     = 4.6e-1
        f_impl = vertcat(vertcat(*fx)-xdot, -rho*s+w_bar-sdot)

        model = AcadosModel()

        model.f_impl_expr = f_impl
        # model.f_expl_expr = f_expl
        model.x = x
        model.xdot = xdot
        if withstabilizingfeedback:
            model.u = v
        else:
            model.u = u
        model.p = []
        model.name = model_name

        return model

    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    # set model
    model = export_stirtank_ode_model()
    ocp.model = model

    # model parameters
    Tf = float(np.genfromtxt(fp.joinpath('mpc_parameters','Tf.txt'), delimiter=','))
    N = 10
    if (nx != (model.x.size()[0]-1))  or (nu != model.u.size()[0]):
        raise Exception("Model nx, nu dimension mismatch")

    nx_ = model.x.size()[0]
    ny = nx_ + nu
    ny_e = nx_
    alpha_f = float(np.genfromtxt(fp.joinpath('mpc_parameters','alpha.txt'), delimiter=','))

    Sinit = odeint(lambda y,t: -rho*y+w_bar, 0, np.linspace(0,Tf,N+1))
    print("Sinit =\n",Sinit,"\n")


    Q = np.reshape(np.genfromtxt(fp.joinpath('mpc_parameters','Q.txt'), delimiter=','), (nx,nx))
    P = np.reshape(np.genfromtxt(fp.joinpath('mpc_parameters','P.txt'), delimiter=','), (nx,nx))
    R = np.reshape(np.genfromtxt(fp.joinpath('mpc_parameters','R.txt'), delimiter=','), (nu,nu))
    Q_ = scipy.linalg.block_diag(Q, 1)
    P_ = scipy.linalg.block_diag(P, 1)
    K = np.reshape(np.genfromtxt(fp.joinpath('mpc_parameters','K.txt'), delimiter=','), (nx, nu)).T
    Kdelta = np.reshape(np.genfromtxt(fp.joinpath('mpc_parameters','Kdelta.txt'), delimiter=','), (nx, nu)).T
        alpha_f = float(np.genfromtxt(fp.joinpath('mpc_parameters','alpha.txt'), delimiter=','))

    ocp.dims.N = N

    print("P = \n", P, "\n")
    print("Q = \n", Q, "\n")
    print("R = \n", R, "\n")


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

    # we just use one sided ug constraint:
    # C*x + D*u <= ug
    # [Lx; Ls]*[x,s] + Lu*u <=1

    nxconstr = 4
    nuconstr = 2
    nconstr = nxconstr+nuconstr

    Lx = np.reshape(np.genfromtxt(fp.joinpath('mpc_parameters','Lx.txt'), delimiter=','), (nx,nconstr)).T
    Lu = np.reshape(np.genfromtxt(fp.joinpath('mpc_parameters','Lu.txt'), delimiter=','), (nu,nconstr)).T
    Ls = np.reshape(np.genfromtxt(fp.joinpath('mpc_parameters','Ls.txt'), delimiter=','), (1,nconstr)).T

    # constrain umin <= Kdelta @ x + v <= umax
    # the imported Lu is of the form: Lx @ x + Lu @ u <= 1
    # therefore, the stabilizing feedback term is
    #            Lu @ Kdelta @ x + Lu @ v <= 1
    #           |---- Lx ----|
    if withstabilizingfeedback:
        Lx[nxconstr:nxconstr+nu, :]      = Lu[nxconstr:nxconstr+nu]@Kdelta
        Lx[nxconstr+nu:nxconstr+2*nu, :] = Lu[nxconstr+nu:nxconstr+2*nu]@Kdelta

    print("Lx = \n", Lx,"\n")
    print("Lu = \n", Lu,"\n")
    print("Ls = \n", Ls,"\n")

    print("C = \n", np.hstack((Lx,Ls)), "\n")
    print("D = \n", Lu, "\n")

    ocp.constraints.C   = Lx
    ocp.constraints.D   = Lu
    ocp.constraints.lg  = -100000*np.ones(nconstr) # set something very small to deactivate lg
    ocp.constraints.ug  = np.ones(nconstr)

    ocp.constraints.Jsg = np.eye(nconstr)
    L2_pen = 1e6
    L1_pen = 1e4
    ocp.cost.Zl = L2_pen * np.ones((nconstr,))
    ocp.cost.Zu = L2_pen * np.ones((nconstr,))
    ocp.cost.zl = L1_pen * np.ones((nconstr,))
    ocp.cost.zu = L1_pen * np.ones((nconstr,))


    ## Terminal set constraint
    ## x' * P * x \leq ( alpha - alpha_s * s_f )^2
    alpha_s = float(np.genfromtxt(fp.joinpath('mpc_parameters','alpha_s.txt'), delimiter=','))
    ocp.constraints.lh_e = np.array([-10000]) # set something very small to deactivate lh_e

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

    mpc = MPCQuadraticCostLxLu( f, nx, nu, N, Tf, Q, R, P, alpha_f, K, Lx, Lu, Kdelta, alpha_reduced=alpha, S=Sinit, Ls=Ls)
    mpc.name = model.name

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

    # set prediction horizon
    ocp.solver_options.tf = Tf

    # ocp.solver_options.print_level = 1
    ocp.solver_options.nlp_solver_max_iter = 1000
    # ocp.solver_options.sim_method_num_stages = 6
    ocp.solver_options.sim_method_newton_iter = 10
    # ocp.solver_options.sim_method_num_steps = 100

    if generate:
        acados_ocp_solver = AcadosOcpSolver(ocp, json_file = 'acados_ocp_' + model.name + '.json')


    # acados_ocp_solver = AcadosOcpSolver(ocp, json_file = 'acados_ocp_' + model.name + '.json')

    ue      = 0.7853
    xe1     = 0.2632
    xe2     = 0.6519
    umin = np.array([0-ue])
    umax = np.array([2-ue])
    xmin = np.array([-0.2,-0.2])
    xmax = np.array([0.2,0.2])
    xmin = np.array([1/Lx[i, i] for i in range(nx)])
    xmax = np.array([1/Lx[nx+i, i] for i in range(nx)])
    umax = np.array([1/Lu[nxconstr+i, i] for i in range(nu)])
    umin = np.array([1/Lu[nxconstr+nu+i, i] for i in range(nu)])
    print("\nxmin =\n ",xmin)
    print("\nxmax =\n ",xmax)
    print("\numin =\n ",umin)
    print("\numax =\n ",umax)
    sampler = RandomSampler(numberofsamples, mpc.nx, randomseed, xmin, xmax)

    # for i in range(N):
    #     print(acados_ocp_solver.get(i,'x'))
    #     print(acados_ocp_solver.get(i,'u'))

    def run(x0):
            # reset to avoid false warmstart
            acados_ocp_solver = AcadosOcpSolver(ocp, json_file = 'acados_ocp_' + model.name + '.json', build = False, generate=False)

            # solve ocp
            acados_ocp_solver.set(0, "lbx", np.append(x0,0))
            acados_ocp_solver.set(0, "ubx", np.append(x0,0))

            Xinit = np.linspace(x0,np.zeros(nx), N+1)
            Uinit = np.zeros((N,nu))

            Kinit = K
            for i in range(N):
                Uinit[i] = Kinit @ Xinit[i]
                Uinit[i] = np.clip(Uinit[i], umin-Kdelta@Xinit[i], umax-Kdelta@Xinit[i])
                Xinit[i+1] = mpc.singlestepsim(Xinit[i], Uinit[i])

            if verbose:
                print("\nx0 =\n",x0)
                print("\nXinit =\n ",Xinit)
                print("\nUinit =\n ",Uinit)
                print("\nfeasible = ",mpc.feasible(Xinit, Uinit, verbose=True))

            for i in range(N):
                acados_ocp_solver.set(i, "x", np.append(Xinit[i], Sinit[i]))
                acados_ocp_solver.set(i, "u", Uinit[i])

            status = acados_ocp_solver.solve()

            # if status != 0 and status !=2:
            #     print('acados acados_ocp_solver returned status {}. Exiting.'.format(status))
            #     print(x0)
            #     acados_ocp_solver.print_statistics()

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
            return X,U, status, computetime


    # experimentname = ""
    # samplesperaxis = 1000

    _,_,_,_, outfile = sampledataset(mpc, run, sampler, experimentname, runtobreak=True, verbose=verbose)
    # print("Outfile",outfile)
    # x0dataset, Udataset, Xdataset, computetimes = import_dataset(mpc, outfile)


    # from plot_stirtank import *
    # plot_feas(x0dataset,np.array([mpc.xmin[0], mpc.xmax[0]]), np.array([mpc.xmin[1], mpc.xmax[1]]))

    if showplot:
        x0dataset, Udataset, Xdataset, computetimes = import_dataset(mpc, outfile)
        
        dimx = 0
        dimy = 1
        plot_feas(x0dataset[:,dimx],x0dataset[:,dimy])
    
        dimx = 0
        dimy = 2
        plot_feas(x0dataset[:,dimx],x0dataset[:,dimy])

    return outfile

if __name__ == "__main__":
    fire.Fire(samplempc)