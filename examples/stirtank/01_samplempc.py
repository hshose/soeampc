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

from soeampc import *
from dynamics.f import f

def export_stirtank_ode_model():

    model_name = 'stirtank'

    # set up states & controls
    x1     = SX.sym('x1')
    x2     = SX.sym('x2')
    s     =  SX.sym('s')
    u      = SX.sym('u')
    # xdot
    x1_dot     = SX.sym('x1_dot')
    x2_dot     = SX.sym('x2_dot')
    s_dot     = SX.sym('s_dot')
    x = vertcat( x1, x2, s)
    xdot = vertcat(x1_dot, x2_dot, s_dot)
    fx = f(vertcat(x1, x2),u)
    fx1 = fx[0]
    fx2 = fx[1]
    rho       = float(np.genfromtxt(fp.joinpath('mpc_parameters','rho_c.txt'), delimiter=',')) # 10
    # rho       = 10
    # we set eta = 0.022
    w_bar     = float(np.genfromtxt(fp.joinpath('mpc_parameters','wbar.txt'), delimiter=',')) # 4.6e-1
    # w_bar     = 4.6e-1
    f_impl = vertcat(fx1-x1_dot,fx2-x2_dot, -rho*s+w_bar-s_dot)

    model = AcadosModel()

    model.f_impl_expr = f_impl
    # model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
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
model = export_stirtank_ode_model()
ocp.model = model

# model parameters
Tf = 20
N = 10
nx = model.x.size()[0]-1
nx_ = model.x.size()[0]
nu = model.u.size()[0]
ny = nx_ + nu
ny_e = nx_
alpha_f = float(np.genfromtxt(fp.joinpath('mpc_parameters','alpha.txt'), delimiter=','))



ue      = 0.7853
xe1     = 0.2632
xe2     = 0.6519
umin = np.array([0-ue])
umax = np.array([2-ue])
xmin = np.array([-0.2,-0.2])
xmax = np.array([0.2,0.2])
cj = np.genfromtxt(fp.joinpath('mpc_parameters','Ls.txt'), delimiter=',')

Q = np.reshape(np.genfromtxt(fp.joinpath('mpc_parameters','Q.txt'), delimiter=','), (nx,nx))
P = np.reshape(np.genfromtxt(fp.joinpath('mpc_parameters','P.txt'), delimiter=','), (nx,nx))
R = np.reshape(np.genfromtxt(fp.joinpath('mpc_parameters','R.txt'), delimiter=','), (nu,nu))
Q_ = scipy.linalg.block_diag(Q, 0)
P_ = scipy.linalg.block_diag(P, 0)

K = np.reshape(np.genfromtxt(fp.joinpath('mpc_parameters','K.txt'), delimiter=','), (nx, nu))

mpc = MPCQuadraticCostBoxConstr(f, nx, nu, N, Tf, Q, R, P, alpha_f, K, xmin, xmax, umin, umax)
mpc.name = model.name

ocp.dims.N = N

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

# ocp.constraints.lbu   = umin
# ocp.constraints.ubu   = umax
# ocp.constraints.idxbu = np.array(range(nu))

# ocp.constraints.lbx   = xmin
# ocp.constraints.ubx   = xmax
# ocp.constraints.idxbx = np.array(range(nx))

# setting general constraints --> lg <= C*x +D*u <= ug
# xmin <= x - Lx s 
#         x + Lx s <= xmax
# 1 <= x/xmin - s 
#      x/xmax + s <= 1

nxconstr = 2
nuconstr = 1
nconstr = 2*(nxconstr+nuconstr)
Ls = np.zeros(nconstr)
Lx = np.zeros((nconstr,nx_))
Lu = np.zeros((nconstr, nu))
ug = np.zeros(nconstr)
lg = np.zeros(nconstr)

Lx[:nxconstr, :-1] = np.diag(1/xmin)
Ls[:nxconstr] = -np.array(cj[0], cj[2])/xmin
lg[:nxconstr] = -100
ug[:nxconstr] = 1

Lx[nxconstr:2*nxconstr, :-1] = np.diag(1/xmax)
Ls[nxconstr:2*nxconstr] = np.array(cj[1], cj[3])/xmax
lg[nxconstr:2*nxconstr] = -100
ug[nxconstr:2*nxconstr] = 1

Lu[2*nxconstr:2*nxconstr+nuconstr, :] = np.diag(1/umin)
Ls[2*nxconstr:2*nxconstr+nuconstr] = -cj[4]/umin
lg[2*nxconstr:2*nxconstr+nuconstr] = -100
ug[2*nxconstr:2*nxconstr+nuconstr] = 1

Lu[2*nxconstr+nuconstr:2*nxconstr+2*nuconstr, :] = np.diag(1/umax)
Ls[2*nxconstr+nuconstr:2*nxconstr+2*nuconstr] = cj[5]/umax
lg[2*nxconstr+nuconstr:2*nxconstr+2*nuconstr] = -100
ug[2*nxconstr+nuconstr:2*nxconstr+2*nuconstr] = 1

Lx[:, -1] = Ls

ocp.constraints.C   = Lx
ocp.constraints.D   = Lu
ocp.constraints.lg  = lg
ocp.constraints.ug  = ug

# Pdelta = np.reshape(np.genfromtxt('parameter/Pdelta.txt', delimiter=','), (nx,nx))
# Ps = np.linalg.norm(scipy.linalg.sqrtm(P)*scipy.linalg.sqrtm(scipy.linalg.inv(Pdelta)))
# print("PS")
# print(Ps)
# ocp.model.con_h_expr_e = ocp.model.x[:nx_].T @ P @ ocp.model.x[:nx_] + Ps*ocp.model.x[-1]
# print(type(ocp.model.con_h_expr_e))
# print(model.con_h_expr_e.shape)
# ocp.constraints.lh_e = np.array([-1])
# ALPHA
# ocp.constraints.uh_e = np.array([alpha_f])

# x0 = np.array([np.deg2rad(45), np.deg2rad(5), np.deg2rad(5), 0, 0, 0, 0, 0])
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

# set prediction horizon
ocp.solver_options.tf = Tf

# ocp.solver_options.print_level = 1
ocp.solver_options.nlp_solver_max_iter = 1000
# ocp.solver_options.sim_method_num_stages = 6
ocp.solver_options.sim_method_newton_iter = 10
# ocp.solver_options.sim_method_num_steps = 100

acados_ocp_solver = AcadosOcpSolver(ocp, json_file = 'acados_ocp_' + model.name + '.json')

# for i in range(N):
#     print(acados_ocp_solver.get(i,'x'))
#     print(acados_ocp_solver.get(i,'u'))

def run(x0):
        # reset to avoid false warmstart
        acados_ocp_solver.reset()

        # solve ocp
        acados_ocp_solver.set(0, "lbx", np.append(x0,0))
        acados_ocp_solver.set(0, "ubx", np.append(x0,0))

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


experimentname = ""
samplesperaxis = 100

_,_,_,_, outfile = sampledataset(mpc, run,samplesperaxis, experimentname)
# print("Outfile",outfile)
x0dataset, Udataset, Xdataset, computetimes = import_dataset(mpc, outfile)


from plot_stirtank import *
plot_feas(x0dataset,np.array([mpc.xmin[0], mpc.xmax[0]]), np.array([mpc.xmin[1], mpc.xmax[1]]))