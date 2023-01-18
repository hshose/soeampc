# from math import *
from casadi import sqrt, SX
def f(xrel,u, force=None):
    
    nx = xrel.shape[0]
    xref = [0.243567, -0.0, -0.515431, 0.5, -0.0, -0.696318, 0.756433, -0.0, -0.515431, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    x = [xrel[i] + xrel[i] for i in range(nx)]

    M = int(( nx/3 - 1 ) / 2)
    
    if not nx == (2*M + 1)*3:
        raise Exception("wrong number of state dimensions calculated!")

    L = 0.033
    D = 1.0
    m = 0.033

    nxpos = (M+1)*3
    nxvel = M*3

    # xpos = SX.sym('xpos', (M+1)*3, 1) # position of fix mass eliminated
    # xvel = SX.sym('xvel', M*3, 1)
    # u = SX.sym('u', nu, 1)
    # xdot = SX.sym('xdot', nx, 1)

    force = [0 for i in range(3*M)]

    for i in range(M):
        force[3*i+2] = - 9.81

    for i in range(M+1):
        if i == 0:
            dist = [x[i*3+j] for j in range(3)]
        else:
            dist = [x[i*3+j] - x[(i-1)*3+j] for j in range(3)]

        scale = D/m*(1-L/ sqrt(dist[0]**2+dist[1]**2+dist[2]**2))
        F = [scale*dist[j] for j in range(3)]

        # mass on the right
        if i < M:
            force[i*3] -=   F[0]
            force[i*3+1] -= F[1]
            force[i*3+2] -= F[2]
    
        # mass on the left
        if i > 0:
            force[(i-1)*3  ] += F[0]
            force[(i-1)*3+1] += F[1]
            force[(i-1)*3+2] += F[2]

    fexpl = []
    for i in range(nxvel):
        fexpl.append(x[nxpos+i])
    for i in range(3):
        fexpl.append(u[i])
    for i in range(nxvel):
        fexpl.append(force[i])

    return fexpl