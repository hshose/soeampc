from casadi import tan
def f(x, u):
    d0=10
    d1=8
    n0=10
    kT=0.91
    g=9.8
    return [x[3], x[4], x[5],
        g*tan(x[6]), g*tan(x[8]), kT*u[2],
        -d1*x[6]+x[7], -d0*x[6]+n0*u[0],
        -d1*x[8]+x[9], -d0*x[8]+n0*u[1]]