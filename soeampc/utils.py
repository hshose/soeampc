from pathlib import Path
from datetime import datetime
import os
import errno
import numpy as np

import importlib
import inspect

from .mpcproblem import *

def export_dataset(mpc, x0dataset, Udataset, Xdataset, computetimes, filename):
    date = datetime.now().strftime("%Y%m%d-%H%M%S")
    Nsamples = np.shape(x0dataset)[0]

    print("\nExporting Dataset with Nvalid",Nsamples,"feasible samples\n")
    
    datasetname = mpc.name+"_N_"+str(Nsamples)+"_"+filename+date
    
    p = Path("datasets").joinpath(datasetname, "data")
    p.mkdir(parents=True,exist_ok=True)
    np.savetxt(p.joinpath("x0.txt") ,  x0dataset,    delimiter=",")
    np.savetxt(p.joinpath("X.txt")  ,  np.reshape( Xdataset, ( Nsamples, mpc.nx*(mpc.N+1))),  delimiter=",")
    np.savetxt(p.joinpath("U.txt")  ,  np.reshape( Udataset, ( Nsamples, mpc.nu*mpc.N)),    delimiter=",")
    np.savetxt(p.joinpath("ct.txt") ,  computetimes, delimiter=",")

    p = Path("datasets").joinpath(datasetname, "parameters")
    p.mkdir(parents=True,exist_ok=True)
    np.savetxt(p.joinpath("nx.txt"),    np.array([mpc.nx]), fmt='%i', delimiter=",")
    np.savetxt(p.joinpath("nu.txt"),    np.array([mpc.nu]), fmt='%i', delimiter=",")
    np.savetxt(p.joinpath("N.txt"),     np.array([mpc.N]),  fmt='%i', delimiter=",")
    np.savetxt(p.joinpath("Tf.txt"),    np.array([mpc.Tf]),           delimiter=",")

    np.savetxt(p.joinpath("xmax.txt"),    np.array([mpc.xmax]), delimiter=",")
    np.savetxt(p.joinpath("xmin.txt"),    np.array([mpc.xmin]), delimiter=",")
    np.savetxt(p.joinpath("umax.txt"),    np.array([mpc.umax]), delimiter=",")
    np.savetxt(p.joinpath("umin.txt"),    np.array([mpc.umin]), delimiter=",")
    
    np.savetxt(p.joinpath("P.txt"),     np.array(mpc.P),              delimiter=",")
    np.savetxt(p.joinpath("Q.txt"),     np.array(mpc.Q),              delimiter=",")
    np.savetxt(p.joinpath("R.txt"),     np.array(mpc.R),              delimiter=",")
    np.savetxt(p.joinpath("alpha.txt"), np.array([mpc.alpha]),        delimiter=",")
    np.savetxt(p.joinpath("K.txt") ,    np.array(mpc.K),              delimiter=",")

    ffile = open(p.joinpath("f.py"),'w')
    ffile.write(inspect.getsource(mpc.f))
    ffile.close()

    print("Exported to directory:\n\t",  Path("datasets").joinpath(datasetname).absolute(),"\n")

    target = datasetname
    link_name=Path("datasets").joinpath("latest")
    try:
        os.symlink(target, link_name)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise e
    return datasetname

def export_model(model, datasetname, modelname):
    p = Path("models").joinpath(datasetname)
    model.save(p.joinpath(modelname))
    link_name=p.joinpath("latest")
    target=modelname
    try:
        os.symlink(target, link_name)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise e


def import_dataset(mpc, file="latest"):
    p = Path("datasets").joinpath(file, "data")
    x0raw       = np.genfromtxt( p.joinpath("x0.txt"), delimiter=",")
    Xraw        = np.genfromtxt( p.joinpath("X.txt"), delimiter=",")
    Uraw        = np.genfromtxt( p.joinpath("U.txt"), delimiter=",")
    computetimes    = np.genfromtxt( p.joinpath("ct.txt"), delimiter=",")

    Nsamples  =  np.shape(x0raw)[0]
    x0dataset =  x0raw.reshape( Nsamples,   mpc.nx )
    Udataset  =  Uraw.reshape(  Nsamples,   mpc.N,   mpc.nu)
    Xdataset  =  Xraw.reshape(  Nsamples,   mpc.N+1, mpc.nx)

    return x0dataset, Udataset, Xdataset, computetimes
    

def import_mpc(file="latest"):
    p = Path("datasets").joinpath(file, "parameters")
    
    nx = int(np.genfromtxt( p.joinpath( 'nx.txt'), delimiter=',', dtype="int"))
    nu = int(np.genfromtxt( p.joinpath( 'nu.txt'), delimiter=',', dtype="int"))
    N  = int(np.genfromtxt( p.joinpath( 'N.txt'),  delimiter=',', dtype="int"))
    Tf = float(np.genfromtxt( p.joinpath( 'Tf.txt'), delimiter=','))
    alpha_f = float(np.genfromtxt( p.joinpath( 'alpha.txt'), delimiter=','))

    xmax = np.genfromtxt( p.joinpath('xmax.txt'),   delimiter=',')
    xmin = np.genfromtxt( p.joinpath('xmin.txt'),   delimiter=',')
    umax = np.genfromtxt( p.joinpath('umax.txt'),   delimiter=',')
    umin = np.genfromtxt( p.joinpath('umin.txt'),   delimiter=',')

    Q = np.reshape( np.genfromtxt( p.joinpath( 'Q.txt' ), delimiter=','), (nx,nx))
    P = np.reshape( np.genfromtxt( p.joinpath( 'P.txt' ), delimiter=','), (nx,nx))
    R = np.reshape( np.genfromtxt( p.joinpath( 'R.txt' ), delimiter=','), (nu,nu))
    K = np.reshape( np.genfromtxt( p.joinpath( 'K.txt' ), delimiter=','), (nx, nu))

    
    spec = importlib.util.spec_from_file_location("f", p.joinpath("f.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    f = mod.f

    return MPCQuadraticCostBoxConstr(f, nx, nu, N, Tf, Q, R, P, alpha_f, K, xmin, xmax, umin, umax)