from pathlib import Path
from datetime import datetime
import numpy as np

def export_dataset(mpc, x0dataset, Udataset, Xdataset, computetimes, filename):
    date = datetime.now().strftime("%Y%m%d-%H%M%S")
    Nsamples = np.shape(x0dataset)[0]

    print("\nExporting Dataset with N=",Nsamples," samples\n")
    
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
    np.savetxt(p.joinpath("Tf.txt"),    np.array([mpc.Tf]), fmt='%i', delimiter=",")
    
    np.savetxt(p.joinpath("P.txt"),     np.array(mpc.P),              delimiter=",")
    np.savetxt(p.joinpath("Q.txt"),     np.array(mpc.Q),              delimiter=",")
    np.savetxt(p.joinpath("R.txt"),     np.array(mpc.R),              delimiter=",")
    np.savetxt(p.joinpath("alpha.txt"), np.array([mpc.alpha]),          delimiter=",")
    np.savetxt(p.joinpath("K.txt") ,    np.array(mpc.K),              delimiter=",")

    return datasetname

def import_dataset(mpc, file):
    p = Path("datasets").joinpath(file, "data")
    x0raw       = np.genfromtxt( p.joinpath("x0.txt"), delimiter=",")
    Xraw        = np.genfromtxt( p.joinpath("X.txt"), delimiter=",")
    Uraw        = np.genfromtxt( p.joinpath("U.txt"), delimiter=",")
    computetimes    = np.genfromtxt( p.joinpath("ct.txt"), delimiter=",")

    print(np.shape(x0raw))
    Nsamples  =  np.shape(x0raw)[0]
    x0dataset =  x0raw.reshape( Nsamples,   mpc.nx )
    Udataset  =  Uraw.reshape(  Nsamples,   mpc.N,   mpc.nu)
    Xdataset  =  Xraw.reshape(  Nsamples,   mpc.N+1, mpc.nx)

    return x0dataset, Udataset, Xdataset, computetimes
