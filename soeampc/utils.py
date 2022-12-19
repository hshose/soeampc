from pathlib import Path
from datetime import datetime
import os
import errno
import numpy as np

import tensorflow.keras as keras

from .mpcproblem import *

def append_to_dataset(mpc, x0dataset, Udataset, Xdataset, computetimes, filename):
    Nsamples = np.shape(x0dataset)[0]
    p = Path("datasets").joinpath(filename, "data")
    p.mkdir(parents=True,exist_ok=True)
    f = open(p.joinpath("x0.txt"), 'a')
    np.savetxt(f,  x0dataset,    delimiter=",")
    f.close()
    f = open(p.joinpath("X.txt"), 'a')
    np.savetxt(f,  np.reshape( Xdataset, ( Nsamples, mpc.nx*(mpc.N+1))),  delimiter=",")
    f.close()
    f = open(p.joinpath("U.txt"), 'a')
    np.savetxt(f  ,  np.reshape( Udataset, ( Nsamples, mpc.nu*mpc.N)),    delimiter=",")
    f.close()
    f = open(p.joinpath("ct.txt"), 'a')
    np.savetxt(f,  computetimes, delimiter=",")
    f.close()

def getdatestring():
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def export_dataset(mpc, x0dataset, Udataset, Xdataset, computetimes, filename, barefilename=False):    
    date = getdatestring()
    Nsamples = np.shape(x0dataset)[0]

    # print("\nExporting Dataset with Nvalid",Nsamples,"feasible samples\n")
    
    datasetname = mpc.name+"_N_"+str(Nsamples)+"_"+filename+date

    if barefilename:
        datasetname=filename
    
    p = Path("datasets").joinpath(datasetname, "data")
    p.mkdir(parents=True,exist_ok=True)
    np.savetxt(p.joinpath("x0.txt") ,  x0dataset,    delimiter=",")
    np.savetxt(p.joinpath("X.txt")  ,  np.reshape( Xdataset, ( Nsamples, mpc.nx*(mpc.N+1))),  delimiter=",")
    np.savetxt(p.joinpath("U.txt")  ,  np.reshape( Udataset, ( Nsamples, mpc.nu*mpc.N)),    delimiter=",")
    np.savetxt(p.joinpath("ct.txt") ,  computetimes, delimiter=",")

    mpc.savetxt(Path("datasets").joinpath(datasetname, "parameters"))

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
    x0raw       = np.loadtxt( p.joinpath("x0.txt"), delimiter=",")
    Xraw        = np.loadtxt( p.joinpath("X.txt"), delimiter=",")
    Uraw        = np.loadtxt( p.joinpath("U.txt"), delimiter=",")
    computetimes    = np.loadtxt( p.joinpath("ct.txt"), delimiter=",")

    Nsamples  =  np.shape(x0raw)[0]
    x0dataset =  x0raw.reshape( Nsamples,   mpc.nx )
    Udataset  =  Uraw.reshape(  Nsamples,   mpc.N,   mpc.nu)
    Xdataset  =  Xraw.reshape(  Nsamples,   mpc.N+1, mpc.nx)

    return x0dataset, Udataset, Xdataset, computetimes
    

def import_mpc(file="latest", mpcclass=MPCQuadraticCostBoxConstr):
    p = Path("datasets").joinpath(file, "parameters")
    mpc = mpcclass.genfromtxt(p)
    return mpc

def import_model(datasetname="latest", modelname="latest"):
    p = Path("models").joinpath(datasetname).joinpath(modelname)
    model = keras.models.load_model(p)
    return model