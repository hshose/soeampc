from pathlib import Path
from datetime import datetime
import os
import errno
import numpy as np
import shutil

from tqdm import tqdm

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

def get_date_string():
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def export_dataset(mpc, x0dataset, Udataset, Xdataset, computetimes, filename, barefilename=False):    
    date = get_date_string()
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

def mergesamples(folder_names, new_dataset_name=get_date_string(), remove_after_merge=False):
    p=Path("datasets")
    mpc = import_mpc(folder_names[0], MPCQuadraticCostLxLu)
    print("file:", folder_names[0])
    x0dataset, Udataset, Xdataset, computetimes = import_dataset(mpc, folder_names[0])
    Nsamples = x0dataset.shape[0]
    exporttempfilename = export_dataset(mpc, x0dataset, Udataset, Xdataset, computetimes, mpc.name+"_N_temp_merged_"+str(new_dataset_name), barefilename=True)
    for f in tqdm(folder_names[1:]):
        print("file:", f)
        x0dataset, Udataset, Xdataset, computetimes = import_dataset(mpc, f)
        Nsamples = Nsamples + x0dataset.shape[0]
        append_to_dataset(mpc, x0dataset, Udataset, Xdataset, computetimes, exporttempfilename)

    print("\ncollected a total of ", str(Nsamples), "sample points")
    exportfilename = mpc.name+"_N_"+str(Nsamples)+"_merged_"+str(new_dataset_name)
    os.rename(p.joinpath(exporttempfilename), p.joinpath(exportfilename))
    print("\nExported merged dataset to:\n")
    print("\t", exportfilename,"\n")

    if remove_after_merge:
        print("\n\nRemoving Folders:\n")
        print("\t", folder_names)
        for f in folder_names:
            shutil.rmtree(p.joinpath(f), ignore_errors=True)

    return exportfilename

def merge_parallel_jobs(merge_list, new_dataset_name=""):
    """merges datasets matching merge_list into single dataset    
    """
    print("\n\n===============================================")
    print("Merging datasets for arrayjobids"+str(merge_list))
    print("===============================================\n")

    path=Path("datasets")
    # print([name for name in os.listdir(p)])

    merge_folders = [folder_name for folder_name in os.listdir(path) if any(str(dataset_name) in folder_name for dataset_name in merge_list) ] 
    
    return mergesamples(merge_folders, new_dataset_name=new_dataset_name, remove_after_merge=True)

def merge_single_parallel_job(dataset_name):
    return merge_parallel_jobs([dataset_name], new_dataset_name=dataset_name)

def print_compute_time_statistics(compute_times):
    print(f"Compute time mean ={ np.mean(compute_times) :.5f} [s]")
    print(f"Compute time max 3 = { np.sort(compute_times[np.argpartition(compute_times, -3)[-3:]])} [s]")
    print(f"Compute time sum = { np.sum(compute_times)/60/60  :.5f} [core-h]")

def mpc_dataset_import(dataset_name, mpc_type=MPCQuadraticCostLxLu):
    mpc = import_mpc(dataset_name, mpc_type)
    X0, V, X, compute_times = import_dataset(mpc, dataset_name)
    return mpc, X0, V, X, compute_times

def print_dataset_statistics(dataset_name):
    mpc = import_mpc(dataset_name, MPCQuadraticCostLxLu)
    x0dataset, Udataset, Xdataset, compute_times = import_dataset(mpc, dataset_name)
    print_compute_time_statistics(compute_times)