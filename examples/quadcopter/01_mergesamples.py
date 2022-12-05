import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..','..'))
from pathlib import Path
import shutil

fp = Path(os.path.dirname(__file__))
os.chdir(fp)

from soeampc.utils import *

import fire

def mergesamples(now, Nsamples):
    
    print("\n\n===============================================")
    print("Merging datasets for Docker_"+str(now))
    print("===============================================\n")

    p=Path("datasets")
    print([name for name in os.listdir(p)])
    copyfolder = [name for name in os.listdir(p) if "Docker_"+str(now)+"_0_" in name ]

    appendfolders = [name for name in os.listdir(p) if "Docker_"+str(now) in name and not "Docker_"+str(now)+"_0_" in name]

    mpc = import_mpc(copyfolder[0])
    x0dataset, Udataset, Xdataset, computetimes = import_dataset(mpc, copyfolder[0])
    filename = export_dataset(mpc, x0dataset, Udataset, Xdataset, computetimes, mpc.name+"_N_"+str(Nsamples)+"_merged_"+str(now), barefilename=True)

    for f in appendfolders:
        x0dataset, Udataset, Xdataset, computetimes = import_dataset(mpc, f)
        append_to_dataset(mpc, x0dataset, Udataset, Xdataset, computetimes, filename)



    removefolders = appendfolders
    removefolders.append(copyfolder[0])

    print("\n\nRemoving Folders:\n")
    print("\t",removefolders)

    for f in removefolders:
        shutil.rmtree(p.joinpath(f), ignore_errors=True)

    return filename

if __name__ == "__main__":
    fire.Fire(mergesamples)