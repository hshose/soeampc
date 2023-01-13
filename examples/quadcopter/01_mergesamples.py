import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..','..'))
from pathlib import Path
import shutil
from tqdm import tqdm

fp = Path(os.path.dirname(__file__))
os.chdir(fp)

from soeampc.utils import *

import fire

# quadcopter_N_80000_merged_20221206-181236
# quadcopter_N_80000_merged_20221208-193938
# quadcopter_N_80000_merged_20221209-210859
# quadcopter_N_12000_Docker_20221208-205651_0_20221209-214049
# quadcopter_N_12000_Docker_20221208-205651_10_20221209-200211
# quadcopter_N_12000_Docker_20221208-205651_11_20221209-211403
# quadcopter_N_12000_Docker_20221208-205651_1_20221209-202937
# quadcopter_N_12000_Docker_20221208-205651_13_20221209-211151
# quadcopter_N_12000_Docker_20221208-205651_14_20221209-212601
# quadcopter_N_12000_Docker_20221208-205651_15_20221209-220429
# quadcopter_N_12000_Docker_20221208-205651_2_20221209-200138
# quadcopter_N_12000_Docker_20221208-205651_3_20221209-214302
# quadcopter_N_12000_Docker_20221208-205651_4_20221209-224814
# quadcopter_N_12000_Docker_20221208-205651_5_20221209-195818
# quadcopter_N_12000_Docker_20221208-205651_6_20221209-210458
# quadcopter_N_12000_Docker_20221208-205651_7_20221209-200847
# quadcopter_N_12000_Docker_20221208-205651_8_20221209-202954
# quadcopter_N_12000_Docker_20221208-205651_9_20221209-204400

def mergeclusterjobs(arrayjobids):
    print("\n\n===============================================")
    print("Merging datasets for arrayjobids"+str(arrayjobids))
    print("===============================================\n")

    p=Path("datasets")
    # print([name for name in os.listdir(p)])

    foldernames = [name for name in os.listdir(p) if any(str(ajobid) in name for ajobid in arrayjobids) ] 
    
    return mergesamples(foldernames, remove_after_merge=True)

def mergesingleclusterjob(id):
    return mergeclusterjobs([id])

def mergedocker(now, Nsamples):
    print("\n\n===============================================")
    print("Merging datasets for Docker_"+str(now))
    print("===============================================\n")

    p=Path("datasets")
    print([name for name in os.listdir(p)])
    copyfolder = [name for name in os.listdir(p) if "Docker_"+str(now)+"_0_" in name ]

    appendfolders = [name for name in os.listdir(p) if "Docker_"+str(now) in name and not "Docker_"+str(now)+"_0_" in name]

    foldernames = copyfolder+appendfolders
    return mergesamples(foldernames, now, remove_after_merge=True)

def mergelist():
    filenames = [   
                    'quadcopter_N_80000_merged_20221206-181236',
                    'quadcopter_N_80000_merged_20221208-193938',
                    'quadcopter_N_80000_merged_20221209-210859',
                    'quadcopter_N_240000_merged_20221208-160424',
                    'quadcopter_N_12000_Docker_20221208-205651_0_20221209-214049',
                    'quadcopter_N_12000_Docker_20221208-205651_10_20221209-200211',
                    'quadcopter_N_12000_Docker_20221208-205651_11_20221209-211403',
                    'quadcopter_N_12000_Docker_20221208-205651_1_20221209-202937',
                    'quadcopter_N_12000_Docker_20221208-205651_13_20221209-211151',
                    'quadcopter_N_12000_Docker_20221208-205651_14_20221209-212601',
                    'quadcopter_N_12000_Docker_20221208-205651_15_20221209-220429',
                    'quadcopter_N_12000_Docker_20221208-205651_2_20221209-200138',
                    'quadcopter_N_12000_Docker_20221208-205651_3_20221209-214302',
                    'quadcopter_N_12000_Docker_20221208-205651_4_20221209-224814',
                    'quadcopter_N_12000_Docker_20221208-205651_5_20221209-195818',
                    'quadcopter_N_12000_Docker_20221208-205651_6_20221209-210458',
                    'quadcopter_N_12000_Docker_20221208-205651_7_20221209-200847',
                    'quadcopter_N_12000_Docker_20221208-205651_8_20221209-202954',
                    'quadcopter_N_12000_Docker_20221208-205651_9_20221209-204400'
                ]
    return mergesamples(filenames)

def mergesamples(foldernames, now=get_date_string(), remove_after_merge=False):
    p=Path("datasets")
    mpc = import_mpc(foldernames[0])
    print("file:", foldernames[0])
    x0dataset, Udataset, Xdataset, computetimes = import_dataset(mpc, foldernames[0])
    Nsamples = x0dataset.shape[0]
    exporttempfilename = export_dataset(mpc, x0dataset, Udataset, Xdataset, computetimes, mpc.name+"_N_temp_merged_"+str(now), barefilename=True)
    for f in tqdm(foldernames[1:]):
        print("file:", f)
        x0dataset, Udataset, Xdataset, computetimes = import_dataset(mpc, f)
        Nsamples = Nsamples + x0dataset.shape[0]
        append_to_dataset(mpc, x0dataset, Udataset, Xdataset, computetimes, exporttempfilename)

    print("\ncollected a total of ", str(Nsamples), "sample points")
    exportfilename = mpc.name+"_N_"+str(Nsamples)+"_merged_"+str(now)
    os.rename(p.joinpath(exporttempfilename), p.joinpath(exportfilename))
    print("\nExported merged dataset to:\n")
    print("\t", exportfilename,"\n")

    if remove_after_merge:
        print("\n\nRemoving Folders:\n")
        print("\t", foldernames)
        for f in foldernames:
            shutil.rmtree(p.joinpath(f), ignore_errors=True)

    return exportfilename

if __name__ == "__main__":
    fire.Fire()