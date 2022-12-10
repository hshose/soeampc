# import tensorflow.keras as keras
# import tensorflow as tf
import numpy as np
# from tensorflow.keras import layers
# from sklearn.model_selection import train_test_split

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..','..'))
from pathlib import Path

import fire

fp = Path(os.path.dirname(__file__))
os.chdir(fp)

from soeampc import *


def approximatempc(dataset="latest"):
    # import latest dataset :-D
    mpc = import_mpc(dataset)
    X, U, _, _ = import_dataset(mpc, dataset)


    # define architectures to be tested
    architectures = np.array([
        # [mpc.nx, 10, mpc.nu*mpc.N],
        # [mpc.nx, 20, mpc.nu*mpc.N],
        # [mpc.nx, 40, mpc.nu*mpc.N],
        # [mpc.nx, 80, mpc.nu*mpc.N],
        # [mpc.nx, 160, mpc.nu*mpc.N],
        # [mpc.nx, 10, 10, mpc.nu*mpc.N],
        # [mpc.nx, 20, 20, mpc.nu*mpc.N],
        # [mpc.nx, 40, 40, mpc.nu*mpc.N],
        # [mpc.nx, 80, 80, mpc.nu*mpc.N],
        # [mpc.nx, 160, 160, mpc.nu*mpc.N],
        # [mpc.nx, 10, 20, mpc.nu*mpc.N],
        # [mpc.nx, 20, 40, mpc.nu*mpc.N],
        # [mpc.nx, 40, 80, mpc.nu*mpc.N],
        # [mpc.nx, 80, 160, mpc.nu*mpc.N],
        # [mpc.nx, 10, 10, 10, mpc.nu*mpc.N],
        # [mpc.nx, 20, 20, 20, mpc.nu*mpc.N],
        # [mpc.nx, 40, 40, 40, mpc.nu*mpc.N],
        # [mpc.nx, 80, 80, 80, mpc.nu*mpc.N],
        # [mpc.nx, 160, 160, 160, mpc.nu*mpc.N],
        # [mpc.nx, 10, 20, 40, mpc.nu*mpc.N],
        # [mpc.nx, 20, 40, 80, mpc.nu*mpc.N],
        # [mpc.nx, 40, 80, 160, mpc.nu*mpc.N]
        # [mpc.nx, 10, 100, 300, 600, mpc.nu*mpc.N]
        [mpc.nx, 200, 400, 400, 400, 200, mpc.nu*mpc.N]
        # [mpc.nx, 10, 20, 40, 60, 60, 60, mpc.nu*mpc.N],
        # [mpc.nx, 10, 10, 10, 10, 10, 10, 10, 10, mpc.nu*mpc.N],
        ])

    # traverse list until architecture is found
    # datasetname = "latest"
    model = hyperparametertuning(mpc, X, U, dataset, architectures)


if __name__=="__main__":
    fire.Fire(approximatempc)
