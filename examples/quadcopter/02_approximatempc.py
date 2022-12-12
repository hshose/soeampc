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
        # [mpc.nx, 200, 400, 400, 400, 200, mpc.nu*mpc.N] # achieved mu=0.06
        [mpc.nx, 200, 400, 600, 800, 600, 400, 200, mpc.nu*mpc.N]
        ])

    # traverse list until architecture is found
    # datasetname = "latest"
    model = hyperparametertuning(mpc, X, U, dataset, architectures)


if __name__=="__main__":
    fire.Fire(approximatempc)
