# import tensorflow.keras as keras
# import tensorflow as tf
import numpy as np
# from tensorflow.keras import layers
# from sklearn.model_selection import train_test_split

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..','..'))
from pathlib import Path

fp = Path(os.path.dirname(__file__))
os.chdir(fp)

import fire

from soeampc.trainampc import architecture_search, retrain_model, import_model
from soeampc.mpcproblem import *
from soeampc.datasetutils import import_dataset

def find_approximate_mpc(dataset="latest"):
    # import latest dataset :-D
    mpc = import_mpc(dataset, MPCQuadraticCostLxLu)
    X, U, _, _ = import_dataset(mpc, dataset)

    # define architectures to be tested
    architectures = np.array([
        # [mpc.nx, 10, 10, mpc.nu*mpc.N],
        [mpc.nx, 50, 50, mpc.nu*mpc.N],
        # [mpc.nx, 40, 80, mpc.nu*mpc.N],
        # [mpc.nx, 10, 10, 10, mpc.nu*mpc.N],
        # [mpc.nx, 40, 40, 40, mpc.nu*mpc.N],
        # [mpc.nx, 10, 20, 40, mpc.nu*mpc.N],
        # [mpc.nx, 20, 40, 80, mpc.nu*mpc.N],
        # [mpc.nx, 40, 80, 160, mpc.nu*mpc.N]
        ])

    hyperparameters = [ {"learning_rate":0.01,  "patience": 1000, "max_epochs": 1000, "batch_size": 10000},
                        {"learning_rate":0.01, "patience": 1000, "max_epochs": 100000, "batch_size": 10000},
                        {"learning_rate":0.005, "patience": 1000, "max_epochs": 100000, "batch_size": 10000},
                        {"learning_rate":0.002, "patience": 1000, "max_epochs": 100000, "batch_size": 10000},
                        {"learning_rate":0.001, "patience": 1000, "max_epochs": 100000, "batch_size": 10000},
                        {"learning_rate":0.0005, "patience": 1000, "max_epochs": 100000, "batch_size": 10000},
                        {"learning_rate":0.0002, "patience": 1000, "max_epochs": 100000, "batch_size": 10000},
                        {"learning_rate":0.0001, "patience": 1000, "max_epochs": 100000, "batch_size": 10000},]

    model = architecture_search(mpc, X, U, hyperparameters=hyperparameters, architectures=architectures)
    return model

def retrain_ampc(datasetname="latest", modelname="latest", maxepochs = 5000, learningrate=1e-3):
    # import latest dataset :-D
    mpc = import_mpc(datasetname, MPCQuadraticCostLxLu)
    X, Y, _, _ = import_dataset(mpc, datasetname)
    
    # X_test, X_train, Y_test, Y_train = train_test_split(X, U, test_size=0.1, random_state=42)
    model = import_model(modelname=modelname)
    model = retrain_model(mpc=mpc, model=model, X=X, Y=Y, architecturestring=modelname.split('_',1)[0], maxepochs = maxepochs, learning_rate=learningrate)

if __name__=="__main__":
    fire.Fire({
        "find_approximate_mpc": find_approximate_mpc,
        "retrain_ampc": retrain_ampc,
    })