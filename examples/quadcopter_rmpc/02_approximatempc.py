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


def approximatempc(dataset="latest", maxepochs=int(1e6), batchsize=int(1e4)):
    # import latest dataset :-D
    mpc = import_mpc(dataset, MPCQuadraticCostLxLu)
    X, U, _, _ = import_dataset(mpc, dataset)


    # define architectures to be tested
    architectures = np.array([
        # [mpc.nx, 200, 400, 400, 400, 200, mpc.nu*mpc.N] # achieved mu=0.06
        [mpc.nx, 200, 400, 600, 800, 600, 400, 200, mpc.nu*mpc.N]
        ])

    # traverse list until architecture is found
    # datasetname = "latest"
    model = hyperparametertuning(mpc, X, U, dataset, architectures, batchsize=batchsize, maxepochs=maxepochs)


def retrainampc(datasetname="latest", modelname="latest", maxepochs = 5000, learningrate=1e-3):
    # import latest dataset :-D
    mpc = import_mpc(datasetname)
    X, Y, _, _ = import_dataset(mpc, datasetname)
    
    # X_test, X_train, Y_test, Y_train = train_test_split(X, U, test_size=0.1, random_state=42)
    model = import_model(datasetname=datasetname, modelname=modelname)
    model = retrainmodel(mpc=mpc, model=model, X=X, Y=Y, architecturestring=modelname.split('_',1)[0], datasetname=datasetname, maxepochs = maxepochs, learning_rate=learningrate)

if __name__=="__main__":
    fire.Fire()
