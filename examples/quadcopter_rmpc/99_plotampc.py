import numpy as np
import tensorflow.keras as keras
# import tensorflow as tf
import numpy as np
# from tensorflow.keras import layers
from sklearn.model_selection import train_test_split


import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..','..'))
from pathlib import Path

fp = Path(os.path.dirname(__file__))
os.chdir(fp)

from soeampc import *

from plot import *

import fire

def plotampc(dataset='latest', model='latest'):
    # dataset = "quadcopter_N_100000_20221129-171745"

    mpc = import_mpc(dataset)
    X, U, _, _ = import_dataset(mpc, dataset)
    X_train, X_test, Y_train, Y_test = train_test_split(X, U, test_size=0.1, random_state=42)
    # X_test, X_train, Y_test, Y_train = train_test_split(X, U, test_size=0.1, random_state=42)
    model = import_model(model)

    # i = 1234

    for i in range(X_test.shape[0]):
        x0 = X_test[i]
        U_MPC = Y_test[i]
        U_NN = model(x0).numpy()[0]

        print("x0 =\n",     x0,     "\n")
        print("U_MPC =\n",  U_MPC,  "\n")
        print("U_NN =\n",   U_NN,   "\n")

        X_MPC = mpc.forwardsim(x0, U_MPC)
        X_NN = mpc.forwardsim(x0, U_NN)

        print("forwardsimcheck MPC:",   mpc.feasible(X_MPC, U_MPC,  verbose=True))
        print("forwardsimcheck NN:",    mpc.feasible(X_NN,  U_NN,   verbose=True))

        plot_quadcopter_ol(mpc,[U_NN,U_MPC], [X_NN, X_MPC], labels=['NN','MPC'])

def plotmpc(dataset='latest'):
    # dataset = "quadcopter_N_100000_20221129-171745"

    mpc = import_mpc(dataset, mpcclass=MPCQuadraticCostLxLu)
    X, U, _, _ = import_dataset(mpc, dataset)
    # X_train, X_test, Y_train, Y_test = train_test_split(X, U, test_size=0.1, random_state=42)
    # X_test, X_train, Y_test, Y_train = train_test_split(X, U, test_size=0.1, random_state=42)

    # i = 1234

    # xmin = np.array([None, None, None, None,    None, None, -math.pi/10,  None, -math.pi/10, None]) 
    # xmax = np.array([ 1,  None,  None,  None,  None,  None,  math.pi/10,  None,  math.pi/10,  None]) 
    # umin = np.array([ -4*math.pi, -4*math.pi, -9.8/0.91       ])
    # umax = np.array([  4*math.pi,  4*math.pi,  2*9.8-9.8/0.91 ])
    
    xmin = np.array([None, None, None, None,    None, None, -math.pi/4,  None, -math.pi/4, None]) 
    xmax = np.array([ 1,  None,  None,  None,  None,  None,  math.pi/4,  None,  math.pi/4,  None]) 
    umin = np.array([ -math.pi/9, -math.pi/9, -9.8/0.91       ])
    umax = np.array([  math.pi/9,  math.pi/9,  2*9.8-9.8/0.91 ])
    

    limits = {
        "xmin": xmin,
        "xmax": xmax,
        "umin": umin,
        "umax": umax
    }

    for i in range(X.shape[0]):
        x0 = X[i]
        U_MPC = U[i]

        print("x0 =\n",     x0,     "\n")
        print("U_MPC =\n",  U_MPC,  "\n")

        X_MPC = mpc.forwardsim(x0, U_MPC)

        print("forwardsimcheck MPC:",   mpc.feasible(X_MPC, U_MPC,  verbose=True))

        plot_quadcopter_ol(mpc,[U_MPC], [X_MPC], labels=['MPC'], limits=limits)


if __name__=='__main__':
    fire.Fire()