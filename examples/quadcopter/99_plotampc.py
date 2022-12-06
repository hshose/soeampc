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

dataset = "quadcopter_N_100000_20221129-171745"

mpc = import_mpc(dataset)
X, U, _, _ = import_dataset(mpc, dataset)
X_train, X_test, Y_train, Y_test = train_test_split(X, U, test_size=0.1, random_state=42)
model = import_model()

i = 1234

x0 = X_test[i]
U_MPC = Y_test[i]
U_NN = model(x0).numpy()[0]

print("x0 =\n", x0, "\n")
print("U_MPC =\n", U_MPC, "\n")
print("U_NN =\n", U_NN, "\n")

X_MPC = mpc.forwardsim(x0, U_MPC)
X_NN = mpc.forwardsim(x0, U_NN)

print("forwardsimcheck MPC:", mpc.feasible(X_MPC, U_MPC, verbose=True))
print("forwardsimcheck NN:", mpc.feasible(X_NN, U_NN, verbose=True))

plot_quadcopter_ol(mpc,[U_NN,U_MPC], [X_NN, X_MPC], labels=['NN','MPC'])