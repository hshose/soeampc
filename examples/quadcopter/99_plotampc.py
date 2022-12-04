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

mpc = import_mpc()
X, U, _, _ = import_dataset(mpc)
X_train, X_test, Y_train, Y_test = train_test_split(X, U, test_size=0.2, random_state=42)
model = import_model()

i = 1234

x0 = X_test[i]
U_MPC = Y_test[i]
U_NN = model(x0).numpy()

print(x0)
print(U_MPC)
print(U_NN)

X_MPC = mpc.forwardsim(x0, U_MPC)
X_NN = mpc.forwardsim(x0, U_NN)

plot_quadcopter_ol(mpc,[U_NN,U_MPC], [X_NN, X_MPC], labels=['NN','MPC'])