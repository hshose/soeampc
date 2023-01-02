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

import fire

def testinitialconditions(dataset='latest', model='latest'):
    # dataset = "quadcopter_N_100000_20221129-171745"

    mpc = import_mpc(dataset)
    X, U, _, _ = import_dataset(mpc, dataset)
    X_train, X_test, Y_train, Y_test = train_test_split(X, U, test_size=0.1, random_state=42)
    model = import_model(model)

    testresult, mu = statisticaltest(mpc, model, X_test)


if __name__=='__main__':
    fire.Fire(testinitialconditions)
