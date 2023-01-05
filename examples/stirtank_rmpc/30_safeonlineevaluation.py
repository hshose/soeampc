import numpy as np
# from sklearn.model_selection import train_test_split

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..','..'))
from pathlib import Path

fp = Path(os.path.dirname(__file__))
os.chdir(fp)

import fire

from soeampc.trainampc import import_model
from soeampc.mpcproblem import *
from soeampc.datasetutils import import_dataset
from soeampc.safeonline import AMPC, SafeOnlineEvaluationAMPC, closed_loop_test

def safe_online_vs_naive(dataset="latest", model="latest", N_samples=int(1e3)):
    # import latest dataset :-D
    mpc = import_mpc(dataset, MPCQuadraticCostLxLu)
    X, U, _, _ = import_dataset(mpc, dataset)
    if N_samples >= X.shape[0]:
        N_samples = X.shape[0]
        print("WARNING: N_samples exceeds size of dataset, will use N_samples =", N_samples,"instead")
    # X_test, X_train, Y_test, Y_train = train_test_split(X, U, test_size=0.1, random_state=42)
    model = import_model(modelname=model)

    naive_controller = AMPC(mpc, model)
    safe_controller = SafeOnlineEvaluationAMPC(mpc, model)

    controllers = [ naive_controller, safe_controller ]
    closed_loop_test(X[:N_samples],U[:N_samples],controllers)

if __name__=="__main__":
    fire.Fire({
        "safe_online_vs_naive": safe_online_vs_naive,
    })