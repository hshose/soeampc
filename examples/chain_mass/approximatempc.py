from soeampc.datasetutils import import_dataset, print_dataset_statistics, merge_single_parallel_job
from soeampc.mpcproblem import *
from soeampc.trainampc import architecture_search, retrain_model, test_ampc, computetime_test_model
import fire
from pathlib import Path
import numpy as np

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


fp = Path(os.path.dirname(__file__))
os.chdir(fp)


def find_approximate_mpc(dataset="latest"):
    """train ampc with different architectures and huperparameters

    Description:
        For each architecture, the hyperparameter list is traversed.

    Args:
        dataset:
            name of the dataset
    Returns:
        tensorflow model
    """
    # import latest dataset :-D
    mpc = import_mpc(dataset, MPCQuadraticCostLxLu)
    X, U, _, _ = import_dataset(mpc, dataset)

    # define architectures to be tested
    architectures = np.array([
        # [mpc.nx, 200, 400, 400, 400, 200, mpc.nu*mpc.N] # achieved mu=0.06
        [mpc.nx, 200, 400, 600, 600, 400, 200, mpc.nu*mpc.N]  # achieved mu=0.06
        # [mpc.nx, 200, 400, 600, 800, 600, 400, 200, mpc.nu*mpc.N]
    ])

    # traverse list until architecture is found
    # datasetname = "latest"
    hyperparameters = [{"learning_rate": 0.01,   "patience": 1000,
                        "max_epochs": 10000, "batch_size": 10000},
                       {"learning_rate": 0.005,  "patience": 1000,
                        "max_epochs": 10000, "batch_size": 10000},
                       {"learning_rate": 0.002,  "patience": 1000,
                        "max_epochs": 10000, "batch_size": 10000},
                       {"learning_rate": 0.001,  "patience": 1000,
                        "max_epochs": 10000, "batch_size": 10000},
                       {"learning_rate": 0.0005, "patience": 1000,
                        "max_epochs": 10000, "batch_size": 10000},
                       {"learning_rate": 0.0002, "patience": 1000,
                        "max_epochs": 10000, "batch_size": 10000},
                       {"learning_rate": 0.0001, "patience": 1000, "max_epochs": 10000, "batch_size": 10000},]

    model = architecture_search(
        mpc, X, U, hyperparameters=hyperparameters, architectures=architectures)
    return model


if __name__ == "__main__":
    fire.Fire({
        "find_approximate_mpc": find_approximate_mpc,
        "retrain_model": retrain_model,
        "test_ampc": test_ampc,
        "print_dataset_statistics": print_dataset_statistics,
        "merge_single_parallel_job": merge_single_parallel_job,
        "computetime_test_model":   computetime_test_model,
    })
