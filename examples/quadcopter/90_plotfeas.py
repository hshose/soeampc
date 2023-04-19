import numpy as np

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..','..'))
from pathlib import Path

fp = Path(os.path.dirname(__file__))
os.chdir(fp)

from soeampc import *

from plot import *

import fire

def plotfeas(dimx, dimy, scalex=1, scaley=1, dataset='latest', onlyx0=False):
    mpc = import_mpc(dataset, mpcclass=MPCQuadraticCostLxLu)
    x0dataset, Udataset, Xdataset, computetimes = import_dataset(mpc, dataset)
    # plot_feas(x0dataset[:,dimx],x0dataset[:,dimy],np.array([mpc.xmin[dimx], mpc.xmax[dimx]]), np.array([mpc.xmin[dimy], mpc.xmax[dimy]]))
    if onlyx0:
        datx = x0dataset[:,dimx]
        daty = x0dataset[:,dimy]
    else:
        datx = Xdataset[:,:,dimx].flatten()
        daty = Xdataset[:,:,dimy].flatten()

    plot_feas(datx, daty)

if __name__=='__main__':
    fire.Fire(plotfeas)