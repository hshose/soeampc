import numpy as np

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..','..'))
from pathlib import Path

fp = Path(os.path.dirname(__file__))
os.chdir(fp)

from soeampc import *

mpc = import_mpc()
x0dataset, Udataset, Xdataset, computetimes = import_dataset(mpc)

from plot import *
dimx = 3
dimy = 4

# plot_feas(x0dataset[:,dimx],x0dataset[:,dimy],np.array([mpc.xmin[dimx], mpc.xmax[dimx]]), np.array([mpc.xmin[dimy], mpc.xmax[dimy]]))


datx = Xdataset[:,:,dimx].flatten()
daty = Xdataset[:,:,dimy].flatten()
scalex = 1
scaley = 1
plot_feas(datx, daty, scalex*np.array([mpc.xmin[dimx], mpc.xmax[dimx]]), scaley*np.array([mpc.xmin[dimy], mpc.xmax[dimy]]))