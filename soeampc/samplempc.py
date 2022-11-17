import numpy as np

from .utils import *

def sampledataset(mpc, run, samplesperaxis, outfile):
    """

    """
    # grid over statespace
    res = np.empty((0,2), float)
    if isinstance(samplesperaxis, np.ndarray) and len(samplesperaxis) == mpc.nx():
        N = np.array(samplesperaxis, dtype=int)
    elif isinstance(samplesperaxis, int) and samplesperaxis > 0:
        N = samplesperaxis*np.ones(mpc.nx, dtype=int)
    else:
        raise Exception('samplesperaxis invalid, expected positive integer or numpy array of len', mpc.nx())
    
    Nsamples= int(np.prod(N))
    X0dataset = np.empty((Nsamples, mpc.nx), float)
    Xdataset =  np.empty((Nsamples, mpc.N+1, mpc.nx), float)
    Udataset =  np.empty((Nsamples, mpc.N, mpc.nu), float)
    computetimes = np.empty(Nsamples, float)

    print("Evaluating MPC on grid")
    i = np.zeros(mpc.nx)
    Nvalid = 0

    def loop_rec(N, n, i):
        nonlocal X0dataset
        nonlocal Xdataset
        nonlocal Udataset
        nonlocal computetimes
        nonlocal Nvalid
        
        if n > 0:
            for j in range(N[n-1]):
                i[n-1] = j
                loop_rec(N, n-1, i)
        else:
            
            x0 = mpc.xmin + (mpc.xmax-mpc.xmin)*i/N
            # mpc.c.reset()
            X, U, status, elapsed = run(x0)
            # print(status)
            if status == 0:
                X0dataset[Nvalid,:] = x0
                Xdataset[Nvalid,:,:]  = X
                Udataset[Nvalid,:,:]  = U
                computetimes[Nvalid] = elapsed
                Nvalid +=1
                # plot_stirtank(np.linspace(0, c.Tf, c.N+1), c.umax, c.umin, c.xmax, c.xmin, U, X)
            # else:
            #     print(status)

    loop_rec(N,mpc.nx,i)
    # print(Nvalid)
    datasetname = export_dataset(mpc, X0dataset[:Nvalid,:], Udataset[:Nvalid,:,:], Xdataset[:Nvalid,:,:], computetimes[:Nvalid], outfile)
    return X0dataset[:Nvalid,:], Udataset[:Nvalid,:,:], Xdataset[:Nvalid,:,:], computetimes[:Nvalid], datasetname

    
def inspectdataset(mpc, file):

    plot_feas(X0dataset[:Nvalid,:],np.array([xmin_init[0], xmax_init[0]]), np.array([xmin_init[1], xmax_init[1]]))
    if args.write:
        plt.savefig("figures/"+ps+"_feasible_set.pgf")
    if args.plots:
        plt.show()
    plt.clf()
    plot_ctdistro(computetimes[:Nvalid])
    if args.write:
        plt.savefig("figures/"+ps+"_compute_time_distro.pgf")
    if args.plots:
        plt.show()

    Ngood = 0
    print("Performing forward sim check on all samples")
    for i in tqdm(range(Nvalid)):
        x0 = X0dataset[i,:]
        U = Udataset[i,:]
        allgood = forwardsimcheck(x0, U, c.Tf, c.N)
        Ngood += allgood

    print("Forward sim checks passed:", Ngood/Nvalid*100,"[%]")