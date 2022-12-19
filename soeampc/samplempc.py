import numpy as np
from tqdm import tqdm

from .utils import export_dataset


def sampledataset(mpc, run, sampler, outfile, runtobreak=False, verbose=False):
    """

    """
    # grid over statespace
    res = np.empty((0,2), float)
    
    X0dataset = np.empty((sampler.Nsamples, mpc.nx), float)
    Xdataset =  np.empty((sampler.Nsamples, mpc.N+1, mpc.nx), float)
    Udataset =  np.empty((sampler.Nsamples, mpc.N, mpc.nu), float)
    computetimes = np.empty(sampler.Nsamples, float)

    print("\n\n===============================================")
    print("Evaluating MPC on grid with",sampler.Nsamples,"points")
    print("===============================================\n")
    i = np.zeros(mpc.nx)
    Nvalid = 0
    n = 0
    with tqdm(total=sampler.Nsamples) as pbar:
        while n < sampler.Nsamples:
            x0 = sampler.sample()
            X, U, status, elapsed = run(x0)
            # print(status)
            if status == 0 or status == 2:
                if verbose:
                    print("acados status: ", status)
                if mpc.feasible(X, U, verbose=True):
                    X0dataset[Nvalid,:] = x0
                    Xdataset[Nvalid,:,:]  = X
                    Udataset[Nvalid,:,:]  = U
                    computetimes[Nvalid] = elapsed
                    Nvalid += 1
                    if runtobreak:
                        pbar.update(1)
                        n += 1

            # if verbose:
                # print("Status",status,"\nforwardsimcheck MPC:", mpc.feasible(X, U, verbose=True),"\n")

            if not runtobreak:
                pbar.update(1)
                n +=1

    print("Got",Nvalid,"feasible solutions for MPC")

    datasetname = export_dataset(
        mpc,
        X0dataset[:Nvalid,:],
        Udataset[:Nvalid,:,:],
        Xdataset[:Nvalid,:,:],
        computetimes[:Nvalid],
        outfile)
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