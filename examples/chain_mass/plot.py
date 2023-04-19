import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def plot_chain_mass_ol(mpc, Utraj, Xtraj, labels, plt_show=True, limits={}):

    N_sim = mpc.N
    nx = mpc.nx

    Tf = mpc.Tf
    t = np.linspace(0, mpc.Tf, mpc.N+1)

    Ts = t[1] - t[0]

    Ntrajs = len(Utraj)
    print(Xtraj[0].shape)

    linestyles = ['solid', 'dotted', 'dashed', 'dashdot']
    looselydashed = (0, (5, 10))
    colors = ['r','g','b','r','g','b','r','g','b','r','g','b','r','g','b','r','g','b']

    nxbatches = int(nx/3)
    xbatches = [[3*i, 3*i+1, 3*i+2] for i in range(nxbatches)]

    ubatches = [[0,1,2]]

    M = int((nx/3-1)/2)
    xlabels = [f"p_{i},{c}" for i in range(M+1) for c in ["x", "y", "z"] ]
    xlabels = xlabels + [f"v_{i},{c}" for i in range(M) for c in ["x", "y", "z"] ]

    # xlabels = ["p_1,x", "p_1,y", "p_1,z", "p_2,x", "p_2,y", "p_2,z", "v_1,x", "v_1,y", "v_1,z"]
    ulabels = ["u_1", "u_2", "u_3"]

    batches = len(xbatches) + len(ubatches)

    for k in range(len(ubatches)):
        plt.subplot(batches, 1, k+1)
        batch = ubatches[k]
        for i in range(Ntrajs):
            V = Utraj[i]
            X = Xtraj[i]
            U = np.array([mpc.stabilizing_feedback_controller(X[j], V[j]) for j in range(V.shape[0])])
            for j in batch:
                line, = plt.step(t, np.append([U[0,j]], U[:,j]), label=labels[i]+" "+ulabels[j], color=colors[j], linestyle=linestyles[i])
        plt.grid()
        # plt.title('predicted trajectory')
        plt.ylabel('inputs u')
        for j in batch:
            if "umin" in limits and not limits["umin"][j] == None:
                plt.hlines(limits["umin"][j], t[0], t[-1], linestyles=looselydashed, color=colors[batch[0]-j], alpha=0.7)
            if "umax" in limits and not limits["umax"][j] == None:
                plt.hlines(limits["umax"][j], t[0], t[-1], linestyles=looselydashed, color=colors[batch[0]-j], alpha=0.7)
        
        # if "umin" in limits and "umax" in limits:
        #     plt.ylim([ 
        #             1.2*np.min([limits["umin"][j] for j in batch if not limits["umin"][j] == None ]),
        #             1.2*np.max([limits["umax"][j] for j in batch if not limits["umax"][j] == None ])
        #             ])
        plt.legend(loc=1)

    for k in range(len(xbatches)):
        batch = xbatches[k]
        plt.subplot(batches, 1, len(ubatches)+k+1)
        for i in range(Ntrajs):
            X = Xtraj[i]
            for j in batch:
                line, = plt.plot(t, X[:, j], label=labels[i]+" "+xlabels[j], color=colors[j], linestyle=linestyles[i])
        plt.ylabel('$x$')
        for j in batch:
            if "xmin" in limits and not limits["xmin"][j] == None:
                plt.hlines(limits["xmin"][j], t[0], t[-1], linestyles=looselydashed, color=colors[batch[0]-j], alpha=0.7)
            if "xmax" in limits and not limits["xmax"][j] == None:
                plt.hlines(limits["xmax"][j], t[0], t[-1], linestyles=looselydashed, color=colors[batch[0]-j], alpha=0.7)       
        # if "xmin" in limits and "xmax" in limits:
        #     plt.ylim([ 
        #             1.2*np.min([limits["xmin"][j] for j in batch if not limits["xmin"][j] == None ]),
        #             1.2*np.max([limits["xmax"][j] for j in batch if not limits["xmax"][j] == None ])
        #             ])
        plt.grid()
        plt.legend(loc=1)

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, hspace=0.4)
    plt.show()
    # return plt

def plot_feas(xfeas, yfeas, xlim=None, ylim=None):
    plt.clf()
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(w=3.4, h=3.4)
    if not isinstance(xlim, type(None)):
        plt.xlim(1.2*xlim)
    if not isinstance(ylim, type(None)):
        plt.ylim(1.2*ylim)
    ax = plt.gca()
    ax.set_aspect('equal')
    # plt.title('Feasible Set')
    plt.xlabel('$x_1$')
    plt.ylabel(r'$x_2$')
    plt.tight_layout()
    plt.scatter(xfeas, yfeas, marker=',', color='blue')
    # import tikzplotlib
    # tikzplotlib.save("figures/mpcfeasibleset.tex", axis_height='2.4in', axis_width='2.4in')
    plt.show()

def plot_feas_notfeas(feas, notfeas, xlim, ylim):
    plt.clf()
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(w=3.4, h=3.4)
    plt.xlim(1.2*xlim)
    plt.ylim(1.2*ylim)
    ax = plt.gca()
    ax.set_aspect('equal')
    # plt.title('Feasible Set')
    plt.xlabel('$x_1$')
    plt.ylabel(r'$x_2$')
    plt.tight_layout()
    plt.scatter(feas[:,0], feas[:,1], marker='.', color='blue')
    plt.scatter(notfeas[:,0], notfeas[:,1], marker='.', color='red')
    import tikzplotlib
    tikzplotlib.save("figures/offlineapproximationtestgrid.tex", axis_height='2.4in', axis_width='2.4in')
    plt.show()


def plot_ctdistro(ct):
    plt.clf()
    computetimes = ct*1000
    logbins = np.geomspace(computetimes.min(), computetimes.max(), 20)
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(w=3.4, h=2.1)
    plt.hist(computetimes, density=False, bins=logbins)
    locs, _ = plt.yticks()
    plt.yticks(locs,np.round(locs/len(computetimes)*100,1))
    # plt.yticks([])
    plt.ylabel('fraction [%]')
    plt.grid()
    lgnd = plt.legend()
    lgnd.remove()
    plt.xlabel('compute time [ms]')
    # import tikzplotlib
    # tikzplotlib.save("figures/mpcctdistro.tex", axis_width='2.4in', axis_height='1.48in')
    # plt.set_yticklabels([])
    plt.tight_layout()