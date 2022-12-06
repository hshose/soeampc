import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def plot_quadcopter_ol(mpc, Utraj, Xtraj, labels, plt_show=True):
    # # latexify plot
    # if latexify:
    #     params = {'backend': 'ps',
    #             'text.latex.preamble': r"\usepackage{gensymb} \usepackage{amsmath}",
    #             'axes.labelsize': 10,
    #             'axes.titlesize': 10,
    #             'legend.fontsize': 10,
    #             'xtick.labelsize': 10,
    #             'ytick.labelsize': 10,
    #             'text.usetex': True,
    #             'font.family': 'serif'
    #     }

    # matplotlib.rcParams.update(params)

    N_sim = mpc.N
    nx = mpc.nx

    Tf = mpc.Tf
    t = np.linspace(0, mpc.Tf, mpc.N+1)

    Ts = t[1] - t[0]

    Ntrajs = len(Utraj)

    linestyles = ['solid', 'dotted', 'dashed', 'dashdot']
    looselydashed = (0, (5, 10))
    colors = ['r','g','b','c','m','y','k', 'darkred', 'navy', 'darkgreen']

    xbatches = [[0,1,2], [3,4,5], [6,8], [7,9]]
    ubatches = [[0,1],[2]]

    batches = len(xbatches) + len(ubatches)

    for k in range(len(ubatches)):
        plt.subplot(batches, 1, k+1)
        batch = ubatches[k]
        for i in range(Ntrajs):
            U = Utraj[i]
            for j in batch:
                line, = plt.step(t, np.append([U[0,1]], U[:,1]), label=labels[i]+" u"+str(j), color=colors[j], linestyle=linestyles[i])
        plt.grid()
        plt.title('predicted trajectory')
        plt.ylabel('inputs u')
        for j in batch:
            plt.hlines(mpc.umax[j], t[0], t[-1], linestyles=looselydashed, color=colors[batch[0]-j], alpha=0.7)
            plt.hlines(mpc.umin[j], t[0], t[-1], linestyles=looselydashed, color=colors[batch[0]-j], alpha=0.7)
        plt.ylim([1.2*np.min([mpc.umin[j] for j in batch]), 1.2*np.max([mpc.umax[j] for j in batch])])
        plt.legend(loc=1)

    for k in range(len(xbatches)):
        batch = xbatches[k]
        plt.subplot(batches, 1, len(ubatches)+k+1)
        for i in range(Ntrajs):
            X = Xtraj[i]
            for j in batch:
                line, = plt.plot(t, X[:, j], label=labels[i]+" x"+str(j), color=colors[j], linestyle=linestyles[i])
        plt.ylabel('$x$')
        for j in batch:
            plt.hlines(mpc.xmax[j], t[0], t[-1], linestyles=looselydashed, color=colors[batch[0]-j],  alpha=0.7)
            plt.hlines(mpc.xmin[j], t[0], t[-1], linestyles=looselydashed, color=colors[batch[0]-j],  alpha=0.7)
        plt.ylim([1.2*np.min([mpc.xmin[j] for j in batch]), 1.2*np.max([mpc.xmax[j] for j in batch])])
        plt.grid()
        plt.legend(loc=1)

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, hspace=0.4)
    plt.show()
    # return plt

def plot_feas(xfeas, yfeas, xlim, ylim):
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


def plot_cl_ampc(shooting_nodes, U, X, Checksim, latexify=False, plt_show=True):
    """
    Params:
        shooting_nodes: time values of the discretization
        u_max: maximum absolute value of u
        U: arrray with shape (N_sim-1, nu) or (N_sim, nu)
        X: arrray with shape (N_sim, nx)
        latexify: latex style plots
    """

    # latexify plot
    if latexify:
        params = {'backend': 'ps',
                'text.latex.preamble': r"\usepackage{gensymb} \usepackage{amsmath}",
                'axes.labelsize': 10,
                'axes.titlesize': 10,
                'legend.fontsize': 10,
                'xtick.labelsize': 10,
                'ytick.labelsize': 10,
                'text.usetex': True,
                'font.family': 'serif'
        }

        matplotlib.rcParams.update(params)


    ue      = 0.7853
    x_min = np.array([-0.2,-0.2])
    x_max = np.array([0.2,0.2])
    u_min = np.array([0-ue])
    u_max = np.array([2-ue])

    N_sim = X.shape[0]
    nx = X.shape[1]

    Tf = shooting_nodes[N_sim-1]
    t = shooting_nodes

    # print(x_min)
    # print(x_max)

    Ts = t[1] - t[0]

    fig, (ax1, ax2) = plt.subplots(2, 1)
    # plt.clf()
    ax1.step(t, np.append([U[0]], U[:]), label=r'$u$', color='b')
    ax1.plot((t[:-1])[Checksim==0], U[Checksim==0], marker='x', linestyle='None', markersize=8, color='red', label='candidate')
    # plt.title('predicted trajectory')
    ax1.set_ylabel('inputs')
    ax1.plot([t[0], t[-1]], [u_max[0],u_max[0]], linestyle='dashed', color='blue', alpha=0.7, label='constraints')
    ax1.plot([t[0], t[-1]], [u_min[0],u_min[0]], linestyle='dashed', color='blue', alpha=0.7)
    ax1.set_xticks([0,2,4,6,8,10,12,14,16,18,20])  # Set label locations_.
    ax1.set_ylim([1.1*u_min, 1.1*u_max])
    ax1.grid()
    # ax1.legend(loc=1)
    ax1.legend(loc = 'center left', bbox_to_anchor=(1.05, 0.5),fancybox=False, framealpha=1)

    ax2.plot(t, X[:, 0], label=r'$x_1$', color='g')
    ax2.set_ylabel('states')
    ax2.plot(t, X[:, 1], label=r'$x_2$', color='purple')
    # ax2.plot((t[:-1])[Checksim==0], (X[:-1])[Checksim==0,1], marker='x', linestyle='None', markersize=8,    color='red')
    ax2.plot([t[0], t[-1]], [x_max[0],x_max[0]], linestyle='dashed', color='green',  alpha=0.7, label='constraints')
    ax2.plot([t[0], t[-1]], [x_min[0],x_min[0]], linestyle='dashed', color='green',  alpha=0.7)
    ax2.set_ylim([1.1*x_min[1], 1.1*x_max[1]])
    ax2.set_xticks([0,2,4,6,8,10,12,14,16,18,20])  # Set label locations.
    ax2.set_xlabel('time [s]')
    ax2.grid()
    # ax2.legend(loc=1)
    ax2.legend(loc = 'center left', bbox_to_anchor=(1.05, 0.5),fancybox=False, framealpha=1)

    # plt.subplots_adjust(left=None, bottom=None, right=None, top=None, hspace=0.4)
    import tikzplotlib
    tikzplotlib.save("figures/NNtrajectory.tex", axis_height='1.4in', axis_width='3.4in')
    # tikzplotlib.save("figures/NNtrajectory.tex")
    plt.show()

def plot_cl_ampc_compare(shooting_nodes, U, X, Checksim, Uc, Xc, latexify=False, plt_show=True):
    """
    Params:
        shooting_nodes: time values of the discretization
        u_max: maximum absolute value of u
        U: arrray with shape (N_sim-1, nu) or (N_sim, nu)
        X: arrray with shape (N_sim, nx)
        latexify: latex style plots
    """

    # latexify plot
    if latexify:
        params = {'backend': 'ps',
                'text.latex.preamble': r"\usepackage{gensymb} \usepackage{amsmath}",
                'axes.labelsize': 10,
                'axes.titlesize': 10,
                'legend.fontsize': 10,
                'xtick.labelsize': 10,
                'ytick.labelsize': 10,
                'text.usetex': True,
                'font.family': 'serif'
        }

        matplotlib.rcParams.update(params)


    ue      = 0.7853
    x_min = np.array([-0.2,-0.2])
    x_max = np.array([0.2,0.2])
    u_min = np.array([0-ue])
    u_max = np.array([2-ue])

    N_sim = X.shape[0]
    nx = X.shape[1]

    Tf = shooting_nodes[N_sim-1]
    t = shooting_nodes

    # print(x_min)
    # print(x_max)

    Ts = t[1] - t[0]

    fig, (ax1, ax2) = plt.subplots(2, 1)
    # plt.clf()
    ax1.step(t, np.append([U[0]], U[:]), label=r'$u$ valid', color='b')
    ax1.step(t, np.append([Uc[0]], Uc[:]), label=r'$u$ naive', color='r')
    ax1.plot((t[:-1])[Checksim==0], U[Checksim==0], marker='o', linestyle='None', markersize=2, color='blue', label='candidate')
    # ax1.plot((t[:-1])[Checksimc==0], Uc[Checksimc==0], marker='x', linestyle='None', markersize=8, color='red', label='error')
    # plt.title('predicted trajectory')
    ax1.set_ylabel('inputs')
    ax1.plot([t[0], t[-1]], [u_max[0],u_max[0]], linestyle='dashed', color='blue', alpha=0.7, label='constraints')
    ax1.plot([t[0], t[-1]], [u_min[0],u_min[0]], linestyle='dashed', color='blue', alpha=0.7)
    ax1.set_xticks([0,2,4,6,8,10,12,14,16,18,20])  # Set label locations_.
    ax1.set_ylim([1.1*u_min, 1.1*u_max])
    ax1.grid()
    # ax1.legend(loc=1)
    ax1.legend(loc = 'center left', bbox_to_anchor=(1.05, 0.5),fancybox=False, framealpha=1)

    ax2.set_ylabel('states')
    ax2.plot(t, X[:, 0], label=r'$x_1$ valid',  color='blue')
    ax2.plot(t, Xc[:, 0], label=r'$x_1$ naive', color='darkred',  linestyle='dashed')
    ax2.plot(t, X[:, 1], label=r'$x_2$ valid',  color='blue', linestyle='dotted' )
    ax2.plot(t, Xc[:, 1], label=r'$x_2$ naive', color='brown' , linestyle='dashdot')
    # ax2.plot((t[:-1])[Checksim==0], (X[:-1])[Checksim==0,1], marker='x', linestyle='None', markersize=8,    color='red')
    ax2.plot([t[0], t[-1]], [x_max[0],x_max[0]], linestyle='dashed', color='green',  alpha=0.7, label='constraints')
    ax2.plot([t[0], t[-1]], [x_min[0],x_min[0]], linestyle='dashed', color='green',  alpha=0.7)
    ax2.set_ylim([1.1*x_min[1], 1.1*x_max[1]])
    ax2.set_xticks([0,2,4,6,8,10,12,14,16,18,20])  # Set label locations.
    ax2.set_xlabel('time [s]')
    ax2.grid()
    # ax2.legend(loc=1)
    ax2.legend(loc = 'center left', bbox_to_anchor=(1.05, 0.5),fancybox=False, framealpha=1)

    # plt.subplots_adjust(left=None, bottom=None, right=None, top=None, hspace=0.4)
    import tikzplotlib
    tikzplotlib.save("figures/NNtrajectorycompare.tex", axis_height='1.4in', axis_width='3.4in')
    # tikzplotlib.save("figures/NNtrajectory.tex")
    plt.show()