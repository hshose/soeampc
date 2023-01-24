import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def plot_quadcopter_cl(mpc, Utraj, Xtraj, feasible, labels, plt_show=True, limits={}, path=None, filename=None):
    plt.clf()
    Ntrajs = len(Utraj)
    nx = mpc.nx
    Ts = mpc.Tf/mpc.N
    N_sim_max = np.max(np.array([len(Utraj[i])+1 for i in range(Ntrajs)]))
    t = np.linspace(0, (N_sim_max-1)*Ts, N_sim_max)

    linestyles = ['solid', 'dotted', 'dashed']
    colors = ['r','g','b','c','m','y','k', 'darkred', 'navy', 'darkgreen']

    xbatches = [[0,1,2], [3,4,5], [6,8], [7,9]]
    ubatches = [[0,1],[2]]

    xlabels = ["$x_1$", "$x_2$", "$x_3$", "$v_1$", "$v_2$", "$v_3$", "$\phi_1$", "$\omega_1$", "$\phi_2$", "$\omega_2$"]
    ulabels = ["$u_1$", "$u_2$", "$u_3$"]

    batches = len(xbatches) + len(ubatches)

    for k in range(len(ubatches)):
        plt.subplot(batches, 1, k+1)
        batch = ubatches[k]
        for i in range(Ntrajs):
            U = Utraj[i]
            X = Xtraj[i]
            f = feasible[i]
            # U = np.array([mpc.stabilizing_feedback_controller(X[j], V[j]) for j in range(V.shape[0])])
            for j in batch:
                line, = plt.step(t[:U.shape[0]+1], np.append([U[0,j]], U[:,j]), label=labels[i]+" "+ulabels[j], color=colors[j], linestyle=linestyles[i])
                line, = plt.plot((t[:U.shape[0]])[f==0], U[f==0,j], marker='x', linestyle='None', markersize=8, color='red')
        plt.grid()
        # plt.ylabel('inputs u')
        for j in batch:
            if "umin" in limits and not limits["umin"][j] == None:
                plt.plot([t[0], t[-1]], [limits["umin"][j],limits["umin"][j]], linestyle='dashed', color=colors[batch[0]-j], alpha=0.7)
            if "umax" in limits and not limits["umax"][j] == None:
                plt.plot([t[0], t[-1]], [limits["umax"][j],limits["umax"][j]], linestyle='dashed', color=colors[batch[0]-j], alpha=0.7)
        
        # if "umin" in limits and "umax" in limits:
        #     plt.ylim([ 
        #             1.2*np.min([limits["umin"][j] for j in batch if not limits["umin"][j] == None ]),
        #             1.2*np.max([limits["umax"][j] for j in batch if not limits["umax"][j] == None ])
        #             ])
        # plt.legend(loc=1)
        plt.legend(loc = 'center left', bbox_to_anchor=(1.05, 0.5),fancybox=False, framealpha=1)

    for k in range(len(xbatches)):
        batch = xbatches[k]
        plt.subplot(batches, 1, len(ubatches)+k+1)
        for i in range(Ntrajs):
            X = Xtraj[i]
            for j in batch:
                line, = plt.plot(t[:np.shape(X)[0]], X[:, j], label=labels[i]+" "+xlabels[j], color=colors[j], linestyle=linestyles[i])
        # plt.ylabel('states x')
        for j in batch:
            if "xmin" in limits and not limits["xmin"][j] == None:
                plt.plot([t[0], t[-1]], [limits["xmin"][j],limits["xmin"][j]], linestyle='dashed', color=colors[batch[0]-j], alpha=0.7)
            if "xmax" in limits and not limits["xmax"][j] == None:
                plt.plot([t[0], t[-1]], [limits["xmax"][j],limits["xmax"][j]], linestyle='dashed', color=colors[batch[0]-j], alpha=0.7)
        # if "xmin" in limits and "xmax" in limits:
        #     plt.ylim([ 
        #             1.2*np.min([limits["xmin"][j] for j in batch if not limits["xmin"][j] == None ]),
        #             1.2*np.max([limits["xmax"][j] for j in batch if not limits["xmax"][j] == None ])
        #             ])
        plt.grid()
        # plt.legend(loc=1)
        plt.legend(loc = 'center left', bbox_to_anchor=(1.05, 0.5),fancybox=False, framealpha=1)

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, hspace=0.4)
    plt.xlabel("time [s]")
    # import tikzplotlib
    # tikzplotlib.save("figures/NNtrajectorycompare.tex", axis_height='1.4in', axis_width='3.4in')
    if plt_show:
        plt.show()


    if not isinstance(path, type(None)) and not isinstance(filename, type(None)):
        if not os.path.isdir(path):
            os.mkdir(path)
        plt.gcf().set_size_inches(20, 15)
        plt.savefig(path.joinpath(f"{filename}.png"),dpi=300)
        import tikzplotlib
        tikzplotlib.save(path.joinpath(f"{filename}_double.tex"), axis_height='1.4in', axis_width='6.8in')
        tikzplotlib.save(path.joinpath(f"{filename}_single.tex"), axis_height='1.4in', axis_width='3.4in')
    
    plt.clf()
    ax = plt.gca()
    ax.set_aspect('equal')
    # plt.title('Feasible Set')
    axis_x = 0
    axis_y = 1
    plt.xlabel(xlabels[axis_x])
    plt.ylabel(xlabels[axis_y])
    for i in range(Ntrajs):
        X = Xtraj[i]
        f = feasible[i]
        plt.plot(X[:,axis_x],X[:,axis_y],label=labels[i], linestyle=linestyles[i], color=colors[i])
        X = Xtraj[i][:-1,:]
        plt.plot(X[f==0,axis_x], X[f==0,axis_y], marker='.', linestyle='None', markersize=8, color='red')
    plt.plot([0.145,0.145], [ np.min([np.min(Xtraj[i][:,axis_y]) for i in range(Ntrajs)]),np.max([np.max(Xtraj[i][:,axis_y]) for i in range(Ntrajs)])], linestyle='dashed', color='red')
    plt.tight_layout()
    # plt.legend(loc = 'center left', bbox_to_anchor=(1.05, 0.5),fancybox=False, framealpha=1)
    # plt.legend(loc = 1)
    plt.legend(loc=2)
    if not isinstance(path, type(None)) and not isinstance(filename, type(None)):
        if not os.path.isdir(path):
            os.mkdir(path)
        plt.gcf().set_size_inches(20, 15)
        plt.savefig(path.joinpath(f"{filename}_xy.png"),dpi=300)
        import tikzplotlib
        tikzplotlib.save(path.joinpath(f"{filename}_xy.tex"), axis_width='3.4in')
    

def plot_quadcopter_ol_V(mpc, Vtraj, labels):
    plt.clf()
    N_sim = mpc.N
    nx = mpc.nx

    Tf = mpc.Tf
    t = np.linspace(0, mpc.Tf, mpc.N+1)

    Ts = t[1] - t[0]

    Ntrajs = len(Vtraj)

    linestyles = ['solid', 'dotted', 'dashed', 'dashdot']
    looselydashed = (0, (5, 10))
    colors = ['r','g','b','c','m','y','k', 'darkred', 'navy', 'darkgreen']

    ulabels = ["u_1", "u_2", "u_3"]

    for i in range(Ntrajs):
        V = Vtraj[i]
        for j in range(3):
            line, = plt.step(t, np.append([V[0,j]], V[:,j]), label=labels[i]+" "+ulabels[j], color=colors[j], linestyle=linestyles[i])
    plt.grid()
    plt.ylabel('inputs v')
    # plt.legend(loc=1)
    plt.legend(loc=1)
    plt.show()

def plot_chain_mass_ol(mpc, Utraj, Xtraj, labels, plt_show=True, limits={}):
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