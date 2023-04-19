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

def plot_quadcopter_ol(mpc, Utraj, Xtraj, labels, plt_show=True, limits={}):

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

    xlabels = ["x_1", "x_2", "x_3", "v_1", "v_2", "v_3", "phi_1", "omega_1", "phi_2", "omega_2"]
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
