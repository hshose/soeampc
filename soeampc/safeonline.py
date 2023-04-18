import numpy as np
from tqdm import tqdm

from .mpcproblem import MPCQuadraticCostLxLu, MPC
from .datasetutils import mpc_dataset_import
from .trainampc import import_model, import_model_mpc
import copy 

class AMPC():
    def __init__(self, mpc, model):
        self.__mpc = mpc
        self.__model = model
        self.__feasible = False

    @property
    def mpc(self):
        return self.__mpc

    @property
    def feasible(self):
        return self.__feasible

    @mpc.setter
    def mpc(self, mpc):
        if isinstance(mpc, MPC):
            self.__mpc = mpc
        else:
            raise Exception('Invalid mpc value, excpected type MPC')
    
    def initialize(self, x0, V):
        pass

    def V(self,x0):
        return np.reshape(self.__model(x0).numpy(), (self.__mpc.N, self.__mpc.nu))

    def __call__(self,x0):
        V = self.V(x0)
        X = self.mpc.forward_simulate_trajectory_clipped_inputs(x0,V)
        self.__feasible = self.mpc.feasible(X, V)
        return V[0], X[1]

class SafeOnlineEvaluationAMPC(AMPC):

    def __init__(self, mpc, model):
        super().__init__(mpc, model)
        # super(SafeOnlineEvaluationAMPC,SafeOnlineEvaluationAMPC).mpc.__set__(self,mpc)
        # super(SafeOnlineEvaluationAMPC,SafeOnlineEvaluationAMPC).model.__set__(self,model)
        self.__V_candidate = None
        self.__X_candidate = None

    def initialize_candidate(self, x0, V_initialize, verbose=True):
        self.__V_candidate = V_initialize
        self.__X_candidate = self.mpc.forward_simulate_trajectory_clipped_inputs(x0,V_initialize)
        self.__feasible = self.mpc.feasible(self.__X_candidate, self.__V_candidate, robust=False)
        if not self.mpc.in_state_and_input_constraints(self.__X_candidate, self.__V_candidate, robust=False) and verbose:
            print("WARNING: Initialization infeasible!")

    def initialize(self,x0,V):
        self.initialize_candidate(x0, self.V(x0))


    @property
    def feasible(self):
        return self.__feasible

    def shift_append_terminal(self,X,V):
        # print("X before shift", X)
        # print("V before shift", V)
        V_shifted = np.zeros((self.mpc.N, self.mpc.nu))
        V_shifted[:-1] = np.copy(V[1:])
        V_shifted[-1] = self.mpc.terminal_controller(X[-1])
        # V = np.append(V[1:], self.__terminal_controller(X[-1]), axis=0)
        X_shifted = self.mpc.forward_simulate_trajectory_clipped_inputs(X[1], V_shifted)
        # print("X after shift", X_shifted)
        # print("V after shift", V_shifted)
        return X_shifted,V_shifted

    def safe_evaluate(self,x,V, enforce_cost_decrease=True):
        # print("V", V)
        X = self.mpc.forward_simulate_trajectory_clipped_inputs(x,V)

        # print("clipped inputs are feasible:", self.mpc.in_state_and_input_constraints(X, V, verbose=True, robust=False, only_states=True))
        # print("V_clipped",V_clipped)

        cd = True
        if enforce_cost_decrease:
            cd = ( self.mpc.cost(X, V) <= self.mpc.cost(self.__X_candidate, self.__V_candidate ) )

        cf = self.mpc.feasible(self.__X_candidate, self.__V_candidate)
        # if not cf:
            # print(f"THIS SHOULD NEVER HAPPEN {self.i}")

        nf = self.mpc.feasible(X,V)
        nf_state = self.mpc.in_state_and_input_constraints(X, V)
        nf_terminal = self.mpc.in_terminal_constraints(X[-1,:])

        self.feasible_debug = {"candidate_feasible":cf, "in_state_and_input_constraints":nf_state, "in_terminal_constraint":nf_terminal, "cost_decrease": cd}

        if (nf and cd) or (not cf): # only check for cost decrease if we want to enforce it, if candidate is infeasible, always take NN, we have no guarantees anyways
            self.__V_candidate = copy.deepcopy(V)
            self.__X_candidate = copy.deepcopy(X)
            self.__feasible = nf
            # print("tic")
        else:
            # print("toc")
            self.__feasible = False

        v = np.copy(self.__V_candidate[0])
        x = np.copy(self.__X_candidate[1])
        # print("this candidate is feasible", self.mpc.in_state_and_input_constraints(self.__X_candidate, self.__V_candidate, verbose=True))

        # if not self.mpc.feasible(self.__X_candidate, self.__V_candidate, verbose=True):
        #     print(f"SECOND THIS SHOULD NEVER HAPPEN {self.i}")
        # v_old_candidate = np.copy(self.__V_candidate)
        # x_old_candidate = np.copy(self.__X_candidate)

        self.__X_candidate, self.__V_candidate = self.shift_append_terminal(self.__X_candidate, self.__V_candidate)
        
        # if not self.mpc.feasible(self.__X_candidate, self.__V_candidate, verbose=True):
        #     print(f"THIRD THIS SHOULD ALSO NEVER HAPPEN {self.i}")
        #     print(f"{self.__V_candidate[:-1]-v_old_candidate[1:]}")
        #     print(f"{self.__X_candidate[:-1]-x_old_candidate[1:]}")

        # self.i +=1
        return v,x

    def __call__(self,x0):
        V = self.V(x0)
        return self.safe_evaluate(x0,V)

class SafeOnlineEvaluationAMPCGroundTruethInit(SafeOnlineEvaluationAMPC):
    def __init__(self, mpc, model):
        super().__init__(mpc, model)

    def initialize(self, x0, V):
        self.i = 0
        return self.initialize_candidate(x0, V)


def closed_loop_experiment(x0, controller, Nsim=1000):
    """simulation for testing controllers
    
    Args:
        x0:
            starting state for closed loop simulation
        V_init:
            initialization control sequence, should be feasible solution to underlying mpc problem
        controllers:
            array of AMPC class instances, must implement `init(x,V)` and `__call__(x)` functions and have property mpc of class `MPC`.
        Nsim:
            maximum closed loop simulation steps for timeout, defaults to 1000

    Returns:
        
    """
    nx = controller.mpc.nx
    nu = controller.mpc.nu

    V_cl                = np.zeros((Nsim-1, nu))
    U_cl                = np.zeros((Nsim-1, nu))
    X_cl                = np.zeros((Nsim,   nx))
    feasible_ampc       = np.zeros((Nsim-1))
    status = 'running'

    X_cl[0,:] = x0
    
    for k in range(Nsim-1):
        # print("\nTIMESTEP ",k, "CONTROLLER", i)
        x0_ol = X_cl[k,:]
        v, x1_ol = controller(x0_ol)
        # x1_ol = controller.mpc.forward_simulate_single_step(x0_ol, v)
        u = controller.mpc.stabilizing_feedback_controller_clipped_inputs(x0_ol, v)
        
        V_cl[k,:]         = np.copy(v)
        U_cl[k,:]         = np.copy(u)
        X_cl[k+1,:]       = np.copy(x1_ol)

        feasible_ampc[k]  = np.copy(controller.feasible)

        feasible_cl = np.copy(controller.mpc.in_state_and_input_constraints(X_cl[:k+2,:], V_cl[:k+1,:], robust=False, verbose=False))
        if not feasible_cl:
            status = 'infeasible_cl'
            # print("SOMETHING IS NOT FEASIBLE")
            # controller.mpc.in_state_and_input_constraints(X_cl[:k+2,:], V_cl[:k+1,:], robust=False, verbose=True)
            # print(feasible_ampc[:k+1])
            # print(U_cl[:k,:])
        
        terminalset_reached = controller.mpc.in_terminal_constraints(x1_ol, robust=False )
        if terminalset_reached:
            status = 'terminal_set_reached'
           
        if status != 'running':
            return status, X_cl[:k+2], U_cl[:k+1], V_cl[:k+1], feasible_ampc[:k+1]
    
    status = 'timeout'
    return status, X_cl, U_cl, V_cl, feasible_ampc

def iterate_controllers(x0, V_init, controllers, Nsim=1000):
    results = []
    for controller in controllers:
        controller.initialize(x0, V_init)
        feasible_init = controller.feasible
        status, X, U, V, feasible = closed_loop_experiment(x0, controller, Nsim=Nsim)
        results.append({"status":status, "X":X, "U":U, "V":V, "feasible": feasible, "feasible_init": feasible_init})
    return results

def closed_loop_test_on_dataset(dataset, model_name, N_samples=int(1e3), N_sim=200):
    """performs closed loop simulation on dataset of initial conditions
    
    Args:
        dataset:
            dataset name to be imported
        model_name:
            name model to be imported
        N_samples:
            number of samples to be evaluated from dataset
    Returns:
    """
    mpc, X, V, _, _ = mpc_dataset_import(dataset)
    if N_samples >= X.shape[0]:
        N_samples = X.shape[0]
        print("WARNING: N_samples exceeds size of dataset, will use N_samples =", N_samples,"instead")
    # X_test, X_train, Y_test, Y_train = train_test_split(X, U, test_size=0.1, random_state=42)
    model = import_model(modelname=model_name)

    naive_controller = AMPC(mpc, model)
    safe_controller = SafeOnlineEvaluationAMPC(mpc, model)
    safe_controller_ground_trueth_init = SafeOnlineEvaluationAMPCGroundTruethInit(mpc, model)

    # controllers = [ naive_controller, safe_controller, safe_controller_ground_trueth_init ]
    controllers = [ naive_controller, safe_controller_ground_trueth_init ]
    controller_names = [ "naive", "safe init" ]
    
    # sample_idx = [104, 112, 131, 154, 242, 484, 514, 667, 738, 835, 866, 976]
    # N_samples = len(sample_idx)
    # X_test = X[sample_idx]
    # V_test = V[sample_idx]

    X_test = X[:N_samples]
    V_test = V[:N_samples]
    results = []
    print("\ntesting controllers on", len(X_test), "initial conditions in closed loop\n")
    for i in tqdm(range(N_samples)):
        x0              = X_test[i]
        V_initialize    = V_test[i]
        simulation_results = iterate_controllers(x0, V_initialize, controllers, Nsim = N_sim)
        results.append(simulation_results)

    # l1 = [results[i][0]["status"] for i in range(N_samples)]
    # l2 = [results[i][1]["status"] for i in range(N_samples)]
    # l3 = [results[i][1]["feasible"] for i in range(N_samples)]
    # l4 = [results[i][1]["feasible_init"] for i in range(N_samples)]
    # print(f"\n\nresults are \n naive: {l1} \n safe init: {l2} \n safe init feasible_init: {l4} \n safe init feasible: {l3}")

    for j in range(len(controllers)):
        status_cl = np.array([results[i][j]["status"] for i in range(N_samples) if (any(results[i][1]["feasible"]) or results[i][1]["feasible_init"])])
        status_cl_safe_NN_init = np.array([results[i][j]["status"] for i in range(N_samples) if results[i][j]["feasible"][0]])
        # print(f"status cl debug = {status_cl}")
        terminal_set_rached_cl = np.mean(status_cl=="terminal_set_reached")
        feasible_cl = np.mean(status_cl!="infeasible_cl")
        feasible_cl_safe_NN_init = np.mean(status_cl_safe_NN_init!="infeasible_cl")
        print(f"Results for controller: {controller_names[j]}: {terminal_set_rached_cl=}, {feasible_cl=}, {feasible_cl_safe_NN_init=}")
    # # naive
    # status_naive = [results[i][j]["status"] for i in range(N_samples) if (results[i][3]["feasible"][0] or results[i][3]["feasible_init"])]
    # j=0

    # print(f"Results for controller: {controller_names[j]}: {terminal_set_rached_cl=}, {feasible_cl=}")

    # # safe
    # j=1
    # status_safe = np.array([results[i][j]["status"] for i in range(N_samples) if (results[i][3]["feasible"][0] or results[i][3]["feasible_init"])])
    # terminal_set_rached_cl = np.mean(status_safe=="terminal_set_reached")
    # feasible_cl = np.mean(status_safe!="infeasible")
    # print(f"Results for controller: {controller_names[j]}: {terminal_set_rached_cl=}, {feasible_cl=}")
    
    # # safe init
    # j=2
    # status_safe_init = [ results[i][j]["status"]!="infeasible_cl" for i in range(N_samples) if (results[i][j]["feasible"][0] or results[i][j]["feasible_init"]) ]
    # aprint = [results[i][j]["feasible_init"] for i in range(N_samples)]
    # print(f"{aprint=}")
    # print(f"{status_initially_feasible=}")
    # mu_cl_feasible = np.mean(status_initially_feasible!="infeasible")
    # print(f"Results for controller: {controller_names[j]}: {mu_cl_feasible=}")
    
    idx_naive_infeas_safe_feas = [i for i in range(N_samples) if (results[i][0]["status"] == "infeasible_cl" ) and ( results[i][1]["status"] != "infeasible_cl")]
    print(f"{idx_naive_infeas_safe_feas=}")
    return
    return [results[idx] for idx in idx_naive_infeas_safe_feas], controller_names, mpc

def closed_loop_test_wtf(dataset, model_name, N_samples=int(1e3)):
    mpc, X, V, _, _ = mpc_dataset_import(dataset)
    if N_samples >= X.shape[0]:
        N_samples = X.shape[0]
        print("WARNING: N_samples exceeds size of dataset, will use N_samples =", N_samples,"instead")
    # X_test, X_train, Y_test, Y_train = train_test_split(X, U, test_size=0.1, random_state=42)
    model = import_model(modelname=model_name)

    controller = SafeOnlineEvaluationAMPCGroundTruethInit(mpc, model)

    X_test = X[:N_samples]
    V_test = V[:N_samples]
    results = []
    print("\ntesting controllers on", len(X_test), "initial conditions in closed loop\n")
    for i in tqdm(range(N_samples)):
        x0              = X_test[i]
        V_initialize    = V_test[i]
        controller.initialize(x0, V_initialize)
        status, X, U, V, feasible = closed_loop_experiment(x0, controller, Nsim=N_samples)
        if ( status == "infeasible_cl" ) and feasible[0]:
            print(f"{X=},{U=},{V=},{feasible=}")

def closed_loop_test_on_sampler(model_name, sampler, N_samples=int(1e3), N_sim=20):
    """performs closed loop simulation on sampled initial conditions
    
    Args:
        model_name:
            name of the model to be tested
        sampler:
            instance of class sampler
        N_samples:
            number of samples to be evaluated
    Returns:
        array of dicts, containing initial states and closed loop trajectories for all controllers at which at least one controller was infeasible
    """
    mpc, model = import_model_mpc(modelname=model_name)

    naive_controller = AMPC(mpc, model)
    safe_controller = SafeOnlineEvaluationAMPC(mpc, model)

    controllers = [ naive_controller, safe_controller ]
    controller_names = [ "naive", "safe" ]
    
    results = []

    print(f"\ntesting controllers on {N_samples} initial conditions in closed loop\n")
    for i in tqdm(range(N_samples)):
        x0 = sampler.sample()
        simulation_results = iterate_controllers(x0, None, controllers, Nsim=N_sim)
        results.append(simulation_results)

    for j in range(len(controllers)):
        mu_cl_in_terminal_set = np.mean(np.array([results[i][j]["status"]=="terminal_set_reached" for i in range(N_samples)]))
        mu_cl_feasible = np.mean(np.array([results[i][j]["status"]!="infeasible_cl" for i in range(N_samples)]))
        print(f"Results for controller: {controller_names[j]}: {mu_cl_feasible=} {mu_cl_in_terminal_set=}")


def closed_loop_test_reason(dataset, model_name, N_samples=int(1e3), N_sim=200):
    mpc, X, V, _, _ = mpc_dataset_import(dataset)
    if N_samples >= X.shape[0]:
        N_samples = X.shape[0]
        print("WARNING: N_samples exceeds size of dataset, will use N_samples =", N_samples,"instead")
    # X_test, X_train, Y_test, Y_train = train_test_split(X, U, test_size=0.1, random_state=42)
    model = import_model(modelname=model_name)

    controller = SafeOnlineEvaluationAMPCGroundTruethInit(mpc, model)
    X_test = X
    # X_test = X[:N_samples]
    V_test = V
    # V_test = V[:N_samples]
    results = []
    print("\ntesting controllers on", len(X_test), "initial conditions in closed loop\n")
    # j = 0
    for i in tqdm(range(N_samples)):
        # feasible_initial_point = False
        x0              = X_test[i]
        V_initialize    = V_test[i]
        controller.initialize(x0, V_initialize)
        # while( not feasible_initial_point ):
            # x0              = X_test[j]
            # V_initialize    = V_test[j]
            # j=j+1
            # controller.initialize(x0, V_initialize)
            # feasible_initial_point = controller.feasible

        V_cl                = np.zeros((N_sim-1, mpc.nu))
        U_cl                = np.zeros((N_sim-1, mpc.nu))
        X_cl                = np.zeros((N_sim,   mpc.nx))
        feasible_ampc       = np.zeros((N_sim-1))
        status = 'running'

        X_cl[0,:] = x0
    
        for k in range(N_sim-1):
            # print("\nTIMESTEP ",k, "CONTROLLER", i)
                    # print("\nTIMESTEP ",k, "CONTROLLER", i)
            x0_ol = X_cl[k,:]
            v, x1_ol = controller(x0_ol)
            # x1_ol = controller.mpc.forward_simulate_single_step(x0_ol, v)
            u = controller.mpc.stabilizing_feedback_controller_clipped_inputs(x0_ol, v)
            
            V_cl[k,:]         = np.copy(v)
            U_cl[k,:]         = np.copy(u)
            X_cl[k+1,:]       = np.copy(x1_ol)

            results.append(controller.feasible_debug)
            
            feasible_cl = np.copy(controller.mpc.in_state_and_input_constraints(X_cl[:k+2,:], V_cl[:k+1,:], robust=False, verbose=False))
            if not feasible_cl:
                status = 'infeasible_cl'
                # print("SOMETHING IS NOT FEASIBLE")
                # controller.mpc.in_state_and_input_constraints(X_cl[:k+2,:], V_cl[:k+1,:], robust=False, verbose=True)
                # print(feasible_ampc[:k+1])
                # print(U_cl[:k,:])
            
            terminalset_reached = controller.mpc.in_terminal_constraints(x1_ol, robust=False )
            if terminalset_reached:
                status = 'terminal_set_reached'
            
            if status != 'running':
                break

    
    # self.feasible_debug = {"candidate_feasible":cf, "in_state_and_input_constraints":nf_state, "in_terminal_constraint":nf_terminal, "cost_decrease": cd}
    N_results = len(results)
    # print([r for r in results])
    # rejectionarray = [not (r["in_state_and_input_constraints"] and r["in_terminal_constraint"] and r["cost_decrease"]) for r in results]
    # print(rejectionarray)
    rejection_rate = np.mean(np.array([not (r["in_state_and_input_constraints"] and r["in_terminal_constraint"] and r["cost_decrease"]) for r in results]))
    rejections = [r for r in results if not (r["in_state_and_input_constraints"] and r["in_terminal_constraint"] and r["cost_decrease"])]
    rejection_from_state_and_input_constraint= np.mean(np.array([not r["in_state_and_input_constraints"] for r in rejections]))
    rejection_from_terminal_constraint= np.mean(np.array([not r["in_terminal_constraint"] for r in rejections]))
    rejection_from_cost_decrease= np.mean(np.array([not r["cost_decrease"] for r in rejections]))

    print(f"{rejection_rate=}")
    print(f"{rejection_from_state_and_input_constraint=}")
    print(f"{rejection_from_terminal_constraint=}")
    print(f"{rejection_from_cost_decrease=}")


    

