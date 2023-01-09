import numpy as np
from tqdm import tqdm

from .mpcproblem import MPCQuadraticCostLxLu, MPC
from .datasetutils import mpc_dataset_import
from .trainampc import import_model
import copy 

class AMPC():
    def __init__(self, mpc, model):
        self.__mpc = mpc
        self.__model = model
        self.__feasible = True

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
        return V[0]

class SafeOnlineEvaluationAMPC(AMPC):

    def __init__(self, mpc, model):
        super().__init__(mpc, model)
        # super(SafeOnlineEvaluationAMPC,SafeOnlineEvaluationAMPC).mpc.__set__(self,mpc)
        # super(SafeOnlineEvaluationAMPC,SafeOnlineEvaluationAMPC).model.__set__(self,model)
        self.__V_candidate = None
        self.__X_candidate = None
        self.__feasible = True

    def initialize_candidate(self, x0, V_initialize):
        self.__V_candidate = V_initialize
        self.__X_candidate = self.mpc.forward_simulate_trajectory_clipped_inputs(x0,V_initialize)
        if not self.mpc.in_state_and_input_constraints(self.__X_candidate, self.__V_candidate, robust=False):
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

        if self.mpc.feasible(X,V) and cd: # only check for cost decrease if we want to enforce it
            self.__V_candidate = copy.deepcopy(V)
            self.__X_candidate = copy.deepcopy(X)
            self.__feasible = True
            # print("tic")
        else:
            # print("toc")
            self.__feasible = False

        v = np.copy(self.__V_candidate[0])

        # print("this candidate is feasible", self.mpc.in_state_and_input_constraints(self.__X_candidate, self.__V_candidate, verbose=True))

        self.__X_candidate, self.__V_candidate = self.shift_append_terminal(self.__X_candidate, self.__V_candidate)
        # print("X_cand", self.__X_candidate)
        # print("V_cand", self.__V_candidate)
        # print("next candidate is feasible", self.mpc.in_state_and_input_constraints(self.__X_candidate, self.__V_candidate, verbose=True))
        return v

    def __call__(self,x0):
        V = self.V(x0)
        return self.safe_evaluate(x0,V)

class SafeOnlineEvaluationAMPCGroundTruethInit(SafeOnlineEvaluationAMPC):
    def __init__(self, mpc, model):
        super().__init__(mpc, model)

    def initialize(self, x0, V):
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
        tuple (status, X_cl, U_cl, V_cl, feasible_ampc) with status string, X, U and V closed loop results of shape (N_controllers, mpc.N, nx/nu) and feasible_ampc is array of bools indicating if the the ampc solution was taken or a backup solution was invoked.
        
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
        v = controller(x0_ol)
        x1_ol = controller.mpc.forward_simulate_single_step(x0_ol, v)
        u = controller.mpc.stabilizing_feedback_controller(x0_ol, v)
        
        V_cl[k,:]         = np.copy(v)
        U_cl[k,:]         = np.copy(u)
        X_cl[k+1,:]       = np.copy(x1_ol)
        
        feasible_ampc[k]  = np.copy(controller.feasible)
                
        feasible_cl = np.copy(controller.mpc.in_state_and_input_constraints(X_cl[:k+2,:], V_cl[:k+1,:], robust=False, verbose=False))
        if not feasible_cl:
            status = 'infeasible_cl'
            print("SOMETHING IS NOT FEASIBLE")
            controller.mpc.in_state_and_input_constraints(X_cl[:k+2,:], V_cl[:k+1,:], robust=False, verbose=True)
            print(feasible_ampc[:k])
            print(U_cl[:k,:])
        
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
        status, X, U, V, feasible = closed_loop_experiment(x0, controller, Nsim=Nsim)
        results.append({"status":status, "X":X, "U":U, "V":V, "feasible": feasible, })
    return results

def closed_loop_test_on_dataset(dataset, model_name, N_samples=int(1e3)):
    """performs closed loop simulation on dataset of initial conditions
    
    Args:
        X_test:
            dataset of initial conditions
        Y_test:
            corresponding open-loop trajectories used for initialization (should be feasible solution to mpc problem at corresponding x0)
        controllers:
            array of controllers to be tested (see closed_loop_experiment)
    Returns:
        array of dicts, containing initial states and closed loop trajectories for all controllers at which at least one controller was infeasible
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

    controllers = [ naive_controller, safe_controller, safe_controller_ground_trueth_init ]
    controller_names = [ "naive", "safe", "safe init" ]
    
    X_test = X[:N_samples]
    V_test = V[:N_samples]
    results = []
    print("\ntesting controllers on", len(X_test), "initial conditions in closed loop\n")
    for i in tqdm(range(N_samples)):
        x0              = X_test[i]
        V_initialize    = V_test[i]
        simulation_results = iterate_controllers(x0, V_initialize, controllers)
        results.append(simulation_results)

    for j in range(len(controllers)):
        mu_cl_feasible = np.mean(np.array([results[i][j]["status"]!="infeasible_cl" for i in range(N_samples)]))
        status_initially_feasible = [ results[i][j]["status"] for i in range(N_samples) if results[i][j]["feasible"][0] ]
        mu_cl_feasible_of_initially_feasible = np.mean(status_initially_feasible!="infeasible_cl")
        print(f"Results for controller: {controller_names[j]}:\n\t {mu_cl_feasible=} \n\t{mu_cl_feasible_of_initially_feasible=}")


        # if status == 'infeasible':
        #     # print(U_cl)
        #     print(status,"\n\n")
        #     results = np.append(results, {"x0":x0, "X_cl": X_cl, "U_cl": U_cl, "V_cl": V_cl, "feasible_ampc": feasible_ampc})
        # if status == 'timeout':
        #     # print(U_cl)
        #     print(status,"\n\n")
        #     # results = np.append(results, {"x0":x0, "X_cl": X_cl, "U_cl": U_cl, "V_cl": V_cl, "feasible_ampc": feasible_ampc})

    
            
            
def closed_loop_test_on_sampler():
    pass