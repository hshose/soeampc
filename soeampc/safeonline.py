import numpy as np
from tqdm import tqdm

from .mpcproblem import MPCQuadraticCostLxLu, MPC
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

    def initialize(self, x0, V_initialize):
        self.__V_candidate = V_initialize
        self.__X_candidate = self.mpc.forward_simulate_trajectory_clipped_inputs(x0,V_initialize)
        if not self.mpc.in_state_and_input_constraints(self.__X_candidate, self.__V_candidate, robust=False):
            print("WARNING: Initialization infeasible!")

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



def closed_loop_experiment(x0, V_init, controllers, Nsim=1000):
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
    nx = controllers[0].mpc.nx
    nu = controllers[0].mpc.nu
    N_controllers = len(controllers)

    V_cl                = np.zeros((N_controllers, Nsim-1, nu))
    U_cl                = np.zeros((N_controllers, Nsim-1, nu))
    X_cl                = np.zeros((N_controllers, Nsim,   nx))
    feasible_ampc       = np.zeros((N_controllers, Nsim-1))
    feasible_cl         = np.zeros((N_controllers, Nsim-1))
    in_terminal_set_cl  = np.zeros((N_controllers, Nsim))

    status = 'running'

    for i in range(N_controllers):
        X_cl[i,0,:] = x0
        controllers[i].initialize(x0, V_init)
    
    for k in range(Nsim-1):
        # simulate all controllers for one step
        for i in range(N_controllers):
            # print("\nTIMESTEP ",k, "CONTROLLER", i)
            x0_ol = X_cl[i,k,:]
            v = controllers[i](x0_ol)
            x1_ol = controllers[i].mpc.forward_simulate_single_step(x0_ol, v)
            u = controllers[i].mpc.stabilizing_feedback_controller(x0_ol, v)
            
            V_cl[i,k,:]         = np.copy(v)
            U_cl[i,k,:]         = np.copy(u)
            X_cl[i,k+1,:]       = np.copy(x1_ol)
            
            feasible_ampc[i,k]  = np.copy(controllers[i].feasible)
            feasible_cl[i,k]    = np.copy(controllers[i].mpc.in_state_and_input_constraints(X_cl[i,:k+1,:], V_cl[i,:k,:], robust=False, verbose=False))
            
            if not feasible_cl[i,k]:
                print("SOMETHING IS NOT FEASIBLE")
                controllers[i].mpc.in_state_and_input_constraints(X_cl[i,:k+1,:], V_cl[i,:k,:], robust=False, verbose=True)
                print(feasible_ampc[i,:k])
                print(U_cl[i,:k,:])
            
            in_terminal_set_cl[i,k+1] = controllers[i].mpc.in_terminal_constraints(x1_ol, robust=False )

        # abort if all controllers reach terminal set or any controller becomes infeasible
        if np.all(in_terminal_set_cl[:,k]):
            # print('allterminal at k ', k)
            status = 'allterminal'

        
        if np.any(feasible_cl[:,k]!=1):
            print(feasible_cl[:,k], k)
            status = 'infeasible'
            
        if status != 'running':
            return status, X_cl[:,:k+1], U_cl[:,:k], V_cl[:,:k], feasible_ampc[:,:k]
    
    status = 'timeout'
    return status, X_cl, U_cl, V_cl, feasible_ampc


def closed_loop_test(X_test, Y_test, controllers):
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
    results = np.array([])
    N_test = len(X_test)
    print("\ntesting controllers on", len(X_test), "initial conditions in closed loop\n")
    for i in tqdm(range(N_test)):
        x0              = X_test[i]
        V_initialize    = Y_test[i]
        status, X_cl, U_cl, V_cl, feasible_ampc = closed_loop_experiment(x0, V_initialize, controllers)
        if status == 'infeasible':
            # print(U_cl)
            print(status,"\n\n")
            results = np.append(results, {"x0":x0, "X_cl": X_cl, "U_cl": U_cl, "V_cl": V_cl, "feasible_ampc": feasible_ampc})
        if status == 'timeout':
            # print(U_cl)
            print(status,"\n\n")
            # results = np.append(results, {"x0":x0, "X_cl": X_cl, "U_cl": U_cl, "V_cl": V_cl, "feasible_ampc": feasible_ampc})
    print("\nfound ", len(results), "initial conditions, for which at least one controller becomes infeasible")
    return results

    
            
            

