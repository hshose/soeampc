from abc import ABC, abstractmethod
import numpy as np
from scipy.integrate import odeint

import importlib
import inspect

from pathlib import Path

__all__ = ['MPC', 'MPCQuadraticCostLxLu', 'import_mpc']

def check_box_constraint(series, lower, upper):
    for s in series:
        if not (all(s<=upper) and all(s>=lower)):
            return False
    return True

class MPC(ABC):
    "Dimensions for OCP"
    def __init__(self):
        super().__init__()
        self.__name = None
        self.__nx = None
        self.__nu = None
        self.__N = None
        self.__Tf = None
        self.__f = None
        self.__f_type = None
        self.__terminal_controller = None
    
    @property
    def name(self):
        """ `name` - unique name of the problem, used to save and load default dataset names and constants
        """
        return self.__name

    @property
    def nx(self):
        return self.__nx

    @property
    def nu(self):
        return self.__nu

    @property
    def N(self):
        return self.__N

    @property
    def Tf(self):
        return self.__Tf

    @property
    def f(self):
        """:math:`f(x,u)` - system dynamics, if cont time, ::math::`\dot{f}=f(x,u)`, in discrete time, ::math::`x^+=f(x,u)`.
        Default: :code:`None`.
        """
        return self.__f

    @property
    def f_type(self):
        """Type of system dynamics function.
        -- string in {CT, DT}
        Default: :code:'CT'.
        """
        return self.__f_type
    
    @property
    def terminal_controller(self):
        """`terminal_controller(x)` - evaluates the therminal controller for given state x
        Default: :code:'None'.
        """
        return self.__terminal_controller

    @property
    def stabilizing_feedback_controller(self):
        """`stabilizing_feedback_controller(x, v)` - evaluates the therminal controller for given state x and inputs v, returns feedback u
        Default: :code:'None'.
        """
        return self.__stabilizing_feedback_controller

    @name.setter
    def name(self, name):
        if isinstance(name, str):
            self.__name = name
        else:
            raise Exception('Invalid name value, expected string')

    @nx.setter
    def nx(self, nx):
        if isinstance(nx, int) and nx > -1:
            self.__nx = nx
        else:
            raise Exception('Invalid nx value, expected non negative int')
    
    @nu.setter
    def nu(self, nu):
        if isinstance(nu, int) and nu > -1:
            self.__nu = nu
        else:
            raise Exception('Invalid nu value, expected non negative int')
    
    @N.setter
    def N(self, N):
        if isinstance(N, int) and N > 0:
            self.__N = N
        else:
            raise Exception('Invalid N value, expected positive int')

    @Tf.setter
    def Tf(self,Tf):
        self.__Tf = Tf


    @f.setter
    def f(self,f):
        if callable(f):
            self.__f = f
        else:
            raise Exception('Invalid f, must be callable')

    @f_type.setter
    def f_type(self, f_type):
        f_types = ('CT', 'DT')
        if f_type in f_types:
            self.__f_type = f_type
        else:
            raise Exception('Invalid f_type value, got', f_type, ' which is not in', f_types)       


    @terminal_controller.setter
    def terminal_controller(self, terminal_controller):
        if callable(terminal_controller):
            self.__terminal_controller = terminal_controller
        else:
            raise Exception('Invalid terminal_controller, must be callable')

    @stabilizing_feedback_controller.setter
    def stabilizing_feedback_controller(self, stabilizing_feedback_controller):
        if callable(stabilizing_feedback_controller):
            self.__stabilizing_feedback_controller = stabilizing_feedback_controller
        else:
            raise Exception('Invalid stabilizing_feedback_controller, must be callable')


    @abstractmethod
    def feasible(self,X,U):
        pass

    @abstractmethod
    def cost(self,X,U):
        pass

    @abstractmethod
    def savetxt(self, outpath):
        pass

    @staticmethod
    @abstractmethod
    def genfromtxt(inpath):
        pass

    def forward_simulate_trajectory(self,x,V):
        Ts = self.__Tf/self.__N
        t = np.linspace(0,self.__Tf,self.__N+1)
        if self.f_type == 'CT':
            def f_pwconst_input(y,t,V,Ts):
                    x = y
                    idx = min( int(t/Ts), np.shape(V)[0]-1)
                    u = self.__stabilizing_feedback_controller(x, V[idx])
                    return tuple(self.__f(x, u))

            X = odeint(f_pwconst_input, x, t, args=(V, Ts))
            return X
        else:
            raise Exception('DT not implemented yet')

    def forward_simulate_single_step(self,x,u):
        Ts = self.__Tf/self.__N
        t = np.linspace(0,Ts,2)
        if self.f_type == 'CT':
            def f_pwconst_input(y,t):
                    x = y
                    ustable = self.__stabilizing_feedback_controller(x, u)
                    return tuple(self.__f(x, ustable))

            X = odeint(f_pwconst_input, x, t)
            return X[-1]
        else:
            raise Exception('DT not implemented yet')

class MPCQuadraticCostLxLu(MPC):

    def __init__(self, f, nx, nu, N, Tf, Q, R, P, alpha, K, Lx, Lu, Kdelta=None, alpha_reduced=None, S=None, Ls=None):
        super().__init__()
        super(MPCQuadraticCostLxLu,MPCQuadraticCostLxLu).N.__set__( self,  N  )
        super(MPCQuadraticCostLxLu,MPCQuadraticCostLxLu).nx.__set__(self,  nx )
        super(MPCQuadraticCostLxLu,MPCQuadraticCostLxLu).nu.__set__(self,  nu )
        super(MPCQuadraticCostLxLu,MPCQuadraticCostLxLu).Tf.__set__(self,  Tf )
        super(MPCQuadraticCostLxLu,MPCQuadraticCostLxLu).f.__set__( self,  f  )
        super(MPCQuadraticCostLxLu,MPCQuadraticCostLxLu).f_type.__set__( self,  'CT'  )


        self.__P = P
        self.__alpha = alpha
        self.__Q = Q
        self.__R = R
        self.__K = K
        self.__S = S
        self.__Ls = Ls

        if alpha_reduced == None:
            self.__alpha_reduced = alpha
        else:
            self.__alpha_reduced = alpha_reduced
        
        # if no Kdelta is set, this reverts to no stabilizing feedback.
        if isinstance(Kdelta, type(None)):
            self.__Kdelta = np.zeros((self.nu, self.nx))
        elif Kdelta.shape[0] == self.nu and Kdelta.shape[1] == self.nx:
            self.__Kdelta = Kdelta
        else:
            raise Exception("Dimension mismatch between Kdelta, nx, and nu!")
    
        # self.__stabilizing_feedback_controller = lambda x,v: self.__Kdelta @ x + v 
        # self.__terminal_controller = lambda x: (self.__K-self.__Kdelta) @ x
        super(MPCQuadraticCostLxLu,MPCQuadraticCostLxLu).stabilizing_feedback_controller.__set__( self,  lambda x,v: self.__Kdelta @ x + v  )
        super(MPCQuadraticCostLxLu,MPCQuadraticCostLxLu).terminal_controller.__set__( self,  lambda x: (self.__K-self.__Kdelta) @ x  )

        if Lx.shape[0] == Lu.shape[0] and Lx.shape[1] == self.nx and Lu.shape[1] == self.nu:
            self.__nconstr = Lx.shape[0]
            self.__Lx = Lx
            self.__Lu = Lu
        else:
            raise Exception("Dimensions mismatch between Lx, Lu, nx, and nu")
        
        self.__umin = 1/np.min(self.Lu, 0) # possibly crude / wrong if not box constraints!!!
        self.__umax = 1/np.max(self.Lu, 0) # possibly crude / wrong if not box constraints!!!

        # # if no explicit Lx and Lu are supplied, we compute them here...
        # if Lx==None and Lu==None:
        #     self.__Lx = np.vstack((np.diag(1/xmax), np.diag(1/xmin)))
        #     self.__Lu = np.vstack((np.diag(1/umax), np.diag(1/umin)))
        #     self.__ug = np.ones(self.__Lx.shape[0]+self.__Lu.shape[0])
        # else:
        #     self.__Lx = Lx
        #     self.__Lu = Lu
        #     self.__ug = np.ones(self.__Lx.shape[0]+self.__Lu.shape[0])

    @property
    def Lx(self):
        return self.__Lx
    
    @property
    def Lu(self):
        return self.__Lu

    @property
    def P(self):
        return self.__P

    @property
    def Q(self):
        return self.__Q

    @property
    def R(self):
        return self.__R

    @property
    def Q(self):
        return self.__Q

    @property
    def alpha(self):
        return self.__alpha

    @property
    def K(self):
        return self.__K

    @property
    def Kdelta(self):
        return self.__Kdelta

    def in_state_and_input_constraints(self, X, U, verbose = False, eps = 1e-4, robust=False, only_states=True):
        # return np.all((self.__Lx@(X[:-1]).T + self.__Lu@U.T <= 1), axis=0)
        if only_states:
            Lx = self.__Lx[ np.all(self.__Lu==0,axis=1) ]
            Lu = self.__Lu[ np.all(self.__Lu==0,axis=1) ]
        else:
            Lx = self.__Lx
            Lu = self.__Lu

        if robust:
            constrineq = ( Lx@(X[:-1]).T + Lu@U.T + self.__Ls@self.__S[:-1].T - 1 <= eps )
        else:
            constrineq = ( Lx@(X[:-1]).T + Lu@U.T - 1 <= eps )

        if verbose and not np.all(constrineq):
            # print(X,U)
            print("\t LxLu constraints violated:   ", np.where(constrineq == False))
            print("constr ineq = ", constrineq)
            print(f"vals  = {(X[:-1])[np.logical_not(np.all(constrineq, axis=0))]}")
        return np.all( constrineq )

    def in_terminal_constraints(self, x, verbose = False, eps = 1e-4, robust=True):
        r = x.T @ self.__P @ x
        if robust:
            alpha = self.__alpha_reduced
        else:
            alpha = self.__alpha
        constrineq = ( r - (alpha**2) <= eps )
        if verbose and not constrineq:
            print("\t Terminal constraint x.T P x = ", r, " <= ", alpha**2, "is violated")
        return constrineq

    def feasible(self,X,U, verbose=False, only_states=True, robust=False, eps=0.0001):
        res = True
        res = res and self.in_state_and_input_constraints(X,U, robust=robust, only_states=only_states, eps=eps)
        res = res and self.in_terminal_constraints(X[-1,:],    robust=robust, eps=eps)
        if verbose and not res:
            print("Infeasible Trajectory")
            print("\tin state constraint:   ", self.in_state_and_input_constraints(X, U, verbose=True, robust=robust, only_states=only_states))
            print("\tin termial constraint: ", self.in_terminal_constraints(X[-1,:], verbose=True  , robust=robust))
        return res
    
    def forward_simulate_trajectory_clipped_inputs(self, x, V):
        Ts = self.Tf/self.N
        t = np.linspace(0,self.Tf,self.N+1)
        if self.f_type == 'CT':
            def f_pwconst_input(y,t,V,Ts):
                    x = y
                    idx = min( int(t/Ts), np.shape(V)[0]-1)
                    u = np.clip(self.stabilizing_feedback_controller(x, V[idx,:]), self.__umin, self.__umax)
                    return tuple(self.f(x, u))

            X = odeint(f_pwconst_input, x, t, args=(V, Ts))
            return X
        else:
            raise Exception('DT not implemented yet')

    def forward_simulate_single_step_clipped_inputs(self,x,v):
        Ts = self.Tf/self.N
        t = np.linspace(0,Ts,2)
        if self.f_type == 'CT':
            def f_pwconst_input(y,t):
                    x = y
                    ustable = np.clip(self.stabilizing_feedback_controller(x, v), self.__umin, self.__umax)
                    return tuple(self.f(x, ustable))

            X = odeint(f_pwconst_input, x, t)
            return X[-1]
        else:
            raise Exception('DT not implemented yet')


    def stabilizing_feedback_controller_clipped_inputs(self, x, v):
        u = np.clip(self.stabilizing_feedback_controller(x,v), self.__umin, self.__umax)
        if any(u > self.__umax) or any(u < self.__umin):
            print(u)
        return u

    def cost(self, X, U, clipped_inputs=True):
        cost = 0
        for k in range(self.N):
            x = X[k]
            v = U[k]
            if clipped_inputs:
                u = np.clip(self.stabilizing_feedback_controller(x, v), self.__umin, self.__umax)
            else:
                u = self.stabilizing_feedback_controller(x, V[idx,:])    
            cost = cost + x@self.__Q@x.T + u@self.__R@u.T
        
        cost = cost+X[self.N]@self.__P@X[self.N].T
        return cost
    
    def savetxt(self, outpath):
        p = outpath
        p.mkdir(parents=True,exist_ok=True)
        with open(p.joinpath('name.txt'), 'w') as file:
            file.write(self.name)

        np.savetxt(p.joinpath("nx.txt"),    np.array([self.nx]), fmt='%i', delimiter=",")
        np.savetxt(p.joinpath("nu.txt"),    np.array([self.nu]), fmt='%i', delimiter=",")
        np.savetxt(p.joinpath("N.txt"),     np.array([self.N]),  fmt='%i', delimiter=",")
        np.savetxt(p.joinpath("Tf.txt"),    np.array([self.Tf]),           delimiter=",")

        np.savetxt(p.joinpath("Lx.txt"),      np.array(self.Lx),      delimiter=",")
        np.savetxt(p.joinpath("Lu.txt"),      np.array(self.Lu),      delimiter=",")
        
        np.savetxt(p.joinpath("P.txt"),       np.array(self.P),       delimiter=",")
        np.savetxt(p.joinpath("Q.txt"),       np.array(self.Q),       delimiter=",")
        np.savetxt(p.joinpath("R.txt"),       np.array(self.R),       delimiter=",")
        np.savetxt(p.joinpath("alpha.txt"),   np.array([self.alpha]), delimiter=",")
        np.savetxt(p.joinpath("K.txt") ,      np.array(self.K),       delimiter=",")
        np.savetxt(p.joinpath("Kdelta.txt"),  np.array(self.Kdelta),  delimiter=",")

        ffile = open(p.joinpath("f.py"),'w')
        ffile.write('from math import *\n')
        ffile.write(inspect.getsource(self.f))
        ffile.close()

    @staticmethod
    def genfromtxt(inpath):
        # p = Path("datasets").joinpath(file, "parameters")
        p = inpath
        nx = int(np.genfromtxt( p.joinpath( 'nx.txt'), delimiter=',', dtype="int"))
        nu = int(np.genfromtxt( p.joinpath( 'nu.txt'), delimiter=',', dtype="int"))
        N  = int(np.genfromtxt( p.joinpath( 'N.txt'),  delimiter=',', dtype="int"))
        Tf = float(np.genfromtxt( p.joinpath( 'Tf.txt'), delimiter=','))
        alpha_f = float(np.genfromtxt( p.joinpath( 'alpha.txt'), delimiter=','))

        Lx = np.genfromtxt( p.joinpath('Lx.txt'),   delimiter=',', ndmin=2)
        Lu = np.genfromtxt( p.joinpath('Lu.txt'),   delimiter=',', ndmin=2)

        Q = np.genfromtxt( p.joinpath( 'Q.txt' ), delimiter=',', ndmin=2)
        P = np.genfromtxt( p.joinpath( 'P.txt' ), delimiter=',', ndmin=2)
        R = np.genfromtxt( p.joinpath( 'R.txt' ), delimiter=',', ndmin=2)
        K = np.genfromtxt( p.joinpath( 'K.txt' ), delimiter=',', ndmin=2)
        Kdelta = np.genfromtxt( p.joinpath( 'Kdelta.txt' ), delimiter=',', ndmin=2)
        
        spec = importlib.util.spec_from_file_location("f", p.joinpath("f.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        f = mod.f

        mpc = MPCQuadraticCostLxLu(f, nx, nu, N, Tf, Q, R, P, alpha_f, K, Lx, Lu, Kdelta)
        with open(p.joinpath('name.txt'), 'r') as file:
            mpc.name = file.read().rstrip()
        return mpc

def import_mpc(file="latest", mpcclass=MPCQuadraticCostLxLu):
    p = Path("datasets").joinpath(file, "parameters")
    mpc = mpcclass.genfromtxt(p)
    return mpc