from abc import ABC, abstractmethod
import numpy as np
from scipy.integrate import odeint

import importlib
import inspect

__all__ = ['MPC', 'MPCQuadraticCostBoxConstr', 'MPCQuadraticCostLxLu']

def checkboxconstraint(series, lower, upper):
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
        self.__terminalcontroller = None
    
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
    def terminalcontroller(self):
        """`terminalcontroller(x)` - evaluates the therminal controller for given state x
        Default: :code:'None'.
        """
        return self.__terminalcontroller

    @property
    def stabilizingfeedbackcontroller(self):
        """`stabilizingfeedbackcontroller(x)` - evaluates the therminal controller for given state x
        Default: :code:'None'.
        """
        return self.__stabilizingfeedbackcontroller

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


    @terminalcontroller.setter
    def terminalcontroller(self, terminalcontroller):
        if callable(terminalcontroller):
            self.__terminalcontroller = terminalcontroller
        else:
            raise Exception('Invalid terminalcontroller, must be callable')

    @stabilizingfeedbackcontroller.setter
    def stabilizingfeedbackcontroller(self, stabilizingfeedbackcontroller):
        if callable(stabilizingfeedbackcontroller):
            self.__stabilizingfeedbackcontroller = stabilizingfeedbackcontroller
        else:
            raise Exception('Invalid stabilizingfeedbackcontroller, must be callable')


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

    def shiftappendterminal(self,X,U):
        U = np.concatenate((U[1:], self.__terminalcontroller(X[-1])))
        X = self.forwardsim(X[1], U)
        return X,U

    def forwardsim(self,x,U):
        Ts = self.__Tf/self.__N
        t = np.linspace(0,self.__Tf,self.__N+1)
        if self.f_type == 'CT':
            def f_pwconst_input(y,t,U,Ts):
                    x = y
                    idx = min( int(t/Ts), np.shape(U)[0]-1)
                    u = self.__stabilizingfeedbackcontroller(x, U[idx,:])
                    return tuple(self.__f(x, u))

            X = odeint(f_pwconst_input, x, t, args=(U, Ts))
            return X
        else:
            raise Exception('DT not implemented yet')

    def singlestepsim(self,x,u):
        Ts = self.__Tf/self.__N
        t = np.linspace(0,Ts,2)
        if self.f_type == 'CT':
            def f_pwconst_input(y,t):
                    x = y
                    ustable = self.__stabilizingfeedbackcontroller(x, u)
                    return tuple(self.__f(x, ustable))

            X = odeint(f_pwconst_input, x, t)
            return X[-1]
        else:
            raise Exception('DT not implemented yet')

class MPCQuadraticCostBoxConstr(MPC):

    def __init__(self, f, nx, nu, N, Tf, Q, R, P, alpha, K, xmin, xmax, umin, umax, Vx, Vu):
        super().__init__()
        super(MPCQuadraticCostBoxConstr,MPCQuadraticCostBoxConstr).N.__set__( self,  N  )
        super(MPCQuadraticCostBoxConstr,MPCQuadraticCostBoxConstr).nx.__set__(self,  nx )
        super(MPCQuadraticCostBoxConstr,MPCQuadraticCostBoxConstr).nu.__set__(self,  nu )
        super(MPCQuadraticCostBoxConstr,MPCQuadraticCostBoxConstr).Tf.__set__(self,  Tf )
        super(MPCQuadraticCostBoxConstr,MPCQuadraticCostBoxConstr).f.__set__( self,  f  )
        super(MPCQuadraticCostBoxConstr,MPCQuadraticCostBoxConstr).f_type.__set__( self,  'CT'  )

        self.__P = P
        self.__alpha = alpha
        self.__xmin = xmin
        self.__xmax = xmax
        self.__umin = umin
        self.__umax = umax
        self.__Vx = Vx
        self.__Vu = Vu
        self.__Q = Q
        self.__R = R
        self.__K = K
        self.__terminalcontroller = lambda x: K@x

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
    def xmin(self):
        return self.__xmin

    @property
    def xmax(self):
        return self.__xmax

    @property
    def umin(self):
        return self.__umin

    @property
    def umax(self):
        return self.__umax

    @property
    def Vx(self):
        return self.__Vx
    
    @property
    def Vu(self):
        return self.__Vu

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

    def instateconstraints(self, X):
        return checkboxconstraint(self.__Vx*X, self.__xmin, self.__xmax)

    def ininputconstraints(self, U):
        return checkboxconstraint(self.__Vu*U, self.__umin, self.__umax)

    def interminalconstraints(self, x):
        r = x.T @ self.__P @ x
        return r <= self.__alpha
        
    def feasible(self,X,U, verbose=False):
        res = True
        res = res and self.instateconstraints(X)
        # res = res and self.ininputconstraints(U)
        res = res and self.interminalconstraints(X[-1,:])
        if verbose and not res:
            print("Infeasible Trajectory")
            print("\tin state constraint:   ", self.instateconstraints(X))
            print("\tin input constraint:   ", self.ininputconstraints(U))
            print("\tin termial constraint: ", self.interminalconstraints(X[-1,:]))
        return res
    
    def cost(self, X, U):
        cost = 0
        for x in X[:-1]:
            cost = cost+x@self.__Q@x.T
        for u in U:
            cost = cost+u*R*u.T
        cost = cost+X[-1]@self.__P@X[-1].T
        return cost

    def savetxt(self, outpath):
        # p = outpath.joinpath("parameters")
        p = outpath
        p.mkdir(parents=True,exist_ok=True)
        with open(p.joinpath('name.txt'), 'w') as file:
            file.write(mpc.name)

        np.savetxt(p.joinpath("nx.txt"),    np.array([self.nx]), fmt='%i', delimiter=",")
        np.savetxt(p.joinpath("nu.txt"),    np.array([self.nu]), fmt='%i', delimiter=",")
        np.savetxt(p.joinpath("N.txt"),     np.array([self.N]),  fmt='%i', delimiter=",")
        np.savetxt(p.joinpath("Tf.txt"),    np.array([self.Tf]),           delimiter=",")

        np.savetxt(p.joinpath("xmax.txt"),  np.array([self.xmax]), delimiter=",")
        np.savetxt(p.joinpath("xmin.txt"),  np.array([self.xmin]), delimiter=",")
        np.savetxt(p.joinpath("umax.txt"),  np.array([self.umax]), delimiter=",")
        np.savetxt(p.joinpath("umin.txt"),  np.array([self.umin]), delimiter=",")
        np.savetxt(p.joinpath("Vx.txt"),    np.array([self.Vx]), delimiter=",")
        np.savetxt(p.joinpath("Vu.txt"),    np.array([self.Vu]), delimiter=",")
        
        np.savetxt(p.joinpath("P.txt"),     np.array(self.P),              delimiter=",")
        np.savetxt(p.joinpath("Q.txt"),     np.array(self.Q),              delimiter=",")
        np.savetxt(p.joinpath("R.txt"),     np.array(self.R),              delimiter=",")
        np.savetxt(p.joinpath("alpha.txt"), np.array([self.alpha]),        delimiter=",")
        np.savetxt(p.joinpath("K.txt") ,    np.array(self.K),              delimiter=",")

        ffile = open(p.joinpath("f.py"),'w')
        ffile.write('from math import *\n')
        ffile.write(inspect.getsource(mpc.f))
        ffile.close()

    @staticmethod
    def genfromtxt(inpath):
        p = inpath
        # p = Path("datasets").joinpath(file, "parameters")
        nx = int(np.genfromtxt( p.joinpath( 'nx.txt'), delimiter=',', dtype="int"))
        nu = int(np.genfromtxt( p.joinpath( 'nu.txt'), delimiter=',', dtype="int"))
        N  = int(np.genfromtxt( p.joinpath( 'N.txt'),  delimiter=',', dtype="int"))
        Tf = float(np.genfromtxt( p.joinpath( 'Tf.txt'), delimiter=','))
        alpha_f = float(np.genfromtxt( p.joinpath( 'alpha.txt'), delimiter=','))

        xmax = np.genfromtxt( p.joinpath('xmax.txt'),   delimiter=',')
        xmin = np.genfromtxt( p.joinpath('xmin.txt'),   delimiter=',')
        umax = np.genfromtxt( p.joinpath('umax.txt'),   delimiter=',')
        umin = np.genfromtxt( p.joinpath('umin.txt'),   delimiter=',')
        Vx = np.genfromtxt( p.joinpath('Vx.txt'),   delimiter=',')
        Vu = np.genfromtxt( p.joinpath('Vu.txt'),   delimiter=',')

        Q = np.reshape( np.genfromtxt( p.joinpath( 'Q.txt' ), delimiter=','), (nx,nx))
        P = np.reshape( np.genfromtxt( p.joinpath( 'P.txt' ), delimiter=','), (nx,nx))
        R = np.reshape( np.genfromtxt( p.joinpath( 'R.txt' ), delimiter=','), (nu,nu))
        K = np.reshape( np.genfromtxt( p.joinpath( 'K.txt' ), delimiter=','), (nx, nu))
        
        spec = importlib.util.spec_from_file_location("f", p.joinpath("f.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        f = mod.f

        mpc = MPCQuadraticCostBoxConstr(f, nx, nu, N, Tf, Q, R, P, alpha_f, K, xmin, xmax, umin, umax, Vx, Vu)
        with open(p.joinpath('name.txt'), 'r') as file:
            mpc.name = file.read().rstrip()
        return mpc


class MPCQuadraticCostLxLu(MPC):

    def __init__(self, f, nx, nu, N, Tf, Q, R, P, alpha, K, Lx, Lu, Kdelta=None):
        super().__init__()
        super(MPCQuadraticCostBoxConstr,MPCQuadraticCostBoxConstr).N.__set__( self,  N  )
        super(MPCQuadraticCostBoxConstr,MPCQuadraticCostBoxConstr).nx.__set__(self,  nx )
        super(MPCQuadraticCostBoxConstr,MPCQuadraticCostBoxConstr).nu.__set__(self,  nu )
        super(MPCQuadraticCostBoxConstr,MPCQuadraticCostBoxConstr).Tf.__set__(self,  Tf )
        super(MPCQuadraticCostBoxConstr,MPCQuadraticCostBoxConstr).f.__set__( self,  f  )
        super(MPCQuadraticCostBoxConstr,MPCQuadraticCostBoxConstr).f_type.__set__( self,  'CT'  )


        self.__P = P
        self.__alpha = alpha
        self.__Q = Q
        self.__R = R
        self.__K = K
        
        # if no Kdelta is set, this reverts to no stabilizing feedback.
        if isinstance(Kdelta, type(None)):
            self.__Kdelta = np.zeros((self.nu, self.nx))
        elif Kdelta.shape[0] == self.nu and Kdelta.shape[1] == self.nx:
            self.__Kdelta = Kdelta
        else: 
            raise Exception("Dimension mismatch between Kdelta, nx, and nu!")
    
        # self.__stabilizingfeedbackcontroller = lambda x,v: self.__Kdelta @ x + v 
        # self.__terminalcontroller = lambda x: (self.__K-self.__Kdelta) @ x
        super(MPCQuadraticCostBoxConstr,MPCQuadraticCostBoxConstr).stabilizingfeedbackcontroller.__set__( self,  lambda x,v: self.__Kdelta @ x + v  )
        super(MPCQuadraticCostBoxConstr,MPCQuadraticCostBoxConstr).terminalcontroller.__set__( self,  lambda x: (self.__K-self.__Kdelta) @ x  )

        if Lx.shape[0] == Lu.shape[0] and Lx.shape[1] == self.nx and Lu.shape[1] == self.nu:
            self.__nconstr = Lx.shape[0]
            self.__Lx = Lx
            self.__Lu = Lu
        else:
            raise Exception("Dimensions mismatch between Lx, Lu, nx, and nu")
        
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

    def instateandinputconstraints(self, X, U, verbose = False, eps = 1e-6):
        # return np.all((self.__Lx@(X[:-1]).T + self.__Lu@U.T <= 1), axis=0)
        constrineq = ( self.__Lx@(X[:-1]).T + self.__Lu@U.T - 1 <= eps )
        if verbose and not np.all(constrineq):
            print("\t LxLu constraints violated:   ", np.where(constrineq == False))
        return np.all( constrineq )

    def interminalconstraints(self, x, verbose = False, eps = 1e-6):
        r = x.T @ self.__P @ x
        constrineq = ( r - (self.__alpha**2) <= eps )
        if verbose and not constrineq:
            print("\t Terminal constraint x.T P x = ", r, " <= ", self.__alpha**2, "is violated")
        return constrineq
        
    def feasible(self,X,U, verbose=False, testinputs=True):
        res = True
        res = res and self.instateandinputconstraints(X,U)
        res = res and self.interminalconstraints(X[-1,:])
        if verbose and not res:
            print("Infeasible Trajectory")
            print("\tin state constraint:   ", self.instateandinputconstraints(X, U, verbose=True))
            print("\tin termial constraint: ", self.interminalconstraints(X[-1,:], verbose=True))
        return res
    
    def cost(self, X, U):
        cost = 0
        for x in X[:-1]:
            cost = cost+x@self.__Q@x.T
        for u in U:
            cost = cost+u*R*u.T
        cost = cost+X[-1]@self.__P@X[-1].T
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

        Lx = np.genfromtxt( p.joinpath('Lx.txt'),   delimiter=',')
        Lu = np.genfromtxt( p.joinpath('Lu.txt'),   delimiter=',')

        Q = np.genfromtxt( p.joinpath( 'Q.txt' ), delimiter=',')
        P = np.genfromtxt( p.joinpath( 'P.txt' ), delimiter=',')
        R = np.genfromtxt( p.joinpath( 'R.txt' ), delimiter=',')
        K = np.genfromtxt( p.joinpath( 'K.txt' ), delimiter=',')
        Kdelta = np.genfromtxt( p.joinpath( 'Kdelta.txt' ), delimiter=',')
        
        spec = importlib.util.spec_from_file_location("f", p.joinpath("f.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        f = mod.f

        mpc = MPCQuadraticCostLxLu(f, nx, nu, N, Tf, Q, R, P, alpha_f, K, Lx, Lu, Kdelta)
        with open(p.joinpath('name.txt'), 'r') as file:
            mpc.name = file.read().rstrip()
        return mpc
