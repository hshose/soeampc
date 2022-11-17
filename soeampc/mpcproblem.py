from abc import ABC, abstractmethod
import numpy as np

__all__ = ['MPC', 'MPCQuadraticCostBoxConstr']

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

    @abstractmethod
    def feasible(self,X,U):
        pass

    @abstractmethod
    def cost(self,X,U):
        pass

    def shiftappendterminal(self,X,U):
        U = np.concatenate((U[1:], self.__terminalcontroller(X[-1])))
        X = self.forwardsim(X[1], U)
        return X,U

    def forwardsim(self,x,U):
        Ts = self.__Tf/self.__N
        t = np.linspace(0,self.__Tf,self.__N+1)
        if self.f_type() == 'CT':
            def f_pwconst_input(y,t,U,Ts):
                    x = y
                    idx = min( int(t/Ts), len(U)-1)
                    u = U[idx]
                    return self.__f(x, u)
            
            X = odeint(f_pwconst_input, x, t, args=(U, Ts))
        else:
            raise Exception('DT not implemented yet')


class MPCQuadraticCostBoxConstr(MPC):

    def __init__(self, f, nx, nu, N, Tf, Q, R, P, alpha, K, xmin, xmax, umin, umax):
        super().__init__()
        super(MPCQuadraticCostBoxConstr,MPCQuadraticCostBoxConstr).N.__set__( self,  N  )
        super(MPCQuadraticCostBoxConstr,MPCQuadraticCostBoxConstr).nx.__set__(self,  nx )
        super(MPCQuadraticCostBoxConstr,MPCQuadraticCostBoxConstr).nu.__set__(self,  nu )
        super(MPCQuadraticCostBoxConstr,MPCQuadraticCostBoxConstr).Tf.__set__(self,  Tf )
        super(MPCQuadraticCostBoxConstr,MPCQuadraticCostBoxConstr).f.__set__( self,  f  )

        self.__P = P
        self.__alpha = alpha
        self.__xmin = xmin
        self.__xmax = xmax
        self.__umin = umin
        self.__umax = umax
        self.__Q = Q
        self.__R = R
        self.__K = K
        self.__terminalcontroller = lambda x: K@x

    
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

    def checkboxconstraint(series, lower, upper):
        for s in series:
            if not (all(s<=upper) and all(s>=lower)):
                return False
        return True

    def instateconstraints(X):
        return checkboxconstraint(X, xmin, xmax)

    def ininputconstraints(U):
        return checkboxconstraint(U, umin, umax)

    def interminalconstraints(x):
        r = x.T @ self.__P @ x
        return r <= self.__alpha
        
    def feasible(self,X,U):
        return self.instateconstraints(X) and self.ininputconstraints(U) and self.interminalconstraint(X[-1,:])
    
    def cost(self, X, U):
        cost = 0
        for x in X[:-1]:
            cost = cost+x@self.__Q@x.T
        for u in U:
            cost = cost+u*R*u.T
        cost = cost+X[-1]@self.__P@X[-1].T
        return cost