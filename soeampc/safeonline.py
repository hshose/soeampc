class SafeOnlineEvaluation():

    def __init__(mpc):
        self.__U_candidate = None
        self.__X_candidate = None
        self.__mpc = mpc

    @property
    def mpc(self):
        return self.__mpc

    @mpc.setter
    def mpc(self, mpc):
        if isinstance(mpc, MPC):
            self.__mpc = mpc
        else:
            raise Exception('Invalid mpc value, excpected type MPC')
    
    def evaluate(self,x,U):
        X = self.mpc.forward_simulate_trajectory(x,U)
        if self.mpc.feasible(X,U) and self.mpc.cost(X,U) <= self.mpc.cost(X_candidate, U_candidate):
                self.__U_candidate = U
                self.__X_candidate = X
        u = self.__U_candidate[0]
        self.__X_candidate, self.__U_candidate = self.mpc.forward_simulate_trajectory(self.__X_candidate, self.__U_candidate)
        return u