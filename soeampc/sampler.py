from abc import ABC, abstractmethod
import numpy as np

from tqdm import tqdm

class Sampler(ABC):
    def __init__(self):
        self.__nmax = None

    @property
    def Nsamples(self):
        return self.__nmax

    @Nsamples.setter
    def Nsamples(self, Nsamples):
        if isinstance(Nsamples, int) and Nsamples > 0:
            self.__nmax = Nsamples
        else:
            raise Exception('Invalid Nsamples value, expected positive int')

    @abstractmethod
    def sample():
        pass


class GridSampler(Sampler):
    def __init__(self, grid):

        # if isinstance(samplesperaxis, np.ndarray) and len(samplesperaxis) == mpc.nx():
        #     N = np.array(samplesperaxis, dtype=int)
        # elif isinstance(samplesperaxis, int) and samplesperaxis > 0:
        #     N = samplesperaxis*np.ones(mpc.nx, dtype=int)
        # else:
        #     raise Exception('samplesperaxis invalid, expected positive integer or numpy array of len', mpc.nx())

        # how many grid points per axis
        # grid = [3, 4, 5]
        self.__grid = grid

        # how many dimensions each output has
        super(GridSampler,GridSampler).Nsamples.__set__( self,  len(self.__grid)  )
        # self.__nmax = len(self.__grid)

        # the "current" index of the grid
        # i starts with [0,0,0],[0,0,1],...,[0,0,4],[0,1,0],...[2,3,5]
        self.__i = np.zeros(self.__nmax)

    def reset(self):
        self.__i = np.zeros(self.__nmax)


    def updatei(self,n):
        """
        Recursively check what dimensions of i to update
        """
        
        if n == self.__nmax-1:
            self.__pbar.update(1)
        
        if self.__i[n] == self.__grid[n]-1:
            if n > 0:
                self.__i[n] = 0
                self.updatei(n-1)        
        else:
            self.__i[n] += 1

    def sample(self):
        if np.all(self.__i == np.zeros(self.__nmax)):
            self.__pbar = tqdm(total = np.prod(self.__grid))
        
        res = self.__i/(self.__grid-1)
        
        if np.all(self.__i == self.__grid-1):
            self.__pbar.close()
            self.__i = np.zeros(self.__nmax)
        else: 
            self.updatei(self.__nmax-1)
        
        return res


class RandomSampler(Sampler):
    def __init__(self, nmax, n, seed):
        super(RandomSampler,RandomSampler).Nsamples.__set__( self,  nmax  )
        # self.__nmax = nmax
        self.__n = n
        self.__rng = np.random.default_rng(seed)
    
    def sample(self):
        return self.__rng.random((self.__n))

