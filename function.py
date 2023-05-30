from abc import ABC, abstractmethod
# ABC stands for Abstract Base Class
import numpy as np

# abstract class
class WellDefinedFunction(ABC):

        # define the () operator
    @abstractmethod
    def __call__(self, x) -> float:
        pass

    # every WellDefinedFunction need to implement the gradient and hessian

    @abstractmethod
    def grad(self, x):
        pass

    @abstractmethod
    def hessian(self, x) -> np.array:
        pass

    @property
    @abstractmethod
    def min_value(self):
        pass


# class Rosenbrock(WellDefinedFunction):

# 0.5x^TQx
class QuadraticFunction(WellDefinedFunction):
    def __init__(self, Q: np.array):
        # assert it is a square matrix
        assert(Q.shape[0] == Q.shape[1])
        self.Q = Q



    def __call__(self, x)->float:
        return 0.5*x.T @ self.Q @ x

    def call_with_vals(self, x, y):
        x = np.array([x, y])
        self.__call__(x)

    def grad(self, x)->np.array:
        return 0.5* (self.Q +self.Q.T)@x



    def hessian(self , x) ->np.array:
          return 0.5*(self.Q+self.Q.T)

    def optimal_alpha(self, x, d) -> float:
        nominator = d.T @ self.grad(x)
        denominator = d.T @ self.Q @ d
        return -nominator/denominator

    @property
    def min_value(self):
        return 0

class RosenbrockFunction(WellDefinedFunction):
    def __init__(self):
        # call parent constructor
        super().__init__()

    def __call__(self, x)->float:
        return (1-x[0])**2+100*(x[1]-x[0]**2)**2

    def grad(self, x)->np.array:
        x1, x2 = (x[0], x[1])
        return np.array([-2*(1-x1)-400*x1*(x2-x1**2), 200*(x2-x1**2)])

    def hessian(self , x) ->np.array:
        x, y = (x[0], x[1])
        return np.array([[1200*x**2-400*y+2, -400*x], [-400*x, 200]])

    @property
    def min_value(self):
        return 0
