import matplotlib.pyplot as plt
import numpy as np
from function import QuadraticFunction
from optimizationMethods import grad_descent, newton_method
import matplotlib.cm as cm
from plots import *

# cls && C:/Users/priel/anaconda3/python.exe c:/Users/priel/Downloads/semesterD/Introduction_to_Optimization/hw/ex2/main.py



def main():
    # 2.5 : question 2
    x0 = np.array([1.5, 2])
    Q = np.diagflat([3, 3])
    f = QuadraticFunction(Q)
    inexact = False
    optimize_and_plot(f, x0, inexact=inexact)

    # 2.5 : question 3
    Q = np.diagflat([10, 1])
    f = QuadraticFunction(Q)
    inexact = False
    optimize_and_plot(f, x0, inexact=inexact)

    # 2.5 : question 4
    inexact = True
    optimize_and_plot(f, x0, inexact=inexact)

    # 2.6 : question 5
    x0 = np.array([0, 0])
    f = RosenbrockFunction()
    inexact = True
    optimize_and_plot(f, x0, inexact=inexact, rosenbrock=True)




if __name__ == '__main__':
    main()
