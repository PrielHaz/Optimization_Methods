from mcholmz import modifiedChol
import numpy as np

def inexact_line_search(f, x, d):
    alpha = 1
    beta = 0.5
    slope_decrease_factor = 0.25
    while (f(x + alpha * d) > f(x) + alpha * slope_decrease_factor * (f.grad(x).T @ d)):
        alpha *= beta
    return alpha

def exact_line_search(f, x, d):
    return f.optimal_alpha(x, d)


def update_method_info(info, x, f, alpha, d):
    info["x"].append(x)
    info["f"].append(f(x))
    info["grad"].append(f.grad(x))
    info["error"].append(f(x) - f.min_value)
    if (alpha is not None):
        info["alpha"].append(alpha)
    if (d is not None):
        info["d"].append(d)
    info["grad_norm"].append(np.linalg.norm(f.grad(x)))

# need to collect all the data about the descent


# minimize f using gradient descent
def grad_descent(f, x0, inexact=False, gradStoppingNorm=10**(-5)):
    info = {"x": [], "f": [], "grad": [], "alpha": [], "d": [], "grad_norm": [], "error": []}

    x = x0
    while (np.linalg.norm(f.grad(x)) >= gradStoppingNorm):
        d = -f.grad(x)
        if inexact:
            alpha = inexact_line_search(f, x, d)
        else:
            alpha = exact_line_search(f, x, d)

        update_method_info(info, x, f, alpha, d)
        x = x - alpha * f.grad(x)
    update_method_info(info, x, f, alpha=None, d=None)
    return info

def solve_newton_equation(f, x):
    #G+diag(e) = L*diag(D)*L'
    L, d, _ = modifiedChol(f.hessian(x))
    D = np.diagflat(d)
    y = np.linalg.solve(L, -f.grad(x))
    z = np.linalg.solve(D, y)
    d = np.linalg.solve(L.T, z)
    return d

# minimize f using newton method
def newton_method(f, x0, inexact=False, gradStoppingNorm=10**(-5)):
    info = {"x": [], "f": [], "grad": [], "alpha": [], "d": [], "grad_norm": [], "error": []}

    x = x0
    while (np.linalg.norm(f.grad(x)) >= gradStoppingNorm):
        d = solve_newton_equation(f, x)
        if inexact:
            alpha = inexact_line_search(f, x, d)
        else:
            alpha = exact_line_search(f, x, d)

        update_method_info(info, x, f, alpha, d)
        x = x + alpha * d

    update_method_info(info, x, f, alpha=None, d=None)

    return info
