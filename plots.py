from optimizationMethods import *
import matplotlib.pyplot as plt
import numpy as np
from function import QuadraticFunction, RosenbrockFunction
import matplotlib.cm as cm

def plot_method_figure(info, f, method_name, inexact=False, values_on_each_axis=100, contour_lines=20):
    x = np.linspace(-2, 2, values_on_each_axis)
    y = np.linspace(-2, 2, values_on_each_axis)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros((values_on_each_axis, values_on_each_axis))
    for i in range(values_on_each_axis):
        for j in range(values_on_each_axis):
            Z[i][j] = f(np.array([X[i][j], Y[i][j]]))

    fig = plt.figure(figsize=(7, 3))
    axis_contour = fig.add_subplot(121)
    axis_3d_function = fig.add_subplot(122, projection='3d')
    CS = axis_contour.contour(X, Y, Z, levels=contour_lines)

    points = np.array(info["x"])
    # seperate x and y values:
    p_x = [point[0] for point in points]
    p_y = [point[1] for point in points]
    axis_contour.plot(p_x, p_y, markersize=3, color="blue", marker="o")
    axis_contour.clabel(CS, inline=1, fontsize=10)
    if (inexact):
        inline_method_name = "inexact line search"
    else:
        inline_method_name = "exact line search"
    axis_contour.set_title(method_name + ", with " + inline_method_name + ". contour lines and method path:")

    axis_3d_function.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
    axis_3d_function.set_title(method_name + ", according 3d function:")

    plt.tight_layout()
    plt.show()


# plotting 1D function
def plot_1D(y, title, x_label, y_label):
    fig = plt.figure(figsize=(7, 3))
    ax = fig.add_subplot(121)
    ax.set_title(title)
    iteration_number = np.arange(len(y))
    ax.plot(iteration_number, y, color="blue", marker="o")
    # y as a log scale:
    plt.yscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.tight_layout()
    plt.show()

def grad_descent_and_plot(f, x0, inexact=False, gradStoppingNorm=10**(-5), rosenbrock=False):
    info = grad_descent(f, x0, inexact, gradStoppingNorm)
    contour_lines = 200 if (rosenbrock) else 20
    plot_method_figure(info, f, "grad descent", inexact=inexact, contour_lines=contour_lines)
    if (rosenbrock):
        plot_1D(info["error"], "grad descent Rosenbrock convergence:", "iteration number", "error:f(x_k)-f(x*))")

def newton_method_and_plot(f, x0, inexact=False, gradStoppingNorm=10**(-5), rosenbrock=False):
    info = newton_method(f, x0, inexact, gradStoppingNorm)
    contour_lines = 200 if (rosenbrock) else 20
    plot_method_figure(info, f, "newton method", inexact=inexact, contour_lines=contour_lines)
    if (rosenbrock):
        plot_1D(info["error"], "newton method Rosenbrock convergence:", "iteration number", "error:f(x_k)-f(x*))")

def optimize_and_plot(f, x0, inexact=False, gradStoppingNorm=10**(-5), rosenbrock=False):
    grad_descent_and_plot(f, x0, inexact, gradStoppingNorm, rosenbrock)
    newton_method_and_plot(f, x0, inexact, gradStoppingNorm, rosenbrock)
