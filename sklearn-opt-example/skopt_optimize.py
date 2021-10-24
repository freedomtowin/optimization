import numpy as np
# import pandas as pd
# from numba import njit
from itertools import product
from ortools.linear_solver import pywraplp


from skopt.benchmarks import branin as _branin
import matplotlib.pyplot as plt
import numpy as np

def branin(x, noise_level=0.):
    return _branin(x) + noise_level * np.random.randn()

from matplotlib.colors import LogNorm


def plot_branin():
    fig, ax = plt.subplots()

    x1_values = np.linspace(-5, 10, 100)
    x2_values = np.linspace(0, 15, 100)
    x_ax, y_ax = np.meshgrid(x1_values, x2_values)
    vals = np.c_[x_ax.ravel(), y_ax.ravel()]
    fx = np.reshape([branin(val) for val in vals], (100, 100))

    cm = ax.pcolormesh(x_ax, y_ax, fx,
                       norm=LogNorm(vmin=fx.min(),
                                    vmax=fx.max()),
                       cmap='viridis_r')

    minima = np.array([[-np.pi, 12.275], [+np.pi, 2.275], [9.42478, 2.475]])
    ax.plot(minima[:, 0], minima[:, 1], "r.", markersize=14,
            lw=0, label="Minima")

    cb = fig.colorbar(cm)
    cb.set_label("f(x)")

    ax.legend(loc="best", numpoints=1)

    ax.set_xlabel("X1")
    ax.set_xlim([-5, 10])
    ax.set_ylabel("X2")
    ax.set_ylim([0, 15])


plot_branin()

T = {}
R = np.zeros((100,2)).astype(int)

X = (np.random.randint(-10,10,size=(100,2))).astype(int)

value = (np.random.randint(-1,1,size=(100))).astype(int)

def custom_boolean_rule(T):
    rule = (((X[:,0]>T[0])|(X[:,1]>T[1]))&(X[:,1]<T[0]))
    workload = np.abs(np.sum(rule)-50)
    benefit = np.sum(value[rule])
    return workload+benefit*-1


def plot_branin():
    fig, ax = plt.subplots()

    x1_values = np.linspace(-10, 10, 100)
    x2_values = np.linspace(-10, 10, 100)
    x_ax, y_ax = np.meshgrid(x1_values, x2_values)
    vals = np.c_[x_ax.ravel(), y_ax.ravel()]
    fx = np.reshape([custom_boolean_rule(val) for val in vals], (100, 100))

    cm = ax.pcolormesh(x_ax, y_ax, fx,
                       norm=LogNorm(vmin=fx.min(),
                                    vmax=fx.max()),
                       cmap='viridis_r')

#     minima = np.array([[-np.pi, 12.275], [+np.pi, 2.275], [9.42478, 2.475]])
#     ax.plot(minima[:, 0], minima[:, 1], "r.", markersize=14,
#             lw=0, label="Minima")

    cb = fig.colorbar(cm)
    cb.set_label("f(x)")

    ax.legend(loc="best", numpoints=1)

    ax.set_xlabel("X1")
    ax.set_xlim([-10, 10])
    ax.set_ylabel("X2")
    ax.set_ylim([-10, 10])
    plt.show()


plot_branin()

func = partial(custom_boolean_rule)
bounds = [(-10.0, 10.0), (-10, 10)]
n_calls = 60


def run(minimizer, n_iter=5):
    return [minimizer(func, bounds, n_calls=n_calls, random_state=n)
            for n in range(n_iter)]

# Random search
dummy_res = run(dummy_minimize)

# Random search
dummy_res = run(dummy_minimize)

# Gaussian processes
gp_res = run(gp_minimize)

# Random forest
rf_res = run(partial(forest_minimize, base_estimator="RF"))

# Extra trees
et_res = run(partial(forest_minimize, base_estimator="ET"))

from skopt.plots import plot_convergence

plot = plot_convergence(("dummy_minimize", dummy_res),
                        ("gp_minimize", gp_res),
                        ("forest_minimize('rf')", rf_res),
                        ("forest_minimize('et)", et_res),
                        true_minimum=0.397887, yscale="log")

plot.legend(loc="best", prop={'size': 6}, numpoints=1)

dummy_res