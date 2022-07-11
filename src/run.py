import numpy as np
import sympy as sym
from sqp import SQP
from test_functions import *
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.lines import Line2D
from scipy.optimize import minimize


matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

color = ['tab:blue',
        'tab:orange',
        'tab:green',
        'tab:red',
        'tab:purple',
        'tab:brown',
        'tab:pink',
        'tab:gray',
        'tab:olive',
        'tab:cyan']


def plot_optimization_history(res, axs, label):
    x_history = res.x_history
    obj_history = res.obj_history

    num_vars = x_history.shape[1]
    num_iters = len(x_history)
    x_plot = np.linspace(1, num_iters, num_iters)

    alpha = 0.8

    axs[0].plot(x_plot, obj_history, label=label, alpha=alpha)
    axs[1].plot(x_plot, x_history[:, 0], label=label, alpha=alpha)
    axs[2].plot(x_plot, x_history[:, 1], label=label, alpha=alpha)
    

def plot_scipy_optimization_history(func, axs, label):
    obj_history = func.opt_history[:, 0]
    x_history = func.opt_history[:, 1:]

    num_iters = len(x_history)
    x_plot = np.linspace(1, num_iters, num_iters)

    alpha = 0.8

    axs[0].plot(x_plot, obj_history, label=label, alpha=alpha)
    axs[1].plot(x_plot, x_history[:, 0], label=label, alpha=alpha)
    axs[2].plot(x_plot, x_history[:, 1], label=label, alpha=alpha)

    axs[0].set_ylabel('objective')
    axs[1].set_ylabel('x1')
    axs[2].set_ylabel('x2')


def test_optimization(func, x0, fname, problem_title):
    res1 = SQP(func, x0, method='simplex')
    # res2 = SQP(func, x0, method='linprog')
    # res3 = SQP(func, x0, method='QP')
    res4 = func.optimize(x0)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4.5))

    plot_optimization_history(res1, axs, 'our SQP')
    # plot_optimization_history(res2, axs, 'linprog')
    # plot_optimization_history(res3, axs, 'SQP')
    plot_scipy_optimization_history(func, axs, 'scipy SQP')

    blue_line = Line2D([0], [0], c=color[0], label='our SQP')
    # orange_line = Line2D([0], [0], c=color[1], label='linprog')
    # green_line = Line2D([0], [0], c=color[2], label='SQP')
    red_line = Line2D([0], [0], c=color[1], label='scipy SQP')

    fig.suptitle(problem_title)
    fig.supxlabel('iterations')
    # fig.legend(handles=[blue_line, orange_line, green_line, red_line], ncol=4, loc='upper center', \
    #     bbox_to_anchor=(0.5, 0.95))
    fig.legend(handles=[blue_line, red_line], ncol=4, loc='upper center', \
        bbox_to_anchor=(0.5, 0.95))
    fig.tight_layout()
    fig.subplots_adjust(top=0.85)
    fig.savefig('figures/python' + fname)
    return res1, res4, func


def booth_constrained_test():
    res1, res2, func = test_optimization(BoothDisk(radius=2.5), [0.0, 0.0], 'boothdiskcons', 'Booth function contrained by circle, constraint active')
    x_to_plot = np.linspace(-3.0, 3.5)
    y_to_plot = np.linspace(-3.0, 3.5)
    X, Y = np.meshgrid(x_to_plot, y_to_plot)
    Z = func.objective_function(X, Y)

    constraint = plt.Circle((0, 0), 2.5, fill=False, edgecolor='k', label='constraint')
    fig, ax = plt.subplots(figsize=(6, 5))
    
    m = ax.contourf(X, Y, Z)
    ax.add_patch(constraint)
    ax.plot(1.0, 3.0, 'ro', label='global minumum')

    ax.plot(res1.x_history[-1, 0], res1.x_history[-1, 1], 'c*', label='our SQP')
    ax.plot(func.opt_history[-1, 1], func.opt_history[-1, 2], 'cx', label='scipy SQP')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    fig.colorbar(m)
    fig.suptitle('Booth function contrained by circle of radius 2.5')

    ax.set_aspect('equal', 'box')
    fig.legend(ncol=4, loc='upper center', \
        bbox_to_anchor=(0.5, 0.95))
    fig.tight_layout()
    fig.subplots_adjust(top=0.85)
    fig.savefig('figures/boothcons_contour')
    # plt.show()


def parabola_constrained_test():
    res1, res2, func = test_optimization(Parabola2D(1.0), [0.0, 0.0], 'parabolacons', 'Parabolic function contrained by line, constraint active')
    x_to_plot = np.linspace(-3.0, 3.5)
    y_to_plot = np.linspace(-3.0, 3.5)
    y_cons = -x_to_plot + 1.0
    X, Y = np.meshgrid(x_to_plot, y_to_plot)
    Z = func.objective_function(X, Y)

    fig, ax = plt.subplots(figsize=(6, 5))
    
    m = ax.contourf(X, Y, Z)
    plt.plot(x_to_plot, y_cons, 'k', label='constraint')
    ax.plot(1.0, 1.0, 'ro', label='global minumum')

    ax.plot(res1.x_history[-1, 0], res1.x_history[-1, 1], 'c*', label='our SQP')
    ax.plot(func.opt_history[-1, 1], func.opt_history[-1, 2], 'cx', label='scipy SQP')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_ylim([-3.0, 3.5])

    fig.colorbar(m)
    fig.suptitle('Parabolic function contrained by line, constraint active')

    ax.set_aspect('equal', 'box')
    fig.legend(ncol=4, loc='upper center', \
        bbox_to_anchor=(0.5, 0.95))
    fig.tight_layout()
    fig.subplots_adjust(top=0.85)
    fig.savefig('figures/parabolacons_contour')
    # plt.show()


def main():

    func_list = [Parabola2D(), RosenbrockDisk(), RosenbrockCubicLine(), \
                 DixonPriceDisk(), SixHumpCamel(), BoothDisk(radius=3.5)]
    x0_list = [[-0.1, -0.1], [0.0, 0.0], [3.0, 2.0], \
               [0.7, 0.7], [3.0 ,2.0], [0.0, 0.0]]
    fname_list = ['parabola', 'rosenbrockdisk', 'rosenbrockcubicline', \
                  'dixonpricedisk', 'sixhumpcamel', 'boothdisk']
    title_list = ['Bowl (2D parabola) constrained by line', 
                  'Rosenbrock function constrained by circle', 
                  'Rosenbrock function constrained by cubic and line', 
                  'Dixon Price function constrained by circle', 
                  'Six Hump Camel Function constrained by circle', 
                  'Booth function contrained by circle, constraint active']

    for i in range(len(func_list)):
        print('\n>>> ' + title_list[i])
        test_optimization(func_list[i], x0_list[i], fname_list[i], title_list[i])

    booth_constrained_test()
    parabola_constrained_test()

    return


if __name__ == '__main__':
    main()