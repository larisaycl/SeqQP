"""
This file contains test functions for constrained optimization 
defined as classes for use in the rest of the SQP pipeline
"""

import numpy as np
import sympy as sym
import math
from scipy.optimize import NonlinearConstraint, LinearConstraint, minimize


class Parabola2D():
    def __init__(self, rhs=2.0):
        dv1, dv2 = sym.symbols('x1, x2') 
        self.dvars = np.array([[dv1], [dv2]])
        self.f = self.objective_function(dv1, dv2)
        self.constraints = self.constraint_definition(dv1, dv2, rhs)
        self.scipy_con = self.scipy_con_definition(rhs)

        d1, d2 = sym.symbols('d1, d2')
        self.dirvars = np.array([[d1], [d2]])

        self.bnds = ((-2.0, 2.0), (-2.0, 2.0))
        self.opts = {'maxiter': 100, 'disp':True}
        
    def objective_function(self, x1, x2):
        f = (x1-1)**2 + (x2-1)**2 + 1
        return f

    def constraint_definition(self, x1, x2, rhs):
        ## x1**2 + x2**2 <= 2
        g1 = x1 + x2
        cons = {'0': {'lhs': g1, 'rhs': rhs}}
        return cons

    def scipy_f_definition(self, x):
        f = (x[0]-1)**2 + (x[1]-1)**2 + 1
        return f

    def scipy_con_definition(self, rhs):
        g1 = lambda x: x[0] + x[1]
        lc = LinearConstraint(np.array([[1.0, 1.0]]), -np.inf, rhs)
        return lc

    def optimize(self, x0):
        first_f = self.scipy_f_definition(x0)
        self.opt_history = np.array([[first_f, x0[0], x0[1]]])
        res = minimize(self.scipy_f_definition, x0, constraints=self.scipy_con, method='SLSQP',\
                    bounds=self.bnds, options=self.opts, callback=self.callback_fun)
        print("Final point: {}".format(res.x))
        return res

    def callback_fun(self, xk):
        # pass
        f_k = self.scipy_f_definition(xk)
        param_k = np.insert(xk, 0, f_k)
        self.opt_history = np.concatenate((self.opt_history, np.array([param_k])), axis=0)
        # print("current xk: {};\tcurrent f: {}".format(xk, f_k))


class RosenbrockDisk():
    def __init__(self):
        dv1, dv2 = sym.symbols('x1, x2') 
        self.dvars = np.array([[dv1], [dv2]])
        self.f = self.objective_function(dv1, dv2)
        self.constraints = self.constraint_definition(dv1, dv2)
        self.scipy_con = self.scipy_con_definition()

        d1, d2 = sym.symbols('d1, d2')
        self.dirvars = np.array([[d1], [d2]])

        self.bnds = ((-1.5, 1.5), (-1.5, 1.5))
        self.opts = {'maxiter': 100, 'disp':True}
        
    def objective_function(self, x1, x2):
        f = (1-x1)**2 + 100*(x2-x1**2)**2
        return f

    def constraint_definition(self, x1, x2):
        ## x1**2 + x2**2 <= 2
        g1 = x1**2 + x2**2
        cons = {'0': {'lhs': g1, 'rhs': 2.0}}
        return cons

    def scipy_f_definition(self, x):
        f = (1-x[0])**2 + 100*(x[1]-x[0]**2)**2
        return f

    def scipy_con_definition(self):
        g1 = lambda x: x[0]**2 + x[1]**2
        nlc = NonlinearConstraint(g1, -np.inf, 2.0)
        return nlc

    def optimize(self, x0):
        first_f = self.scipy_f_definition(x0)
        self.opt_history = np.array([[first_f, x0[0], x0[1]]])
        res = minimize(self.scipy_f_definition, x0, constraints=self.scipy_con, method='SLSQP', \
                    bounds=self.bnds, options=self.opts, callback=self.callback_fun)
        print("Final point: {}".format(res.x))
        return res

    def callback_fun(self, xk):
        # pass
        f_k = self.scipy_f_definition(xk)
        param_k = np.insert(xk, 0, f_k)
        self.opt_history = np.concatenate((self.opt_history, np.array([param_k])), axis=0)
        # print("current xk: {};\tcurrent f: {}".format(xk, f_k))


class RosenbrockCubicLine():    # doesn't work very well
    def __init__(self):
        dv1, dv2 = sym.symbols('x1, x2') 
        self.dvars = np.array([[dv1], [dv2]])
        self.f = self.objective_function(dv1, dv2)
        self.constraints = self.constraint_definition(dv1, dv2)
        self.scipy_con = self.scipy_con_definition()

        d1, d2 = sym.symbols('d1, d2')
        self.dirvars = np.array([[d1], [d2]])

        self.bnds = ((-1.5, 1.5), (-1.5, 1.5))
        self.opts = {'maxiter': 100, 'disp':True}
    
    def objective_function(self, x1, x2):
        f = (1-x1)**2 + 100*(x2-x1**2)**2
        return f

    def constraint_definition(self, x1, x2):
        g1 = (x1-1)**3 - x2 + 1.0
        g2 = x1 + x2
        cons = {'0': {'lhs': g1, 'rhs': 0}, '1': {'lhs': g2, 'rhs': 2.0}}
        return cons
        
    def scipy_f_definition(self, x):
        f = (1-x[0])**2 + 100*(x[1]-x[0]**2)**2
        return f

    def scipy_con_definition(self):
        g1 = lambda x: (x[0]-1)**3 - x[1] + 1.0
        lc = LinearConstraint(np.array([[1.0, 1.0]]), -np.inf, 2.0)
        nlc = NonlinearConstraint(g1, -np.inf, 0.0)
        return [lc, nlc]

    def optimize(self, x0):
        first_f = self.scipy_f_definition(x0)
        self.opt_history = np.array([[first_f, x0[0], x0[1]]])
        res = minimize(self.scipy_f_definition, x0, constraints=self.scipy_con, method='SLSQP', \
                    bounds=self.bnds, options=self.opts, callback=self.callback_fun)
        print("Final point: {}".format(res.x))
        return res

    def callback_fun(self, xk):
        # pass
        f_k = self.scipy_f_definition(xk)
        param_k = np.insert(xk, 0, f_k)
        self.opt_history = np.concatenate((self.opt_history, np.array([param_k])), axis=0)
        # print("current xk: {};\tcurrent f: {}".format(xk, f_k))
        

class Arora97():
    def __init__(self):
        dv1, dv2 = sym.symbols('x1, x2') 
        self.dvars = np.array([[dv1], [dv2]])
        self.f = self.objective_function(dv1, dv2)
        self.constraints = self.constraint_definition(dv1, dv2)

        d1, d2 = sym.symbols('d1, d2')
        self.dirvars = np.array([[d1], [d2]])
    
    def objective_function(self, x1, x2):
        f = (x1-3)**2 + (x2-3)**2
        return f

    def constraint_definition(self, x1, x2):
        g1 = x1 + x2
        cons = {'0': {'lhs': g1, 'rhs': 4.0}}
        return cons


class DixonPriceDisk():
    def __init__(self):
        dv1, dv2 = sym.symbols('x1, x2') 
        self.dvars = np.array([[dv1], [dv2]])
        self.f = self.objective_function(dv1, dv2)
        self.constraints = self.constraint_definition(dv1, dv2)
        self.scipy_con = self.scipy_con_definition()

        d1, d2 = sym.symbols('d1, d2')
        self.dirvars = np.array([[d1], [d2]])

        self.bnds = ((-10.0, 10.0), (-10.0, 10.0))
        self.opts = {'maxiter': 100, 'disp':True}
        
    def objective_function(self, x1, x2):
        f = (1-x1)**2 + 2*(2*x2**2 - x1)**2
        return f

    def constraint_definition(self, x1, x2):
        ## x1**2 + x2**2 <= 2
        g1 = x1**2 + x2**2
        cons = {'0': {'lhs': g1, 'rhs': 2.0}}
        return cons

    def scipy_f_definition(self, x):
        f = (1-x[0])**2 + 2*(2*x[1]**2 - x[0])**2
        return f

    def scipy_con_definition(self):
        g1 = lambda x: x[0]**2 + x[1]**2
        nlc = NonlinearConstraint(g1, -np.inf, 2.0)
        return nlc

    def optimize(self, x0):
        first_f = self.scipy_f_definition(x0)
        self.opt_history = np.array([[first_f, x0[0], x0[1]]])
        res = minimize(self.scipy_f_definition, x0, constraints=self.scipy_con, method='SLSQP', \
                    bounds=self.bnds, options=self.opts, callback=self.callback_fun)
        print("Final point: {}".format(res.x))
        return res

    def callback_fun(self, xk):
        # pass
        f_k = self.scipy_f_definition(xk)
        param_k = np.insert(xk, 0, f_k)
        self.opt_history = np.concatenate((self.opt_history, np.array([param_k])), axis=0)
        # print("current xk: {};\tcurrent f: {}".format(xk, f_k))


class SixHumpCamel():
    def __init__(self):
        dv1, dv2 = sym.symbols('x1, x2') 
        self.dvars = np.array([[dv1], [dv2]])
        self.f = self.objective_function(dv1, dv2)
        self.constraints = self.constraint_definition(dv1, dv2)
        self.scipy_con = self.scipy_con_definition()

        d1, d2 = sym.symbols('d1, d2')
        self.dirvars = np.array([[d1], [d2]])

        self.bnds = ((-3.0, 3.0), (-2.0, 2.0))
        self.opts = {'maxiter': 100, 'disp':True}
        
    def objective_function(self, x1, x2):
        f = (4 - 2.1*x1**2 + (x1**4 / 3))*x1**2 + x1*x2 + (-4 + 4*x2**2)*x2**2
        return f

    def constraint_definition(self, x1, x2):
        ## x1**2 + x2**2 <= 2
        g1 = x1**2 + x2**2
        cons = {'0': {'lhs': g1, 'rhs': 2.0}}
        return cons

    def scipy_f_definition(self, x):
        f = (4 - 2.1*x[0]**2 + (x[0]**4 / 3))*x[0]**2 + x[0]*x[1] + (-4 + 4*x[1]**2)*x[1]**2
        return f

    def scipy_con_definition(self):
        g1 = lambda x: x[0]**2 + x[1]**2
        nlc = NonlinearConstraint(g1, -np.inf, 2.0)
        return nlc

    def optimize(self, x0):
        first_f = self.scipy_f_definition(x0)
        self.opt_history = np.array([[first_f, x0[0], x0[1]]])
        res = minimize(self.scipy_f_definition, x0, constraints=self.scipy_con, method='SLSQP', \
                    bounds=self.bnds, options=self.opts, callback=self.callback_fun)
        print("Final point: {}".format(res.x))
        return res

    def callback_fun(self, xk):
        # pass
        f_k = self.scipy_f_definition(xk)
        param_k = np.insert(xk, 0, f_k)
        self.opt_history = np.concatenate((self.opt_history, np.array([param_k])), axis=0)
        # print("current xk: {};\tcurrent f: {}".format(xk, f_k))


class BoothDisk():
    def __init__(self, radius=3.5):
        dv1, dv2 = sym.symbols('x1, x2') 
        self.dvars = np.array([[dv1], [dv2]])
        self.f = self.objective_function(dv1, dv2)
        self.constraints = self.constraint_definition(dv1, dv2, radius)
        self.scipy_con = self.scipy_con_definition(radius)

        d1, d2 = sym.symbols('d1, d2')
        self.dirvars = np.array([[d1], [d2]])

        self.bnds = ((-10.0, 10.0), (-10.0, 10.0))
        self.opts = {'maxiter': 100, 'disp':True}
        
    def objective_function(self, x1, x2):
        f = (x1 + 2*x2 - 7)**2 + (2*x1 + x2 - 5)**2
        return f

    def constraint_definition(self, x1, x2, radius):
        ## x1**2 + x2**2 <= 2
        g1 = x1**2 + x2**2
        cons = {'0': {'lhs': g1, 'rhs': radius**2}}
        return cons

    def scipy_f_definition(self, x):
        f = (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2
        return f

    def scipy_con_definition(self, radius):
        g1 = lambda x: x[0]**2 + x[1]**2
        nlc = NonlinearConstraint(g1, -np.inf, radius**2)
        return nlc

    def optimize(self, x0):
        first_f = self.scipy_f_definition(x0)
        self.opt_history = np.array([[first_f, x0[0], x0[1]]])
        res = minimize(self.scipy_f_definition, x0, constraints=self.scipy_con, method='SLSQP', \
                    bounds=self.bnds, options=self.opts, callback=self.callback_fun)
        print("Final point: {}".format(res.x))
        return res

    def callback_fun(self, xk):
        # pass
        f_k = self.scipy_f_definition(xk)
        param_k = np.insert(xk, 0, f_k)
        self.opt_history = np.concatenate((self.opt_history, np.array([param_k])), axis=0)
        # print("current xk: {};\tcurrent f: {}".format(xk, f_k))
