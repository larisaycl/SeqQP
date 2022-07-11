import numpy as np
from scipy.optimize import linprog, minimize, LinearConstraint
import sympy as sym


class SQP():
    def __init__(self, test_function, x0=[0.0, 0.0], method='QP', maxiter=50):
        """
        Limitations of our SQP implmentation:
            + 2 variable optimization
            + up to 2 constraints 
        """
        self.method = method 

        # pull items from original function to minimize
        self.dvars = test_function.dvars
        self.dirvars = test_function.dirvars
        self.f = test_function.f
        self.g = test_function.constraints

        self.dim = len(self.dirvars)
        self.num_g = len(self.g.keys())

        self.hessian = self._hessian_(self.f)
        self.f_approx, self.gradf = self.objective_quadratic_approximation(self.f)
        self.g_approx, self.g_rhs = self.linearize_constraints(self.g)

        # print("f_quad: \n{}; \ngradf: {}".format(self.f_approx, self.gradf))
        # print("g: {}; type: {}".format(self.g_approx, type(self.g_approx)))
        # print("hessian: \n{}".format(self.hessian))

        x_to_sub = {}
        for i in range(self.dim):
            x_to_sub[self.dvars[i, 0]] = x0[i]
        self.x_history = np.array([x0])
        self.obj_history = [self.f.subs(x_to_sub)]

        self.x0 = x0
        iter_count = 0
        while True:
            x1 = self.optimize(self.x0)
            iter_count += 1

            x_to_sub = {}
            for i in range(self.dim):
                x_to_sub[self.dvars[i, 0]] = x1[i]
            self.x_history = np.concatenate((self.x_history, np.array([x1])), axis=0)
            self.obj_history.append(self.f.subs(x_to_sub))

            x_change = np.linalg.norm(x1 - self.x0)
            
            if x_change < 2e-5:
                break
            if iter_count > maxiter:
                break
            self.x0 = x1

        print("Method: {}\nNumber of iterations: {}\nFunction value: {}; Final point: {}".format(\
            self.method, iter_count, self.obj_history[-1], x1))
        # self.x_history = np.delete(self.x_history, 0, 0)

    def _grad_function_(self, f, vars):
        """
        Helper function for grad of a multivariate function
        """
        gradf = np.empty((self.dim, 1), dtype=object)
        for i in range(self.dim):
            gradf[i, 0] = sym.diff(f, vars[i, 0])
        return gradf
    
    def _hessian_(self, f):
        """
        Helper function to calculate the hessian of a multivarate function
        Requires the _grad_function to calculate the derivatives.
        """
        hessian = np.empty((self.dim, self.dim), dtype=object)
        f_grad = self._grad_function_(f, self.dvars)
        for i in range(self.dim):
            for j in range(self.dim):
                hessian[i, j] = sym.diff(f_grad[i, 0], self.dvars[j, 0])

        return hessian

    def objective_quadratic_approximation(self, f):
        """
        This function does quadratic approximation of the objective function

        Args:
            + f - symbolic objective function
        """
        gradf = self._grad_function_(f, self.dvars)
        f_quad = gradf.T.dot(self.dirvars) + 0.5 * self.dirvars.T @ self.hessian @ self.dirvars
        return f_quad[0, 0], gradf # strip unnecessary dimensions

    def linearize_constraints(self, g_dict):
        """
        This function does linear approximation of the constraints

        Args:
            + g - dict 
                {'item': {'lhs': constraint with dvars, 
                          'rhs': constant}}
        """
        g_lin_arr = {}
        g_rhs_vec = []
        for i in range(self.num_g):
            g = g_dict[str(i)]['lhs']
            gradg = self._grad_function_(g, self.dvars)
            g_lin = gradg.T.dot(self.dirvars)
            g_lin_arr[str(i)] = g_lin[0, 0]
            g_rhs_vec.append(g_dict[str(i)]['rhs'])
        return g_lin_arr, g_rhs_vec

    def optimize(self, x0):
        """
        This function does one iteration of SQP.
            - finds the search direction using simplex method (scipy implementation)
            - finds the step size (golden section search)
        HOW: 
            - formulate the kkt condition as a linear programming problem
            NOTE: refer to Arora Chapter 9.5 - QP problems using simplex method

        Args:
            + x0 (list or 1d array) - current search point
        """
        """ get f, g in terms of d only """
        x_to_sub = {}
        for i in range(self.dim):
            x_to_sub[self.dvars[i, 0]] = x0[i]
        # print("x to sub: {}".format(x_to_sub))
        self.fsubs = self.f_approx.subs(x_to_sub)
        # print("\nfsubs: {}".format(self.fsubs))
        c_vec = []
        for i in range(self.dim):
            c_vec.append(-self.gradf[i, 0].subs(x_to_sub))

        A_ub_arr = []   # this array is m x n, each row is 1 constraint
        A_slack_arr = []
        slack_vars = []
        mu_vars = []
        g_subs_dict = {}
        for j in range(self.num_g):
            A_ub = []
            A_slack = []
            g_subs = self.g_approx[str(j)].subs(x_to_sub)
            g_subs_dict[str(j)] = g_subs - self.g_rhs[j]

            slack_vec = [0] * self.num_g
            slack_vec[j] = 1
            
            for i in range(self.dim):
                gj_coeff = g_subs.coeff(self.dirvars[i, 0])
                A_ub.append(gj_coeff)
                A_slack.append(gj_coeff)

            A_slack.extend([0] * self.num_g)    # 0 coefficients of mu
            A_slack.extend(slack_vec)           # coefficients of slack variables

            A_ub_arr.append(A_ub)
            A_slack_arr.append(A_slack)
            c_vec.append(self.g_rhs[j])
            slack_vars.append(sym.symbols('s%d' % j))
            mu_vars.append(sym.symbols('mu%d' % j))
        
        A_ub_arr = np.array(A_ub_arr)   # m x n
        A_slack_arr = np.array(A_slack_arr)

        """
        Build the B matrix as defined in the Arora QP KKT condition formulation 
        [ H     A(n x m)   0(n x m) ]     A here is A_ub_arr.T, n x m
        [ A.T   0(m x m)   I(m x m) ]     H here is n x n 
        """
        H = np.empty((self.dim, self.dim))
        for i in range(self.dim):
            for j in range(self.dim):
                H[i, j] = self.hessian[i, j].subs(x_to_sub)
        left_col = np.concatenate((H, A_ub_arr), axis=0)
        mid_col = np.concatenate((A_ub_arr.T, np.zeros((self.num_g, self.num_g))), axis=0)
        right_col = np.concatenate((np.zeros((self.dim, self.num_g)), np.eye(self.num_g)), axis=0)
        B_mat = np.concatenate((left_col, mid_col, right_col), axis=1)

        """ formulate the artificial cost function """
        X_vec = np.concatenate((self.dirvars.reshape(self.dim), mu_vars, slack_vars))
        w0 = np.sum(np.array(c_vec))
        C_vec = -np.sum(B_mat, axis=0)   # rowwise sum (get a row vector)
        w = w0 + C_vec @ X_vec.T
        w_coeff = []
        for i in range(len(X_vec)):
            w_coeff.append(w.coeff(X_vec[i]))

        if self.method == 'simplex':
            simplex_tableau = self._initialize_tableau(B_mat, w_coeff, c_vec)
            final_tableau = self.simplex_method(simplex_tableau)

            # print("final tableau: \n{}".format(final_tableau))
            search_dir =[]
            for i in range(self.dim):
                if np.sum(final_tableau[:,i]) == 1:
                    ones_idx = np.where(final_tableau[:,i] == 1)
                    if len(ones_idx[0]) > 1:
                        search_dir.append(0.0)
                    else:
                        search_dir.append(final_tableau[ones_idx[0][0],-1])
                else:
                    search_dir.append(0.0)

        elif self.method == 'linprog':
            res = self._call_simplex(B_mat, w_coeff, c_vec)
            search_dir = res.x[0:len(self.dirvars)]

        elif self.method == 'QP':
            res = self._scipy_qp(A_ub_arr)
            search_dir = res.x[0:len(self.dirvars)]

        # print("\nComputed search direction is: {}".format(search_dir)) 

        """ write the next point in terms of d and alpha, and find alpha """
        alpha = sym.symbols('a')
        x1 = x0 + alpha * np.array(search_dir)
        x1_to_sub = {}
        for i in range(self.dim):
            x1_to_sub[self.dvars[i, 0]] = x1[i]

        f_xkplus1 = self.f.subs(x1_to_sub)
        g_xkplus1 = {}
        for j in range(self.num_g):
            g_xkplus1[str(j)] = self.g[str(j)]['lhs'].subs(x1_to_sub)

        gs_result = self.golden_section(f_xkplus1, g_xkplus1, 0.0, 10.0)
        # print("result from golden section: {}".format(gs_result))

        x1 = x0 + gs_result * np.array(search_dir)
        # print("new x1: {}".format(x1))
        # print("==================================================")
        return x1

    def _initialize_tableau(self, B_mat, w_coeff, D_vec):
        """
        Helper function to initialize the tableau

        Args:
            + B_mat - B matrix as defined in Arora
            + w_coeff - coefficients of the artificial cost functin
            + D_vec - D vector as defined in Arora (rhs column)
        """        
        # find elements in D that are negative
        # idx = np.argwhere(np.array(D_vec) < 0)
        # print("idx: {}".format(idx))
        # if np.size(idx):
        #     for i in idx:
        #         try:
        #             D_vec[i] = -D_vec[i]
        #             B_mat[i, :] = np.negative(B_mat[i, :])
        #         except:
        #             i = i[0]
        #             D_vec[i] = -D_vec[i]
        #             B_mat[i, :] = np.negative(B_mat[i, :])

        # this line adds the columns for the Y artificial variables
        B_mat_slack = np.concatenate((B_mat, np.eye(len(B_mat))), axis=1)
        w_coeff.extend([0]*len(B_mat))
        tableau = np.concatenate((B_mat_slack, np.array([w_coeff])), axis=0)
        tab_col = D_vec
        tab_col.append(-np.sum(D_vec))
        tab_col = np.array([tab_col]).T
        tableau = np.concatenate((tableau, tab_col), axis=1)
        # print("simplex tableau: \n{}".format(tableau))
        return tableau

    def simplex_method(self, tableau):
        """
        Simplex solver

        Args:
            + tableau - Takes in the simplex tableau previously defined

        selection of non basic to become basic is arbitrary, 
        convention is to choose the one with the smallest value in the cost (bottom row)
        NOTE: only the negative values are considered, if it is all positive, then it is optimal
        selection of basic to become non-basic is fixed
        divide the B column (last column) by the positive elements in the pivot column
        NOTE, the B column must be all positive
        then select the row with the smallest factor

        For now, it simple returns the final tableau after the iteratons. I haven't extensively tested the
        algorithm with various functions, but it solves the hard coded test case example below

        Test case example:
            - assume the passed in tableau is defined below
            - the last row has the reduced variable coeeficient and the last element is the cost function
            - other rows are coefficents of the design variables with the columns representing the design variables
            - At the begining of the process, the design variables are all non-basic and thus zero. thus zero is 
              passed into the function to evaluate the cost function.
        tableau = np.array([[2, 0, 1, -1, 0, 0, 1, -1, 1, 0, 0, 0, 6], 
                            [0, 2, 1, 0, -1, 0, -3, 3, 0, 1, 0, 0, 6], 
                            [1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 4], 
                            [1, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1], 
                            [-4, 0, -2, 1, 1, -1, 2, -2, 0, 0, 0, 0, -17]],dtype=float)
        """
        tableau = np.array(tableau,dtype=float)
        # print("The new tableau is \n{}".format(tableau))
        counter = 0
        
        for i in range(self.dim + self.num_g + 1):
            counter += 1
            #pivot column is the column in thae last row that has the lowest value
            pivot_column = int(np.where(tableau[-1]==tableau[-1, :-1].min())[0][0])
            #pivot_column_history.append(pivot_column)

            # print("The Pivot_column is {} \n".format(pivot_column))
            #The basic column that will be changed to non_basic is calculated by dividing the last column by the pivot column
            #trying to remove negative values from the pivot column
            pivot_col = tableau[:,pivot_column].copy()
            for i in range(len(pivot_col)):
                if pivot_col[i] <= 0:
                    pivot_col[i] = 0.0001
            # print("the element to divide by is: {}".format(tableau[:,pivot_column]))
            # print("the new element to divide by is: {}".format(pivot_col))
            b_np = tableau[:,-1]/pivot_col
            # print("The entire array for determining the pivot row is {} \n".format(b_np))
            b_np = b_np[:-1]
            # print("The array for determining the pivot row is {} \n".format(b_np))
            #The minimum in that array is selected as the pivot_row

            pivot_row = int(np.where(b_np==b_np.min())[0][0])
            pivot_element_loc = [pivot_row,pivot_column]
            pivot_element = float(tableau[pivot_row,pivot_column])

            if pivot_element == 0.0:
                ## find a nonzero element in the pivot column
                idx = np.nonzero(tableau[:-1,pivot_column])[0]
                # print("idx: {}".format(idx))
                for j in range(len(idx)):
                    if np.abs(tableau[idx[j],pivot_column]) <= 1e-6:
                        pass
                    else:
                        pivot_row = idx[j]
                pivot_element_loc = [pivot_row,pivot_column]
                pivot_element = float(tableau[pivot_row,pivot_column])

            # print("The Pivot_row is {} \n".format(pivot_row))
            # print("The Pivot_element location is {} \n".format(pivot_element_loc))
            # print("The Pivot_element is {} \n".format(pivot_element))

            tableau[pivot_row,:] = tableau[pivot_row,:]/pivot_element

            for i in range(self.num_g+self.dim+1):  
                if i == pivot_row:
                    continue
                tableau[i,:] = tableau[i,:] - (tableau[pivot_row,:] * tableau[i,pivot_column])
            # print("The new tableau is:\n {} \n".format(tableau))
            if np.abs(tableau[-1, -1]) <= 1e-10 :
                break
        # print("Simplex ran {} times\n".format(counter))
        return tableau

    def _scipy_qp(self, gsubs):
        """
        Helper function to call scipy.optimize.minimize

        Args:
            + gsubs - coefficients of g
        """
        # print("constraint matrix: \n{}".format(gsubs))
        cons = LinearConstraint(gsubs, self.g_rhs, self.g_rhs)
        bnds = []
        for i in range(self.dim):
            bnds.append((None, None))
        bnds = tuple(bnds)
        opts = {'maxiter':100, 'disp':False}

        res = minimize(self._f_for_scipy, [5.0, 5.0], method='SLSQP', constraints=cons, \
                        bounds=bnds, options=opts)
        return res
    
    def _f_for_scipy(self, d0):
        """
        Helper function that defines objective function as a function of d0
        NOTE: For use with scipy QP
        
        Args:
            + d0 - current search direction
        """
        d1 = d0[0]
        d2 = d0[1]
        obj = self.fsubs.evalf(subs={self.dirvars[0, 0]:d1, self.dirvars[1, 0]:d2})
        return obj

    def _call_simplex(self, B_mat, w_coeff, b_vec):
        """
        Helper function to call scipy.optimize.linprog

        Args:
            + B_mat - B matrix as defined in Arora
            + w_coeff - coefficients of artificial cost function
            + b_vec - D vector as defined in Arora
        """
        B_mat_slack = np.concatenate((B_mat, np.eye(len(B_mat))), axis=1)
        w_coeff.extend([0]*len(B_mat))

        # print("\n>>> INPUTS TO SIMPLEX:")
        # print("c: {}".format(w_coeff))
        # print("A_eq: \n{}".format(B_mat_slack))
        # print("b_eq: {}".format(b_vec))

        res = linprog(c=w_coeff, A_eq=B_mat_slack, b_eq=b_vec)
        return res

    def golden_section(self, f, g, lower_start, upper_start, threshold=0.001):
        """
        Function that implements golden section search, used for solving for alpha

        Args:
            + f - f in terms of alpha
            + g - g in terms of alpha
            + lower_start - lower bound of initial interval
            + upper_start - upper bound of initial interval
            + threshold - golden section accuracy requirement
        """
        alpha = sym.symbols('a')
        iteration=[]
        counter=0

        while True:
            iteration.append(counter)
            a = lower_start + 0.382*(upper_start-lower_start)
            b = lower_start + 0.618*(upper_start-lower_start)
            fx1 = self._descent_function(f, g, {alpha:a})
            fx2 = self._descent_function(f, g, {alpha:b})
            if fx1 > fx2:
                lower_start = a
            elif fx2 > fx1:
                upper_start = b 
            else:
                lower_start = a
                upper_start = b
            if np.abs(lower_start-upper_start) <= threshold:
                break
            counter +=1
        # print('golden section ran {} times'.format(counter))
        return (lower_start+upper_start)/2

    def _penalty_function(self, g, a_to_sub):
        """
        Helper function to define penalty function for descent function

        Args:
            + g - the constraints
                NOTE: the input g is already substituted, the only unknown is alpha
            + a_to_sub - alpha to substitute (golden section query points)
        """
        constraints_eval = []
        for i in range(self.num_g):
            cont2 = g[str(i)]
            g_eval = g[str(i)].subs(a_to_sub) - self.g[str(i)]['rhs']
            if g_eval > 0:
                constraints_eval.append(g_eval)
        if any(constraints_eval):
            return max(constraints_eval)*10
        else:
            return 0
    
    def _descent_function(self, f, g, a_to_sub):
        """
        Helper function to define the descent function for Golden Section search
        """
        return f.subs(a_to_sub) + self._penalty_function(g, a_to_sub)
