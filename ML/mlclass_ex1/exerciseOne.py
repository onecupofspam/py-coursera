# This Python file uses the following encoding: utf-8
import pdb

from numpy import *
from numpy.linalg import *

from matplotlib.pyplot import * 
from scipy import optimize


def warmUpExercise():
    '''WARMUPEXERCISE Example function in octave
       A = WARMUPEXERCISE() is an example function that returns the 5x5 identity matrix'''

    # ============= YOUR CODE HERE ==============
    # Instructions: Return the 5x5 identity matrix 
    #               In python, we return values by specifying which
    #               variable(s) to return in the last line of the file,
    #               so just set A accordingly. 
    A = eye(5)
    # ===========================================
    return A

def to_matrix(X):
    return matrix(X)

def to_design(X):
    m = size(X,0)  # number of training examples
    n = size(X,1)  # number of features including theta_0
    X_design = column_stack((ones(m), X))
    return X_design

def plotData(x, y):
    '''PLOTDATA Plots the data points x and y into a new figure 
       PLOTDATA(x,y) plots the data points and gives the figure axes labels of
       population and profit.'''
    fig = figure()
    # ====================== YOUR CODE HERE ======================
    # Instructions: Plot the training data into a figure using the 
    #               "figure" and "plot" commands. Set the axes labels using
    #               the "xlabel" and "ylabel" commands. Assume the 
    #               population and revenue data have been passed in
    #               as the x and y arguments of this function.
    #
    plot(x, y, 'rx', markersize=10)         # Plot the data
    ylabel('Profit in $10,000s')            # Set the y−axis label
    xlabel('Population of City in 10000s')  # Set the x−axis label
    # ============================================================
    return fig


def computeCost(theta, X, y):
    '''COMPUTECOST Compute cost for linear regression
       J = COMPUTECOST(theta, X, y) computes the cost of using theta as the
       parameter for linear regression to fit the data points in X and y'''
    # Initialize some useful values
    # Assume X is the design matrix, [ones(m,1), data(:,:)]
    m = size(X,0)  # number of training examples
    n = size(X,1)  # number of features including theta_0
    # You need to return the following variables correctly 
    J = 0
    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta
    #               You should set J to the cost.
    # use "pdb.set_trace()" to drop into the debugger at this point
    predictions = matrix(X) * matrix(theta).transpose()
    errors = predictions - matrix(y)
    sumOfSquaredErrors = sum( transpose(errors) * errors )
    J = sumOfSquaredErrors / (2 * m);
    # =========================================================================
    return J

def computeCostMulti(theta, X, y):
    return computeCost(theta, X, y)

def normalEqn(X, y):
    '''NORMALEQN Computes the closed-form solution to linear regression
       NORMALEQN(X,y) computes the closed-form solution to linear
       regression using the normal equations.'''

    # Assume X is the design matrix, [ones(m,1), data(:,:)]
    m = size(X,0)  # number of training examples
    n = size(X,1)  # number of features including theta_0
    X_M = to_matrix(X)
    X_T = X_M.transpose()
    # ====================== YOUR CODE HERE ======================
    # Instructions: Complete the code to compute the closed form solution
    #               to linear regression and put the result in theta.
    #
    y_m = to_matrix(y)
    theta_n = pinv(X_T * X_M) * X_T * y_m.transpose()
    theta = theta_n.transpose()
    # ============================================================
    return theta


def gradientDescentMulti(X, y, theta, alpha, num_iters):
    return gradientDescent(X, y, theta, alpha, num_iters)

def gradientDescent(X, y, theta, alpha, num_iters):
    #GRADIENTDESCENT Performs gradient descent to learn theta
    #   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by
    #   taking num_iters gradient steps with learning rate alpha

    # Initialize some useful values
    debug = False;
    # Assume X is the design matrix, [ones(m,1), data(:,:)]
    m = size(X,0)  # number of training examples
    n = size(X,1)  # number of features including theta_0
    J_history = zeros((num_iters, 1))
    if (debug):
        print("\n Starting vectorized gradient descent using SciPy \n")
    #[theta, J_history] = gradientDescentImpl(X,y,theta,alpha,num_iters)
    (xopt,fopt,gopt,Bopt,func_calls,grad_calls,warnflag) = optimize.fmin_bfgs(computeCost, \
                                                           x0, fprime=optimize.rosen_der, \
                                                           maxiter=num_iters,full_output=True)

    #res = optimize.minimize(computeCost, x0=initial_theta, args=(X,y), \
    #                        method='BFGS', jac=True, options={'maxiter':num_iters})
    theta = res.x
    cost = res.fun
    return (theta, J_history)



def featureNormalize(X):
    #FEATURENORMALIZE Normalizes the features in X
    #   FEATURENORMALIZE(X) returns a normalized version of X where
    #   the mean value of each feature is 0 and the standard deviation
    #   is 1. This is often a good preprocessing step to do when
    #   working with learning algorithms.

    # You need to set these values correctly
    X_norm = X
    mu = zeros(size(X, 1))
    sigma = zeros(size(X, 1))
    # ====================== YOUR CODE HERE ======================
    # Instructions: First, for each feature dimension, compute the mean
    #               of the feature and subtract it from the dataset,
    #               storing the mean value in mu. Next, compute the
    #               standard deviation of each feature and divide
    #               each feature by it's standard deviation, storing
    #               the standard deviation in sigma.
    #
    #               Note that X is a matrix where each column is a
    #               feature and each row is an example. You need
    #               to perform the normalization separately for
    #               each feature.
    #
    # Hint: You might find the 'mean' and 'std' functions useful.
    #




    # ============================================================

    return X_norm, mu, sigma

