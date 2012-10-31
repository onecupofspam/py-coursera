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
    A = array([])
    # ===========================================
    return A


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

    # ============================================================
    return fig


def computeCost(X, y, theta):
    '''COMPUTECOST Compute cost for linear regression
       J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
       parameter for linear regression to fit the data points in X and y'''
    # Initialize some useful values
    # Assume X is the design matrix, [ones(m,1), data(:,:)]
    # m: number of training examples
    # n: number of features including theta_0
    [m,n] = shape(X) # number of training examples
    # You need to return the following variables correctly 
    J = 0
    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta
    #               You should set J to the cost.
    # use "pdb.set_trace()" to drop into the debugger at this point

    # =========================================================================
    return J

def computeCostMulti(X, y, theta):
    return computeCost(X, y, theta)

def normalEqn(X, y):
    '''NORMALEQN Computes the closed-form solution to linear regression
       NORMALEQN(X,y) computes the closed-form solution to linear
       regression using the normal equations.'''

    # Assume X is the design matrix, [ones(m,1), data(:,:)]
    # m: number of training examples
    # n: number of features including theta_0
    [m,n] = shape(X)
    theta = zeros(2)
    # ====================== YOUR CODE HERE ======================
    # Instructions: Complete the code to compute the closed form solution
    #               to linear regression and put the result in theta.
    #

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
    # m: number of training examples
    # n: number of features including theta_0
    [m,n] = shape(X);
    J_history = zeros((num_iters, 1))
    theta = zeros(2)
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

