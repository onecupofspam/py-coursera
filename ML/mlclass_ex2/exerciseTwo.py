# This Python file uses the following encoding: utf-8
import pdb

from numpy import *
from numpy.linalg import *

from matplotlib.pyplot import * 

from scipy import optimize



def plotData(X, y):
    #PLOTDATA Plots the data points x and y into a new figure
    #   PLOTDATA(x,y) plots the data points with + for the positive examples
    #   and o for the negative examples. X is assumed to be a Mx2 matrix.

    # Create New Figure
    fig = figure()
    hold(True)
    # ====================== YOUR CODE HERE ======================
    # Instructions: Plot the positive and negative examples on a
    #               2D plot, using the option 'k+' for the positive
    #               examples and 'ko' for the negative examples.
    #



    # ============================================================
    return fig

def sigmoid(z):
    #SIGMOID Compute sigmoid functoon
    #   J = SIGMOID(z) computes the sigmoid of z.

    # You need to return the following variables correctly
    g = zeros(shape(z))

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the sigmoid of each value of z (z can be a matrix,
    #               vector or scalar).




    # =============================================================

    return g

def predict(theta, X):
    #PREDICT Predict whether the label is 0 or 1 using learned logistic
    #regression parameters theta
    #   p = PREDICT(theta, X) computes the predictions for X using a
    #   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)
    m = size(X, 1) # Number of training examples

    # You need to return the following variables correctly
    p = zeros(m)
    # ====================== YOUR CODE HERE ======================
    # Instructions: Complete the following code to make predictions using
    #               your learned logistic regression parameters.
    #               You should set p to a vector of 0's and 1's (or booleans)
    #

    # =========================================================================
    return p

def costFunction(theta, X, y):
    #COSTFUNCTION Compute cost and gradient for logistic regression
    #   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
    #   parameter for logistic regression and the gradient of the cost
    #   w.r.t. to the parameters.

    # Initialize some useful values
    m = len(y) # number of training examples

    # You need to return the following variables correctly
    J = 0
    grad = zeros(size(theta))

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta.
    #               You should set J to the cost.
    #               Compute the partial derivatives and set grad to the partial
    #               derivatives of the cost w.r.t. each parameter in theta
    #
    # Note: grad should have the same dimensions as theta
    #




    # =============================================================

    return J, grad


def costFunctionReg(theta, X, y, lambda_):
    #COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
    #   J = COSTFUNCTIONREG(theta, X, y, lambda_) computes the cost of using
    #   theta as the parameter for regularized logistic regression and the
    #   gradient of the cost w.r.t. to the parameters.

    # Initialize some useful values
    m = len(y) # number of training examples
    lambda_ = float(lambda_)

    # You need to return the following variables correctly
    J = 0
    grad = zeros(size(theta))
    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta.
    #               You should set J to the cost.
    #               Compute the partial derivatives and set grad to the partial
    #               derivatives of the cost w.r.t. each parameter in theta



    # =============================================================
    return J, grad

def plotDecisionBoundary(theta, X, y):
    #PLOTDECISIONBOUNDARY Plots the data points X and y into a new figure with
    #the decision boundary defined by theta
    #   PLOTDECISIONBOUNDARY(theta, X,y) plots the data points with + for the
    #   positive examples and o for the negative examples. X is assumed to be
    #   a either
    #   1) Mx3 matrix, where the first column is an all-ones column for the
    #      intercept.
    #   2) MxN, N>3 matrix, where the first column is all-ones

    # Plot Data
    from mlclass_ex2 import plotData, mapFeature

    fig = plotData(X[:,1:], y)
    hold(True)

    if size(X, 1) <= 3:
        # Only need 2 points to define a line, so choose two endpoints
        plot_x = array([min(X[:,1])-2,  max(X[:,1])+2])

        # Calculate the decision boundary line
        plot_y = (-1/theta[2]) * (theta[1] * plot_x + theta[0])

        # Plot, and adjust axes for better viewing
        plot(plot_x, plot_y)

        # Legend, specific for the exercise
        legend(('Admitted', 'Not admitted', 'Decision Boundary'), numpoints=1)
        axis([30, 100, 30, 100])
    else:
        u = linspace(-1, 1.5, 50)

        # Evaluate z = theta*x over the grid
        z = frompyfunc(lambda x,y: mapFeature(x,y).dot(theta), 2, 1).outer(u,u)

        z = z.T # important to transpose z before calling contour

        # Plot z = 0
        # Notice you need to specify the level as [0]
        contour(u, u, z, [0], linewidth=2)

    hold(False)

    return fig

def mapFeature(X1, X2):
    # MAPFEATURE Feature mapping function to polynomial features
    #
    #   MAPFEATURE(X1, X2) maps the two input features
    #   to quadratic features used in the regularization exercise.
    #
    #   Returns a new feature array with more features, comprising of
    #   X1, X2, X1**2, X2**2, X1*X2, X1*X2**2, etc..
    #
    #   Inputs X1, X2 must be the same size
    #

    degree = 6
    out = [ones(size(X1))]
    for i in range(1, degree+1):
        for j in range(i+1):
            out.append(X1 ** (i-j) * X2 ** j)

    if isscalar(X1):
        return hstack(out)  # if inputs are scalars, return a vector
    else:
        return column_stack(out)

