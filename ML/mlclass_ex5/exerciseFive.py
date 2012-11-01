# This Python file uses the following encoding: utf-8
import pdb

from numpy import *
from numpy.linalg import *

from matplotlib.pyplot import * 

from scipy.optimize import minimize
from itertools import count


def linearRegCostFunction(X, y, theta, lambda_):
    #LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
    #regression with multiple variables
    #   J, grad = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the
    #   cost of using theta as the parameter for linear regression to fit the
    #   data points in X and y. Returns the cost in J and the gradient in grad

    # Initialize some useful values
    m = len(y) # number of training examples

    # You need to return the following variables correctly
    J = 0
    grad = zeros(shape(theta))

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost and gradient of regularized linear
    #               regression for a particular choice of theta.
    #
    #               You should set J to the cost and grad to the gradient.
    #


    
    

    # =========================================================================

    return J, grad

def trainLinearReg(X, y, lambda_):
    #TRAINLINEARREG Trains linear regression given a dataset (X, y) and a
    #regularization parameter lambda
    #   [theta] = TRAINLINEARREG (X, y, lambda) trains linear regression using
    #   the dataset (X, y) and regularization parameter lambda. Returns the
    #   trained parameters theta.
    #

    # Initialize Theta
    initial_theta = zeros(size(X, 1))

    # Create "short hand" for the cost function to be minimized
    costFunction = lambda t: linearRegCostFunction(X, y, t, lambda_)
    # Now, costFunction is a function that takes in only one argument
    # setup an iteration counter
    counter = count()
    # Minimize using CG algorithm
    res = minimize(costFunction, initial_theta, method='CG', jac=True,
                   options={'maxiter': 200}, callback=lambda _:counter.next())
    theta = res.x
    print "Iteration %5d | Cost: %e" % (counter.next(), res.fun)

    return theta

def learningCurve(X, y, Xval, yval, lambda_):
    #LEARNINGCURVE Generates the train and cross validation set errors needed
    #to plot a learning curve
    #   error_train, error_val = LEARNINGCURVE(X, y, Xval, yval, lambda)
    #       returns the train and cross validation set errors for a learning
    #       curve. In particular, it returns two vectors of the same length
    #       - error_train and error_val. Then, error_train[i] contains the
    #       training error for i+1 examples (and similarly for error_val[i]).
    #
    #   In this function, you will compute the train and test errors for
    #   dataset sizes from 1 up to m. In practice, when working with larger
    #   datasets, you might want to do this in larger intervals.
    #

    # Number of training examples
    m = size(X, 0)

    # You need to return these values correctly
    error_train = zeros(m)
    error_val   = zeros(m)

    # ====================== YOUR CODE HERE ======================
    # Instructions: Fill in this function to return training errors in
    #               error_train and the cross validation errors in error_val.
    #               i.e., error_train[i] and
    #               error_val[i] should give you the errors
    #               obtained after training on i+1 examples.
    #
    # Note: You should evaluate the training error on the first i training
    #       examples (i.e., X[:i, :] and y[:i]).
    #
    #       For the cross-validation error, you should instead evaluate on
    #       the _entire_ cross validation set (Xval and yval).
    #
    # Note: If you are using your cost function (linearRegCostFunction)
    #       to compute the training and cross validation error, you should
    #       call the function with the lambda argument set to 0.
    #       Do note that you will still need to use lambda when running
    #       the training to obtain the theta parameters.
    #
    # Hint: You can loop over the examples with the following:
    #
    #       for i in range(m):
    #           # Compute train/cross validation errors using training examples
    #           # X[:i+1, :] and y[:i+1], storing the result in
    #           # error_train[i] and error_val[i]
    #           ....
    #

    # ---------------------- Sample Solution ----------------------


    
    
    
    

    # -------------------------------------------------------------

    # =========================================================================

    return error_train, error_val

def featureNormalize(X):
    #FEATURENORMALIZE Normalizes the features in X
    #   FEATURENORMALIZE(X) returns a normalized version of X where
    #   the mean value of each feature is 0 and the standard deviation
    #   is 1. This is often a good preprocessing step to do when
    #   working with learning algorithms.
    mu = mean(X, 0)
    sigma = std(X, 0, ddof=1)
    X_norm = (X - mu) / sigma
    # ============================================================
    return X_norm, mu, sigma

def plotFit(min_x, max_x, mu, sigma, theta, p):
    #PLOTFIT Plots a learned polynomial regression fit over an existing figure.
    #Also works with linear regression.
    #   PLOTFIT(min_x, max_x, mu, sigma, theta, p) plots the learned polynomial
    #   fit with power p and feature normalization (mu, sigma).

    # Hold on to the current figure
    hold(True)

    # We plot a range slightly bigger than the min and max values to get
    # an idea of how the fit will vary outside the range of the data points
    x = arange(min_x - 15, max_x + 25.01, 0.05)

    # Map the X values
    X_poly = polyFeatures(x, p)
    X_poly = (X_poly - mu) / sigma

    # Add ones
    X_poly = column_stack((ones(size(x)), X_poly))

    # Plot
    plot(x, dot(X_poly, theta), '--', linewidth=2)

    # Hold off to the current figure
    hold(False)


def polyFeatures(X, p):
    #POLYFEATURES Maps X (1D vector) into the p-th power
    #   X_poly = POLYFEATURES(X, p) takes a vector X (size m) and
    #   maps each example into its polynomial features where
    #   X_poly[i, :] = [X[i], X[i]**2, X[i]**3, ..., X[i]**p]
    #

    # You need to return the following variables correctly.
    X_poly = zeros((size(X), p))

    # ====================== YOUR CODE HERE ======================
    # Instructions: Given a vector X, return a matrix X_poly where the p-th
    #               column of X contains the values of X to the p-th power.
    #
    #




    # =========================================================================

    return X_poly


def validationCurve(X, y, Xval, yval):
    #VALIDATIONCURVE Generate the train and validation errors needed to
    #plot a validation curve that we can use to select lambda
    #   lambda_vec, error_train, error_val = VALIDATIONCURVE(X, y, Xval, yval)
    #       returns the train and validation errors (in error_train, error_val)
    #       for different values of lambda. You are given the training set (X,
    #       y) and validation set (Xval, yval).
    #

    # Selected values of lambda (you should not change this)
    lambda_vec = array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1., 3., 10.])

    # You need to return these variables correctly.
    error_train = zeros_like(lambda_vec)
    error_val = zeros_like(lambda_vec)

    # ====================== YOUR CODE HERE ======================
    # Instructions: Fill in this function to return training errors in
    #               error_train and the validation errors in error_val. The
    #               vector lambda_vec contains the different lambda parameters
    #               to use for each calculation of the errors, i.e,
    #               error_train(i), and error_val(i) should give
    #               you the errors obtained after training with
    #               lambda = lambda_vec(i)
    #
    # Note: You can loop over lambda_vec with the following:
    #
    #       for i in range(len(lambda_vec)):
    #           lambda_ = lambda_vec[i]
    #           # Compute train / val errors when training linear
    #           # regression with regularization parameter lambda_
    #           # You should store the result in error_train[i]
    #           # and error_val[i]
    #           ....
    #
    #





    # =========================================================================

    return lambda_vec, error_train, error_val
