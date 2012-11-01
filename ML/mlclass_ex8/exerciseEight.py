# This Python file uses the following encoding: utf-8
import pdb

from numpy import *

from matplotlib.pyplot import * 

from scipy import optimize


def estimateGaussian(X):
    #ESTIMATEGAUSSIAN This function estimates the parameters of a
    #Gaussian distribution using the data in X
    #   mu, sigma2 = estimateGaussian(X),
    #   The input X is the dataset with each n-dimensional data point in one row
    #   The output is an n-dimensional vector mu, the mean of the data set
    #   and the variances sigma^2, another n-dimensional vector
    #

    # Useful variables
    m, n = shape(X)

    # You should return these values correctly
    mu = zeros(n)
    sigma2 = zeros(n)

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the mean of the data and the variances
    #               In particular, mu[i] should contain the mean of
    #               the data for the i-th feature and sigma2[i]
    #               should contain variance of the i-th feature.
    #


    

    # =============================================================

    return mu, sigma2


def multivariateGaussian(X, mu, Sigma2):
    #MULTIVARIATEGAUSSIAN Computes the probability density function of the
    #multivariate gaussian distribution.
    #    p = MULTIVARIATEGAUSSIAN(X, mu, Sigma2) Computes the probability
    #    density function of the examples X under the multivariate gaussian
    #    distribution with parameters mu and Sigma2. If Sigma2 is a matrix, it is
    #    treated as the covariance matrix. If Sigma2 is a vector, it is treated
    #    as the \sigma^2 values of the variances in each dimension (a diagonal
    #    covariance matrix)
    #
    k = len(mu)

    if Sigma2.ndim == 1:
        Sigma2 = diag(Sigma2)

    X = X - mu
    p = (2 * pi) ** (- k / 2.) * linalg.det(Sigma2) ** (-0.5) * \
        exp(-0.5 * sum(dot(X, linalg.pinv(Sigma2)) * X, 1))

    return p

def visualizeFit(X, mu, sigma2):
    #VISUALIZEFIT Visualize the dataset and its estimated distribution.
    #   VISUALIZEFIT(X, mu, sigma2) This visualization shows you the
    #   probability density function of the Gaussian distribution. Each example
    #   has a location (x1, x2) that depends on its feature values.
    #

    coords = linspace(0,30,61)
    X1, X2 = meshgrid(coords, coords)
    Z = multivariateGaussian(column_stack((X1.ravel(),X2.ravel())), mu, sigma2)
    Z = reshape(Z, shape(X1))

    plot(X[:, 0], X[:, 1],'bx')
    hold(True)
    # Do not plot if there are infinities
    if not any(isinf(Z)):
        contour(X1, X2, Z, power(10., arange(-20,0,3)))
    hold(False)


def selectThreshold(yval, pval):
    #SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
    #outliers
    #   bestEpsilon, bestF1 = SELECTTHRESHOLD(yval, pval) finds the best
    #   threshold to use for selecting outliers based on the results from a
    #   validation set (pval) and the ground truth (yval).
    #

    bestEpsilon = 0
    bestF1 = 0
    F1 = 0

    for epsilon in linspace(min(pval), max(pval), 1001):

        # ====================== YOUR CODE HERE ======================
        # Instructions: Compute the F1 score of choosing epsilon as the
        #               threshold and place the value in F1. The code at the
        #               end of the loop will compare the F1 score for this
        #               choice of epsilon and set it to be the best epsilon if
        #               it is better than the current choice of epsilon.
        #
        # Note: You can use predictions = (pval < epsilon).astype(float) to
        #       get a binary vector of 0's and 1's of the outlier predictions




        # =============================================================

        if F1 > bestF1:
           bestF1 = F1
           bestEpsilon = epsilon

    return bestEpsilon, bestF1

def loadMovieList():
    #GETMOVIELIST reads the fixed movie list in movie_ids.txt and returns a
    #list of the titles
    #   movieList = GETMOVIELIST() reads the fixed movie list in movie_ids.txt
    #   and returns a list of the titles in movieList.

    movieList = []

    ## Read the fixed movieulary list
    with open('movie_ids.txt') as fid:
        for line in fid:
            movieName = line.split(' ', 1)[1].strip()
            movieList.append(movieName)

    return movieList


def cofiCostFunc(params, Y, R, num_users, num_movies, num_features, lambda_):
    #COFICOSTFUNC Collaborative filtering cost function
    #   J, grad = COFICOSTFUNC(params, Y, R, num_users, num_movies, num_features, lambda_)
    #   returns the cost and gradient for the collaborative filtering problem.
    #

    # Unfold the U and W matrices from params
    X = reshape(params[:num_movies*num_features], (num_movies, num_features), order='F')
    Theta = reshape(params[num_movies*num_features:], (num_users, num_features), order='F')


    # You need to return the following values correctly
    J = 0
    X_grad = zeros(shape(X))
    Theta_grad = zeros(shape(Theta))

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost function and gradient for collaborative
    #               filtering. Concretely, you should first implement the cost
    #               function (without regularization) and make sure it is
    #               matches our costs. After that, you should implement the
    #               gradient and use the checkCostFunction routine to check
    #               that the gradient is correct. Finally, you should implement
    #               regularization.
    #
    # Notes: X - num_movies  x num_features matrix of movie features
    #        Theta - num_users  x num_features matrix of user features
    #        Y - num_movies x num_users matrix of user ratings of movies
    #        R - num_movies x num_users matrix, where R(i, j) = 1 if the
    #            i-th movie was rated by the j-th user
    #
    # You should set the following variables correctly:
    #
    #        X_grad - num_movies x num_features matrix, containing the
    #                 partial derivatives w.r.t. to each element of X
    #        Theta_grad - num_users x num_features matrix, containing the
    #                     partial derivatives w.r.t. to each element of Theta
    #





    # =============================================================

    grad = hstack((X_grad.ravel('F'), Theta_grad.ravel('F')))

    return J, grad



def checkCostFunction(lambda_=0):
    #CHECKCOSTFUNCTION Creates a collaborative filering problem
    #to check your cost function and gradients
    #   CHECKCOSTFUNCTION(lambda_) Creates a collaborative filering problem
    #   to check your cost function and gradients, it will output the
    #   analytical gradients produced by your code and the numerical gradients
    #   (computed using computeNumericalGradient). These two gradient
    #   computations should result in very similar values.

    ## Create small problem
    X_t = random.rand(4, 3)
    Theta_t = random.rand(5, 3)

    # Zap out most entries
    Y = dot(X_t, Theta_t.T)
    Y[random.rand(*shape(Y)) > 0.5] = 0
    R = where(Y == 0, 0, 1)

    ## Run Gradient Checking
    X = random.randn(*shape(X_t))
    Theta = random.randn(*shape(Theta_t))
    num_users = size(Y, 1)
    num_movies = size(Y, 0)
    num_features = size(Theta_t, 1)

    numgrad = computeNumericalGradient(
        lambda t: cofiCostFunc(t, Y, R, num_users, num_movies, num_features, lambda_),
        hstack((X.ravel('F'), Theta.ravel('F'))))

    cost, grad = cofiCostFunc(hstack((X.ravel('F'), Theta.ravel('F'))), Y, R,
                              num_users, num_movies, num_features, lambda_)

    print column_stack((numgrad, grad))

    print 'The above two columns you get should be very similar.'
    print '(Left-Your Numerical Gradient, Right-Analytical Gradient)\n'

    diff = linalg.norm(numgrad-grad) / linalg.norm(numgrad+grad)
    print 'If your backpropagation implementation is correct, then'
    print 'the relative difference will be small (less than 1e-9).'
    print '\nRelative Difference: %g' % diff


def computeNumericalGradient(J, theta):
    #COMPUTENUMERICALGRADIENT Computes the gradient using "finite differences"
    #and gives us a numerical estimate of the gradient.
    #   numgrad = COMPUTENUMERICALGRADIENT(J, theta) computes the numerical
    #   gradient of the function J around theta. Calling y = J(theta) should
    #   return the function value at theta.

    # Notes: The following code implements numerical gradient checking, and
    #        returns the numerical gradient.It sets numgrad[i] to (a numerical
    #        approximation of) the partial derivative of J with respect to the
    #        i-th input argument, evaluated at theta. (i.e., numgrad[i] should
    #        be the (approximately) the partial derivative of J with respect
    #        to theta[i].)
    #

    numgrad = zeros(shape(theta))
    perturb = zeros(shape(theta))
    e = 1e-4
    for p in ndindex(shape(theta)):
        # Set perturbation vector
        perturb[p] = e
        loss1, _ = J(theta - perturb)
        loss2, _ = J(theta + perturb)
        # Compute Numerical Gradient
        numgrad[p] = (loss2 - loss1) / (2*e)
        perturb[p] = 0

    return numgrad

def normalizeRatings(Y, R):
    #NORMALIZERATINGS Preprocess data by subtracting mean rating for every
    #movie (every row)
    #   Ynorm, Ymean = NORMALIZERATINGS(Y, R) normalized Y so that each movie
    #   has a rating of 0 on average, and returns the mean rating in Ymean.
    #
    m, n = shape(Y)
    Ymean = zeros(m)
    Ynorm = zeros(shape(Y))
    for i in range(m):
        idx = where(R[i, :] == 1)
        Ymean[i] = mean(Y[i, idx])
        Ynorm[i, idx] = Y[i, idx] - Ymean[i]

    return Ynorm, Ymean



