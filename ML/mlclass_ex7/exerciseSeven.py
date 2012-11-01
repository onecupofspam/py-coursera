# This Python file uses the following encoding: utf-8
import pdb

from numpy import *
from numpy.linalg import *

from matplotlib.pyplot import * 

from scipy import optimize


def findClosestCentroids(X, centroids):
    #FINDCLOSESTCENTROIDS computes the centroid memberships for every example
    #   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
    #   in idx for a dataset X where each row is a single example.
    #   idx = vector of centroid assignments (i.e. each entry in range [1..K])
    #

    # Set K
    K = size(centroids, 0)

    # You need to return the following variables correctly.
    idx = zeros(size(X, 0))

    # ====================== YOUR CODE HERE ======================
    # Instructions: Go over every example, find its closest centroid, and store
    #               the index inside idx at the appropriate location.
    #               Concretely, idx(i) should contain the index of the centroid
    #               closest to example i. Hence, it should be a value in the
    #               range 1..K
    #
    # Note: You can use a for-loop over the examples to compute this.
    #




    # =============================================================

    return idx


def computeCentroids(X, idx, K):
    #COMPUTECENTROIDS returs the new centroids by computing the means of the
    #data points assigned to each centroid.
    #   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by
    #   computing the means of the data points assigned to each centroid. It is
    #   given a dataset X where each row is a single data point, a vector
    #   idx of centroid assignments (i.e. each entry in range [1..K]) for each
    #   example, and K, the number of centroids. You should return a matrix
    #   centroids, where each row of centroids is the mean of the data points
    #   assigned to it.
    #

    # Useful variables
    m, n = shape(X)

    # You need to return the following variables correctly.
    centroids = zeros((K, n))


    # ====================== YOUR CODE HERE ======================
    # Instructions: Go over every centroid and compute mean of all points that
    #               belong to it. Concretely, the row vector centroids(i, :)
    #               should contain the mean of the data points assigned to
    #               centroid i.
    #
    # Note: You can use a for-loop over the centroids to compute this.
    #




    # =============================================================

    return centroids

def runkMeans(X, initial_centroids, max_iters, plot_progress=False):
    #RUNKMEANS runs the K-Means algorithm on data matrix X, where each row of X
    #is a single example
    #   centroids, idx = RUNKMEANS(X, initial_centroids, max_iters, plot_progress=false)
    #   runs the K-Means algorithm on data matrix X, where each
    #   row of X is a single example. It uses initial_centroids used as the
    #   initial centroids. max_iters specifies the total number of interactions
    #   of K-Means to execute. plot_progress is a True/False flag that
    #   indicates if the function should also plot its progress as the
    #   learning happens. This is set to False by default. runkMeans returns
    #   centroids, a K x n matrix of the computed centroids and idx, a vector of
    #   size m of centroid assignments (i.e. each entry in range [1..K])
    #

    # Plot the data if we are plotting progress
    if plot_progress:
        fig = figure()
        hold(True)

    # Initialize values
    m, n = shape(X)
    K = size(initial_centroids, 0)
    centroids = initial_centroids
    previous_centroids = centroids
    idx = zeros(m)

    # Run K-Means
    for i in range(max_iters):

        # Output progress
        print 'K-Means iteration %d/%d...' % (i+1, max_iters)

        # For each example in X, assign it to the closest centroid
        idx = findClosestCentroids(X, centroids)

        # Optionally, plot progress here
        if plot_progress:
            plotProgresskMeans(X, centroids, previous_centroids, idx, K, i)
            previous_centroids = centroids
            fig.show()
            print 'Press enter to continue.'
            raw_input()

        # Given the memberships, compute new centroids
        centroids = computeCentroids(X, idx, K)

    # Hold off if we are plotting progress
    if plot_progress:
        hold(True)

    return centroids, idx


def plotProgresskMeans(X, centroids, previous, idx, K, i):
    #PLOTPROGRESSKMEANS is a helper function that displays the progress of
    #k-Means as it is running. It is intended for use only with 2D data.
    #   PLOTPROGRESSKMEANS(X, centroids, previous, idx, K, i) plots the data
    #   points with colors assigned to each centroid. With the previous
    #   centroids, it also plots a line between the previous locations and
    #   current locations of the centroids.
    #

    # Plot the examples
    plotDataPoints(X, idx, K)

    # Plot the centroids as black x's
    plot(centroids[:,0], centroids[:,1], 'x', mec='k', ms=10, mew=3)

    # Plot the history of the centroids with lines
    for j in range(size(centroids, 0)):
        drawLine(centroids[j, :], previous[j, :], 'b')

    # Title
    title('Iteration number #%d' % (i+1))


def plotDataPoints(X, idx, K):
    #PLOTDATAPOINTS plots data points in X, coloring them so that those with the same
    #index assignments in idx have the same color
    #   PLOTDATAPOINTS(X, idx, K) plots data points in X, coloring them so that those
    #   with the same index assignments in idx have the same color

    # Plot the data
    scatter(X[:,0], X[:,1], 100, idx, cmap=cm.hsv, vmax=K+1, facecolors='none')

def drawLine(p1, p2, *args, **kwargs):
    #DRAWLINE Draws a line from point p1 to point p2
    #   DRAWLINE(p1, p2) Draws a line from point p1 to point p2 and holds the
    #   current figure
    plot(array([p1[0], p2[0]]), array([p1[1], p2[1]]), *args, **kwargs)

def kMeansInitCentroids(X, K):
    #KMEANSINITCENTROIDS This function initializes K centroids that are to be
    #used in K-Means on the dataset X
    #   centroids = KMEANSINITCENTROIDS(X, K) returns K initial centroids to be
    #   used with the K-Means on the dataset X
    #

    # You should return this values correctly
    centroids = zeros((K, size(X, 1)))

    # ====================== YOUR CODE HERE ======================
    # Instructions: You should set centroids to randomly chosen examples from
    #               the dataset X
    #




    # =============================================================

    return centroids


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


def pca(X):
    #PCA Run principal component analysis on the dataset X
    #   U, s = pca(X) computes eigenvectors of the covariance matrix of X
    #   Returns the eigenvectors U, the eigenvalues in s
    #

    # Useful values
    m, n = shape(X)

    # You need to return the following variables correctly.
    U = zeros((n,n))
    s = zeros(n)

    # ====================== YOUR CODE HERE ======================
    # Instructions: You should first compute the covariance matrix. Then, you
    #               should use the "svd" function (in numpy.linalg) to compute
    #               the eigenvectors and eigenvalues of the covariance matrix.
    #
    # Note: When computing the covariance matrix, remember to divide by m (the
    #       number of examples).
    #





    # =========================================================================

    return U, s


def projectData(X, U, K):
    #PROJECTDATA Computes the reduced data representation when projecting only
    #on to the top k eigenvectors
    #   Z = projectData(X, U, K) computes the projection of
    #   the normalized inputs X into the reduced dimensional space spanned by
    #   the first K columns of U. It returns the projected examples in Z.
    #

    # You need to return the following variables correctly.
    Z = zeros((size(X, 0), K))

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the projection of the data using only the top K
    #               eigenvectors in U (first K columns).
    #               For the i-th example X[i,:], the projection on to the k-th
    #               eigenvector is given as follows:
    #                    x = X[i, :]
    #                    projection_k = dot(x, U[:, k])
    #




    # =============================================================

    return Z


def recoverData(Z, U, K):
    #RECOVERDATA Recovers an approximation of the original data when using the
    #projected data
    #   X_rec = RECOVERDATA(Z, U, K) recovers an approximation the
    #   original data that has been reduced to K dimensions. It returns the
    #   approximate reconstruction in X_rec.
    #

    # You need to return the following variables correctly.
    X_rec = zeros((size(Z, 0), size(U, 0)))

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the approximation of the data by projecting back
    #               onto the original space using the top K eigenvectors in U.
    #
    #               For the i-th example Z[i,:], the (approximate)
    #               recovered data for dimension j is given as follows:
    #                    v = Z[i, :]
    #                    recovered_j = dot(v, U[j, 1:K])
    #
    #               Notice that U[j, 1:K] is a row vector.
    #



    # =============================================================

    return X_rec

from numpy import *
from matplotlib.pyplot import *

def displayData(X, example_width=None):
#DISPLAYDATA Display 2D data in a nice grid
#   display_array = DISPLAYDATA(X, example_width) displays 2D data
#   stored in X in a nice grid in the current figure. It returns
#   the displayed array.

    # Set example_width automatically if not passed in
    if example_width is None:
        example_width = int(sqrt(size(X, 1))+0.5)

    # Compute rows, cols
    m, n = shape(X)
    example_height = n / example_width

    # Compute number of items to display
    display_rows = int(sqrt(m))
    display_cols = (m + display_rows - 1) / display_rows

    # Between images padding
    pad = 1

    # Setup blank display
    display_array = - ones((pad + display_rows * (example_height + pad),
                            pad + display_cols * (example_width + pad)))

    # Copy each example into a patch on the display array
    curr_ex = 0
    for j in range(display_rows):
        for i in range(display_cols):
            if curr_ex >= m:
                break
            # Copy the patch

            # Get the max value of the patch
            max_val = max(abs(X[curr_ex, :]))
            pos_x = pad + i * (example_width + pad)
            pos_y = pad + j * (example_height + pad)
            display_array[pos_y : pos_y + example_height, pos_x : pos_x + example_width] = \
                reshape(X[curr_ex, :], (example_height, example_width), order='F') / max_val
            curr_ex += 1
        if curr_ex >= m:
           break

    # Display Image
    imshow(display_array, interpolation='none', cmap=cm.gray)

    # Do not show axis
    axis('off')

    return display_array


