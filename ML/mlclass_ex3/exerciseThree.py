# This Python file uses the following encoding: utf-8
import pdb

from numpy import *
from numpy.linalg import *

from matplotlib.pyplot import * 

from scipy import optimize


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


def sigmoid(z):
    #SIGMOID Compute sigmoid functoon
    #   J = SIGMOID(z) computes the sigmoid of z.
    J = 1. / (1. + exp(-z))
    return  J


def oneVsAll(X, y, num_labels, lambda_):
    #ONEVSALL trains multiple logistic regression classifiers and returns all
    #the classifiers in a matrix all_theta, where the i-th row of all_theta
    #corresponds to the classifier for label i
    #   all_theta = ONEVSALL(X, y, num_labels, lambda) trains num_labels
    #   logisitc regression classifiers and returns each of these classifiers
    #   in a matrix all_theta, where the i-th row of all_theta corresponds
    #   to the classifier for label i

    # Some useful variables
    m, n = shape(X)

    # You need to return the following variables correctly
    all_theta = zeros((num_labels, n + 1))

    # Add ones to the X data matrix
    X = column_stack((ones(m), X))

    # ====================== YOUR CODE HERE ======================
    # Instructions: You should complete the following code to train num_labels
    #               logistic regression classifiers with regularization
    #               parameter lambda.
    #
    # Hint: You can use y == c to obtain a vector of booleans that tell use
    #       whether the ground truth is true/false for this class.
    #
    # Note: For the python version of this assignment, we recommend using
    #       scipy.optimize.minimize with the CG method (fmin_cg).
    #       It is okay to use a for-loop (for c in range(1,num_labels+1)) to
    #       loop over the different classes.
    #
    #
    # Example Code for minimize:
    #
    #     # Set Initial theta
    #     initial_theta = zeros(n + 1)
    #
    #     # Run minimize to obtain the optimal theta
    #     # This function will return a Result object. Theta can be retrieved in
    #     # the 'x' attribute and the cost in the 'fun' attribute.
    #     res = optimize.minimize(lrCostFunction, initial_theta, args=(X,(y == c),lambda_), \
    #                             method='CG', jac=True, options={'maxiter':50})
    #     theta, cost = res.x, res.fun




    # =========================================================================

    return all_theta


def lrCostFunction(theta, X, y, lambda_):
    #LRCOSTFUNCTION Compute cost and gradient for logistic regression with
    #regularization
    #   J, grad = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
    #   theta as the parameter for regularized logistic regression and the
    #   gradient of the cost w.r.t. to the parameters.

    # Initialize some useful values
    m = len(y) # number of training examples

    # You need to return the following variables correctly
    J = 0
    grad = zeros(shape(theta))

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta.
    #               You should set J to the cost.
    #               Compute the partial derivatives and set grad to the partial
    #               derivatives of the cost w.r.t. each parameter in theta
    #
    # Hint: The computation of the cost function and gradients can be
    #       efficiently vectorized. For example, consider the computation
    #
    #           sigmoid(dot(X, theta))
    #
    #       Each element of the resulting array will contain the value of the
    #       prediction for that example. You can make use of this to vectorize
    #       the cost function and gradient computations.
    #
    # Hint: When computing the gradient of the regularized cost function,
    #       there're many possible vectorized solutions, but one solution
    #       looks like:
    #           grad = (unregularized gradient for logistic regression)
    #           temp = theta.copy()
    #           temp[0] = 0    # because we don't add anything for j = 0
    #           grad = grad + YOUR_CODE_HERE (using the temp variable)
    #




    # =============================================================

    return J, grad

def predict(Theta1, Theta2, X):
    #PREDICT Predict the label of an input given a trained neural network
    #   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
    #   trained weights of a neural network (Theta1, Theta2)

    # Useful values
    m = size(X, 0)
    num_labels = size(Theta2, 0)

    # You need to return the following variables correctly
    p = zeros(m)

    # ====================== YOUR CODE HERE ======================
    # Instructions: Complete the following code to make predictions using
    #               your learned neural network. You should set p to a
    #               vector containing labels between 1 to num_labels.
    #
    # Hint: The argmax function might come in useful. In particular, the argmax
    #       function returns the index of the max element, for more information
    #       see 'help(argmax)'. If your examples are in rows, then, you can
    #       use argmax(A, 1) to obtain the max for each row.
    #



    # =========================================================================
    return p


def predictOneVsAll(all_theta, X):
    #PREDICT Predict the label for a trained one-vs-all classifier. The labels
    #are in the range 1..K, where K = size(all_theta, 0).
    #  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
    #  for each example in the matrix X. Note that X contains the examples in
    #  rows. all_theta is a matrix where the i-th row is a trained logistic
    #  regression theta vector for the i-th class. You should set p to a vector
    #  of values from 1..K (e.g., p = [1, 3, 1, 2] predicts classes 1, 3, 1, 2
    #  for 4 examples)

    m = size(X, 0)
    num_labels = size(all_theta, 0)

    # You need to return the following variables correctly
    p = zeros(m)

    # Add ones to the X data matrix
    X = column_stack((ones(m), X))

    # ====================== YOUR CODE HERE ======================
    # Instructions: Complete the following code to make predictions using
    #               your learned logistic regression parameters (one-vs-all).
    #               You should set p to a vector of predictions (from 1 to
    #               num_labels).
    #
    # Hint: This code can be done all vectorized using the argmax function.
    #       In particular, the argmax function returns the index of the
    #       max element, for more information see 'help(argmax)'. If your examples
    #       are in rows, then, you can use argmax(A, 1) to obtain the max
    #       for each row.
    #



    # =========================================================================

    return p

