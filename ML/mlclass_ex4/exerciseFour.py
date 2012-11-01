# This Python file uses the following encoding: utf-8
import pdb

from numpy import *
from numpy.linalg import *

from matplotlib.pyplot import * 

from scipy import optimize

from nnCostFunction import nnCostFunction

def sigmoid(z):
    #SIGMOID Compute sigmoid functoon
    #   J = SIGMOID(z) computes the sigmoid of z.
    J = 1. / (1. + exp(-z))
    return J

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


def checkNNGradients(lambda_=0):
    #CHECKNNGRADIENTS Creates a small neural network to check the
    #backpropagation gradients
    #   CHECKNNGRADIENTS(lambda_) Creates a small neural network to check the
    #   backpropagation gradients, it will output the analytical gradients
    #   produced by your backprop code and the numerical gradients (computed
    #   using computeNumericalGradient). These two gradient computations should
    #   result in very similar values.
    #

    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5

    # We generate some 'random' test data
    Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size)
    Theta2 = debugInitializeWeights(num_labels, hidden_layer_size)
    # Reusing debugInitializeWeights to generate X
    X  = debugInitializeWeights(m, input_layer_size - 1)
    y  = 1 + (arange(m)+1) % num_labels

    # Unroll parameters
    nn_params = hstack((Theta1.ravel(order='F'), Theta2.ravel(order='F')))

    # Short hand for cost function
    costFunc = lambda p: nnCostFunction(p, input_layer_size, hidden_layer_size,
                                        num_labels, X, y, lambda_)

    cost, grad = costFunc(nn_params)
    numgrad = computeNumericalGradient(costFunc, nn_params)

    # Visually examine the two gradient computations.  The two columns
    # you get should be very similar.
    print column_stack((numgrad, grad))
    print 'The above two columns you get should be very similar.'
    print '(Left-Your Numerical Gradient, Right-Analytical Gradient)\n'

    # Evaluate the norm of the difference between two solutions.
    # If you have a correct implementation, and assuming you used EPSILON = 0.0001
    # in computeNumericalGradient.m, then diff below should be less than 1e-9
    diff = linalg.norm(numgrad-grad) / linalg.norm(numgrad+grad)

    print 'If your backpropagation implementation is correct, then'
    print 'the relative difference will be small (less than 1e-9).\n'
    print 'Relative Difference: %g\n' % diff


def debugInitializeWeights(fan_out, fan_in):
    #DEBUGINITIALIZEWEIGHTS Initialize the weights of a layer with fan_in
    #incoming connections and fan_out outgoing connections using a fixed
    #strategy, this will help you later in debugging
    #   W = DEBUGINITIALIZEWEIGHTS(fan_in, fan_out) initializes the weights
    #   of a layer with fan_in incoming connections and fan_out outgoing
    #   connections using a fix set of values
    #
    #   Note that W should be set to a matrix of size(1 + fan_in, fan_out) as
    #   the first row of W handles the "bias" terms
    #

    # Set W to zeros
    W = zeros((fan_out, 1 + fan_in))

    # Initialize W using "sin", this ensures that W is always of the same
    # values and will be useful for debugging
    W = reshape(sin(arange(W.size)+1), shape(W), order='F') / 10

    # =========================================================================

    return W


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


def nnCostFunction(nn_params,
                   input_layer_size,
                   hidden_layer_size,
                   num_labels,
                   X, y, lambda_):
    #NNCOSTFUNCTION Implements the neural network cost function for a two layer
    #neural network which performs classification
    #   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
    #   X, y, lambda) computes the cost and gradient of the neural network. The
    #   parameters for the neural network are "unrolled" into the vector
    #   nn_params and need to be converted back into the weight matrices.
    #
    #   The returned parameter grad should be a "unrolled" vector of the
    #   partial derivatives of the neural network.
    #

    # Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
    # for our 2 layer neural network

    Theta1 = reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                     (hidden_layer_size, input_layer_size + 1), order='F')

    Theta2 = reshape(nn_params[hidden_layer_size * (input_layer_size + 1):],
                     (num_labels, hidden_layer_size + 1), order='F')

    # Setup some useful variables
    m = size(X, 0)

    # You need to return the following variables correctly
    J = 0
    Theta1_grad = zeros(shape(Theta1))
    Theta2_grad = zeros(shape(Theta2))

    # ====================== YOUR CODE HERE ======================
    # Instructions: You should complete the code by working through the
    #               following parts.
    #
    # Part 1: Feedforward the neural network and return the cost in the
    #         variable J. After implementing Part 1, you can verify that your
    #         cost function computation is correct by verifying the cost
    #         computed in ex4.m
    #
    # Part 2: Implement the backpropagation algorithm to compute the gradients
    #         Theta1_grad and Theta2_grad. You should return the partial derivatives of
    #         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
    #         Theta2_grad, respectively. After implementing Part 2, you can check
    #         that your implementation is correct by running checkNNGradients
    #
    #         Note: The vector y passed into the function is a vector of labels
    #               containing values from 1..K. You need to map this vector into a
    #               binary vector of 1's and 0's to be used with the neural network
    #               cost function.
    #
    #         Hint: We recommend implementing backpropagation using a for-loop
    #               over the training examples if you are implementing it for the
    #               first time.
    #
    # Part 3: Implement regularization with the cost function and gradients.
    #
    #         Hint: You can implement this around the code for
    #               backpropagation. That is, you can compute the gradients for
    #               the regularization separately and then add them to Theta1_grad
    #               and Theta2_grad from Part 2.
    # -------------------------------------------------------------

    # =========================================================================

    # Unroll gradients
    grad = hstack((Theta1_grad.ravel(order='F'), Theta2_grad.ravel(order='F')))

    return J, grad

def predict(Theta1, Theta2, X):
    #PREDICT Predict the label of an input given a trained neural network
    #   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
    #   trained weights of a neural network (Theta1, Theta2)

    # Useful values
    m = size(X, 0)
    num_labels = size(Theta2, 0)

    # You need to return the following variables correctly
    p = zeros(size(X, 0))

    h1 = sigmoid(dot(column_stack((ones(m), X)), Theta1.T))
    h2 = sigmoid(dot(column_stack((ones(m), h1)), Theta2.T))
    p = argmax(h2, 1) + 1

    # =========================================================================
    return p

def sigmoidGradient(z):
    #SIGMOIDGRADIENT returns the gradient of the sigmoid function
    #evaluated at z
    #   g = SIGMOIDGRADIENT(z) computes the gradient of the sigmoid function
    #   evaluated at z. This should work regardless if z is a matrix or a
    #   vector. In particular, if z is a vector or matrix, you should return
    #   the gradient for each element.

    g = zeros(shape(z))

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the gradient of the sigmoid function evaluated at
    #               each value of z (z can be a matrix, vector or scalar).



    # =============================================================
    return g


def randInitializeWeights(L_in, L_out):
    #RANDINITIALIZEWEIGHTS Randomly initialize the weights of a layer with L_in
    #incoming connections and L_out outgoing connections
    #   W = RANDINITIALIZEWEIGHTS(L_in, L_out) randomly initializes the weights
    #   of a layer with L_in incoming connections and L_out outgoing
    #   connections.
    #
    #   Note that W should be set to a matrix of size(L_out, 1 + L_in) as
    #   the first row of W handles the "bias" terms
    #

    # You need to return the following variables correctly
    W = zeros((L_out, 1 + L_in))

    # ====================== YOUR CODE HERE ======================
    # Instructions: Initialize W randomly so that we break the symmetry while
    #               training the neural network.
    #
    # Note: The first row of W corresponds to the parameters for the bias units
    #





    # =========================================================================

    return W
