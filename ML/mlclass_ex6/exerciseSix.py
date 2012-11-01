# This Python file uses the following encoding: utf-8
import pdb

from os import path
import re

from functools import wraps

from numpy import *
from numpy.linalg import *

from matplotlib.pyplot import * 

from scipy import optimize

from porter import PorterStemmer

from svmTrain import svmTrain
from svmPredict import svmPredict

from numpy import *
from matplotlib.pyplot import *

def plotData(X, y):
    #PLOTDATA Plots the data points X and y into a new figure
    #   PLOTDATA(x,y) plots the data points with + for the positive examples
    #   and o for the negative examples. X is assumed to be a Mx2 matrix.
    #
    # Note: This was slightly modified such that it expects y = 1 or y = 0

    # Find Indices of Positive and Negative Examples
    pos = where(y == 1)
    neg = where(y == 0)

    # Plot Examples
    plot(X[pos, 0], X[pos, 1], 'k+',linewidth=1, markersize=7)
    hold(True)
    plot(X[neg, 0], X[neg, 1], 'ko', markerfacecolor='y', markersize=7)
    hold(False)


def visualizeBoundary(X, y, model):
    #VISUALIZEBOUNDARY plots a non-linear decision boundary learned by the SVM
    #   VISUALIZEBOUNDARYLINEAR(X, y, model) plots a non-linear decision
    #   boundary learned by the SVM and overlays the data on it

    # Plot the training data on top of the boundary
    plotData(X, y)

    # Make classification predictions over a grid of values
    x1plot = linspace(min(X[:,0]), max(X[:,0]), 100)
    x2plot = linspace(min(X[:,1]), max(X[:,1]), 100)
    [X1, X2] = meshgrid(x1plot, x2plot)
    vals = zeros(shape(X1))
    for i in range(size(X1, 1)):
       this_X = column_stack((X1[:, i], X2[:, i]))
       vals[:, i] = svmPredict(model, this_X)

    # Plot the SVM boundary
    hold(True)
    contour(X1, X2, vals, [0], color='b')
    hold(False)

def visualizeBoundaryLinear(X, y, model):
    #VISUALIZEBOUNDARYLINEAR plots a linear decision boundary learned by the
    #SVM
    #   VISUALIZEBOUNDARYLINEAR(X, y, model) plots a linear decision boundary
    #   learned by the SVM and overlays the data on it
    w = model.w
    b = model.b
    xp = linspace(min(X[:,0]), max(X[:,0]), 100)
    yp = - (w[0]*xp + b) / w[1]
    plotData(X, y)
    hold(True)
    plot(xp, yp, '-b')
    hold(False)



def getVocabDict():
    #GETVOCABDICT reads the fixed vocabulary list in vocab.txt and returns a
    #dictionary mapping words to integers
    #   vocabList = GETVOCABDICT() reads the fixed vocabulary list in vocab.txt
    #   and returns a dictionary of the words.

    ## Read the fixed vocabulary list
    with open('vocab.txt') as f:
        # Store all dictionary words in a python dict which maps strings to integers
        vocab = {}
        for line in f:
            id, word = line.split()
            vocab[word] = int(id)

    return vocab

def processEmail(email_contents):
    #PROCESSEMAIL preprocesses a the body of an email and
    #returns a list of word_indices
    #   word_indices = PROCESSEMAIL(email_contents) preprocesses
    #   the body of an email and returns a list of indices of the
    #   words contained in the email.
    #

    # Load Vocabulary
    vocab = getVocabDict()

    # Init return value
    word_indices = []

    # ========================== Preprocess Email ===========================

    # Find the Headers ( \n\n and remove )
    # Uncomment the following lines if you are working with raw emails with the
    # full headers

    # hdrstart = email_contents.find('\n\n')
    # email_contents = email_contents[hdrstart+2:]

    # Lower case
    email_contents = email_contents.lower()

    # Strip all HTML
    # Looks for any expression that starts with < and ends with >
    # and does not have any < or > in the tag and replace it with a space
    email_contents = re.sub('<[^<>]+>', ' ', email_contents)

    # Handle Numbers
    # Look for one or more characters between 0-9
    email_contents = re.sub('[0-9]+', 'number', email_contents)

    # Handle URLS
    # Look for strings starting with http:// or https://
    email_contents = re.sub('(http|https)://[^\s]*', 'httpaddr', email_contents)

    # Handle Email Addresses
    # Look for strings with @ in the middle
    email_contents = re.sub('[^\s]+@[^\s]+', 'emailaddr', email_contents)

    # Handle $ sign
    email_contents = re.sub('[$]+', 'dollar', email_contents)


    # ========================== Tokenize Email ===========================

    # Output the email to screen as well
    print '\n==== Processed Email ====\n'

    # Process file
    l = 0
    porterStemmer = PorterStemmer()
    # Tokenize and also get rid of any punctuation
    sep = '[ \@\$\/\#\.\-\:\&\*\+\=\[\]\?\!\(\)\{\},\'\"\>\_\<\;\%\n\r]+'
    for s in re.split(sep, email_contents):
        # Remove any non alphanumeric characters
        s = re.sub('[^a-zA-Z0-9]', '', s)

        # Stem the word
        s = porterStemmer.stem(s.strip())

        # Skip the word if it is too short
        if len(s) < 1:
           continue

        # Look up the word in the dictionary and add to word_indices if
        # found
        # ====================== YOUR CODE HERE ======================
        # Instructions: Fill in this function to add the index of str to
        #               word_indices if it is in the vocabulary. At this point
        #               of the code, you have a stemmed word from the email in
        #               the variable s. You should look up s in the
        #               vocabulary dictionary (vocab). If a match exists, you
        #               should add the index of the word to the word_indices
        #               vector. Concretely, if s = 'action', then you should
        #               add to word_indices the value under the key 'action'
        #               in vocab. For example, if vocab['action'] = 18, then,
        #               you should add 18 to the word_indices vector
        #               (e.g., word_indices.append(18) ).
        #




        # =============================================================


        # Print to screen, ensuring that the output lines are not too long
        if l + len(s) + 1 > 78:
            print
            l = 0
        print s,
        l += len(s) + 1

    # Print footer
    print '\n========================='

    return array(word_indices)



def dataset3Params(X, y, Xval, yval):
    #EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
    #where you select the optimal (C, sigma) learning parameters to use for SVM
    #with RBF kernel
    #   C, sigma = EX6PARAMS(X, y, Xval, yval) returns your choice of C and
    #   sigma. You should complete this function to return the optimal C and
    #   sigma based on a cross-validation set.
    #

    # You need to return the following variables correctly.
    C = 1.0
    sigma = 0.3

    # ====================== YOUR CODE HERE ======================
    # Instructions: Fill in this function to return the optimal C and sigma
    #               learning parameters found using the cross validation set.
    #               You can use svmPredict to predict the labels on the cross
    #               validation set. For example,
    #                   predictions = svmPredict(model, Xval)
    #               will return the predictions on the cross validation set.
    #
    #  Note: You can compute the prediction error using
    #        mean(predictions != yval)
    #



    # =========================================================================

    return C, sigma


def linearKernel(x1, x2):
    #LINEARKERNEL returns a linear kernel between x1 and x2
    #   sim = linearKernel(x1, x2) returns a linear kernel between x1 and x2
    #   and returns the value in sim

    # Ensure that x1 and x2 are vectors
    x1 = ravel(x1, order='F')
    x2 = ravel(x2, order='F')

    # Compute the kernel
    sim = dot(x1,x2)      # dot product

    return sim

def gaussianKernel(x1, x2, sigma):
    #RBFKERNEL returns a radial basis function kernel between x1 and x2
    #   sim = gaussianKernel(x1, x2) returns a gaussian kernel between x1 and x2
    #   and returns the value in sim

    # Ensure that x1 and x2 are column vectors
    x1 = ravel(x1, order='F')
    x2 = ravel(x2, order='F')

    # You need to return the following variables correctly.
    sim = 0.

    # ====================== YOUR CODE HERE ======================
    # Instructions: Fill in this function to return the similarity between x1
    #               and x2 computed using a Gaussian kernel with bandwidth
    #               sigma
    #
    #




    # =============================================================

    return sim


