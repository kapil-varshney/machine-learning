#  Machine Learning Online Class
#  Exercise 1: Linear regression with multiple variables
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  linear regression exercise.
#
#  You will need to complete the following functions in this
#  exericse:
#
#
#     plotData.m
#     gradientDescentMulti.m
#     computeCostMulti.m
#     featureNormalize.m
#     normalEqn.m
#
#  For this part of the exercise, you will need to change some
#  parts of the code below for various experiments (e.g., changing
#  learning rates).
#

# Initialization
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# ================ Part 1: Feature Normalization ================


def feature_normalize(X):

    #   featureNormalize Normalizes the features in X
    #   It returns a normalized version of X where
    #   the mean value of each feature is 0 and the standard deviation
    #   is 1. This is often a good preprocessing step to do when
    #   working with learning algorithms.

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
    X_norm = np.copy(X)
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)

    for i in range(X.shape[1]):
        X_norm[:,i] = (X[:, i] - mu[i]) / sigma[i]

    return X_norm, mu, sigma

    # ============================================================

print('Loading data ...\n')

# Load Data
data = np.loadtxt('ex1data2.txt', delimiter=',')
n = data.shape[1]
X = data[:, 0:(n-1)]
y = data[:, n]
m = len(y)  # number of training examples

# Print out some data points
print('First 10 examples from the dataset: \n')
print(data[0:10,:])

print('Program paused. Press enter to continue.\n')
input()

# Scale features and set them to zero mean
print('Normalizing Features ...\n')

X, mu, sigma = feature_normalize(X)

# Add intercept term to X
X = np.hstack((np.ones((m,1)), X))

'''
 ================ Part 2: Gradient Descent ================

% ====================== YOUR CODE HERE ======================
% Instructions: We have provided you with the following starter
%               code that runs gradient descent with a particular
%               learning rate (alpha). 
%
%               Your task is to first make sure that your functions - 
%               computeCost and gradientDescent already work with 
%               this starter code and support multiple variables.
%
%               After that, try running gradient descent with 
%               different values of alpha and see which one gives
%               you the best result.
%
%               Finally, you should complete the code at the end
%               to predict the price of a 1650 sq-ft, 3 br house.
%
% Hint: By using the 'hold on' command, you can plot multiple
%       graphs on the same figure.
%
% Hint: At prediction, make sure you do the same feature normalization.
%
'''


def compute_cost_function(X, y, theta):

    """
    Compute cost for linear regression with multiple variables
    This function computes the cost of using theta as the
    parameter for linear regression to fit the data points in X and y
    """

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta
    #               You should set J to the cost.

    J = (0.5 / len(y)) * (np.square(X.dot(theta) - y)).sum()
    return J

    # =========================================================================

def gradient_descent_multi(X,y,theta, alpha, num_iters):

    """
    Performs gradient descent to learn theta
    It updates theta by
    taking num_iters gradient steps with learning rate alpha
    """

    m = len(y)  # number of training examples
    J_history = np.zeros((num_iters, 1))

    for i in range(num_iters):

        # ====================== YOUR CODE HERE ======================
        # Instructions: Perform a single gradient step on the parameter vector
        #               theta.
        #
        # Hint: While debugging, it can be useful to print out the values
        #       of the cost function (compute_cost_multi) and gradient here.
        #

        # ============================================================

        # Save the cost J in every iteration
        J_history(iter) = computeCostMulti(X, y, theta);

print('Running gradient descent ...\n')

# Choose some alpha value
alpha = 0.01
num_iters = 400

# Init Theta and Run Gradient Descent
theta = np.zeros((n, 1))
[theta, J_history] = gradient_descent_multi(X, y, theta, alpha, num_iters)

# Plot the convergence graph
fig = plt.figure()
#plt(1:numel(J_history), J_history, '-b', linewidth=2)
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')

# Display gradient descent's result
print('Theta computed from gradient descent: \n')
print(' %f \n', theta)
print('\n')

# Estimate the price of a 1650 sq-ft, 3 br house
# ====================== YOUR CODE HERE ======================
# Recall that the first column of X is all-ones. Thus, it does
# not need to be normalized.
price = 0  # You should change this


# ============================================================

print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):\n $%f\n' %price)

print('Program paused. Press enter to continue.\n')
input()