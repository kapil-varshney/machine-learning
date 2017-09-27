import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


class Ex1:

    # Machine Learning Online Class - Exercise 1: Linear Regression

    #  Instructions
    # ------------

    # This file contains code that helps you get started on the
    # linear exercise. You will need to complete the following functions
    # in this exercise:

    # warmUpExercise
    # plotData
    # gradientDescent
    # computeCost
    # gradientDescentMulti
    # computeCostMulti
    # featureNormalize
    # normalEqn

    # x refers to the population size in 10,000s
    # y refers to the profit in $10,000s

    def warmUpExercise(self):
        # ============= YOUR CODE HERE ==============
        # Instructions: Return the 5x5 identity matrix
        return np.identity(5)

    def plotData(self, x, y):

        # PLOTDATA Plots the data points x and y into a new figure
        # PLOTDATA(x,y) plots the data points and gives the figure axes labels of
        # population and profit.
        #
        #
        # ====================== YOUR CODE HERE ======================
        # Instructions: Plot the training data into a figure using the
        #               "figure" and "plot" commands. Set the axes labels using
        #               the "xlabel" and "ylabel" commands. Assume the
        #               population and revenue data have been passed in
        #               as the x and y arguments of this function.
        fig = plt.figure()
        plt.plot(x, y,'rx',markersize=10)
        plt.grid(True)
        plt.xlabel('Population (in 10,000s)')
        plt.ylabel('Revenue (in $10,000s')
        plt.show()


    def computeCost(self, X, y, theta):

        # Compute cost for linear regression
        # J = computeCost(X, y, theta) computes the cost of using theta as the
        # parameter for linear regression to fit the data points in X and y

        # Initialize some useful values
        m = len(y)  # number of training examples

        # You need to return the following variables correctly
        J = 0

        # ====================== YOUR CODE HERE ======================
        # Instructions: Compute the cost of a particular choice of theta
        #               You should set J to the cost.

        J = (1/(2*m)) * np.square(np.dot(X, theta) - y).sum()
        return J

        # =========================================================================

    def gradientDescent(self, X, y, theta, alpha, num_iters):
        # Gradient Descent Performs gradient descent to learn theta
        # theta = gradientDescent(X, y, theta, alpha, num_iters) updates theta by
        # taking num_iters gradient steps with learning rate alpha

        # Initialize some useful values
        m = len(y) # number of training examples
        J_history = np.zeros((num_iters, 1))

        for iter in range(num_iters):

            # ====================== YOUR CODE HERE ======================
            # Instructions: Perform a single gradient step on the parameter vector
            #               theta.
            #
            # Hint: While debugging, it can be useful to print out the values
            #       of the cost function (computeCost) and gradient here.
            temp_theta0 = theta[0,0] - (alpha/m) * ((np.dot(X, theta) - y) * (X[:, 0].reshape(m, 1))).sum()
            temp_theta1 = theta[1,0] - (alpha/m) * ((np.dot(X, theta) - y) * (X[:, 1].reshape(m, 1))).sum()

            theta[0,0] = temp_theta0
            theta[1,0] = temp_theta1
            #print(theta)
            # ============================================================

            # Save the cost J in every iteration
            J_history[iter] = self.computeCost(X, y, theta)
        return theta


    def main(self):

        # ==================== Part 1: Basic Function ====================
        # Complete warmUpExercise()
        print ('Running warmUpExercise ... \n')
        print ('5x5 Identity Matrix: \n')
        #print(self.warmUpExercise())

        print ('Program paused. Press enter to continue.\n')
        input()

        # ======================= Part 2: Plotting =======================
        print ('Plotting Data ...\n')
        data = np.loadtxt('ex1data1.txt', delimiter = ',')
        X = data[:, 0]
        y = data[:, 1]
        m = len(y) # number of training examples

        # Plot Data
        # Note: You have to complete the code in plotData()
        self.plotData(X, y)

        print ('Program paused. Press enter to continue.\n')
        input()

        # =================== Part 3: Cost and Gradient descent ===================

        # Add a column of ones to x
        X = np.concatenate((np.ones((m,1)), data[:,0].reshape(97,1)), axis = 1)
        # Initialize fitting parameters
        theta = np.zeros((2,1))

        # Some gradient descent settings
        iterations = 1500
        alpha = 0.01

        # Compute and Display initial Cost
        print('\nTesting the cost function ...\n')

        # Reshape y to be a vector of size (mx1)
        J = self.computeCost(X, y.reshape(m,1), theta)
        print('With theta = [0 ; 0]\nCost computed =',J)
        print('\nExpected cost value (approx) 32.07\n')

        # Further testing of the Cost function
        J = self.computeCost(X, y.reshape(m,1), np.array([[-1],[2]]))
        print('With theta = [-1 ; 2]\nCost computed =', J)
        print('\nExpected cost value (approx) 54.24\n')

        print('Program paused. Press enter to continue.\n')

        # Run Gradient Descent
        print('\nRunning Gradient Descent ...\n')
        theta = self.gradientDescent(X, y.reshape(m,1), theta, alpha, iterations)

        # Print theta to screen
        print('Theta found by gradient descent:\n')
        print(theta)
        print('\nExpected theta values (approx)\n')
        print(' -3.6303\n  1.1664\n\n')

        # Plot the linear fit
        plt.figure()
        plt.plot(X[:,1], y, 'rx', markersize = 10, label = 'Training Data')
        plt.plot(X[:,1], X.dot(theta), 'b-', label = 'Hypothesis: h(x) = %0.2f + %0.2f' %(theta[0,0], theta[1,0]))
        plt.grid(True)
        plt.ylabel('Profit in $10,000s')
        plt.xlabel('Population of city in 10,000s')
        plt.legend()
        plt.show()

        # Predict values for population sizes of 35,000 and 70,000
        predict1 = np.array([1, 3.5]).reshape(1,2).dot(theta)
        print('For population = 35,000, we predict a profit of \n', predict1*10000)
        predict2 = np.array([1, 7]).reshape(1,2).dot(theta)
        print('For population = 70,000, we predict a profit of \n', predict2*10000)

        print('Program paused. Press enter to continue.\n')

        # ============= Part 4: Visualizing J(theta_0, theta_1) =============
        print('Visualizing J(theta_0, theta_1) ...\n')

        # Grid over which we will calculate J
        theta0_vals = np.linspace(-10, 10, 100)
        theta1_vals = np.linspace(-1, 4, 100)

        # initialize J_vals to a matrix of 0's
        J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

        # Fill out J_vals
        for i in range (len(theta0_vals)):
            for j in range(len(theta1_vals)):
                t = np.array([[theta0_vals[i]], [theta1_vals[j]]])
                J_vals[i,j] = self.computeCost(X, y.reshape(m,1), t)

        # Surface plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        a, b = np.meshgrid(theta0_vals, theta1_vals)
        ax.plot_surface(a, b, J_vals, cmap = cm.coolwarm)
        ax.set_xlabel(r'$\theta_0$', fontsize = 15)
        ax.set_ylabel(r'$\theta_1$', fontsize = 15)
        plt.show()

        # Contour plot (Note to self: Contours don't seem to be accurate)
        fig = plt.figure()
        cp = plt.contour(a, b, J_vals, levels = np.logspace(-2, 3, 20))
        plt.plot(theta[0,0],theta[1,0], 'rx', markersize = 10, linewidth =2)
        plt.xlabel(r'$\theta_0$', fontsize = 15)
        plt.ylabel(r'$\theta_1$', fontsize = 15)
        plt.show()

if __name__ == '__main__':
    Ex1().main()
