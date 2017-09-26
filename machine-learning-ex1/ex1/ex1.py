import numpy as np
import matplotlib.pyplot as plt


class Ex1:

    # Machine Learning Online Class - Exercise 1: Linear Regression

    #  Instructions
    # ------------

    # This file contains code that helps you get started on the
    # linear exercise. You will need to complete the following functions
    # in this exercise:

    # warmUpExercise.m
    # plotData.m
    # gradientDescent.m
    # computeCost.m
    # gradientDescentMulti.m
    # computeCostMulti.m
    # featureNormalize.m
    # normalEqn.m

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
        # J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
        # parameter for linear regression to fit the data points in X and y

        # Initialize some useful values
        m = len(y) # number of training examples

        # You need to return the following variables correctly
        J = 0

        # ====================== YOUR CODE HERE ======================
        # Instructions: Compute the cost of a particular choice of theta
        #               You should set J to the cost.

        J = (1/(2*m)) * np.square(np.dot(X, theta) - y).sum()
        return J

        # =========================================================================


    def main(self):

        # ==================== Part 1: Basic Function ====================
        # Complete warmUpExercise()
        print ('Running warmUpExercise ... \n')
        print ('5x5 Identity Matrix: \n')
        print(self.warmUpExercise())

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

        '''
        fprintf('Program paused. Press enter to continue.\n');
        pause;
        
        fprintf('\nRunning Gradient Descent ...\n')
        % run gradient descent
        theta = gradientDescent(X, y, theta, alpha, iterations);
        
        % print theta to screen
        fprintf('Theta found by gradient descent:\n');
        fprintf('%f\n', theta);
        fprintf('Expected theta values (approx)\n');
        fprintf(' -3.6303\n  1.1664\n\n');
        
        % Plot the linear fit
        hold on; % keep previous plot visible
        plot(X(:,2), X*theta, '-')
        legend('Training data', 'Linear regression')
        hold off % don't overlay any more plots on this figure
        
        % Predict values for population sizes of 35,000 and 70,000
        predict1 = [1, 3.5] *theta;
        fprintf('For population = 35,000, we predict a profit of %f\n',...
            predict1*10000);
        predict2 = [1, 7] * theta;
        fprintf('For population = 70,000, we predict a profit of %f\n',...
            predict2*10000);
        
        fprintf('Program paused. Press enter to continue.\n');
        pause;
        '''

if __name__ == '__main__':
    Ex1().main()
