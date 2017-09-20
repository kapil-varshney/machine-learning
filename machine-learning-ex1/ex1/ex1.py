import numpy as np
import matplotlib as mpl


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

    def plotData(self):
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
        plotData(X, y)

        print ('Program paused. Press enter to continue.\n')
        input()


if __name__ == '__main__':
    Ex1().main()
