import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Perceptron:
    """
    eta = learning rate
    n_iter = iterations
    w = weights
        w[0] = bias
        w[1:] = features
    """
    def __init__(self, file_name = 'iris.data', eta = 0.1, n_iter = 10):
        self.eta = eta
        self.errors = []
        self.n_iter = n_iter        

        """ building dataframe """
        df = pd.read_csv(file_name)
            
        """ building an array of -1 and 1, setosa(-1) from versicolor(1) """
        y = df.iloc[0:100, 4].values        
        y = np.where(y=='Iris-setosa', -1, 1)
        
        """ building features matrix, sepal.length and petal.length """
        self.X = df.iloc[0:100, [0, 2]].values        

        """ initial weights are zero """
        self.w = np.zeros(self.X.shape[1]+1)

        """ predicting and updating the weights in case of any error """
        for _ in range(n_iter):
            error_count = 0        
            for xi, target in zip(self.X,y):
                predicted_value = self.predict(xi)
                self.update_weights(target, predicted_value, xi)
                error_count += int(self.w[0] != 0.0)        
            self.errors.append(error_count)


    def predict(self, xi):
        """unit step function used as activation"""
        net_input = np.dot(xi, self.w[1:])+self.w[0]
        predicted_value = np.where(net_input >= 0.0, 1, -1)        
        return predicted_value


    def update_weights(self, target, predicted_value, xi):
        """updating weights if predicted error"""
        w_delta = self.eta * (target - predicted_value)    
        self.w[1:] += w_delta * xi
        self.w[0] += w_delta
        return w_delta


    def plot(self):
        """Ploting scatterplot of data"""
        plt.scatter(self.X[:50, 0], self.X[:50, 1], color = 'red', marker = 'o', 
            label = 'setosa')
        plt.scatter(self.X[50:100, 0], self.X[50:100, 1], color = 'blue', marker = 'x', 
            label = 'versicolor')
        plt.xlabel('petal length')
        plt.ylabel('versicolor length')
        plt.legend(loc='lower right')

        """
        plotting line
        to plot the line we need to calculate it's slope and y-intercept
        the formula described below gives us what we need:
        x2 = -(w1/w2)x1-(b/w2)
        """
        slope = -1*(self.w[1]/self.w[2])
        intercept = -1*(self.w[0]/self.w[2])        
        axes = plt.gca()
        x_vals = np.array(axes.get_xlim())
        y_vals = intercept + slope * x_vals
        plt.plot(x_vals, y_vals, '-')
        plt.show()


    def plot_errors(self):
        """ plotting errors """
        plt.plot(range(1, len(self.errors)+1), self.errors, marker='o')
        plt.xlabel('epochs')
        plt.ylabel('misclassifications')
        plt.show()

