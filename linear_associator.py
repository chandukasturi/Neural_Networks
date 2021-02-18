# Kasturi, Chandra Shekhar
# 1001-825-454
# 2020-10-11
# Assignment-02-01

import numpy as np


class LinearAssociator(object):
    def __init__(self, input_dimensions=2, number_of_nodes=4, transfer_function="Hard_limit"):
        """
        Initialize linear associator model
        :param input_dimensions: The number of dimensions of the input data
        :param number_of_nodes: number of neurons in the model
        :param transfer_function: Transfer function for each neuron. Possible values are:
        "Hard_limit", "Linear".
        """
        self.input_Dimensions = input_dimensions
        self.number_of_Nodes = number_of_nodes
        self.transferFunction = transfer_function
        self.weights = np.array([],[])
        self.initialize_weights()

    def initialize_weights(self, seed=None):
        """
        Initialize the weights, initalize using random numbers.
        If seed is given, then this function should
        use the seed to initialize the weights to random numbers.
        :param seed: Random number generator seed.
        :return: None
        """
        if seed == None:
            """Does not change the seed value to any random number as it is specified none"""
            seed = None
        else:
            seed = np.random.seed(seed)
        weight = np.random.randn(self.number_of_Nodes, self.input_Dimensions)
        self.weights = weight

    def set_weights(self, W):
        """
         This function sets the weight matrix.
         :param W: weight matrix
         :return: None if the input matrix, w, has the correct shape.
         If the weight matrix does not have the correct shape, this function
         should not change the weight matrix and it should return -1.
         """
        weight = np.random.randn(self.number_of_Nodes, self.input_Dimensions)
        if (W.shape == weight.shape):
            self.weights = W
        else:
            return -1


    def get_weights(self):
        """
         This function should return the weight matrix(Bias is included in the weight matrix).
         :return: Weight matrix
         """
        return self.weights
    def predict(self, X):
        """
        Make a prediction on an array of inputs
        :param X: Array of input [input_dimensions,n_samples].
        :return: Array of model outputs [number_of_nodes ,n_samples]. This array is a numerical array.
        """
        net = (np.dot(self.weights, X))
        if self.transferFunction == 'Hard_limit':
            actual_value = (net >= 0).astype(int)
        elif self.transferFunction == 'Linear':
            actual_value = net
        return actual_value


    def fit_pseudo_inverse(self, X, y):
        """
        Given a batch of data, and the targets,
        this function adjusts the weights using pseudo-inverse rule.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes,n_samples]
        """
        if X.shape[0] < X.shape[1]:
            X = X.T
            XP = np.dot(np.linalg.inv(np.dot(X.T, X)), X.T)
            self.weights = np.dot(y, XP.T)
        else:
            XP = np.dot(np.linalg.inv(np.dot(X.T, X)), X.T)
            self.weights = np.dot(y, XP)

    def train(self, X, y, batch_size=5, num_epochs=50, alpha=0.4, gamma=0.9, learning="Delta"):
        """
        Given a batch of data, and the necessary hyperparameters,
        this function adjusts the weights using the learning rule.
        Training should be repeated num_epochs time.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes,n_samples].
        :param num_epochs: Number of times training should be repeated over all input data
        :param batch_size: number of samples in a batch
        :param num_epochs: Number of times training should be repeated over all input data
        :param alpha: Learning rate
        :param gamma: Controls the decay
        :param learning: Learning rule. Possible methods are: "Filtered", "Delta", "Unsupervised_hebb"
        :return: None
        """
        X_split = np.array_split(X, batch_size, 1)
        y_split = np.array_split(y, batch_size, 1)
        for i in range(num_epochs):
            for b in range(batch_size):
                actual_value = self.predict(X_split[b])
                if learning.lower() == 'delta':
                    error = y_split[b] - actual_value
                    self.weights = self.weights +  alpha * np.dot(error, X_split[b].T)
                elif learning.lower == 'unsupervised_hebb':
                    self.weights = self.weights + alpha * np.dot(actual_value, X_split[b].T)
                elif learning.lower == 'filtered':
                    self.weights = self.weights + alpha * np.dot(y_split[b], X_split[b].T)
                    self.weights = (1 - gamma) * self.weights


    def calculate_mean_squared_error(self, X, y):
        """
        Given a batch of data, and the targets,
        this function calculates the mean squared error (MSE).
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes,n_samples]
        :return mean_squared_error
        """
        actual_value = self.predict(X)
        error = y - actual_value
        mean_squared_error = np.mean(error ** 2)
        return mean_squared_error