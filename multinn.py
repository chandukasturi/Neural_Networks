# Kasturi, Chandra Shekhar
# 1001-825-454
# 2020-10-25
# Assignment-03-01

# %tensorflow_version 2.x
import tensorflow as tf
import numpy as np


class MultiNN(object):
    def __init__(self, input_dimension):
        """
        Initialize multi-layer neural network
        :param input_dimension: The number of dimensions for each input data sample
        """
        self.inputDimension = input_dimension
        self.weights = []

    def add_layer(self, num_nodes, transfer_function="Linear"):
        """
         This function adds a dense layer to the neural network
         :param num_nodes: number of nodes in the layer
         :param transfer_function: Activation function for the layer. Possible values are:
        "Linear", "Relu","Sigmoid".
         :return: None
         """
        number_of_layers = len(self.weights)
        if number_of_layers > 0:
            self.weights.append(Multi_NN(self.weights[number_of_layers - 1].numberOfNodes, num_nodes, transfer_function))
        else:
            self.weights.append(Multi_NN(self.inputDimension, num_nodes, transfer_function))

    def get_weights_without_biases(self, layer_number):
        """
        This function should return the weight matrix (without biases) for layer layer_number.
        layer numbers start from zero.
         :param layer_number: Layer number starting from layer 0. This means that the first layer with
          activation function is layer zero
         :return: Weight matrix for the given layer (not including the biases). Note that the shape of the weight matrix should be
          [input_dimensions][number of nodes]
         """
        weights_without_biases = self.weights[layer_number].weights.numpy()
        return weights_without_biases

    def get_biases(self, layer_number):
        """
        This function should return the biases for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param layer_number: Layer number starting from layer 0
         :return: Weight matrix for the given layer (not including the biases).
         Note that the biases shape should be [1][number_of_nodes]
         """
        biases = self.weights[layer_number].biases.numpy()
        return biases
    def set_weights_without_biases(self, weights, layer_number):
        """
        This function sets the weight matrix for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param weights: weight matrix (without biases). Note that the shape of the weight matrix should be
          [input_dimensions][number of nodes]
         :param layer_number: Layer number starting from layer 0
         :return: none
         """
        self.weights[layer_number].weights.assign(weights)
    def set_biases(self, biases, layer_number):
        """
        This function sets the biases for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
        :param biases: biases. Note that the biases shape should be [1][number_of_nodes]
        :param layer_number: Layer number starting from layer 0
        :return: none
        """
        self.weights[layer_number].biases.assign(biases)

    def calculate_loss(self, y, y_hat):
        """
        This function calculates the sparse softmax cross entropy loss.
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
         the desired (true) class.
        :param y_hat: Array of actual output values [n_samples][number_of_classes].
        :return: loss
        """
        loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(y, y_hat)
        return loss

    def predict(self, X):
        """
        Given array of inputs, this function calculates the output of the multi-layer network.
        :param X: Array of input [n_samples,input_dimensions].
        :return: Array of outputs [n_samples,number_of_classes ]. This array is a numerical array.
        """
        number_of_layers = len(self.weights)
        for layerNumber in range(number_of_layers):
            predicted_X = self.weights[layerNumber].predict(X)
            X = predicted_X
        return X
    def train(self, X_train, y_train, batch_size, num_epochs, alpha=0.8):
        """
         Given a batch of data, and the necessary hyperparameters,
         this function trains the neural network by adjusting the weights and biases of all the layers.
         :param X: Array of input [n_samples,input_dimensions]
         :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
         the desired (true) class.
         :param batch_size: number of samples in a batch
         :param num_epochs: Number of times training should be repeated over all input data
         :param alpha: Learning rate
         :return: None
         """
        dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        dataset = dataset.batch(batch_size)
        number_of_layers = len(self.weights)
        for epoch in range(num_epochs):
            for datapoints, (x, y) in enumerate(dataset):
                with tf.GradientTape(persistent=True) as tape:
                    predicted_value = self.predict(x)
                    loss = self.calculate_loss(y, predicted_value)
                for num_layer in range(number_of_layers):
                    old_weights, old_bias = tape.gradient(loss, [self.weights[num_layer].weights,
                                                              self.weights[num_layer].biases])
                    delta_weight = alpha * old_weights
                    delta_bias = alpha * old_bias
                    self.weights[num_layer].weights.assign_sub(delta_weight)
                    self.weights[num_layer].biases.assign_sub(delta_bias)
                del tape

    def calculate_percent_error(self, X, y):
        """
        Given input samples and corresponding desired (true) output as indexes,
        this method calculates the percent error.
        For each input sample, if the predicted class output is not the same as the desired class,
        then it is considered one error. Percent error is number_of_errors/ number_of_samples.
        Note that the predicted class is the index of the node with maximum output.
        :param X: Array of input [n_samples,input_dimensions]
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
        the desired (true) class.
        :return percent_error
        """

        tensor_predicted_value = tf.math.argmax(self.predict(X), axis=1, output_type=tf.dtypes.int32)
        percent_error = 1 - ((tf.math.count_nonzero(tf.math.equal(tf.constant(y), tensor_predicted_value)).numpy()) / y.shape[0])
        return percent_error

    def calculate_confusion_matrix(self, X, y):
        """
        Given input samples and corresponding desired (true) outputs as indexes,
        this method calculates the confusion matrix.
        :param X: Array of input [n_samples,input_dimensions]
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
        the desired (true) class.
        :return confusion_matrix[number_of_classes,number_of_classes].
        Confusion matrix should be shown as the number of times that
        an image of class n is classified as class m.
        """
        tensor_predicted_value = tf.math.argmax(self.predict(X), axis=1, output_type=tf.dtypes.int32)
        confusionTensor = tf.math.confusion_matrix(tf.constant(y), tensor_predicted_value, num_classes=10).numpy()
        return confusionTensor


class Multi_NN(object):
    def __init__(self, inputDimensions=2, numberOfNodes=0, transferFunction="linear"):
        self.inputDimensions = inputDimensions
        self.numberOfNodes = numberOfNodes
        self.transferFunction = transferFunction.lower()
        initial_weights = np.random.randn(self.inputDimensions, self.numberOfNodes)
        initail_bias = np.random.randn(1, self.numberOfNodes)
        self.weights = tf.Variable(initial_weights)
        self.biases = tf.Variable(initail_bias)

    def predict(self, X):
        net_value = tf.matmul(X, self.weights)
        net_value = net_value + self.biases
        if self.transferFunction == 'linear':
            return net_value
        elif self.transferFunction == 'sigmoid':
            return tf.math.sigmoid(net_value)
        elif self.transferFunction == 'relu':
            return tf.nn.relu(net_value)



