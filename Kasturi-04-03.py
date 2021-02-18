# Kasturi, Chandra Shekhar
# 1001-825-454
# 2020-11-09
# Assignment-04-03



import pytest
import numpy as np

import os
import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Flatten, InputLayer, Input
from tensorflow.keras.datasets import cifar10
from cnn import CNN


def test_train():
    cnn = CNN()
    batch_size=10
    num_epochs=10
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    samples = 100

    X_train = X_train[0:samples, :]
    X_train = X_train.astype('float32') / 255

    test_samples = 20
    X_test = X_test[0:test_samples, :]
    X_test = X_test.astype('float32') / 255

    from tensorflow.keras.utils import to_categorical

    y_train = to_categorical(y_train, 10)
    y_train = y_train[0:samples]
    y_test = to_categorical(y_test, 10)
    y_test = y_test[0:test_samples]
    cnn.add_input_layer(shape=(32, 32, 3))
    cnn.append_conv2d_layer(num_of_filters=64, kernel_size=(3, 3), activation='relu', name="conv1")
    cnn.append_conv2d_layer(num_of_filters=32, kernel_size=(3, 3), activation='relu', name="conv2")
    cnn.append_flatten_layer(name="flat1")
    cnn.append_dense_layer(num_nodes=10, activation="relu", name="dense1")
    cnn.set_optimizer(optimizer="SGD")
    cnn.set_loss_function(loss="hinge")
    cnn.set_metric(metric='accuracy')
    LossList = cnn.train(X_train=X_train, y_train=y_train, batch_size=batch_size, num_epochs=num_epochs)

    assert LossList is not None




def test_evaluate():
    cnn = CNN()
    batch_size = 10
    num_epochs = 10
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    samples = 100

    X_train = X_train[0:samples, :]
    X_train = X_train.astype('float32') / 255

    test_samples = 10
    X_test = X_test[0:test_samples, :]
    X_test = X_test.astype('float32') / 255

    from tensorflow.keras.utils import to_categorical

    y_train = to_categorical(y_train, 10)
    y_train = y_train[0:samples, :]
    y_test = to_categorical(y_test, 10)
    y_test = y_test[0:test_samples, :]
    cnn.add_input_layer(shape=(32, 32, 3))
    cnn.append_conv2d_layer(num_of_filters=64, kernel_size=(3, 3), activation='relu', name="conv1")
    cnn.append_conv2d_layer(num_of_filters=32, kernel_size=(3, 3), activation='relu', name="conv2")
    cnn.append_flatten_layer(name="flat1")
    cnn.append_dense_layer(num_nodes=10, activation="relu", name="dense1")
    cnn.set_optimizer(optimizer="SGD")
    cnn.set_loss_function(loss="hinge")
    cnn.set_metric(metric='accuracy')
    ListofLoss = cnn.train(X_train=X_train, y_train=y_train, batch_size=None, num_epochs=num_epochs)

    (loss, metric) = cnn.evaluate(X=X_test, y=y_test)
    
    assert loss < 5
    assert metric < 2
