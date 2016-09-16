#! /usr/bin/env python
# -*- coding: utf-8 -*-
# S.D.G

# Imports
from icyTorpedo.layers import iterlayers
from icyTorpedo.costfunctions import SquaredError
from icyTorpedo.learningrates import FixedRate
import numpy as np


__author__ = 'Ben Johnston'
__revision__ = '0.1'
__date__ = 'Friday 16 September  20:55:50 AEST 2016'
__license__ = 'MPL v2.0'


class baseNetwork(object):

    def __init__(self,
                 network_layers=None,
                 targets=None, 
                 train_data=(None, None),
                 valid_data=(None, None),
                 test_data=(None, None),
                 eta=FixedRate(0.01),
                 costfunction=SquaredError,
                 max_epochs=20,
                 regression=False,
                 verbose=False,
                 *args, **kwargs):

        # Get the architecture of the network
        self.network_layers = network_layers

        # Define the output layer as the last layer in the list
        self.output_layer = self.network_layers[-1]

        # Determine the input layer by traversing the inputs of the output layer
        layer = self.output_layer

        while layer.input_layer is not None:
            layer = layer.input_layer

        # Define the input layer as the first layer in the list
        self.input_layer = layer 

        # Define characteristics of the network
        self.cost_function = costfunction() 
        self.targets=targets
        self.eta = eta
        self.max_epochs = max_epochs
        self.verbose = verbose

        # Reserve some variables for storing training data
        self.x_train, self.y_train = train_data 
        self.x_valid, self.y_valid = valid_data 
        self.x_test, self.y_test = test_data 

        # Flag to indicate if the problem is a regression problem
        self.regression = regression

    def forwardprop(self):
        """Iterate through each of the layers and compute the activations at each node"""
        for layer in iterlayers(self.output_layer):
            layer.a_h() # Compute the activations

    def backprop(self, targets):
        """Execute back propogation
        
        First compute the rate of the error at the output 
        delta_o = dC/da * d_linearity 
                = costfunction' * linearity'

        dC_db = delta_o; Change of cost with respect to bias
        dC_dw = np.dot(activation_prev_layer, delta_o)

        The error at other layers can be calculated by

        delta_l = np.dot(weights_prev_layer, delta_prev_layer) * linearity'(h_layer)

        """

        # Start at the output layer
        layer = self.output_layer
        delta = self.cost_function.prime(output=layer.a, target=targets) * \
                self.output_layer.linearity.prime(layer.h)
        layer.delta = delta
        layer.dc_db = delta
        layer.dc_dw = np.dot(layer.input_layer.a.T, delta) 
        w_1 = layer.W
        layer = layer.input_layer

        # Backprop over remaining layers
        while layer.input_layer is not None:
            # delta = (w^l+1 * delta^(l+1)) * sigma'(z^l)
            delta = np.dot(delta, w_1.T) * layer.linearity.prime(layer.h)

            layer.delta = delta
            layer.dc_db = delta
            layer.dc_dw = np.dot(layer.input_layer.a.T, delta) 

            w_1 = layer.W
            layer = layer.input_layer

    def updateweights(self):
        """Update the weights of the network in each layer:

        W -= eta * dC/dw
        b -= eta * db/dw
        """

        layer = self.output_layer

        # Get the learning rate, if the learning rate changes we only want
        # to do this once per weight update
        learning_rate = self.eta()

        while layer.input_layer is not None:
            layer.W -= learning_rate * layer.dc_dw
            layer.b -= learning_rate * layer.dc_db

            layer = layer.input_layer


    def train(self):
        """Train the neural network
        
        Simple version, apply each of the training examples in the same order per iteration
        
        """

        for epoch in range(self.max_epochs):

            train_err = 0

            # Implement training
            for x_train, y_train in zip(self.x_train, self.y_train):

                x_train = x_train.reshape((1, -1))
                y_train = y_train.reshape((1, -1))

                # Apply the sample
                self.input_layer.set_inputs(x_train)

                # Update the weights using training set
                self.forwardprop()
                self.backprop(y_train)
                self.updateweights()

                # Calculate the training error
                train_pred = self.predict(x_train)

                train_err += self.cost_function(output=train_pred,
                                                target=y_train)

            valid_err = 0
            correct_class = 0

            # Check against validation set
            for x_valid, y_valid in zip(self.x_valid, self.y_valid):

                x_valid = x_valid.reshape((1, -1))
                y_valid = y_valid.reshape((1, -1))
                
                valid_pred = self.predict(x_valid)

                valid_err += self.cost_function(output=valid_pred,
                                                target=y_valid)
                 
                # If this is a categorisation problem determine if correctly labeled
                if not self.regression and self.output_layer.a.argmax() == y_train.argmax():
                    correct_class += 1

            import pdb;pdb.set_trace()

        if self.verbose:
            printable_str = ["%0.4f, " % x for x in self.output_layer.a[0]]
            print("{:^10}{}".format(epoch, "".join(printable_str)))

    def predict(self, inputs):

        self.input_layer.set_inputs(inputs)
        self.forwardprop()

        return self.output_layer.a
