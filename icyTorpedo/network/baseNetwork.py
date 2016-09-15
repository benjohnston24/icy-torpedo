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
__date__ = 'Thursday 15 September  10:04:49 AEST 2016'
__license__ = 'MPL v2.0'


class baseNetwork(object):

    def __init__(self,
                 targets=None, 
                 eta=FixedRate(0.01),
                 costfunction=SquaredError,
                 max_epochs=20,
                 *args, **kwargs):

        self.costfunction = costfunction() 
        self.targets=targets
        self.eta = eta
        self.max_epochs = max_epochs

        self.x_train, self.y_train = (None, None)
        self.x_valid, self.y_valid = (None, None)
        self.x_test, self.y_test = (None, None)

    def network_defn(self):
        """Built from Layer classes, ensure the outpu layer is called self.network"""
        pass  # pragma: no cover

    def forwardprop(self):
        """Iterate through each of the layers and compute the activations at each node"""
        for layer in iterlayers(self.network):
            layer.a_h() # Compute the activations

    def backprop(self):
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
        layer = self.network
        delta = self.costfunction.prime(output=layer.a, target=self.targets) * \
                self.network.linearity.prime(layer.h)
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

        layer = self.network

        # Get the learning rate, if the learning rate changes we only want
        # to do this once per weight update
        learning_rate = self.eta()

        while layer.input_layer is not None:
            layer.W -= learning_rate * layer.dc_dw
            layer.b -= learning_rate * layer.dc_db

            layer = layer.input_layer


    def train(self):
        """Train the neural network"""

        import pdb;pdb.set_trace()
        for epoch in range(self.max_epochs):

            self.forwardprop()
            printable_str = ["%0.4f, " % x for x in self.network.a[0]]
            print("{:^10}{}".format(epoch, "".join(printable_str)))

            self.backprop()
            self.updateweights()

    def predict(self, inputs):

        self.l_in.set_inputs(inputs)
        self.forwardprop()

        return self.network.a
