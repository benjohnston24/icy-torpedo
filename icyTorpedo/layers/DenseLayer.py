#! /usr/bin/env python
# -*- coding: utf-8 -*-
# S.D.G

# Imports
import numpy as np
from .baseLayer import baseLayer
from icyTorpedo.linearities import Sigmoid, Linear

__author__ = 'Ben Johnston'
__revision__ = '0.1'
__date__ = 'Wednesday 14 September  22:22:09 AEST 2016'
__license__ = 'MPL v2.0'


class DenseLayer(baseLayer):
    """Dense Layer class aka standard neural network layer

    Parameters
    -----------

    inputs :  The input to this layer.  A layer inheriting from baseLayer
        or a tuple of the shape of the layer
    name   :  A string of the name for the layer


    Returns
    -----------

    None

    """
    def __init__(self, 
            num_units=1,
            linearity=Sigmoid(),
            bias=1,
            name="Dense Layer",
            *args,
            **kwargs):
        super(DenseLayer, self).__init__(
                name=name,
                num_units=num_units,
                *args, 
                **kwargs)

        self.linearity = linearity

        self.initialise_weights(bias)


    def initialise_weights(self, bias=1):

        """Initialise the weights for the layer and assign to self.W

        Parameters
        -----------
        
        bias :  Include bias units in the weights matrix.  If bias units not required, set to 0

        Returns
        -----------

        None
        """



        self.bias_units = bias

        # Weights
        # Produce random numbers between -0.5 and 0.5 
        # Include biases within the weights.  Biases are the first column of the 
        # inputs
        weights_shape = (self.input_layer.num_units + bias, self.num_units)

#        self.W = (np.random.randn(weights_shape[0], weights_shape[1]) - 0.5) * 0.1
        self.W = np.random.uniform(-0.05, 0.05, weights_shape)


    def h_x(self, *args, **kwargs):
        """Compute the non linearised activations
        
        Parameters
        -----------
        
        None

        Returns
        -----------

        h(x) = sum(wx)

        Note: bias units units are included as the first column of non 
        linearised activations

        """
        inputs = np.hstack((
            np.ones((self.input_layer.a.shape[0],1)),
            self.input_layer.a))
 
        # If the linearity is linear and pseudo-inverse selected
        if isinstance(self.linearity, Linear) and self.linearity.pseudo_inverse \
                and (kwargs['targets'] is not None):
            # If the output layer is linear use the pseudo inverse to calculate weights
            # W = TA+ where W are the weights of the output layer, T the target values
            # and A+ the Moore-Penrose inverse of the hidden layer activations
            # P. de Chazal, J. Tapson and A. van Schaik, "A comparison of extreme learning machines 
            # and back-propagation trained feed-forward networks processing the mnist database," 
            # 2015 IEEE International Conference on Acoustics,
            # Speech and Signal Processing (ICASSP), South Brisbane, QLD, 2015, pp. 2165-2168.
            # doi: 10.1109/ICASSP.2015.7178354
            # Add the bias units to the input
            a_plus = np.linalg.pinv(inputs)

            # W = TA+
            self.W = np.dot(a_plus, kwargs['targets'])

        ## Add the biases
        self.h = np.dot(inputs, self.W)
        return self.h


    def a_h(self, *args, **kwargs):
        """Compute the feedforward calculations for the layer

        a = linearity(h_x)
        """
        self.a = self.linearity(self.h_x(*args, **kwargs))

        return self.a

    def __str__(self):

        return "%s: %d [linearity: %s]" % \
                (self.name, self.num_units, self.linearity.name)
