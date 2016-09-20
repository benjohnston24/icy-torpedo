#! /usr/bin/env python
# -*- coding: utf-8 -*-
# S.D.G

# Imports
import numpy as np
from .baseLayer import baseLayer
from icyTorpedo.linearities import Sigmoid

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
            linearity=Sigmoid,
            name="Dense Layer",
            *args,
            **kwargs):
        super(DenseLayer, self).__init__(
                name=name,
                num_units=num_units,
                *args, 
                **kwargs)

        self.linearity = linearity() 

        self.initialise_weights()

    def initialise_weights(self):

        # Weights
        # Produce a normal distribution of mean 0 and standard deviation 0.4
        # Include biases within the weights.  Biases are the first column of the 
        # inputs
        # Add 1 for the biases
        self.W = np.random.randn(self.input_layer.num_units + 1, self.num_units) * 0.4 


    def h_x(self):
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
        ## Add the biases
        inputs = np.hstack((
            np.ones((self.input_layer.a.shape[0],1)),
            self.input_layer.a))
        self.h = np.dot(inputs, self.W)
        #self.h = np.dot(self.input_layer.a, self.W)
        return self.h


    def a_h(self):
        """Compute the feedforward calculations for the layer

        a = linearity(h_x)
        """
        self.a = self.linearity(self.h_x())
        return self.a

    def __str__(self):

        return "%s: %d [linearity: %s]" % \
                (self.name, self.num_units, self.linearity.name)
