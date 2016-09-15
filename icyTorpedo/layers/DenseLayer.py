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
            hidden_units, 
            linearity=Sigmoid,
            **kwargs):
        super(DenseLayer, self).__init__(**kwargs)

        self.num_units = hidden_units
        self.linearity = linearity() 

        self.initialise_weights()

    def initialise_weights(self):

        # Weights
        self.W = np.random.randn(self.input_shape, self.num_units) 

        # Biases
        self.b = np.random.randn(1, self.num_units)

    def h_x(self):
        """Compute the non linearised activations
        
        Parameters
        -----------
        
        None

        Returns
        -----------

        h(x) = sum(wx) + b

        """
        self.h = np.dot(self.input_layer.a, self.W) + self.b
        return self.h


    def a_h(self):
        """Compute the feedforward calculations for the layer

        a = linearity(h_x)
        """
        self.a = self.linearity(self.h_x())
        return self.a
