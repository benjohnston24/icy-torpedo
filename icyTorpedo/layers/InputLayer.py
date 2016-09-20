#! /usr/bin/env python
# -*- coding: utf-8 -*-
# S.D.G

# Imports
import numpy as np
from .baseLayer import baseLayer

__author__ = 'Ben Johnston'
__revision__ = '0.1'
__date__ = 'Wednesday 14 September  22:26:43 AEST 2016'
__license__ = 'MPL v2.0'


class InputLayer(baseLayer):
    """Input Layer class 

    Parameters
    -----------

    inputs :  Input values as an np.array with dimensions (#samples, #features) 
    name   :  A string of the name for the layer
    """

    #def __init__(self, inputs=None, name=None, **kwargs):
    def __init__(self, num_units=None, name="Input Layer", **kwargs):

        self.input_layer = None

        self.num_units = num_units

        self.name = name

    def set_inputs(self, input_values):

        num_samples, self.num_units = input_values.shape
        # Activations for input layer are just the inputs to the network
        # Add bias units
        self.a = np.hstack((np.ones((num_samples, 1)), input_values))
        self.input_shape = self.num_units + 1  # Add the bias units

    def __str__(self):

        return "%s: %d" % (self.name, self.num_units)
