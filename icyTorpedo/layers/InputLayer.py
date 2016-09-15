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
    def __init__(self, num_units=None, name=None, **kwargs):

        self.input_layer = None

        self.num_units = num_units

        #if inputs is not None:
        #    self.input_shape, self.num_units = inputs.shape
        #else:
        #    self.input_shape, self.num_units = (None, None)

        self.name = name

    def set_inputs(self, input_values):

        # Activations g(h)
        self.a = input_values
