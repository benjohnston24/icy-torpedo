#! /usr/bin/env python
# -*- coding: utf-8 -*-
# S.D.G

# Imports
import numpy as np


__author__ = 'Ben Johnston'
__revision__ = '0.1'
__date__ = 'Wednesday 14 September  22:21:35 AEST 2016'
__license__ = 'MPL v2.0'


class baseLayer(object):
    """Base layer class

    Parameters
    -----------

    input_layer :  The input layer to this layer.  A layer inheriting from baseLayer
    name   :  A string of the name for the layer
    """

    def __init__(self, input_layer=None, num_units=None, name=None):

        #self.input_shape = input_layer.num_units
        self.input_layer = input_layer

        self.name = name

        self.num_units = num_units  # Add bias units


def iterlayers(output_layer):
    """Iterate through the layer list from not including inputs to outputs

    Parameters
    ------------

    output_layer :  The final layer of the network
    
    
    """

    layers = []

    layer = output_layer

    while layer.input_layer:
        layers.append(layer)
        layer = layer.input_layer

    layers.reverse()

    for layer in layers:
        yield layer
