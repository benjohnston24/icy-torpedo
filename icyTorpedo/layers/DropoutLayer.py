#! /usr/bin/env python
# -*- coding: utf-8 -*-
# S.D.G

"Dropout Layer"

# Imports
import numpy as np
from .DenseLayer import DenseLayer


__author__ = 'Ben Johnston'
__revision__ = '0.1'
__date__ =
__license__ = 'MPL v2.0'


class DropoutLayer(DenseLayer):
    """Drop out layer for a neural network
    Randomly set the outputs of the previous layers to zero

    Parameters
    -----------

    input_layer :  The input to this layer.  A layer inheriting from baseLayer
        or a tuple of the shape of the layer
    rate        :  The dropout rate for the layer [default is 0.5, i.e. set 50%
        of outputs to zero]
    name        :  A string of the name for the layer


    Returns
    -----------

    None

    """

    def __init__(self,
            input_layer=None,
            rate=0.5,
            name="Dropout Layer",
            *args,
            **kwargs,
            )
        super(DropoutLayer, self).__init__(
                name=name,
                num_units=input_layer.num_units,
                linearity=Linear()
                *args,
                **kwargs,
                )

        self.rate = rate
